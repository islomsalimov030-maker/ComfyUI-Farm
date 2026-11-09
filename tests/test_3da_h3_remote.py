"""Remote bridge with real farm client + in-memory HTTP, never a live POST."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('remote_pack_nodes', ROOT / 'comfyui/ComfyUI-3DA/nodes.py')
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'repo': str(ROOT), 'h3_remote': {'url': 'https://example.com', 'timeout_s': 20, 'poll_s': .001}}
        self.env = {'COMFY_USER': 'mock-user-only', 'COMFY_PASS': 'mock-password-only'}
        self.image = torch.zeros((1, 768, 1344, 3))
        self.mask = torch.full((1, 768, 1344), .25)
        b = io.BytesIO()
        Image.new('RGB', (1344, 768), (10, 20, 30)).save(b, format='PNG')
        self.png = b.getvalue()
        self.requests = []
        self.uploads = []
        self.submitted = None
        self.histories = 0
        self.failure = None

    def handler(self, req):
        self.requests.append(req)
        expected = 'Basic ' + base64.b64encode((self.env['COMFY_USER'] + ':' + self.env['COMFY_PASS']).encode()).decode()
        self.assertEqual(req.headers['authorization'], expected)
        if self.failure == req.url.path:
            return httpx.Response(502, text='upstream error mock-password-only ' + expected)
        if req.url.path == '/upload/image':
            content = req.read()
            self.assertIn(b'3da_manual', content)
            self.assertIn(b'name="overwrite"\r\n\r\nfalse', content)
            start = content.index(b'\x89PNG')
            end = content.index(b'\r\n--', start)
            with Image.open(io.BytesIO(content[start:end])) as im:
                self.uploads.append(np.array(im))
            return httpx.Response(200, json={'name': f'file{len(self.uploads)}.png', 'subfolder': '3da_manual'})
        if req.url.path == '/prompt':
            self.submitted = json.loads(req.content)['prompt']
            return httpx.Response(200, json={'prompt_id': 'mock-pid'})
        if req.url.path == '/history/mock-pid':
            self.histories += 1
            if self.histories == 1:
                return httpx.Response(200, json={})
            return httpx.Response(200, json={'mock-pid': {
                'status': {'completed': True, 'status_str': 'success'},
                'outputs': {k: {'images': [{'filename': k + '.png', 'subfolder': '3DA_manual', 'type': 'output'}]}
                            for k in ('save', 'save_raw')}}})
        if req.url.path == '/view':
            self.assertIn(req.url.params['filename'], ['save.png', 'save_raw.png'])
            return httpx.Response(200, content=self.png)
        self.fail('Unexpected request ' + str(req.url))

    def execute(self, handler=None):
        original = nodes.remote_client
        def factory(cfg):
            client = original(cfg)
            client.http.close()
            client.http = httpx.Client(base_url=cfg['h3_remote']['url'],
                                      transport=httpx.MockTransport(handler or self.handler),
                                      auth=(self.env['COMFY_USER'], self.env['COMFY_PASS']))
            return client
        with patch.dict(nodes.os.environ, self.env), patch.object(nodes, 'config', return_value=self.cfg), \
                patch.object(nodes, 'remote_client', side_effect=factory):
            return nodes.H3Remote().execute(self.image, self.image, self.image, self.image,
                                            self.mask, 'SFW prompt supplied by geometry', 777)

    def test_upload_submit_poll_download_and_mask_polarity(self):
        final, raw, info = self.execute()
        self.assertEqual(final.shape, self.image.shape)
        self.assertEqual(raw.shape, self.image.shape)
        self.assertAlmostEqual(final[0, 0, 0, 0].item(), 10/255, places=6)
        self.assertEqual(json.loads(info)['prompt_id'], 'mock-pid')
        self.assertEqual(len(self.uploads), 5)
        self.assertTrue(np.all(self.uploads[-1] == 64))
        self.assertEqual(self.uploads[-1].ndim, 2)  # opaque luminance, not inverted alpha
        self.assertEqual(self.submitted['D1_noise']['inputs']['noise_seed'], 777)
        self.assertEqual(self.submitted['prompt_text']['inputs']['text_a'], 'SFW prompt supplied by geometry')
        self.assertEqual(self.submitted['strength_input']['inputs']['image'], '3da_manual/file5.png')
        self.assertEqual(sum(r.url.path == '/prompt' for r in self.requests), 1)
        self.assertEqual(self.histories, 2)

    def test_http_errors_at_each_stage_do_not_raise_or_leak(self):
        for path in ('/upload/image', '/prompt', '/history/mock-pid', '/view'):
            with self.subTest(path=path):
                self.failure = path
                self.histories = 0
                final, raw, info = self.execute()
                self.assertEqual(final.shape, (1, 64, 64, 3))
                self.assertIn('status=error', info)
                self.assertIn('HTTP 502', info)
                self.assertIn('upstream error', info)
                self.assertNotIn(self.env['COMFY_PASS'], info)
                self.assertNotIn(base64.b64encode(b'mock-user-only:mock-password-only').decode(), info)

    def test_execution_error_and_missing_output(self):
        for status, outputs in [({'status_str': 'error', 'messages': [['execution_error', {'exception_message': 'bad model'}]]}, {}),
                                ({'completed': True, 'status_str': 'success'}, {})]:
            def handler(req):
                if req.url.path.startswith('/history/'):
                    return httpx.Response(200, json={'mock-pid': {'status': status, 'outputs': outputs}})
                return self.handler(req)
            _, _, info = self.execute(handler)
            self.assertIn('status=error', info)
            self.assertIn('prompt_id=mock-pid', info)

    def test_invalid_json_auth_and_timeout(self):
        def handler(req):
            if req.url.path == '/prompt':
                return httpx.Response(401, text='authentication failed')
            return self.handler(req)
        self.assertIn('HTTP 401', self.execute(handler)[2])
        def malformed(req):
            return httpx.Response(200, text='not json')
        self.assertIn('status=error', self.execute(malformed)[2])
        self.cfg['h3_remote']['timeout_s'] = .0001
        self.assertIn('NOT cancelled', self.execute()[2])
        self.assertFalse(any(r.url.path in ('/interrupt', '/queue', '/free') for r in self.requests))

    def test_local_validation_prevents_submission(self):
        with patch.object(nodes, 'config', return_value=self.cfg), patch.object(nodes, 'remote_client') as client:
            for prompt, mask in [('', self.mask), ('SFW', torch.zeros((1, 64, 64))),
                                 ('SFW', torch.full_like(self.mask, float('nan')))]:
                result = nodes.H3Remote().execute(self.image, self.image, self.image, self.image, mask, prompt, 1)
                self.assertIn('status=error', result[2])
            client.assert_not_called()

    def test_no_credentials_in_config_and_missing_env(self):
        with patch.dict(nodes.os.environ, {'COMFY_USER': '', 'COMFY_PASS': ''}):
            with self.assertRaisesRegex(ValueError, 'COMFY_USER'):
                nodes.remote_client(self.cfg)
        self.cfg['h3_remote']['url'] = 'https://user:password@example.com'
        with self.assertRaisesRegex(ValueError, 'must not contain credentials'):
            nodes.remote_client(self.cfg)


if __name__ == '__main__':
    unittest.main()
