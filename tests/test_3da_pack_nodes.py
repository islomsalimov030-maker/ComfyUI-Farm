"""Thin bridge tests; no server, subprocess work, or GPU."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

PATH = Path(__file__).resolve().parents[1] / 'comfyui/ComfyUI-3DA/nodes.py'
spec = importlib.util.spec_from_file_location('pack_nodes', PATH)
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class NodeTests(unittest.TestCase):
    def test_registered_io(self):
        self.assertEqual(set(nodes.NODE_CLASS_MAPPINGS), {'3DALocation', '3DAPlacePose', '3DAProp', '3DAH3Remote'})
        self.assertEqual(len(nodes.PlacePose.RETURN_TYPES), 7)
        self.assertEqual(len(nodes.CLASSES), 9)

    def test_error_does_not_raise_and_returns_status(self):
        image = torch.zeros((1, 16, 16, 3))
        with patch.object(nodes, 'invoke', side_effect=RuntimeError('boom')):
            loc, plate, objects, preview = nodes.Location().execute(image, 'x')
            self.assertIs(preview, image)
            self.assertEqual(json.loads(loc)['status'], 'error')
            self.assertIs(plate, image)
            result = nodes.PlacePose().execute(loc, image, 'stand', 'floor', '', '', '', 1.62, '', 'grey', 1., 101)
            self.assertEqual(len(result), 7)
            self.assertIn('status=error', result[5])
            self.assertEqual(result[2].shape, (1, 64, 64))

    def test_place_block_on_fail_only_accepted_statuses_forward_prompt(self):
        image = torch.ones((1, 16, 16, 3))
        mask = torch.ones((1, 16, 16))
        statuses = ('ok', 'partial', 'contact_fail', 'wrong_surface', 'no_target',
                    'out_of_frame', 'unknown', '', {'status': 'ok'})
        for status in statuses:
            with self.subTest(status=status):
                payload = dict(status=status, h3_prompt='SFW geometry prompt', playblast='p', init='i',
                               strength='m', silhouette='s', debug='d', report={'kept': True})
                with patch.object(nodes, 'invoke', return_value=payload), \
                     patch.object(nodes, 'load_image', return_value=image), \
                     patch.object(nodes, 'load_mask', return_value=mask):
                    result = nodes.PlacePose().execute('{}', image, 'stand', 'floor', '', '', '',
                                                       1.62, '', 'grey', 1., 101, block_on_fail=True)
                accepted = isinstance(status, str) and status in ('ok', 'partial')
                self.assertEqual(result[4], 'SFW geometry prompt' if accepted else '')
                self.assertIs(result[0], image)
                self.assertIs(result[1], image)
                self.assertIs(result[2], mask)
                self.assertIs(result[6], image)
                self.assertIn('kept', result[5])
                if not accepted:
                    with patch.object(nodes, 'config', return_value={'repo': str(PATH.parents[2]),
                                      'h3_remote': {}}), patch.object(nodes, 'remote_client') as client:
                        remote = nodes.H3Remote().execute(image, image, image, image, mask, result[4], 101)
                    self.assertIn('Empty H3 prompt', remote[2])
                    client.assert_not_called()

    def test_location_preview_slot_and_fallback(self):
        plate, preview = torch.zeros((1, 4, 4, 3)), torch.ones((1, 4, 4, 3))
        self.assertEqual(nodes.Location.RETURN_NAMES, ('location', 'plate', 'objects', 'objects_preview'))
        for present in (True, False):
            payload = dict(location='{}', plate='plate', objects=[{'id': 'surf_1'}])
            if present:
                payload['objects_preview'] = 'preview'
            with patch.object(nodes, 'invoke', return_value=payload), patch.object(
                    nodes, 'load_image', side_effect=lambda p: preview if p == 'preview' else plate):
                result = nodes.Location().execute(plate, 'new')
            self.assertIs(result[1], plate)
            self.assertIs(result[3], preview if present else plate)
            self.assertEqual(json.loads(result[2]), payload['objects'])

    def test_manual_defaults_forward_prompt_and_all_new_fields(self):
        image, mask = torch.ones((1, 4, 4, 3)), torch.ones((1, 4, 4))
        payload = dict(status='contact_fail', h3_prompt='test payload', playblast='p', init='i',
                       strength='m', silhouette='s', debug='d')
        for fields in ({}, dict(body_dx=.3, body_dy=-.2, body_dz=.4, body_dyaw=20.,
                                body_dpitch=1., body_droll=-2., pose_text='test payload', props='[]')):
            with patch.object(nodes, 'invoke', return_value=payload) as call, \
                    patch.object(nodes, 'load_image', return_value=image), \
                    patch.object(nodes, 'load_mask', return_value=mask):
                result = nodes.PlacePose().execute('{}', image, 'stand', 'floor', '', '', '',
                                                   1.62, '', 'grey', 1., 101, **fields)
            self.assertEqual(result[4], 'test payload')
            request = call.call_args.args[1]
            self.assertIs(request['block_on_fail'], False)
            self.assertEqual(request['seed'], 101)
            self.assertEqual(request['props'], '[]')
            for name in ('body_dx', 'body_dy', 'body_dz', 'body_dyaw', 'body_dpitch', 'body_droll'):
                self.assertEqual(request[name], fields.get(name, 0.))
            self.assertEqual(request['pose_text'], fields.get('pose_text', ''))

    def test_invalid_transform_or_props_never_invokes_cli(self):
        image = torch.ones((1, 4, 4, 3))
        for fields in ({'body_dx': float('nan')}, {'body_droll': float('inf')},
                       {'props': '{}'}, {'props': '[{}]'}, {'block_on_fail': 'false'}):
            with patch.object(nodes, 'invoke') as call:
                result = nodes.PlacePose().execute('{}', image, 'stand', 'floor', '', '', '',
                                                   1.62, '', 'grey', 1., 101, **fields)
            call.assert_not_called()
            self.assertIn('status=error', result[5])

    def test_prop_json_chain_and_library_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'head').mkdir()
            (root / 'head' / 'prop.json').write_text(json.dumps(dict(name='Head', size_m=.3,
                                               description_en='', prop_strength=.5)), encoding='utf-8')
            args = dict(prop_id='head', anchor='char_head', x=.3, y=0., z=.4,
                        yaw=0., pitch=0., roll=0., scale=1.)
            with patch.object(nodes, 'prop_library', return_value=root):
                self.assertEqual(nodes.prop_choices(), ['head'])
                result = nodes.Prop().execute(**args)[0]
                entries = json.loads(nodes.Prop().execute(**args, props_in=result)[0])
                self.assertEqual(len(entries), 2)
                self.assertEqual(entries[0]['prop_strength'], .35)
                self.assertEqual(entries[0]['anchor'], 'char_head')
                # Explicit widget value, not an undocumented metadata sentinel.
                self.assertEqual(json.loads(nodes.Prop().execute(**args, prop_strength=.5)[0])[0]['prop_strength'], .5)
                for bad in ({'prop_id': '../head'}, {'prop_id': 'missing'}, {'anchor': 'bone'},
                            {'scale': 0.}, {'x': float('nan')}, {'roll': float('inf')},
                            {'prop_strength': -1.}, {'props_in': '{}'}, {'props_in': '[{}]'}):
                    with self.subTest(bad=bad), self.assertRaises((ValueError, OSError)):
                        nodes.Prop().execute(**{**args, **bad})
                (root / 'head' / 'prop.json').write_text('{"name": 1}')
                self.assertEqual(nodes.prop_choices(), [''])
                with self.assertRaises(ValueError):
                    nodes.Prop().execute(**args)
        self.assertEqual(nodes.Prop.INPUT_TYPES()['required']['prop_strength'][1]['default'], .35)

    def test_timeout_does_not_raise_in_node(self):
        image = torch.zeros((1, 16, 16, 3))
        with patch.object(nodes, 'invoke', side_effect=subprocess.TimeoutExpired('CLI', 600)):
            result = nodes.Location().execute(image, 'x')
            self.assertIn('600', result[2])

    def test_last_thirty_lines_only(self):
        msg = nodes.error_text(RuntimeError('\n'.join('line' + str(i) for i in range(100))))
        self.assertNotIn('line69\n', msg)
        self.assertIn('line70\n', msg)
        self.assertIn('line99', msg)

    def test_diagnostics_redact_secrets(self):
        with patch.dict(nodes.os.environ, {'FARM_AUTH': 'sensitive-test-value'}):
            msg = nodes.error_text('failed sensitive-test-value https://person:password@example.com')
        self.assertNotIn('sensitive-test-value', msg)
        self.assertNotIn('person:password', msg)

    def test_invoke_timeout_terminates_worker_tree(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            script = repo / 'tools/scene3d/pack_cli.py'
            script.parent.mkdir(parents=True)
            script.write_text('# stub')
            cfg = dict(repo=td, farm_root=td, python='python')
            proc = MagicMock()
            proc.pid = 123456
            proc.communicate.side_effect = [subprocess.TimeoutExpired('CLI', 600), ('', '')]
            with patch.object(nodes, 'config', return_value=cfg), \
                 patch.object(nodes.subprocess, 'Popen', return_value=proc), \
                 patch.object(nodes.subprocess, 'run') as stop, \
                 patch.object(nodes.os, 'killpg', create=True) as killpg:
                with self.assertRaisesRegex(RuntimeError, 'process-tree termination requested'):
                    nodes.invoke('location', {'location_id': 'x'}, {})
                if nodes.os.name == 'nt':
                    self.assertEqual(stop.call_args.args[0], ['taskkill', '/PID', '123456', '/T', '/F'])
                else:
                    killpg.assert_called_once()

    def test_image_mask_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'x.png'
            nodes.save_image(torch.ones((1, 3, 4, 3)), p)
            self.assertEqual(nodes.load_image(p).shape, (1, 3, 4, 3))
            self.assertEqual(nodes.load_mask(p).shape, (1, 3, 4))
            self.assertEqual(float(nodes.load_mask(p).min()), 1.)
            with self.assertRaises(ValueError):
                nodes.save_image(torch.ones((2, 3, 4, 3)), p)


if __name__ == '__main__':
    unittest.main()
