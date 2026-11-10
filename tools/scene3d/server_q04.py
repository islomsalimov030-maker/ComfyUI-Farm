"""Supervised 3DA HTTP runs. Credentials come only from the environment.

No reboot operation is provided: a restart requires separate chat approval.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit
import uuid

import requests


def redact(value):
    text = str(value)
    for key, secret in os.environ.items():
        if secret and re.search(r'PASS|TOKEN|SECRET|AUTH|API_KEY|COMFY_USER', key, re.I):
            text = text.replace(secret, '[redacted]')
    return re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@', r'\1[redacted]@', text)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact(json.dumps(value, ensure_ascii=False, indent=2)), encoding='utf-8')


class Server:
    def __init__(self, url, out):
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
            raise ValueError('Expected a base URL without credentials/query/fragment')
        self.url = url.rstrip('/')
        self.out = out
        self.session = requests.Session()
        self.session.auth = (os.environ['COMFY_USER'], os.environ['COMFY_PASS'])

    def request(self, method, path, **kwargs):
        response = self.session.request(method, self.url + path, timeout=120,
                                        allow_redirects=False, **kwargs)
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f'{method} {path.split("?")[0]}: HTTP {response.status_code}')
        return response

    def health(self):
        stats = self.request('GET', '/system_stats').json()
        info = self.request('GET', '/object_info').json()
        names = ['3DALocation', '3DAPlacePose', '3DAProp', '3DAH3Remote', '3DANativeGuides',
                 '3DANativeComposite', '3DASceneEdits', '3DASceneExport', '3DACoreWeights']
        system = stats['system']
        result = dict(torch=system.get('pytorch_version'), comfyui=system.get('comfyui_version'),
                      nodes=[n for n in names if n in info])
        save(self.out / 'health.json', result)
        print(json.dumps(result), flush=True)
        if result['torch'] != '2.13.0+cu130' or result['comfyui'] != '0.44.2' or len(result['nodes']) != 9:
            raise RuntimeError('S1 health gate failed; stop server work')

    def wait_empty(self, minutes=40):
        deadline = time.monotonic() + minutes * 60
        while time.monotonic() < deadline:
            queue = self.request('GET', '/queue').json()
            if not queue.get('queue_running') and not queue.get('queue_pending'):
                return
            time.sleep(15)
        raise TimeoutError('Shared queue remained busy; nothing submitted')

    def logs(self, label):
        response = self.request('GET', '/internal/logs/raw')
        try:
            data = response.json()
            if isinstance(data, dict):
                data = data.get('entries', data.get('logs', data))
            entries = data if isinstance(data, list) else [data]
            lines = []
            for entry in entries:
                message = entry.get('m', entry.get('message', str(entry))) if isinstance(entry, dict) else str(entry)
                lines.extend(message.splitlines())
        except ValueError:
            lines = response.text.splitlines()
        tail = redact('\n'.join(lines[-40:]))
        (self.out / f'{label}.logs.txt').write_text(tail, encoding='utf-8')
        return tail

    def history(self, prompt_id, label, minutes):
        deadline = time.monotonic() + minutes * 60
        print(f'{label}: queued {prompt_id}', flush=True)
        while time.monotonic() < deadline:
            history = self.request('GET', '/history/' + prompt_id).json().get(prompt_id)
            if history and (history.get('status', {}).get('completed') or
                            history.get('status', {}).get('status_str') == 'error'):
                save(self.out / f'{label}.history.json', history)
                messages = history.get('status', {}).get('messages', [])
                stamps = {name: data.get('timestamp') for name, data in messages}
                start = stamps.get('execution_start')
                end = stamps.get('execution_success') or stamps.get('execution_error')
                result = dict(status=history['status']['status_str'],
                              seconds=(end - start) / 1000 if start is not None and end is not None else None)
                reported_errors = [value for output in history.get('outputs', {}).values()
                                   for value in output.get('text', [])
                                   if isinstance(value, str) and value.startswith('status=error')]
                if reported_errors:
                    result.update(graph_status=result['status'], status='error', node_errors=reported_errors)
                count = 0
                for node, output in history.get('outputs', {}).items():
                    if output.get('text'):
                        print(redact(json.dumps({'node': node, 'text': output['text']})), flush=True)
                    for item in output.get('images', []):
                        response = self.request('GET', '/view', params={
                            'filename': item['filename'], 'subfolder': item.get('subfolder', ''),
                            'type': item.get('type', 'output')})
                        suffix = Path(item['filename']).suffix
                        dest = self.out / f'{label}_{node}_{count}{suffix}'
                        dest.write_bytes(response.content)
                        count += 1
                result['images'] = count
                save(self.out / f'{label}.summary.json', result)
                self.logs(label)
                print(json.dumps(result), flush=True)
                if result['status'] != 'success':
                    errors = [d for n, d in messages if n == 'execution_error']
                    print(redact(json.dumps(errors, ensure_ascii=False)), flush=True)
                return history
            time.sleep(15)
        raise TimeoutError(f'{label} still running after {minutes} min; do not resubmit')

    def weights(self):
        self.wait_empty()
        result = self.request('POST', '/prompt', json={'prompt': {'1': {
            'class_type': '3DACoreWeights', 'inputs': {'moge': True, 'sam_3d_body': True}}},
            'client_id': str(uuid.uuid4())}).json()
        if result.get('node_errors') or not result.get('prompt_id'):
            save(self.out / 'weights.rejected.json', result)
            raise RuntimeError('CoreWeights validation failed')
        self.history(result['prompt_id'], 'weights', 40)
        for node in ['LoadMoGeModel', 'SAM3DBody_Loader']:
            info = self.request('GET', '/object_info/' + node).json()
            save(self.out / f'{node}.json', info)
            print(json.dumps(info), flush=True)

    def workflow(self, graph_path, label, h3, minutes):
        from playwright.sync_api import sync_playwright
        graph = json.loads(graph_path.read_text(encoding='utf-8-sig'))
        # Address nodes by title/type; these titles identify the three H3 branches.
        for node in graph['nodes']:
            if node.get('title') in ['Движок H3 (не трогать)', 'Сохранить · финальная вклейка', 'Сохранить · raw H3']:
                node['mode'] = 0 if h3 else 2
        self.wait_empty()
        captured = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(http_credentials={
                'username': os.environ['COMFY_USER'], 'password': os.environ['COMFY_PASS']})
            page = context.new_page()
            browser_errors = []
            page.on('pageerror', lambda error: browser_errors.append(str(error)))
            page.on('console', lambda message: browser_errors.append(message.text) if message.type == 'error' else None)
            def capture(response):
                if urlsplit(response.url).path in ('/prompt', '/api/prompt') and response.request.method == 'POST':
                    captured.append(response.json())
            page.on('response', capture)
            page.goto(self.url, wait_until='domcontentloaded', timeout=120000)
            page.wait_for_function('window.app && app.graph', timeout=120000)
            page.evaluate('(graph) => app.loadGraphData(graph)', graph)
            self.wait_empty()
            queued = page.evaluate('() => app.queuePrompt(0)')
            if not captured:
                page.wait_for_timeout(2000)
                save(self.out / f'{label}.frontend.json', {
                    'queued': queued, 'errors': browser_errors,
                    'dialogs': page.locator('[role="dialog"]').all_text_contents(),
                    'queuePromptSource': page.evaluate('() => app.queuePrompt.toString()')})
                page.screenshot(path=str(self.out / f'{label}.frontend.png'))
            browser.close()
        if len(captured) != 1 or captured[0].get('node_errors') or not captured[0].get('prompt_id'):
            save(self.out / f'{label}.rejected.json', captured)
            raise RuntimeError('Frontend submission failed; not resubmitting')
        return self.history(captured[0]['prompt_id'], label, minutes)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--out', type=Path, default=Path('.tmp_3da_port/server_Q04'))
    parser.add_argument('action', choices=['health', 'weights', 'run'])
    parser.add_argument('--graph', type=Path, default=Path('workflows/3DA_scene_server_v1.json'))
    parser.add_argument('--label', default='A')
    parser.add_argument('--h3', action='store_true')
    parser.add_argument('--minutes', type=int, default=30)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    server = Server(args.url, args.out)
    try:
        if args.action == 'health':
            server.health()
        elif args.action == 'weights':
            server.weights()
        else:
            server.workflow(args.graph, args.label, args.h3, args.minutes)
    except Exception as exc:
        print(redact(f'{type(exc).__name__}: {exc}'), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
