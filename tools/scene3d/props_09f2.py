"""Bounded 09f2 verification using existing geometry CLI + H3 Remote. SFW only."""
from pathlib import Path
import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

REPO = Path(__file__).resolve().parents[2]
STATE = Path('C:/work/scene3d_work/farm/props_09f2')
STATE.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools/scene3d/plate'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def composite_mask(strength):
    """CPU reference of core ThresholdMask/GrowMask/ImageBlur, for metrics only."""
    binary = np.asarray(strength) > 0
    grown = binary_dilation(binary, iterations=16)
    return gaussian_filter(grown.astype(np.float32), sigma=8., radius=8, mode='mirror')


def prepare():
    import yaml
    cfg = yaml.safe_load(Path('C:/work/ComfyUI-next/custom_nodes/ComfyUI-3DA/config.yaml').read_text())
    ids = sorted(p.parent.name for p in (Path(cfg['farm_root']) / 'props').glob('*/prop.json'))
    if not ids:
        raise ValueError('No imported SFW prop')
    # Input paths/shot selection belong to the experiment, never to placement rules.
    for shot in ('Q01', 'Q04', 'Q12', 'T01', 'T05'):
        specs = json.loads((REPO / 'eval/shot_spec_gold.json').read_text())['specs']
        spec = next(s for s in specs if s['id'].startswith(shot + '_'))
        old = STATE.parent / 'props_09e' / (shot + '_prop_request.json')
        if old.exists():
            req = json.loads(old.read_text())
        else:
            location = spec.get('loc_id', spec.get('location_id'))
            if not location:
                # Resolve existing prepared room from the experiment runner mapping.
                rows = json.loads((REPO / 'tools/scene3d/plate/pose_round_prompts.json').read_text())['prompts']
                row = next(r for r in rows if r['id'] == spec['id'])
                import pose_round
                location = Path(pose_round.PLATE.format(row['loc'])).name
            req = dict(config=cfg, location=str(Path(cfg['farm_root']) / 'locations' / location / 'profile.json'),
                       pose_ref=str(Path('C:/work/scene3d_work/refs_v2') / (spec['id'] + '.png')),
                       pose_class=spec['pose_class'], target_object=spec['target_object'],
                       support_parts=','.join(spec['support_parts']), secondary_contacts='',
                       height_m=1.62, body_json=str(Path(cfg['farm_root']) / 'characters/stacy/body.json'),
                       proxy='grey', strength=1., seed=101, block_on_fail=False)
        req['config'] = cfg
        req['props'] = '[]'
        req.update(body_dx=0., body_dy=0., body_dz=0., body_dyaw=0., body_dpitch=0., body_droll=0.)
        save(STATE / (shot + '_pose_request.json'), req)
        if shot in ('Q01', 'Q04'):
            prop = dict(prop_id=ids[0], anchor='char_head' if shot == 'Q01' else 'world',
                        x=0., y=0., z=.35, yaw=0., pitch=0., roll=0., scale=1., size_m=.30, prop_strength=.35)
            req['props'] = json.dumps([prop])
            if shot == 'Q01':
                req['body_dx'] = -.10
            save(STATE / (shot + '_prop_request.json'), req)
    print('prepared SFW requests', flush=True)


def preview(shot, kind='prop'):
    req = STATE / f'{shot}_{kind}_request.json'
    out = STATE / f'{shot}_{kind}_preview.json'
    t = time.monotonic()
    with (STATE / f'{shot}_{kind}.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-I', str(REPO / 'tools/scene3d/pack_cli.py'),
                                 'place', '--request', str(req), '--output', str(out)], cwd=REPO,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUTF8='1'))
        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True, timeout=15)
            raise
    result = json.loads(out.read_text(encoding='utf-8'))
    result['preview_wall_s'] = round(time.monotonic() - t, 3)
    save(out, result)
    print(shot, kind, result.get('status'), result.get('timings'), result.get('error'), flush=True)
    return result


def remote(shot):
    from farm.comfy_client import ComfyClient
    result = json.loads((STATE / f'{shot}_prop_preview.json').read_text(encoding='utf-8'))
    if result.get('status') == 'error':
        raise ValueError(result['error'])
    req = json.loads((STATE / f'{shot}_prop_request.json').read_text())
    client = ComfyClient('http://127.0.0.1:8187')
    dest = STATE / f'{shot}_remote'
    dest.mkdir(exist_ok=True)
    if (dest / 'submitted.json').exists():
        raise ValueError('Already submitted; inspect history before retry')
    graph, inputs = {}, {}
    try:
        if any(client.queue().get(k) for k in ('queue_running', 'queue_pending')):
            raise ValueError('PC queue not empty')
        images = {key: Path(result[key]) for key in ('playblast', 'init', 'strength')}
        images.update(plate=Path(result['init']).parent / 'plate.png',
                      ref_char=Path('C:/work/scene3d_work/comfy_mirror/input/0003.png'))
        for key, path in images.items():
            name = client.upload_image(path, name=f'09f2_{shot}_{key}_{time.time_ns()}.png', subfolder='3da_props')
            graph[key] = {'class_type': 'LoadImage', 'inputs': {'image': name}, '_meta': {'title': 'Input ' + key}}
            inputs[key] = [key, 0]
        graph['mask'] = {'class_type': 'ImageToMask', 'inputs': {'image': ['strength', 0], 'channel': 'red'}, '_meta': {'title': 'Strength white'}}
        inputs.update(strength=['mask', 0], prompt=result['h3_prompt'], seed=101)
        graph['remote'] = {'class_type': '3DAH3Remote', 'inputs': inputs, '_meta': {'title': '3DA H3 Remote'}}
        for key, slot in [('final', 0), ('raw', 1)]:
            graph[key] = {'class_type': 'SaveImage', 'inputs': {'images': ['remote', slot], 'filename_prefix': f'3DA_props09f2/{shot}_{key}'}, '_meta': {'title': key}}
        graph['info'] = {'class_type': 'PreviewAny', 'inputs': {'source': ['remote', 2]}, '_meta': {'title': 'Remote info'}}
        pid = client.submit(graph)
        save(dest / 'submitted.json', {'prompt_id': pid})
        t = time.monotonic()
        while time.monotonic() - t < 1200:
            entry = client.history(pid)
            if entry and client._finished(entry):
                client._check_entry(pid, entry)
                break
            time.sleep(2)
        else:
            raise TimeoutError('Own H3 not cancelled; inspect prompt id')
        save(dest / 'history.json', entry)
        for key in ('final', 'raw'):
            d = entry['outputs'][key]['images'][0]
            (dest / (key + '.png')).write_bytes(client.view(d['filename'], d.get('subfolder', ''), d.get('type', 'output')))
        save(dest / 'result.json', {'total_remote_wall_s': round(time.monotonic() - t, 3), 'info': entry['outputs']['info']})
        print(shot, 'remote', entry['outputs']['info'], flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'preview', 'remote'])
    ap.add_argument('shot', nargs='?')
    ap.add_argument('--kind', default='prop', choices=['prop', 'pose'])
    args = ap.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'preview': preview(args.shot, args.kind)
    else: remote(args.shot)
