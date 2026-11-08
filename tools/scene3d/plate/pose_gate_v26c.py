"""09.11c Q04 VAE-relative gates; isolated state, no automatic resubmission.

R has no sampler. G0/G1 use the unchanged D1_s101 recipe and core final
composite. Saved PNGs (not JPEG evidence) are the metric inputs. GPU execution
requires --execute-gpu; any failed gate blocks every later stage.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

import pose_gate_v26 as old

r = old.r
STATE = r.farm_env.ROOT / 'pose_round' / 'h3_v26_9c_pack'
OUT_SUB = 'plate3d_v26_9c_pack'
RECIPE = r.STATE / 'Q04_G0.api.json'


def by_title(graph, title):
    matches = [n for n in graph.values() if n.get('_meta', {}).get('title') == title]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one node titled {title!r}')
    return matches[0]


def prune(graph):
    keep = set()
    def visit(key):
        if key in keep:
            return
        node = graph[key]
        keep.add(key)
        for value in node['inputs'].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                visit(str(value[0]))
    for key, node in graph.items():
        if node['class_type'] in ('SaveImage', 'PreviewAny'):
            visit(key)
    return {k: v for k, v in graph.items() if k in keep}


def gate_graph(recipe, name, strength_file):
    """Transform our saved recipe by its unique titles, never remote node IDs."""
    if name not in ('R', 'G0', 'G1'):
        raise ValueError(name)
    g = copy.deepcopy(recipe)
    title = lambda key: '3DA v26 · ' + key
    latent_title = title('D1_lat') + ' · h3_img2img_latent v0.2.0'
    latent = by_title(g, latent_title)
    latent['inputs']['always_encode'] = True
    by_title(g, title('strength_image'))['inputs']['image'] = strength_file
    save = by_title(g, title('save'))
    save['inputs']['filename_prefix'] = f'{OUT_SUB}/{name}/raw'
    decode = by_title(g, title('D1_decode'))
    if name == 'R':
        # Explicit VAE roundtrip: the encoded latent feeds the SAME decoder.
        latent_key = next(k for k, n in g.items() if n is latent)
        decode['inputs']['samples'] = [latent_key, 0]
    def link(key):
        node = by_title(g, title(key))
        return [next(k for k, n in g.items() if n is node), 0]
    def add(key, typ, **inputs):
        if key in g:
            raise ValueError(key)
        g[key] = {'class_type': typ, 'inputs': inputs,
                  '_meta': {'title': f'3DA 09.11c · {key}'}}
    add('final_composite', 'ImageCompositeMasked', destination=link('plate_scale'),
        source=link('D1_select'), mask=link('strength_mask'), x=0, y=0, resize_source=False)
    add('save_composite', 'SaveImage', images=['final_composite', 0],
        filename_prefix=f'{OUT_SUB}/{name}/composite')
    for key, source in (('init', 'init_scale'), ('plate', 'plate_scale')):
        add('save_' + key, 'SaveImage', images=link(source), filename_prefix=f'{OUT_SUB}/{name}/{key}')
    add('mask_rgb', 'MaskToImage', mask=link('strength_mask'))
    add('save_mask', 'SaveImage', images=['mask_rgb', 0], filename_prefix=f'{OUT_SUB}/{name}/mask')
    return prune(g)


def scores(raw, init, composite, plate, mask, baseline):
    outside = np.asarray(mask.convert('L')) == 0
    if not outside.any():
        raise ValueError('No pixels outside strength map')
    return {'ssim': old.gate_metric(raw, init),
            'outside_ssim': old.gate_metric(raw, init, outside),
            'composite_outside_ssim': old.gate_metric(composite, plate, outside),
            'outside_pixels': int(outside.sum()),
            'outside_changed_pixels': int(np.any(np.asarray(composite) != np.asarray(plate), axis=2)[outside].sum()),
            'threshold': None if baseline is None else baseline - .01}


def run(name, execute_gpu=False):
    if not execute_gpu:
        raise RuntimeError('GPU requires --execute-gpu')
    gates_file = STATE / 'gates.json'
    gates = r.load_json(gates_file) if gates_file.exists() else {}
    if name in gates:
        raise RuntimeError('Saved submission exists; inspect, never resubmit blindly')
    if name != 'R' and gates.get('R', {}).get('status') != 'success':
        raise RuntimeError('R required first')
    if name == 'G1' and not gates.get('G0', {}).get('passed'):
        raise RuntimeError('G0 failed/missing: STOP')
    recipe = r.load_json(RECIPE)
    pid = r.base.full_id('Q04')
    directory = r.scene_dir(pid)
    rel = f'{r.loc_of(pid)}_s2/{pid}/gate_09c_{name}_strength.png'
    local_mask = r.farm_env.COMFY / 'input' / rel
    size = tuple(r.ts.SIZE)
    if name == 'G1':
        sil = r.silhouette(directory / 'silhouette.png', size)
        strength = r.proxy.strength_map(sil, 1.0, gates['R']['cell_px'])
        Image.fromarray(np.rint(strength * 255).astype('uint8')).convert('RGB').save(local_mask)
    else:
        Image.new('RGB', size).save(local_mask)
    graph = gate_graph(recipe, name, rel)
    # The saved G0 recipe scales strength from render to output. Here strength
    # is built at output resolution, so this resize is an exact identity.
    oi = r.base.qr.api('/object_info')
    r.check_object_info(graph, oi)
    queue = r.base.qr.api('/queue')
    if queue['queue_running'] or queue['queue_pending']:
        raise RuntimeError('Queue occupied; nothing submitted')
    for node in graph.values():
        if node['class_type'] == 'LoadImage':
            path = Path(node['inputs']['image'])
            r.base.remote_sync.upload(r.farm_env.COMFY / 'input' / path, path.parent.as_posix())
    r.save_json(STATE / f'Q04_{name}.api.json', graph)
    prompt_id = r.ab.post(graph)
    gate = {'prompt_id': prompt_id, 'status': 'submitted', 'node_commit': '38ade3f', 'recipe': str(RECIPE)}
    gates[name] = gate
    r.save_json(gates_file, gates)
    previous_state = r.STATE
    r.STATE = STATE
    try:
        gate['status'] = r.wait_strict(prompt_id)
    finally:
        r.STATE = previous_state
        r.save_json(gates_file, gates)
    if gate['status'] != 'success':
        raise RuntimeError(f'{name}: {gate["status"]}; STOP')
    history = r.base.qr.api('/history/' + prompt_id)[prompt_id]
    r.save_json(STATE / f'history_{name}.json', history)
    outputs = history['outputs']
    def image(key):
        info = outputs[key]['images']
        if len(info) != 1:
            raise RuntimeError(f'{key}: expected one selected image, got {len(info)}')
        im = info[0]
        path = r.farm_env.COMFY / 'output' / im.get('subfolder', '') / im['filename']
        return Image.open(path).convert('RGB')
    raw, init, comp, plate, mask = [image(k) for k in ('save', 'save_init', 'save_composite', 'save_plate', 'save_mask')]
    grid = old.parse_info(outputs['latent_info'])
    if grid['cell_px'] != 16:
        raise RuntimeError('cell != 16 px; STOP, coordinator decision required')
    result = scores(raw, init, comp, plate, mask, gates.get('R', {}).get('ssim'))
    gate.update(result, cell_px=grid['cell_px'], latent_info=grid['info'])
    times = {kind: detail['timestamp'] for kind, detail in history['status'].get('messages', []) if 'timestamp' in detail}
    gate['timestamps_ms'] = times
    if 'execution_start' in times and 'execution_success' in times:
        gate['elapsed_s'] = (times['execution_success'] - times['execution_start']) / 1000
    if name == 'R':
        gate['passed'] = True  # R is a measured baseline, not an absolute-quality gate.
    else:
        score = result['ssim'] if name == 'G0' else result['outside_ssim']
        gate['passed'] = score >= result['threshold'] and result['composite_outside_ssim'] >= .995
    target = r.base.REPO / 'docs/experiments/img'
    old.evidence(raw, init, target / f'gate_{name}.jpg')
    if name == 'G1':
        old.evidence(comp, init, target / 'gate_G1_composite.jpg')
    r.save_json(gates_file, gates)
    print(json.dumps(gate, ensure_ascii=False), flush=True)
    if not gate['passed']:
        raise SystemExit(f'{name} FAIL: STOP session')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('gate', choices=('R', 'G0', 'G1'))
    ap.add_argument('--execute-gpu', action='store_true')
    args = ap.parse_args()
    run(args.gate, args.execute_gpu)
