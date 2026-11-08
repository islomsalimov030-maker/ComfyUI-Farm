"""Split verification: real local pack manifest -> existing server H3/core graph.

NOT an end-to-end test of server-installed 3DA nodes. Never runs for error or
rejected geometry. One submission per shot, saved state prevents blind retries.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pose_gate_v26c as gate

r = gate.r


def build(manifest, shot, input_dir):
    graph = r.load_json(gate.RECIPE)
    title = lambda key: '3DA v26 · ' + key
    lookup = lambda key: gate.by_title(graph, title(key))
    folder = f'3DA_manual_09c/{shot}'
    for key in ('figure_source', 'silhouette_image', 'strength_image'):
        filename = {'figure_source': 'playblast.png', 'silhouette_image': 'silhouette.png',
                    'strength_image': 'strength.png'}[key]
        lookup(key)['inputs']['image'] = folder + '/' + filename
    lookup('ref_room')['inputs']['image'] = folder + '/plate.png'
    lookup('clay')['inputs']['image'] = folder + '/playblast.png'
    lookup('r2v')['inputs']['prompt'] = manifest['h3_prompt']
    r.require_lint(manifest['h3_prompt'], True)
    # Pass the actual Place Pose init output, not a recomputed proxy variant.
    graph['pack_init'] = r.ab.node('LoadImage', image=folder + '/init.png')
    graph['pack_init']['_meta'] = {'title': '3DA split · pack init'}
    latent = next(n for n in graph.values() if n['class_type'] == 'H3LatentFromImage')
    latent['inputs']['image'] = ['pack_init', 0]
    graph['composite'] = r.ab.node('ImageCompositeMasked', destination=['plate_scale', 0],
                                  source=['D1_select', 0], mask=['strength_mask', 0], x=0, y=0, resize_source=False)
    graph['composite']['_meta'] = {'title': '3DA split · final composite'}
    lookup('save')['inputs'].update(images=['composite', 0], filename_prefix=f'{gate.OUT_SUB}/manual/{shot}/final')
    for key, source in [('raw', ['D1_select', 0]), ('init', ['pack_init', 0]),
                        ('plate', ['plate_scale', 0]), ('mask', ['strength_scale', 0])]:
        graph['save_' + key] = r.ab.node('SaveImage', images=source,
                                        filename_prefix=f'{gate.OUT_SUB}/manual/{shot}/{key}')
        graph['save_' + key]['_meta'] = {'title': '3DA split · save ' + key}
    return gate.prune(graph)


def run(manifest_file, location_file, shot, execute_gpu=False):
    if not execute_gpu:
        raise RuntimeError('--execute-gpu required')
    manifest, location = r.load_json(manifest_file), r.load_json(location_file)
    if manifest.get('status') not in ('ok', 'partial'):
        raise RuntimeError('Geometry rejected/error: no H3 submitted')
    state = gate.STATE / 'manual' / shot
    result_path = state / 'result.json'
    if result_path.exists():
        raise RuntimeError('Saved submission exists; inspect, never resubmit blindly')
    inputs = r.farm_env.COMFY / 'input/3DA_manual_09c' / shot
    inputs.mkdir(parents=True, exist_ok=True)
    size = r.ts.SIZE
    files = {k: manifest[k] for k in ('playblast', 'init', 'strength', 'silhouette')}
    files['plate'] = location['plate']
    for name, path in files.items():
        Image.open(path).convert('RGB').resize(size, Image.Resampling.NEAREST if name in ('silhouette', 'strength')
                                               else Image.Resampling.LANCZOS).save(inputs / (name + '.png'))
    graph = build(manifest, shot, inputs)
    r.check_object_info(graph)
    queue = r.base.qr.api('/queue')
    if queue['queue_running'] or queue['queue_pending']:
        raise RuntimeError('Queue busy, nothing submitted')
    for node in graph.values():
        if node['class_type'] == 'LoadImage':
            rel = Path(node['inputs']['image'])
            r.base.remote_sync.upload(r.farm_env.COMFY / 'input' / rel, rel.parent.as_posix())
    r.save_json(state / 'graph.api.json', graph)
    prompt_id = r.ab.post(graph)
    result = {'shot': shot, 'prompt_id': prompt_id, 'status': 'submitted',
              'mode': 'split local geometry -> server H3; NOT server pack end-to-end',
              'geometry_status': manifest['status'], 'geometry_timings': manifest.get('timings', {})}
    r.save_json(result_path, result)
    prev = r.STATE
    r.STATE = state
    try:
        result['status'] = r.wait_strict(prompt_id)
    finally:
        r.STATE = prev
        r.save_json(result_path, result)
    if result['status'] != 'success':
        raise RuntimeError('H3 failed; stop')
    history = r.base.qr.api('/history/' + prompt_id)[prompt_id]
    r.save_json(state / 'history.json', history)
    def image(key):
        im = history['outputs'][key]['images'][0]
        path = r.farm_env.COMFY / 'output' / im.get('subfolder', '') / im['filename']
        return Image.open(path).convert('RGB')
    final, raw, init, plate, mask = [image(k) for k in ('save', 'save_raw', 'save_init', 'save_plate', 'save_mask')]
    outside = np.asarray(mask.convert('L')) == 0
    ssim = gate.old.gate_metric(final, plate, outside)
    changed = int(np.any(np.asarray(final) != np.asarray(plate), axis=-1)[outside].sum())
    times = {k: d['timestamp'] for k, d in history['status'].get('messages', []) if 'timestamp' in d}
    result.update(outside_ssim=ssim, changed_outside_pixels=changed, passed=ssim >= .995,
                  h3_elapsed_s=(times['execution_success'] - times['execution_start']) / 1000,
                  node_timings='per-node timestamps unavailable in API history')
    gate.old.evidence(final, init, r.base.REPO / f'docs/experiments/img/manual_{shot}.jpg')
    gate.old.evidence(raw, init, state / 'raw_evidence.jpg')
    r.save_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--location', required=True)
    ap.add_argument('--shot', choices=('Q04', 'Q11'), required=True)
    ap.add_argument('--execute-gpu', action='store_true')
    a = ap.parse_args()
    run(a.manifest, a.location, a.shot, a.execute_gpu)
