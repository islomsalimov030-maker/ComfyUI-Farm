"""09.11a Q04 gates. G0 RGB SSIM vs init, then G1 background outside compact map.

Requires installed h3_img2img_latent v0.2.0; never resubmits saved gates.
python pose_gate_v26.py G0 --execute-gpu
python pose_gate_v26.py G1 --execute-gpu
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_h3_v26 as r


def init_image(directory, size):
    plate = Image.open(directory / 'plate_rgb.png').convert('RGB').resize(size, Image.Resampling.LANCZOS)
    figure = Image.open(directory / 'playblast.png').convert('RGB').resize(size, Image.Resampling.LANCZOS)
    mask = Image.open(directory / 'silhouette.png').convert('L').resize(size, Image.Resampling.NEAREST)
    return Image.composite(figure, plate, mask)


def parse_info(info):
    text = '\n'.join(info.get('text', []))
    cell = re.search(r'cell_px=([0-9.]+)', text)
    if not cell:
        raise ValueError('No measured cell_px in node info; do not infer latent grid')
    return {'cell_px': float(cell[1]), 'info': text}


def gate_metric(image, init, region=None):
    # RGB SSIM: equal-channel average of spatial SSIM, no grayscale shortcut.
    a, b = np.asarray(image, float), np.asarray(init, float)
    region = np.ones(a.shape[:2], bool) if region is None else region
    return float(np.mean([r.ssim_masked(a[..., c], b[..., c], region) for c in range(3)]))


def evidence(image, init, path):
    from PIL import ImageDraw
    difference = Image.fromarray(np.clip(np.abs(np.asarray(image, int) - np.asarray(init, int)) * 4, 0, 255).astype('uint8'))
    sheet = Image.new('RGB', (image.width * 3, image.height + 24), '#181818')
    draw = ImageDraw.Draw(sheet)
    for i, (im, name) in enumerate(((image, 'output'), (init, 'init composite'), (difference, '|difference| x4'))):
        sheet.paste(im, (i * image.width, 24))
        draw.text((i * image.width + 8, 5), name, fill='white')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=94)


def run_gate(name, execute_gpu=False):
    if not execute_gpu:
        raise SystemExit('Pass --execute-gpu for the single authorized Q04 gate')
    gates_file = r.STATE / 'gates.json'
    gates = r.load_json(gates_file) if gates_file.is_file() else {}
    if name in gates:
        raise RuntimeError('Saved gate exists: inspect it; never resubmit blindly')
    if name == 'G1' and not gates.get('G0', {}).get('passed'):
        raise RuntimeError('G0 not passed; STOP G1 and D')
    pid = r.base.full_id('Q04')
    r.do_prep([pid], str(r.farm_env.ROOT / 'characters/stacy/body.json'), prepare_strength=name == 'G1')
    directory = r.scene_dir(pid)
    if name == 'G0':
        Image.new('RGB', Image.open(directory / 'silhouette.png').size).save(directory / 'G0_strength.png')
    graph = r.graph_variant(pid, 'D1')
    if name == 'G0':
        graph['strength_image']['inputs']['image'] = f'{r.loc_of(pid)}_s2/{pid}/G0_strength.png'
    graph['save']['inputs']['filename_prefix'] = f'{r.OUT_SUB}/{pid}/{name}'
    oi = r.base.qr.api('/object_info')
    node_inputs = oi['H3LatentFromImage']['input']
    if 'always_encode' not in node_inputs.get('required', {}) or 'mask' not in node_inputs.get('optional', {}):
        raise RuntimeError('Server node not updated to always_encode+mask; no GPU submitted')
    r.check_object_info(graph, oi)
    queue = r.base.qr.api('/queue')
    if queue['queue_running'] or queue['queue_pending']:
        raise RuntimeError('Queue occupied; gate not submitted')
    for node in graph.values():
        if node['class_type'] == 'LoadImage':
            rel = Path(node['inputs']['image'])
            r.base.remote_sync.upload(r.farm_env.COMFY / 'input' / rel,
                                      rel.parent.as_posix() if str(rel.parent) != '.' else '')
    r.save_json(r.STATE / f'Q04_{name}.api.json', graph)
    prompt_id = r.ab.post(graph)
    gate = {'prompt_id': prompt_id, 'status': 'submitted', 'node_version': '0.2.0'}
    gates[name] = gate
    r.save_json(gates_file, gates)
    gate['status'] = r.wait_strict(prompt_id)
    r.save_json(gates_file, gates)
    if gate['status'] != 'success':
        raise RuntimeError(f'{name} did not complete; stop')
    grid = parse_info(r.load_json(r.STATE / f'latent_info_{prompt_id}.json'))
    r.save_json(r.STATE / 'latent_grid.json', grid)
    path = r.base.latest(r.farm_env.COMFY / 'output' / r.OUT_SUB / pid, name)
    image = Image.open(path).convert('RGB')
    init = init_image(directory, image.size)
    init.save(directory / 'gate_init.png')
    outside = None
    if name == 'G1':
        from scipy.ndimage import distance_transform_edt
        sil = r.silhouette(directory / 'silhouette.png', image.size)
        outside = distance_transform_edt(~sil) >= 3 * grid['cell_px']
    score = gate_metric(image, init, outside)
    gate.update(ssim=score, passed=score >= (.98 if name == 'G0' else .97), file=str(path),
                cell_px=grid['cell_px'], latent_info=grid['info'],
                mask_shape=[image.height, image.width],
                silhouette_source='Blender body-only film_transparent alpha threshold 0.5')
    r.save_json(gates_file, gates)
    evidence(image, init, r.base.REPO / f'docs/experiments/img/gate_{name}.jpg')
    if name == 'G1':
        pp = r.load_json(r.STATE / 'prompts_v26.json')
        target, _ = r.target_points(pid, pp[pid]['plan'], image.size)
        r.seam_crop(image, target).save(r.base.REPO / 'docs/experiments/img/gate_G1_seam.png')
    print(json.dumps(gate, ensure_ascii=False), flush=True)
    if not gate['passed']:
        raise SystemExit(f'{name} failed; STOP pilot D, no workaround')


def run_g0(execute_gpu=False):
    return run_gate('G0', execute_gpu)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('gate', choices=('G0', 'G1'))
    ap.add_argument('--execute-gpu', action='store_true')
    args = ap.parse_args()
    run_gate(args.gate, args.execute_gpu)
