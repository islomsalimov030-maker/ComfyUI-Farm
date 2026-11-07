"""Five SFW table rows x four seeds, next8189 only; no FacePass or new weights.
Pod-side runner. Durable submission intent, frame timing, deadline, CPU QA and contact sheets.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import build_ab as ab
import quality_round as qr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import farm_env  # noqa: E402
ROOT = farm_env.ROOT
STATE = ROOT / 'table_round'
COMFY = farm_env.COMFY
SEEDS = (101, 202, 303, 404)
VARIANT = 'T_R1s_pose05'
SIZE = (1344, 768)
SIT = 'S3D_SIT01_bed_edge'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    qr.save(data, path)


def request(path, data=None):
    req = urllib.request.Request(ab.URL + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def setup_sit():
    """Add a separate table-only pose; never alter sit_upright or unrelated pod fixes."""
    p = HERE.parent / 'poses.py'
    text = p.read_text()
    if '"sit_hands_knees"' not in text:
        marker = '    "sit_upright":'
        addition = ('    # Table-only seated pose; baseline sit_upright is unchanged.\n'
                    '    "sit_hands_knees": {"contact": "seat", "steps": _sit(88, 88, 2) + [\n'
                    '        ("@reach", "L", "lowerleg01.L", (0.0, 0.03, 0.13)),\n'
                    '        ("@reach", "R", "lowerleg01.R", (0.0, 0.03, 0.13))]},\n')
        assert marker in text
        shutil.copy2(p, STATE / 'poses.before_sit.py')
        p.write_text(text.replace(marker, addition + marker, 1))
    p = ROOT / 'table_specs/mainroom3.json'
    spec = json.loads(p.read_text())
    shutil.copy2(p, STATE / 'mainroom3.before_sit.json')
    scene = next(s for s in spec['scenes'] if s['name'] == SIT)
    scene.update(pose='sit_hands_knees', seat=[0.4, 0.70])
    save(spec, p)
    save({'pose': 'sit_hands_knees', 'seat': scene['seat'], 'baseline_unchanged': 'sit_upright',
          'steps': '_sit(88,88,2) + bilateral @reach lowerleg01 (0,.03,.13)'}, STATE / 'sit_changes.json')


def render_sit():
    cmd = farm_env.bpy_cmd(HERE / 'render_plate_scene.py') + [
           '--plate', str(ROOT / 'plate/mainroom3'), '--spec', str(ROOT / 'table_specs/mainroom3.json'),
           '--mannequin', os.environ.get('FARM_MANNEQUIN', str(ROOT / 'stacy_mannequin_162_pod.blend')), '--out',
           str(COMFY / 'input/table3d_m3'), '--only', SIT, '--samples', '32']
    subprocess.run(cmd, check=True)


def prompt(spec, scene):
    text = qr.clay_prompt(spec, scene, shadow=True)
    old = 'She wears an opaque grey athletic T-shirt, opaque full-length navy trousers and white sneakers.'
    new = ('She wears an opaque white short-sleeved cotton T-shirt, light-blue straight-leg jeans '
           'and white lace-up sneakers. The clothing is fully opaque and covers her torso and legs.')
    assert old in text
    text = text.replace(old, new).replace('One single photorealistic photograph.',
        'One single photorealistic static image. <Picture 2> is only an identity reference sheet, not a layout.')
    text = text.replace('Scale anchor:', 'True-to-life scale: she is never enlarged to fill the frame. Scale anchor:')
    text = text.replace('retention_analysis:\n', 'retention_analysis:\n<Subject 1>: attribute_transfer - facial identity; wear the specified white T-shirt and light-blue jeans.\n')
    sections = ['subject_definitions:', 'summary:', 'retention_analysis:', 'detailed_description:',
                'overall_soundscape:', 'non_diegetic_music:']
    assert [text.index(s) for s in sections] == sorted(text.index(s) for s in sections)
    assert 'studio' not in text.lower()
    assert '\nThe lighting is preserved unchanged throughout.\n' in text
    assert '\nColors and saturation stay unchanged from the reference image.\n' in text
    return text


def graph_for(spec, scene, seed, maps):
    ab.LOC[:] = [maps.name]
    ab.SIZE[:] = SIZE
    recipe = dict(name=VARIANT, model='base', depth_end=1., pose_strength=.5,
                  clay_video=True, playblast=True, steps=25, control='pose')
    cfg = dict(size=SIZE, steps=25, cfg=1., models={'base': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'})
    g = qr.make_graph(spec, scene, seed, recipe, cfg)
    by_title = {n['_meta']['title']: n for n in g.values()}
    by_title['3DA quality · clay']['inputs']['image'] = f'{maps.name}/{scene["name"]}/playblast.png'
    by_title['3DA quality · r2v']['inputs']['prompt'] = prompt(spec, scene)
    by_title['3DA quality · save']['inputs']['filename_prefix'] = f'plate3d_table/{scene["name"]}/{VARIANT}_s{seed}'
    assert all(n['class_type'] != 'LoraLoaderModelOnly' for n in g.values())
    assert by_title['3DA quality · ks']['inputs']['cfg'] == 1.
    assert 'ap_depth' not in g and 'ap_pose' in g
    for n in g.values():
        for value in n['inputs'].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in g, value
    return g


def prepare():
    jobs = []
    for number in (1, 2, 3):
        maps = COMFY / f'input/table3d_m{number}'
        plate = maps / '_plate'
        plate.mkdir(exist_ok=True)
        for name in ('plate_rgb.png', 'plate.json'):
            if not (plate / name).exists():
                shutil.copy2(ROOT / f'plate/mainroom{number}' / name, plate / name)
        spec = json.loads((ROOT / f'table_specs/mainroom{number}.json').read_text())
        for scene in spec['scenes']:
            assert (maps / scene['name'] / 'playblast.png').exists()
            assert (maps / scene['name'] / 'pose25.json').exists()
        subprocess.run([sys.executable, str(HERE.parent / 'draw_h3_maps.py'), str(maps)], check=True)
        ab.LOC[:] = [maps.name]
        Path(ab.SHEETS).mkdir(exist_ok=True, parents=True)
        ab.prep(str(maps), spec)
        for scene in spec['scenes']:
            if scene['name'] == SIT:
                assert scene['pose'] == 'sit_hands_knees', 'SIT must be fixed and visually approved first'
                assert (STATE / 'sit_visual_approved.json').exists(), 'Read SIT playblast before launching'
            for seed in SEEDS:
                name = f'{scene["name"]}_{VARIANT}_s{seed}'
                graph = graph_for(spec, scene, seed, maps)
                save(graph, STATE / f'{name}.api.json')
                jobs.append((name, scene['name'], seed, graph))
    assert len(jobs) == 20
    return jobs


def generate(budget):
    jobs = prepare()
    path = STATE / 'runs.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    start = time.monotonic()
    save({'wall_start': now(), 'budget_seconds': budget, 'expected_frames': 20,
          'recipe': 'R1s + pose .5 through .5; ref2va base; 25 steps; cfg1; 1344x768; no turbo; no FacePass',
          'downloaded_models': []}, STATE / 'session.json')
    try:
        for name, scene, seed, graph in jobs:
            previous = next((r for r in rows if r['name'] == name), None)
            if previous:
                if previous['status'] == 'success':
                    continue
                raise RuntimeError(f'Existing non-success {name}; reconcile manually, do not resubmit')
            if budget - (time.monotonic() - start) < 100:
                raise RuntimeError('Budget guard: less than 100 seconds left; no further frames submitted')
            queue = request('/queue')
            if queue['queue_running'] or queue['queue_pending']:
                raise RuntimeError('next8189 queue occupied')
            row = dict(name=name, scene=scene, variant=VARIANT, seed=seed, status='submitting', wall_start=now())
            rows.append(row)
            save(rows, path)
            t = time.monotonic()
            row.update(prompt_id=ab.post(graph), status='submitted')
            save(rows, path)
            status, result = qr.wait(row['prompt_id'], min(300, max(1, budget - (time.monotonic() - start))))
            row.update(status=status, result=result, seconds=round(time.monotonic() - t, 2), wall_end=now())
            save(rows, path)
            print(name, status, row['seconds'], flush=True)
            if status != 'success':
                raise RuntimeError('Stop after non-success')
    finally:
        request('/free', {'unload_models': True, 'free_memory': True})
        save({'wall_end': now(), 'seconds': round(time.monotonic() - start, 2),
              'success': sum(r['status'] == 'success' for r in rows)}, STATE / 'generation_end.json')


def metrics():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=farm_env.FACE_PKGS,
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    for number in (1, 2, 3):
        maps = COMFY / f'input/table3d_m{number}'
        spec = json.loads((ROOT / f'table_specs/mainroom{number}.json').read_text())
        paths = []
        for scene in spec['scenes']:
            dest = STATE / f'pose_{scene["name"]}.json'
            subprocess.run([farm_env.PY_POSE, '-I', '-u', str(HERE / 'pose_metrics.py'),
                            '--glob', str(COMFY / f'output/plate3d_table/{scene["name"]}/*.png'),
                            '--maps', str(maps), '--thresholds', str(HERE / 'quality_round.yaml'),
                            '--out', str(dest)], check=True, env=env)
            paths.append(str(dest))
        subprocess.run([sys.executable, '-u', str(HERE / 'quality_report.py'), '--poses', ','.join(paths),
                        '--maps', str(maps), '--group', 'T', '--out', str(STATE / f'metrics_m{number}')],
                       check=True, env=env)
    contact()


def contact():
    from PIL import Image, ImageDraw
    rows = json.loads((STATE / 'runs.json').read_text())
    scenes = list(dict.fromkeys(r['scene'] for r in rows))
    sheet = Image.new('RGB', (480 * 4, 300 * len(scenes)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row in rows:
        if row['status'] != 'success':
            continue
        images = row['result']['images']
        im = images[0]
        path = COMFY / 'output' / im.get('subfolder', '') / im['filename']
        x, y = SEEDS.index(row['seed']) * 480, scenes.index(row['scene']) * 300
        draw.text((x + 4, y + 3), f'{row["scene"]} s{row["seed"]} {row["seconds"]:.0f}s', fill='black')
        sheet.paste(Image.open(path).convert('RGB').resize((480, 274)), (x, y + 24))
    sheet.save(STATE / 'contact.jpg', quality=92)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['setup-sit', 'render-sit', 'build', 'generate', 'metrics', 'all', 'contact'], default='all')
    p.add_argument('--budget-seconds', type=int, default=1500)
    a = p.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    if a.mode == 'setup-sit':
        setup_sit()
    elif a.mode == 'render-sit':
        render_sit()
    elif a.mode == 'build':
        prepare()
    elif a.mode == 'contact':
        contact()
    else:
        if a.mode in ('generate', 'all'):
            generate(a.budget_seconds)
        if a.mode in ('metrics', 'all'):
            metrics()
    print('TABLE_DONE', a.mode, now(), flush=True)


if __name__ == '__main__':
    main()
