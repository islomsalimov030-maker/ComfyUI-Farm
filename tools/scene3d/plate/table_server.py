"""3D-A step 4 (07.11b): SFW test table (docs/tables/stacy_sfw_test.json, 5 rows x 4 seeds) with the photo recipe
S3 comp2 on the studio server: L_C2 (playblast _s2 as <Video 1>, base ref2va, 25 steps, no turbo + SAM3 person mask in
the same job) -> comp2 locally (person over photo x (1 - 0.8 shadow)) -> PCK -> best frame per row -> face pass.

Playblasts: render_plate_scene.py into $FARM_COMFY/input/table3d_m<N>_s2 (photo emission 2.5, M3 with gobo), specs in
$FARM_ROOT/table_specs/mainroom<N>.json (render_specs of the table; mainroom3 light = scenes_mainroom3.json, gobo).
  python table_server.py --mode build|generate|post   (source farm_env.sh, FARM_MODS = Stacy mods)
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_ab as ab  # noqa: E402
import composite_round as cr  # noqa: E402
import farm_env  # noqa: E402
import quality_round as qr  # noqa: E402
import table_round as tr  # noqa: E402

STATE = farm_env.ROOT / 'table_server'
SEEDS = (101, 202, 303, 404)
SIZE = (1344, 768)
RECIPE = dict(name='L_C2', model='base', depth_end=1.0, pose_strength=0.5, clay_video=True, playblast=True,
              playblast_suffix='_s2', steps=25, sam=True)
CFG = dict(size=SIZE, steps=25, cfg=1.0, models={'base': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'})
ORDER = ['S3D_STD01_window_cup', 'S3D_SIT01_bed_edge', 'S3D_CRO01_mat_shoelace', 'S3D_LIE01_bed_read',
         'S3D_STD02_bench_look_back']


def rows():
    out = []
    for n in (1, 2, 3):
        spec = json.loads((farm_env.ROOT / f'table_specs/mainroom{n}.json').read_text(encoding='utf-8'))
        for scene in spec['scenes']:
            out.append((f'table3d_m{n}', spec, scene))
    return sorted(out, key=lambda r: ORDER.index(r[2]['name']))


def graph(loc, spec, scene, seed):
    ab.LOC[:] = [loc]
    ab.SIZE[:] = SIZE
    g = qr.make_graph(spec, scene, seed, RECIPE, CFG)
    g['r2v']['inputs']['prompt'] = tr.prompt(spec, scene)        # table outfit + light sentence, asserted
    pre = f"plate3d_table/{scene['name']}/L_C2"
    g['save']['inputs']['filename_prefix'] = f'{pre}_s{seed}'
    g['sam_save']['inputs']['filename_prefix'] = f'{pre}_mask_s{seed}'
    return g


def build():
    jobs = []
    STATE.mkdir(parents=True, exist_ok=True)
    for loc, spec, scene in rows():
        for seed in SEEDS:
            g = graph(loc, spec, scene, seed)
            name = f"{scene['name']}_s{seed}"
            (STATE / f'{name}.api.json').write_text(json.dumps(g, indent=1), encoding='utf-8')
            jobs.append((name, loc, scene['name'], seed, g))
    return jobs


def now():
    return datetime.now(timezone.utc).isoformat()


def generate(limit):
    jobs = build()
    path = STATE / 'runs.json'
    runs = json.loads(path.read_text()) if path.exists() else []
    if farm_env.REMOTE:
        import remote_sync
        for loc in sorted({j[1] for j in jobs}):
            remote_sync.upload(farm_env.COMFY / 'input' / f'{loc}_rgb.png', '')
            for name, l, scene, seed, g in jobs:
                if l == loc and seed == SEEDS[0]:
                    remote_sync.upload(farm_env.COMFY / 'input' / f'{loc}_s2' / scene / 'playblast.png', f'{loc}_s2/{scene}')
    t_all = time.monotonic()
    sent = 0
    for name, loc, scene, seed, g in jobs:
        prev = next((r for r in runs if r['name'] == name), None)
        if prev and prev['status'] == 'success':
            continue
        if prev:
            raise SystemExit(f'{name}: previous {prev["status"]}; reconcile manually')
        if sent >= limit:
            print('[table] job limit reached', flush=True)
            break
        q = qr.api('/queue')
        if q['queue_running'] or q['queue_pending']:
            raise SystemExit('Queue is occupied; refusing to add work')
        row = dict(name=name, loc=loc, scene=scene, seed=seed, attempt=SEEDS.index(seed) + 1, wall_start=now())
        t = time.monotonic()
        row['prompt_id'] = ab.post(g)
        row['status'] = 'submitted'
        runs.append(row)
        qr.save(runs, path)
        sent += 1
        status, result = qr.wait(row['prompt_id'], 1200)
        row.update(status=status, seconds=round(time.monotonic() - t, 1), wall_end=now(),
                   images=[im for im in (result.get('images') or [])] if isinstance(result, dict) else [])
        qr.save(runs, path)
        print(f'[table] {name}: {status} {row["seconds"]}s', flush=True)
        if status != 'success':
            raise SystemExit('Stop after non-success')
    (STATE / 'generation_wall.json').write_text(json.dumps({'seconds': round(time.monotonic() - t_all, 1),
                                                             'sent': sent, 'end': now()}))


def post():
    """comp2 for every generated frame (local)."""
    out = farm_env.COMFY / 'output' / 'plate3d_table'
    inp = farm_env.COMFY / 'input'
    for loc, spec, scene in rows():
        for seed in SEEDS:
            d = out / scene['name']
            gen = sorted(d.glob(f'L_C2_s{seed}_*.png'))
            mask = sorted(d.glob(f'L_C2_mask_s{seed}_*.png'))
            if not gen or not mask:
                print('[table] missing', scene['name'], seed)
                continue
            cr.comp2(inp / f'{loc}_rgb.png', inp / f'{loc}_s2' / scene['name'] / 'shadow.png', gen[-1], mask[-1],
                     d / f'comp2_s{seed}.png')
    print('[table] comp2 done', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['build', 'generate', 'post'], required=True)
    p.add_argument('--limit', type=int, default=20, help='max server jobs in this call')
    a = p.parse_args()
    if a.mode == 'build':
        print(f'[table] {len(build())} graphs', flush=True)
    elif a.mode == 'generate':
        generate(a.limit)
    else:
        post()


if __name__ == '__main__':
    main()
