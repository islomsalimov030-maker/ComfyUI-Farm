"""Measurement glue (07.11d): stage comp2 outputs + target pose25 (shifted into the plan crop) and run pose_metrics.py."""
import json, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, 'tools/scene3d/plate')
import pose_round as pr
COMFY = Path(os.environ['FARM_COMFY']); ROOT = Path(os.environ['FARM_ROOT']) / 'pose_round'
tag, picks = sys.argv[1], sys.argv[2].split(',')          # picks: M/id[:plan]
stage = ROOT / f'pck_{tag}'; maps = stage / '_maps'
shutil.rmtree(stage, ignore_errors=True)
loc = {p['id']: p['loc'] for p in pr.prompts()}
info = {}
for pk in picks:
    mid, _, plan = pk.partition(':'); m, pid = mid.split('/')
    name = f"{m}_{pid}{'_' + plan if plan else ''}"
    src = COMFY / 'output' / 'plate3d_poses' / name / 'comp2_s101.png'
    d = stage / 'img' / name; d.mkdir(parents=True); shutil.copy(src, d / f'{name}.png')
    base = COMFY / 'input' / f"pose_{m}_m{loc[pid]}_s2" / pid
    t = json.loads((base / 'pose25.json').read_text())
    if plan:
        pj = json.loads((COMFY / 'input' / f"pose_{m}_m{loc[pid]}_{plan}_s2" / pid / 'plan.json').read_text())
        x0, y0, x1, y1 = pj['rect_render']
        t['keypoints_body25'] = [[k[0] - x0, k[1] - y0, *k[2:]] for k in t['keypoints_body25']]
        t['size'] = [x1 - x0, y1 - y0]
        info[name] = {'rect': pj['rect_render'], 'zoom': pj['zoom'], 'fig_h_crop_px': pj['figure_bbox'][3] - pj['figure_bbox'][1]}
    (maps / name).mkdir(parents=True); (maps / name / 'pose25.json').write_text(json.dumps(t))
out = stage / 'pck.json'
subprocess.run([os.environ['FARM_PY_POSE'], '-I', '-u', 'tools/scene3d/plate/pose_metrics.py', '--glob', str(stage / 'img' / '*' / '*.png'),
                '--maps', str(maps), '--out', str(out), '--weights', os.environ['FARM_DWPOSE'],
                '--thresholds', 'tools/scene3d/plate/quality_round_local.yaml'], check=True)
res = {r['scene']: {'pck_strict': round(r['pck_strict'], 2), 'people': r['people'], **info.get(r['scene'], {})} for r in json.loads(out.read_text())}
(stage / 'summary.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res))
