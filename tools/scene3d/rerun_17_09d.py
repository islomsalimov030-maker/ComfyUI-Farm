"""Bounded 09d existing SFW round; preserve backups and explicit NEW/STALE."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('C:/work/scene3d_work/farm')
STATE = ROOT / 'rerun_09d'
PY = 'C:/work/ComfyUI-next/venv/Scripts/python.exe'
RUNNER = REPO / 'tools/scene3d/posev2/run_v2.py'
REFS = ROOT.parent / 'refs_v2_npz'
ENV = dict(os.environ, FARM_ROOT=str(ROOT), FARM_COMFY='C:/work/ComfyUI-next',
           FARM_BLENDER='C:/Program Files/Blender Foundation/Blender 5.0/blender.exe',
           FARM_BPY_PKGS='C:/work/venvs/blender_pkgs',
           FARM_MANNEQUIN='C:/work/scene3d_work/stacy_mannequin_162.blend',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def run(command, timeout, log):
    with log.open('w', encoding='utf-8') as stream:
        p = subprocess.Popen(command, cwd=str(REPO), env=ENV, stdout=stream, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        save(STATE / 'process.json', {'pid':p.pid, 'command':command, 'timeout_s':timeout})
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'], capture_output=True, timeout=15)
            p.wait(timeout=15)
            return 124


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE / 'started.json').exists():
        raise RuntimeError('09d run already started; do not overwrite evidence')
    roundroot = ROOT / 'pose_round/V2'
    backup = ROOT / ('pose_round/V2_before_09d_' + time.strftime('%Y%m%d_%H%M%S'))
    shutil.copytree(roundroot, backup)
    before = {f.parent.name: f.stat().st_mtime_ns for f in roundroot.glob('*/fit.json')}
    save(STATE / 'started.json', {'time':time.time(), 'backup':str(backup), 'before_mtime_ns':before,
                                'placement_timeout_s':1800})
    t0 = time.monotonic()
    code = run([PY,str(RUNNER),'place','--refs',str(REFS),'--comfy','C:/work/ComfyUI-next'],
               1800, STATE / 'place.log')
    elapsed = time.monotonic()-t0
    # IDs come from reviewed SFW gold, not arbitrary directory contents.
    gold = json.loads((REPO / 'eval/shot_spec_gold.json').read_text(encoding='utf-8'))
    if isinstance(gold, dict):
        gold = gold.get('specs', gold.get('shots', gold))
    if isinstance(gold, dict):
        ids = list(gold)
    else:
        ids = [r['id'] for r in gold]
    rows=[]
    for pid in ids:
        f=roundroot/pid/'fit.json'
        new=f.is_file() and f.stat().st_mtime_ns != before.get(pid)
        d=json.loads(f.read_text(encoding='utf-8')) if f.is_file() else {}
        rows.append({'id':pid,'freshness':'NEW' if new else 'STALE','status':d.get('status','missing'),
                     'fit_sha256':hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None,
                     'orient':d.get('orient'),'first_failure':d.get('validation',{}).get('first_failure')})
    save(STATE/'result.json', {'exit_code':code,'placement_s':elapsed,'rows':rows})
    newids=[r['id'] for r in rows if r['freshness']=='NEW' and (roundroot/r['id']/'body.obj').is_file()]
    if newids:
        render=run([PY,str(RUNNER),'render','--only',','.join(newids)],600,STATE/'render.log')
    else:
        render=None
    result=json.loads((STATE/'result.json').read_text(encoding='utf-8'))
    result['render_exit_code']=render
    save(STATE/'result.json',result)
    print(json.dumps({'exit_code':code,'placement_s':elapsed,'NEW':len(newids),'render_exit_code':render}),flush=True)


if __name__=='__main__':
    main()
