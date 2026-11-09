"""T05 uses the existing table-seed runner; no missing profile anchor is invented."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('C:/work/scene3d_work/farm')
STATE=ROOT/'props_09f2'
PY='C:/work/ComfyUI-next/venv/Scripts/python.exe'
ENV=dict(os.environ,FARM_ROOT=str(ROOT),FARM_COMFY='C:/work/ComfyUI-next',
    FARM_BLENDER='C:/Program Files/Blender Foundation/Blender 5.0/blender.exe',
    FARM_BPY_PKGS='C:/work/venvs/blender_pkgs',FARM_MANNEQUIN='C:/work/scene3d_work/stacy_mannequin_162.blend',
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUTF8='1')
if __name__=='__main__':
    shot='T05_barre_standing_split';start=time.time();t=time.monotonic()
    for mode,extra,limit in [('place',['--refs','C:/work/scene3d_work/refs_v2_npz','--comfy','C:/work/ComfyUI-next'],600),('render',[],180)]:
        with (STATE/f'T05_{mode}.log').open('w',encoding='utf-8') as log:
            p=subprocess.Popen([PY,str(REPO/'tools/scene3d/posev2/run_v2.py'),mode,'--only',shot]+extra,env=ENV,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            try:code=p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=15);raise
            if code:raise RuntimeError(f'T05 {mode} exit {code}; see own local log')
    fit=ROOT/'pose_round/V2'/shot
    if (fit/'fit.json').stat().st_mtime<start:raise RuntimeError('Stale fit')
    report=json.loads((fit/'fit.json').read_text(encoding='utf-8'))
    files=list(Path('C:/work/ComfyUI-next/input').glob(f'pose_V2_m*_s2/{shot}/playblast.png'))
    if len(files)!=1 or files[0].stat().st_mtime<start:raise RuntimeError('Missing or stale playblast')
    result=dict(status=report['status'],playblast=str(files[0]),init=str(files[0]),debug=str(fit/'overlay.jpg'),
        report=report,timings={'total_s':round(time.monotonic()-t,3),'runner':'existing table seed'},preview_wall_s=round(time.monotonic()-t,3))
    (STATE/'T05_pose_preview.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'shot':'T05','status':report['status'],'flags':report.get('flags'),'orient':report.get('orient')},ensure_ascii=True),flush=True)
