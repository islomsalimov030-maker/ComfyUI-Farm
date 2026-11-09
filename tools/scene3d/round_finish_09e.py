"""Bounded remainder round + render; no old fit labelled as a new result."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('C:/work/scene3d_work/farm')
ENV=dict(os.environ,FARM_ROOT=str(ROOT),FARM_COMFY='C:/work/ComfyUI-next',
         FARM_BLENDER='C:/Program Files/Blender Foundation/Blender 5.0/blender.exe',
         FARM_BPY_PKGS='C:/work/venvs/blender_pkgs',FARM_MANNEQUIN='C:/work/scene3d_work/stacy_mannequin_162.blend',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUTF8='1')
PY='C:/work/ComfyUI-next/venv/Scripts/python.exe'
RUN=str(REPO/'tools/scene3d/posev2/run_v2.py')


def bounded(args,name,timeout):
    log=ROOT/'props_09e'/name
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.Popen([PY,RUN]+args,env=ENV,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=15)
            return 124

if __name__=='__main__':
    # Six other rows have already been rerun in this session, do not repeat them.
    done={'Q04_bed_side_lie','Q10_all_fours_cat','T01_sofa_legs_tucked','T03_armchair_side_sit','T04_floor_sit_knees_up','T05_barre_standing_split'}
    gold=json.loads((REPO/'eval/shot_spec_gold.json').read_text())['specs']
    remaining=[s['id'] for s in gold if s['id'] not in done]
    start=time.time();code=bounded(['place','--refs','C:/work/scene3d_work/refs_v2_npz','--comfy','C:/work/ComfyUI-next','--only',','.join(remaining)],'remainder_place.log',1800)
    render=bounded(['render'],'all_render.log',600)
    rows=[]
    for s in gold:
        p=ROOT/'pose_round/V2'/s['id']/'fit.json';d=json.loads(p.read_text(encoding='utf-8'))
        fresh=s['id'] in done or p.stat().st_mtime>=start
        rows.append({'id':s['id'],'freshness':'NEW' if fresh else 'STALE','status':d['status'],
                     'flags':d.get('flags',[]),'first_failure':d.get('validation',{}).get('first_failure')})
    (ROOT/'props_09e/round_result.json').write_text(json.dumps({'place_exit':code,'render_exit':render,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print('round remainder',code,'render',render,'NEW',sum(r['freshness']=='NEW' for r in rows),flush=True)
