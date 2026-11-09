"""Bounded explicit manual translation candidates, always freshly validated.

An experiment on supplied data, not a pose-ID rule in the placement engine.
"""
import argparse,itertools,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def main():
 p=argparse.ArgumentParser(__doc__)
 for key in ('out','config','refs'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--shot',required=True);a=p.parse_args()
 import yaml
 cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
 os.environ.update(FARM_ROOT=cfg['farm_root'],FARM_COMFY=cfg['comfy'])
 sys.path[:0]=[str(ROOT/'tools/scene3d/posev2'),str(ROOT/'tools/scene3d/plate')]
 import manual_pose,place,profiles,shot_spec,pose_fit as pf,run_v2
 directory=a.out/'pose_round/V2'/a.shot;r=json.loads((directory/'fit.json').read_text(encoding='utf-8'))
 ref=pf.load_ref(a.refs/(a.shot+'.ref.npz'));spec=shot_spec.ShotSpec.load(directory/'shot_spec.json')
 prompt=next(x for x in run_v2.pr.prompts() if x['id']==a.shot);loc=run_v2.loc_id(prompt['loc'])
 plate=run_v2.attach_geometry_assets(pf.Plate(cfg['location_asset_dirs'][loc]));profile=profiles.LocationProfile.load(loc,Path(cfg['farm_root']))
 X,F=pf.load_obj_mesh(directory/'body.obj');K=np.load(directory/'label_joints.npz')['kp'];groups=[]
 with np.load(directory/'fit_state.npz') as d:
  for item in d['groups']:
   name,surf,role=str(item).split(':');groups.append((name,d['g_'+name],surf,role))
 base=dict(X=X,KP=K,reg=pf.regions(ref['V'],ref['kp']),groups=groups,report=r)
 H=r['body_dims']['H'];horizontal=[-.1*H,-.02*H,0,.02*H,.1*H];vertical=[-.01*H,0,.01*H]
 grid=sorted(itertools.product(horizontal,horizontal,vertical),key=lambda v:np.linalg.norm(v))
 rows=[];best=None
 for dx,dy,dz in grid:
  values=dict(body_dx=dx,body_dy=dy,body_dz=dz)
  A,J,checked=manual_pose.revalidate(base,ref,plate,profile,spec,run_v2.table_seed(prompt,plate),H,values)
  failures=[q for q in checked['validation']['checks'] if not q['ok']]
  score=({'ok':0,'partial':1,'out_of_frame':3,'contact_fail':4,'wrong_surface':5}.get(checked['status'],6),len(failures),
      float(checked.get('back_contact_gap_m') or 0),np.linalg.norm([dx,dy,dz]))
  rows.append(dict(values=values,status=checked['status'],failures=failures,score=list(score)))
  if best is None or score<best[0]:best=(score,A,J,checked,values)
  if checked['status']=='ok':break
 report=dict(selected=best[4],status=best[3]['status'],evaluated=len(rows),candidates=rows)
 (directory/'manual_candidate_search.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 if best[0][0]>=5:raise RuntimeError('No supported candidate; previous geometry retained')
 checked=best[3];checked['review_candidate']='Historical V2 shape plus explicit bounded manual translation; fresh full validator. trace/ belongs to rejected automatic candidate.'
 manual_pose.write_transformed(directory,best[1],best[2],F,checked)
 print(json.dumps({k:report[k] for k in ('selected','status','evaluated')}),flush=True)

if __name__=='__main__':main()
