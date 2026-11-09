"""Fetch own completed job9 and replay only production local composition.

No upload or POST. Transport is replaced with the exact saved server tensors.
"""
import argparse,importlib.util,json,os,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/scene3d'))

def main():
 p=argparse.ArgumentParser(__doc__)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
 import requests,yaml
 cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'));r=a.out/'isolated_Q01'
 pid=json.loads((r/'studio_submitted.json').read_text())['prompt_id']
 with requests.Session() as client:
  client.auth=(os.environ['COMFY_USER'],os.environ['COMFY_PASS']);url=cfg['h3_remote']['url'].rstrip('/')
  response=client.get(url+'/history/'+pid,timeout=30);response.raise_for_status();entry=response.json()[pid]
  (a.out/'h3_history'/f'{pid}.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8')
  assert entry['status']['status_str']=='success'
  graph=entry['prompt'][2]
  for title,name in [('Сохранить · финальная вклейка','body_h3_composite.png'),('Сохранить · raw H3','raw.png')]:
   key=next(k for k,n in graph.items() if n.get('_meta',{}).get('title')==title)
   d=entry['outputs'][key]['images'][0];response=client.get(url+'/view',params=d,timeout=30);response.raise_for_status();(r/name).write_bytes(response.content)
  facts=json.loads((a.out/'h3_actual_mapping.json').read_text(encoding='utf-8'))
  engine=next(n for n in graph.values() if n.get('_meta',{}).get('title')=='Движок · r2v')
  facts['jobs']=[x for x in facts['jobs'] if x['prompt_id']!=pid]+[dict(prompt_id=pid,status='success',
    reference_inputs={k:v for k,v in engine['inputs'].items() if k.startswith('ref_images.')},mode='rendered_separately')]
  (a.out/'h3_actual_mapping.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
 spec=importlib.util.spec_from_file_location('recompose_nodes',ROOT/'comfyui/ComfyUI-3DA/nodes.py');nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)
 guides=json.loads((r/'body_guides.json').read_text(encoding='utf-8'))
 body={k:nodes.load_image(guides[k]) for k in ('plate','init','playblast','prop_rgb')}
 body.update(strength=nodes.load_mask(guides['strength']),prop_alpha=nodes.load_mask(guides['prop_alpha']),prompt=guides['prompt'],render_prompt=guides['render_prompt'])
 job=Path(guides['plate']).parent
 repair=nodes.load_image(job/'prop_repair.png');mask=nodes.load_mask(job/'prop_repair_mask.png')
 strength=nodes.load_mask(job/'strength.png')
 old_info=json.loads((r/'output_14_0.txt').read_text(encoding='utf-8'))
 # Keep the actual tight-support score; it failed .9999 despite zero changed
 # outside pixels because SSIM windows straddle the visible prop boundary.
 if (r/'result.json').exists() and not (r/'result_before_background_guard.json').exists():
  (r/'result_before_background_guard.json').write_bytes((r/'result.json').read_bytes())
 final_h3=nodes.load_image(r/'body_h3_composite.png');raw_h3=nodes.load_image(r/'raw.png')
 def transport(*args,**kwargs):return final_h3,raw_h3,json.dumps(old_info)
 with patch.object(nodes,'config',return_value=cfg),patch.object(nodes,'remote_run',side_effect=transport):
  final,raw,info=nodes.H3Remote().execute(body['plate'],nodes.load_image(a.out/'h3/Q01/ref_char.png'),body['playblast'],body['init'],strength,
    guides['render_prompt'],101,prop_repair=repair,prop_repair_mask=mask,body_guides=body,prop_handling='rendered_separately')
 detail=json.loads(info);assert detail['prompt_id']==pid
 nodes.save_image(final,r/'final.png')
 from prop_composite import restore_isolated
 _,support,guard=restore_isolated(final_h3,body['prop_rgb'],body['prop_alpha'],repair,mask,body['strength'])
 Image.fromarray(support.astype('uint8')*255).save(r/'final_composite_support.png')
 Image.fromarray(guard.astype('uint8')*255).save(r/'background_guard.png')
 detail['verification']='live full manual job9 transport; updated local compositor replayed through production H3Remote.execute with exact server tensors; no new H3'
 (r/'recomposition.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Fetched own history and replayed local composition',pid,detail['background_guard_pixels'])

if __name__=='__main__':main()
