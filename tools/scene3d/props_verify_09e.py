"""Bounded requested SFW prop tests. Preview CLI then existing H3 Remote on PC."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
STATE=Path('C:/work/scene3d_work/farm/props_09e')


def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')


def requests():
    import yaml
    cfg=yaml.safe_load(Path('C:/work/ComfyUI-next/custom_nodes/ComfyUI-3DA/config.yaml').read_text())
    cfg['props_source_dir']='C:/Users/user/Downloads/3d_objects'
    ids=sorted(p.parent.name for p in (Path(cfg['farm_root'])/'props').glob('*/prop.json'))
    if not ids:raise ValueError('No imported SFW prop')
    for shot,loc,pid,cls,target in [('Q01','mainroom1','Q01_stand_hand_face','stand','floor'),('Q04','mainroom3','Q04_bed_side_lie','lie','bed')]:
        prop=dict(prop_id=ids[0],anchor='char_head',x=.3,y=0.,z=.4,yaw=0.,pitch=0.,roll=0.,scale=1.,prop_strength=.35)
        req=dict(config=cfg,location=str(Path(cfg['farm_root'])/'locations'/loc/'profile.json'),
                 pose_ref=str(Path('C:/work/scene3d_work/refs_v2')/(pid+'.png')),pose_class=cls,target_object=target,
                 height_m=1.62,body_json=str(Path(cfg['farm_root'])/'characters/stacy/body.json'),proxy='grey',strength=1.,seed=101,
                 block_on_fail=False,props=json.dumps([prop]),pose_text=('She stands with her left hand beside her face.' if shot=='Q01' else 'She lies on her side with her head resting on her bent arm.'),
                 body_dx=.3 if shot=='Q01' else 0.,body_dyaw=20. if shot=='Q01' else 0.)
        save(STATE/(shot+'_prop_request.json'),req)
    print('prepared Q01,Q04 head prop scale1 x+.3 z+.4; Q01 body dx+.3 yaw20')


def preview(shot):
    req=STATE/(shot+'_prop_request.json');out=STATE/(shot+'_prop_preview.json')
    command=[sys.executable,'-I',str(REPO/'tools/scene3d/pack_cli.py'),'place','--request',str(req),'--output',str(out)]
    t=time.monotonic()
    p=subprocess.Popen(command,cwd=REPO,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:code=p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=15);raise
    result=json.loads(out.read_text());result['preview_wall_s']=round(time.monotonic()-t,3);save(out,result)
    print(shot,result.get('status'),result.get('timings'),result.get('error'))


def remote(shot):
    sys.path.insert(0,str(REPO))
    from farm.comfy_client import ComfyClient
    result=json.loads((STATE/(shot+'_prop_preview.json')).read_text())
    if result.get('status')=='error':raise ValueError(result['error'])
    req=json.loads((STATE/(shot+'_prop_request.json')).read_text())
    c=ComfyClient('http://127.0.0.1:8188')
    graph={}
    try:
        if any(c.queue().get(k) for k in ('queue_running','queue_pending')):raise ValueError('PC queue not empty')
        inputs={}
        images={'plate':Path(req['config']['location_asset_dirs'][('mainroom1' if shot=='Q01' else 'mainroom3')])/'plate_rgb.png',
                'ref_char':Path('C:/work/scene3d_work/comfy_mirror/input/0003.png'),
                'playblast':Path(result['playblast']),'init':Path(result['init']),'strength':Path(result['strength'])}
        for i,(key,path) in enumerate(images.items()):
            name=c.upload_image(path,name='09e_'+shot+'_'+key+'_'+str(time.time_ns())+'.png',subfolder='3da_props')
            graph[key]={'class_type':'LoadImage','inputs':{'image':name},'_meta':{'title':'Input '+key}}
            inputs[key]=[key,0]
        graph['mask']={'class_type':'ImageToMask','inputs':{'image':['strength',0],'channel':'red'},'_meta':{'title':'Strength white'}}
        inputs.update(strength=['mask',0],prompt=result['h3_prompt'],seed=101)
        graph['remote']={'class_type':'3DAH3Remote','inputs':inputs,'_meta':{'title':'3DA H3 Remote'}}
        for key,slot in [('final',0),('raw',1)]:graph[key]={'class_type':'SaveImage','inputs':{'images':['remote',slot],'filename_prefix':'3DA_props09e/'+shot+'_'+key},'_meta':{'title':key}}
        graph['info']={'class_type':'PreviewAny','inputs':{'source':['remote',2]},'_meta':{'title':'Remote info'}}
        dest=STATE/(shot+'_remote');dest.mkdir(exist_ok=True)
        if (dest/'submitted.json').exists():raise ValueError('Already submitted; inspect history before retry')
        t=time.monotonic();pid=c.submit(graph);save(dest/'submitted.json',{'prompt_id':pid});save(dest/'graph.json',graph)
        while time.monotonic()-t<1200:
            entry=c.history(pid)
            if entry and c._finished(entry):c._check_entry(pid,entry);break
            time.sleep(2)
        else:raise TimeoutError('Own H3 not cancelled; inspect prompt id')
        save(dest/'history.json',entry)
        for key in ('final','raw'):
            d=entry['outputs'][key]['images'][0];(dest/(key+'.png')).write_bytes(c.view(d['filename'],d.get('subfolder',''),d.get('type','output')))
        save(dest/'result.json',{'total_remote_wall_s':round(time.monotonic()-t,3),'info':entry['outputs']['info']})
        print(shot,'remote complete',entry['outputs']['info'])
    finally:c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['requests','preview','remote']);p.add_argument('shot',nargs='?',choices=['Q01','Q04']);a=p.parse_args()
    {'requests':requests,'preview':lambda:preview(a.shot),'remote':lambda:remote(a.shot)}[a.mode]()
