"""Exercise updated Prop -> PlacePose through actual PC API, H3 muted/absent."""
from pathlib import Path
import json
import sys
import time
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from farm.comfy_client import ComfyClient

if __name__=='__main__':
    root=Path('C:/work/scene3d_work/farm/props_09e');req=json.loads((root/'Q04_prop_request.json').read_text())
    c=ComfyClient('http://127.0.0.1:8187');g=json.loads((REPO/'workflows/3DA_manual_v1.api.json').read_text())
    try:
        for title,path in [('Плита комнаты','C:/work/scene3d_work/plate/mainroom3/plate_rgb.png'),('Референс позы · SFW',req['pose_ref'])]:
            node=next(n for n in g.values() if n.get('_meta',{}).get('title')==title)
            node['inputs']['image']=c.upload_image(path,subfolder='3da_props',name='09e_node_'+Path(path).name)
        loc=next(n for n in g.values() if n['class_type']=='3DALocation');loc['inputs']['location_id']='mainroom3'
        place=next(n for n in g.values() if n['class_type']=='3DAPlacePose')
        for key in ['pose_class','target_object','height_m','body_json','proxy','strength','pose_text','block_on_fail']:
            place['inputs'][key]=req[key]
        g['prop']={'class_type':'3DAProp','inputs':json.loads(req['props'])[0], '_meta':{'title':'3DA Prop'}}
        place['inputs']['props']=['prop',0]
        # Output traversal executes only these previews, never Remote/H3.
        for key in ['save','preview_raw_H3','preview_remote_info','remote']:
            g.pop(key,None)
        t=time.monotonic();pid=c.submit(g);(root/'node_preview_submitted.json').write_text(json.dumps({'prompt_id':pid}))
        while time.monotonic()-t<600:
            e=c.history(pid)
            if e and c._finished(e):c._check_entry(pid,e);break
            time.sleep(2)
        else:raise TimeoutError('PC preview timeout')
        (root/'node_preview_history.json').write_text(json.dumps(e,ensure_ascii=False,indent=2),encoding='utf-8')
        (root/'node_preview_graph.json').write_text(json.dumps(g,ensure_ascii=False,indent=2),encoding='utf-8')
        text=e['outputs']['preview_status']['text'][0]
        status,payload=text.split('\n',1);body=json.loads(payload)
        print('actual node preview',pid,status,'wall',round(time.monotonic()-t,3),'times',body['timings'])
    finally:c.close()
