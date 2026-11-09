"""Reapply the corrected core alpha to an existing raw H3; no new generation."""
from pathlib import Path
import json
import sys
import time
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from farm.comfy_client import ComfyClient
STATE=Path('C:/work/scene3d_work/farm/props_09f2')
if __name__=='__main__':
    shot=sys.argv[1]
    p=json.loads((STATE/f'{shot}_prop_preview.json').read_text())
    c=ComfyClient('http://127.0.0.1:8187')
    try:
        source=json.loads((REPO/'workflows/3DA_h3_server_api.json').read_text(encoding='utf-8'))
        graph={k:n for k,n in source.items() if k.startswith('composite_')}
        for key,path in {'plate':Path(p['init']).parent/'plate.png','raw':STATE/f'{shot}_remote/raw.png','strength':p['strength']}.items():
            name=c.upload_image(path,name=f'recomp_{shot}_{key}_{time.time_ns()}.png')
            graph[key]={'class_type':'LoadImage','inputs':{'image':name},'_meta':{'title':key}}
        graph['strength_mask']={'class_type':'ImageToMask','inputs':{'image':['strength',0],'channel':'red'},'_meta':{'title':'Strength'}}
        graph['final']={'class_type':'ImageCompositeMasked','inputs':{'destination':['plate',0],'source':['raw',0],'mask':['composite_mask',0],'x':0,'y':0,'resize_source':False},'_meta':{'title':'Final'}}
        graph['save']={'class_type':'SaveImage','inputs':{'images':['final',0],'filename_prefix':'3DA09f2/recomposite'},'_meta':{'title':'Save'}}
        pid=c.submit(graph);t=time.monotonic()
        while time.monotonic()-t<60:
            e=c.history(pid)
            if e and c._finished(e):c._check_entry(pid,e);break
            time.sleep(.5)
        else:raise TimeoutError('recomposite')
        d=e['outputs']['save']['images'][0]
        dest=STATE/f'{shot}_remote/final.png'
        if dest.exists() and not dest.with_name('final_before_sigma_fix.png').exists():dest.rename(dest.with_name('final_before_sigma_fix.png'))
        dest.write_bytes(c.view(d['filename'],d.get('subfolder',''),d.get('type','output')))
        (STATE/f'{shot}_remote/recomposite.json').write_text(json.dumps({'prompt_id':pid,'note':'Same raw H3; corrected physical sigma8px alpha, core nodes only'}))
        print(shot,pid,'recomposited')
    finally:c.close()
