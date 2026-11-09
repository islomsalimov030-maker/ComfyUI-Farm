"""Install ready Q01 PC preset outside repo; retain generic tracked templates."""
from pathlib import Path
import json
import shutil
import time

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('C:/work/scene3d_work/farm')

def main():
    graph=json.loads((ROOT/'manual_e2e_09d/Q01/graph.json').read_text(encoding='utf-8'))
    ui=json.loads((REPO/'workflows/3DA_manual_v1.json').read_text(encoding='utf-8'))
    titles={n['_meta']['title']:n for n in graph.values()}
    for n in ui['nodes']:
        title=n.get('title','')
        if title not in titles:continue
        inputs=titles[title]['inputs']
        if n['type']=='LoadImage':
            n['widgets_values']=[inputs['image']]
            n['widgets_values_named']={'image':inputs['image']}
        elif n['type']=='3DALocation':
            n['widgets_values']=[inputs['location_id']]
            n['widgets_values_named']={'location_id':inputs['location_id']}
        elif n['type']=='3DAPlacePose':
            names=['pose_class','target_object','support_parts','secondary_contacts','instance_hint',
                   'height_m','body_json','proxy','strength']
            n['widgets_values']=[inputs[k] for k in names]
            n['widgets_values_named']={k:inputs[k] for k in names}
    dest=Path('C:/work/ComfyUI-next/user/default/workflows/3DA_manual_v1.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        backup=ROOT/'pack_backup_09d'/('manual_before_'+time.strftime('%H%M%S')+'.json')
        shutil.copy2(dest,backup)
    dest.write_text(json.dumps(ui,ensure_ascii=False,indent=2),encoding='utf-8')
    downloads=Path.home()/'Downloads/3DA_COMFY_PACK_2026-11-09d'
    downloads.mkdir(parents=True,exist_ok=True)
    shutil.copy2(dest,downloads/'3DA_manual_v1.json')
    for name in ['3DA_h3_server_v1.json','3DA_h3_server_api.json']:
        shutil.copy2(REPO/'workflows'/name,downloads/name)
    print('Ready Q01 preset:',dest)
    print('Server manual workflow copied to Downloads; no server install')

if __name__=='__main__':main()
