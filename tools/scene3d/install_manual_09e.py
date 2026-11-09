"""Ready local Q04 preset; keep tracked template Prop bypass unchanged."""
from pathlib import Path
import json
import shutil
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('C:/work/scene3d_work/farm/props_09e')

if __name__=='__main__':
    ui=json.loads((REPO/'workflows/3DA_manual_v1.json').read_text(encoding='utf-8'))
    graph=json.loads((ROOT/'node_preview_graph.json').read_text(encoding='utf-8'))
    bytitle={n['_meta']['title']:n for n in graph.values()}
    prop=graph['prop']['inputs']
    for n in ui['nodes']:
        title=n.get('title')
        if n['type']=='3DAProp':
            n['mode']=0;values=n.get('widgets_values_named',{});values.update(prop)
            n['widgets_values_named']=values
            names=['prop_id','anchor','x','y','z','yaw','pitch','roll','scale','prop_strength']
            n['widgets_values']=[prop[k] for k in names]
        elif title in bytitle:
            inp=bytitle[title]['inputs']
            if n['type']=='LoadImage':n['widgets_values']=[inp['image']];n['widgets_values_named']={'image':inp['image']}
            elif n.get('widgets_values_named'):
                names=list(n['widgets_values_named'])
                for k in names:
                    if k in inp and not isinstance(inp[k],list):n['widgets_values_named'][k]=inp[k]
                # Preserve seed control_after_generate separately for PrimitiveInt.
                if n['type'] in ('3DALocation','3DAPlacePose'):n['widgets_values']=[n['widgets_values_named'][k] for k in names]
        if title in ['3DA H3 Remote','Сохранить · финальная вклейка','Превью · raw_H3','Статус · H3 Remote']:
            n['mode']=2 # safe ready preview first; user enables H3 explicitly
    dest=Path('C:/work/ComfyUI-next/user/default/workflows/3DA_props_Q04_09e.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(ui,ensure_ascii=False,indent=2),encoding='utf-8')
    downloads=Path.home()/'Downloads/3DA_PROPS_2026-11-09e';downloads.mkdir(parents=True,exist_ok=True)
    shutil.copy2(dest,downloads/dest.name)
    shutil.copy2(REPO/'workflows/3DA_manual_v1.json',downloads/'3DA_manual_v1.json')
    print(dest)
