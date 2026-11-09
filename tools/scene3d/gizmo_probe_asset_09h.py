"""Load an ordinary library test GLB in the existing live core viewport."""
import argparse
import json
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--prop-id',required=True)
    a=p.parse_args();import yaml
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    source=Path(cfg['farm_root'])/'props'/a.prop_id/'model.glb'
    name='3d/3da_'+a.prop_id+'.glb'
    target=Path(cfg['comfy'])/'input'/name;shutil.copy2(source,target)
    setup=json.loads((a.out/'gizmo_setup.json').read_text(encoding='utf-8'))
    initial=json.loads((a.out/'translate_mouse_state.json').read_text())['gizmo']
    with sync_playwright() as pw:
        b=pw.chromium.connect_over_cdp('http://127.0.0.1:9351')
        page=next(p for c in b.contexts for p in c.pages if p.url.startswith('http://127.0.0.1:8187'))
        page.evaluate('''async data=>{
          const n=window.reviewNode,w=n.widgets.find(w=>w.name==='model_file');
          w.value=data.name;if(w.callback)await w.callback(data.name);
          const prop=app.graph._nodes.find(n=>n.type==='3DAProp');
          prop.widgets.find(w=>w.name==='prop_id').value=data.propId;
          n.connect(1,prop,prop.inputs.findIndex(i=>i.name==='model_3d_info'));
          n.connect(0,prop,prop.inputs.findIndex(i=>i.name==='model_3d'));
        }''',dict(name=name,propId=a.prop_id))
        page.wait_for_timeout(2000)
        state=page.evaluate('''data=>{const l=window.reviewLoad;
          l.setCameraState(data.camera);l.applyGizmoTransform(data.position,
            {x:Math.PI/6,y:-Math.PI/9,z:Math.PI/18},{x:1.1,y:.8,z:1.3});
          return {model:l.getModelInfo(),camera:l.getCameraState()};
        }''',dict(camera=setup['camera'],position=initial['position']))
        (a.out/'synthetic_setup.json').write_text(json.dumps(state,indent=2),encoding='utf-8')


if __name__=='__main__':main()
