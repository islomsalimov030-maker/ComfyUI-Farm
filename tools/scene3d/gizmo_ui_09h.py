"""Actual local ComfyUI browser evidence. No request interception or test assets."""
import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=['open','capture','overview','drag','export','variant','widgets','gizmo_mode','close'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,default=9351)
    p.add_argument('--name',default='baseline')
    p.add_argument('--value',default='')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as pw:
        if a.mode=='open':
            try:urllib.request.urlopen(f'http://127.0.0.1:{a.port}/json/version',timeout=1);running=True
            except OSError:running=False
            if not running:
                cmd=[pw.chromium.executable_path,'--headless=new',f'--remote-debugging-port={a.port}',
                     f'--user-data-dir={a.out/"browser_profile"}','--window-size=1800,1200',
                     '--force-device-scale-factor=1','--no-first-run','--no-default-browser-check','about:blank']
                proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                    creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
                (a.out/'browser_pid.txt').write_text(str(proc.pid))
            for _ in range(40):
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{a.port}/json/version',timeout=1);break
                except OSError:time.sleep(.25)
        browser=pw.chromium.connect_over_cdp(f'http://127.0.0.1:{a.port}')
        if a.mode=='close':browser.close();return
        ctx=browser.contexts[0]
        page=next((p for p in ctx.pages if p.url.startswith('http://127.0.0.1:8187')),None)
        if page is None:page=ctx.new_page()
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        if a.mode=='open':
            urls=[]
            page.on('request',lambda r:urls.append(r.url))
            page.goto('http://127.0.0.1:8187',wait_until='networkidle',timeout=60000)
            page.wait_for_function('()=>!!window.app?.graph',timeout=60000)
            # Core's optional named restore avoids positional migration of
            # force-input widgets in the installed frontend. Native setting,
            # scoped to this dedicated browser profile; no custom extension.
            awaitable=page.evaluate("async()=>{await app.ui.settings.setSettingValue('Comfy.Workflow.NamedValuesRestore',true);return app.ui.settings.getSettingValue('Comfy.Workflow.NamedValuesRestore');}")
            if awaitable is not True:raise RuntimeError('Enable Comfy.Workflow.NamedValuesRestore before opening this preset')
            workflow=json.loads((a.out/'3DA_manual_gizmo.preview.json').read_text(encoding='utf-8'))
            page.evaluate('async w=>{await window.app.loadGraphData(w,true,true,"3DA manual gizmo");}',workflow)
            page.wait_for_timeout(3000)
            service=next((u for u in urls if re.search(r'/load3dService-[^/]+\.js$',u)),None)
            if not service:raise RuntimeError('Actual load3dService module not observed')
            state=page.evaluate('''async url=>{
              const n=app.graph._nodes.find(n=>n.type==='Load3DAdvanced');
              const mod=await import(url);const service=mod.t();
              window.reviewLoad=await service.getLoad3dAsync(n);window.reviewNode=n;
              app.canvas.ds.scale=.85;app.canvas.ds.offset=[-n.pos[0]+40,-n.pos[1]+70];app.canvas.setDirty(true,true);
              const l=window.reviewLoad;return {model:l.getModelInfo(),camera:l.getCameraState(),
                snapshot:await n.widgets.find(w=>w.name==='viewport_state').serializeValue(),
                canvases:[...document.querySelectorAll('canvas')].map(c=>({width:c.width,height:c.height,rect:c.getBoundingClientRect().toJSON()}))};
            }''',service)
            page.wait_for_timeout(1000)
            (a.out/'ui_open.json').write_text(json.dumps(dict(state=state,errors=errors,service=service),indent=2),encoding='utf-8')
        elif a.mode=='variant':
            state=page.evaluate('''v=>{const l=window.reviewLoad;const initial=JSON.parse(v);
                l.applyGizmoTransform(initial.position,initial.rotation,initial.scale);return l.getModelInfo();}''',a.value)
            (a.out/(a.name+'_applied.json')).write_text(json.dumps(state,indent=2))
            page.wait_for_timeout(500)
        elif a.mode=='widgets':
            page.evaluate('''v=>{const n=app.graph._nodes.find(n=>n.type==='3DAPlacePose');
              for(const [name,value]of Object.entries(JSON.parse(v))){const w=n.widgets.find(w=>w.name===name);
                if(!w)throw Error('Missing native widget '+name);w.value=value;w.callback?.(value,app.canvas,n);}
              app.canvas.setDirty(true,true);}''',a.value)
        elif a.mode=='drag':
            coords=[float(x) for x in a.value.split(',')]
            page.mouse.move(*coords[:2]);page.mouse.down();page.mouse.move(*coords[2:],steps=16);page.mouse.up();page.wait_for_timeout(500)
        elif a.mode=='gizmo_mode':
            page.evaluate('mode=>window.reviewLoad.setGizmoMode(mode)',a.value)
            page.wait_for_timeout(300)
        elif a.mode=='overview':
            page.evaluate('()=>{app.canvas.ds.scale=.4;app.canvas.ds.offset=[180,150];app.canvas.setDirty(true,true);}')
            page.wait_for_timeout(400)
        if a.mode in ('open','capture','overview','drag','variant','widgets','gizmo_mode'):
            state=page.evaluate('''async()=>{const l=window.reviewLoad,n=window.reviewNode;
              const state=await n.widgets.find(w=>w.name==='viewport_state').serializeValue();
              const vertices=[];l.modelManager.currentModel?.updateMatrixWorld(true);
              l.modelManager.currentModel?.traverse(o=>{if(!o.isMesh)return;const a=o.geometry.attributes.position;
                const step=Math.max(1,Math.ceil(a.count/512));
                for(let i=0;i<a.count;i+=step){const p=o.position.clone().fromBufferAttribute(a,i).applyMatrix4(o.matrixWorld);vertices.push([p.x,-p.z,p.y]);}});
              const c=l.renderer?.domElement||l.canvas;const rect=c?.getBoundingClientRect();
              return {snapshot:state,gizmo:l.getGizmoTransform(),vertices_world_zup:vertices,rect:rect?.toJSON(),
                captured:await l.captureScene(1344,768)};}''')
            for key,value in state.pop('captured').items():
                if isinstance(value,str) and value.startswith('data:'):
                    (a.out/(a.name+'_'+key+'.png')).write_bytes(base64.b64decode(value.split(',',1)[1]))
            (a.out/(a.name+'_state.json')).write_text(json.dumps(state,indent=2),encoding='utf-8')
            page.screenshot(path=str(a.out/(a.name+'_ui.png')))
        if a.mode=='export':
            result=page.evaluate('async()=>await app.graphToPrompt()')
            if any(n['class_type']=='3DAH3Remote' for n in result['output'].values()):
                raise RuntimeError('Preview export unexpectedly contains active Remote')
            (a.out/(a.name+'_export.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(mode=a.mode,name=a.name,errors=errors),ensure_ascii=True))


if __name__=='__main__':main()
