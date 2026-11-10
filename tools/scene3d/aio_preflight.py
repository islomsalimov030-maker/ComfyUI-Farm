"""Read-only AIO inventory and schema audit. Does not install or submit jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def schema_entry(info, name):
    fields={**info.get('input',{}).get('required',{}),**info.get('input',{}).get('optional',{})}
    if name in fields:return fields[name]
    parent=name.split('.')[0]
    if '.' in name and parent in fields and fields[parent][0]=='COMFY_AUTOGROW_V3':return ('*',)
    if info.get('name')=='ImpactSwitch' and re.fullmatch(r'input[1-9][0-9]*',name):return ('*',)
    return None


def clean(graph, oi):
    """Drop UI-only button values from annotations, not real backend inputs."""
    for n in graph.values():
        info=oi.get(n['class_type'])
        if not info:continue
        n['inputs']={k:v for k,v in n['inputs'].items() if schema_entry(info,k) is not None}
    return graph


def inventory(graph, oi):
    missing=sorted({n['class_type'] for n in graph.values()}-set(oi))
    invalid=[];models=[]
    for key,n in graph.items():
        info=oi.get(n['class_type']);title=n.get('_meta',{}).get('title',n['class_type'])
        if not info:continue
        for field in info.get('input',{}).get('required',{}):
            if field not in n['inputs']:
                if info['input']['required'][field][0]=='COMFY_AUTOGROW_V3' and any(k.startswith(field+'.') for k in n['inputs']):continue
                invalid.append(dict(title=title,input=field,error='missing required input'))
        for name,value in n['inputs'].items():
            entry=schema_entry(info,name)
            if entry is None:
                invalid.append(dict(title=title,input=name,error='unknown input'));continue
            if isinstance(value,list) and len(value)==2 and str(value[0]) in graph:
                source=graph[str(value[0])];source_info=oi.get(source['class_type'])
                if source_info and not 0<=value[1]<len(source_info['output']):invalid.append(dict(title=title,input=name,error='output slot out of range'))
            elif name in ('unet_name','model_name','clip_name','vae_name','lora_name','model_file') and isinstance(value,str) and value:
                choices=entry[0] if isinstance(entry[0],list) else (entry[1].get('options',[]) if len(entry)>1 else [])
                models.append(dict(title=title,input=name,file=value,listed=value in choices if choices else None))
    return dict(missing_classes=missing,schema_errors=invalid,models=models,
                note='listed model files and clean schemas do not prove runtime execution')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--object-info',type=Path,required=True)
    p.add_argument('--workflows',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    oi=json.loads(a.object_info.read_text(encoding='utf-8'));reports={}
    for file in sorted(a.workflows.glob('*.api.json')):
        graph=json.loads(file.read_text(encoding='utf-8'));reports[file.name]=inventory(graph,oi)
    root=Path(__file__).resolve().parents[2]
    files=['__init__.py','requirements.txt','comfyui/ComfyUI-3DA/nodes.py','comfyui/ComfyUI-3DA/__init__.py',
           'comfyui/ComfyUI-3DA/pyproject.toml','tools/scene3d/pack_cli.py','tools/scene3d/scene_edit.py','tools/scene3d/scene_export_blender.py',
           'tools/scene3d/posev2/manual_pose.py',*['web/3da/'+p.name for p in (root/'web/3da').glob('*.js')]]
    result=dict(workflows=reports,source_files={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({f:dict(missing_classes=v['missing_classes'],schema_errors=len(v['schema_errors']),unlisted_models=[m['file'] for m in v['models'] if m['listed'] is False]) for f,v in reports.items()},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
