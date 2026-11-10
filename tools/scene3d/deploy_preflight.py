"""Run under the Comfy service user after installation; never installs or restarts.

Config and report are private. Runtime availability is not a geometry-forward PASS.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request
import yaml

CLASSES=['3DALocation','3DAProp','3DAPlacePose','3DASceneExport','3DASceneEdits','3DANativeGuides','3DANativeComposite',
         'LoadMoGeModel','MoGeInference','MoGeGeometryToFOV','SAM3DBody_Loader','SAM3DBody_Predict']


def run(args):
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'));checks={};blockers=[]
    def check(name,ok,**detail):
        checks[name]=dict(ok=bool(ok),**detail)
        if not ok:blockers.append(name)
    for key in ['repo','python','geometry_python','blender','mannequin','comfy','farm_root']:
        value=cfg.get(key);check('path:'+key,bool(value) and Path(value).exists())
    for loc,folder in cfg.get('location_asset_dirs',{}).items():
        for f in ['plate.json','plate_rgb.png','plate_mesh.obj','depth_loc.png']:
            check('plate:'+loc+':'+f,(Path(folder)/f).is_file())
        profile=Path(cfg['farm_root'])/'locations'/loc/'profile.json'
        check('profile:'+loc,profile.is_file())
        if profile.is_file():
            data=json.loads(profile.read_text(encoding='utf-8'))
            for ob in data['objects']:
                for key in ['top_grid_npz','pts_npz','backrest_npz','mask_png']:
                    if ob.get(key):check('profile_ref:'+loc+':'+ob['id']+':'+key,(profile.parent/ob[key]).is_file())
    props=[]
    for metadata in (Path(cfg['farm_root'])/'props').glob('*/prop.json'):
        d=json.loads(metadata.read_text(encoding='utf-8'));model=metadata.parent/d.get('model','model.glb')
        check('prop:'+metadata.parent.name,model.is_file())
        props.append(dict(id=metadata.parent.name,size_m=d.get('size_m'),license_status=d.get('license_status','unknown')))
    checks['props']=props
    for name,command in [('blender',[cfg['blender'],'--background','--factory-startup','--disable-autoexec','--version']),
                         ('farm_python',[cfg['python'],'-I','-c','import numpy,scipy,PIL,yaml,httpx;print("imports_ok")']),
                         ('geometry_python',[cfg['geometry_python'],'-I','-c','import torch,torchvision,scipy,transformers;assert torch.cuda.is_available(),"CUDA unavailable";print("imports_ok")'])]:
        try:
            r=subprocess.run(command,capture_output=True,text=True,timeout=60)
            check('runtime:'+name,r.returncode==0,exit_code=r.returncode)
        except (OSError,subprocess.TimeoutExpired):check('runtime:'+name,False)
    try:
        opener=urllib.request.build_opener()
        if os.environ.get('COMFY_USER'):
            passwords=urllib.request.HTTPPasswordMgrWithDefaultRealm();passwords.add_password(None,args.url,os.environ['COMFY_USER'],os.environ.get('COMFY_PASS',''))
            opener=urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(passwords))
        with opener.open(args.url.rstrip('/')+'/object_info',timeout=20) as r:oi=json.load(r)
        missing=[k for k in CLASSES if k not in oi];check('registered_classes',not missing,missing=missing)
    except Exception as e:check('registered_classes',False,error=type(e).__name__)
    directories=[Path(cfg['comfy'])/'input',Path(cfg['comfy'])/'output',*[Path(cfg['farm_root'])/v for v in ['cache/editor','cache/placement','pack_requests','pack/jobs','props']]]
    for directory in directories:
        label='writable:'+str(directory)
        if args.probe_writes:
            try:
                directory.mkdir(parents=True,exist_ok=True)
                with tempfile.TemporaryFile(dir=directory) as f:f.write(b'3da_preflight');f.flush()
                check(label,True)
            except OSError:check(label,False)
        else:checks[label]=dict(status='NOT_RUN',reason='use --probe-writes under the actual service user')
    warnings=[]
    try:
        r=subprocess.run([cfg['geometry_python'],'-I','-c','import cv2'],capture_output=True,timeout=30)
        if r.returncode:warnings.append('OpenCV unavailable in geometry env: optional source_vertical line detection falls back; install it to enable the full diagnostic')
    except (OSError,subprocess.TimeoutExpired):warnings.append('OpenCV diagnostic not checked')
    return dict(schema=1,blockers=blockers,warnings=warnings,checks=checks,remaining_live_acceptance=['geometry_forward','Blender_scene_export','gizmo_camera_TRS_roundtrip',
        'negative_camera_and_frame','native_photo_with_existing_mods_and_facepass','original_video','ready_video_facepass','combined_VRAM_peak'],
        production_ready=False,note='No install, restart, downloads, queue submission, global free or interruption performed')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--url',default='http://127.0.0.1:8188');p.add_argument('--out',type=Path,required=True);p.add_argument('--probe-writes',action='store_true');a=p.parse_args()
    result=run(a);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(blockers=result['blockers'],production_ready=False),ensure_ascii=True));return 1 if result['blockers'] else 0


if __name__=='__main__':raise SystemExit(main())
