"""Prepare a normal core Load3DAdvanced workflow from actual render metadata.

All private paths are CLI arguments/config data; exported graphs use Comfy
input-relative assets. Does not run H3, fit a body, or call Fit to viewer.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/scene3d')]
import wfkit
from prop_gizmo import B, camera_config


def build(preset, cfg, request, preview, background, out):
    w=copy.deepcopy(preset)
    by_title={n.get('title'):n for n in w['nodes']}
    by_title['3DA Place Pose']['widgets_values_named']['block_on_fail']=True
    # Installed frontend mis-restores converted seed/control widgets. Keep
    # these native numeric widgets, avoiding version-dependent hidden slots.
    seed_node=by_title.get('Seed · общий для геометрии и H3')
    removed=set()
    for n in w['nodes']:
        if n['type'] not in ('3DAPlacePose','3DAH3Remote'):continue
        for inp in list(n.get('inputs',[])):
            if inp['name']=='seed':
                removed.add(inp['link']);n['inputs'].remove(inp)
        n.setdefault('widgets_values_named',{})['seed']=101
    w['links']=[l for l in w['links'] if l[0] not in removed]
    if seed_node:
        w['nodes'].remove(seed_node)
    for n in w['nodes']:
        for output in n.get('outputs',[]):
            if output.get('links'):output['links']=[l for l in output['links'] if l not in removed]
    by_title['3DA H3 Remote']['widgets_values_named']['prop_handling']='rendered_separately'
    prop=by_title['3DA Prop']
    # Repair positional widgets in inherited presets using the live schema.
    # widgets_values_named is a tooling annotation; core does not restore it.
    import requests
    oi=requests.get('http://127.0.0.1:8187/object_info',timeout=15).json()
    for n in w['nodes']:
        info=oi.get(n['type'])
        if not info or 'widgets_values_named' not in n:continue
        schema={**info['input'].get('required',{}),**info['input'].get('optional',{})}
        for inp in n.get('inputs',[]):
            entry=schema.get(inp['name'],[None,{}])
            if len(entry)>1 and entry[1].get('forceInput'):inp.pop('widget',None)
        named=n['widgets_values_named'];values=[]
        for key,entry in schema.items():
            typ=entry[0];opts=entry[1] if len(entry)>1 else {}
            if opts.get('forceInput') or not (isinstance(typ,list) or typ in ('INT','FLOAT','STRING','BOOLEAN','COMBO')):continue
            default=opts.get('default',typ[0] if isinstance(typ,list) and typ else '' if typ=='STRING' else False if typ=='BOOLEAN' else 0)
            values.append(named.get(key,default))
            if key in ('seed','noise_seed') or opts.get('control_after_generate'):
                values.append('fixed');named['control_after_generate']='fixed'
        n['widgets_values']=values
    # Removing converted seed inputs shifts later target socket indices.
    for n in w['nodes']:
        for index,inp in enumerate(n.get('inputs',[])):
            for link in w['links']:
                if link[0]==inp.get('link'):link[4]=index
    pid=prop['widgets_values_named']['prop_id']
    metadata=json.loads((Path(cfg['farm_root'])/'props'/pid/'prop.json').read_text(encoding='utf-8'))
    model=Path(cfg['farm_root'])/'props'/pid/metadata.get('model','model.glb')
    def stage(src, folder):
        stem=hashlib.sha256(src.read_bytes()).hexdigest()[:16]+src.suffix.lower()
        # Core LoadImage's combo enumerates only the input directory itself,
        # even though the backend accepts subfolder paths. Root names avoid
        # false missing-media warnings on an ordinary workflow open.
        name=('3da_'+stem) if folder=='3da_image' else folder+'/'+stem
        dst=Path(cfg['comfy'])/'input'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
        return name
    model_name=stage(model,'3d/3da')
    bg_name=stage(Path(background),'3da_gizmo')
    location=json.loads(Path(request['location']).read_text(encoding='utf-8'))['loc_id']
    plate_dir=Path(cfg['location_asset_dirs'][location])
    plate=json.loads((plate_dir/'plate.json').read_text(encoding='utf-8'))
    camera=camera_config(plate,[1344,768])
    facts=preview['report']['props'][0]
    numeric=json.loads(request['props'])[0] if isinstance(request['props'],str) else request['props'][0]
    # Current numeric render: T(center) R(EulerXYZ) S(factor) T(-cachecenter).
    r=Rotation.from_euler('xyz',[numeric.get(k,0) for k in ('roll','pitch','yaw')],degrees=True).as_matrix()
    # The historic preview can reference a metre-size cache while today's GLB
    # is already normalized to .30 m. Reusing its old factor shrinks twice.
    probe=ROOT/'tools/scene3d/probe_gizmo_asset.py'
    signature=hashlib.sha256(model.read_bytes()+probe.read_bytes()+str(Path(cfg['blender']).stat().st_mtime_ns).encode()).hexdigest()
    bounds_file=Path(cfg['farm_root'])/'cache/gizmo_assets'/signature/'bounds.json'
    if not bounds_file.exists():
        bounds_file.parent.mkdir(parents=True,exist_ok=True)
        with (bounds_file.parent/'probe.log').open('w',encoding='utf-8') as log:
            subprocess.run([cfg['blender'],'-b','--factory-startup','--disable-autoexec','--python-exit-code','1',
                '--python',str(probe),'--','--model',str(model),'--out',str(bounds_file)],stdout=log,stderr=subprocess.STDOUT,timeout=60,check=True)
    asset=json.loads(bounds_file.read_text(encoding='utf-8'))
    bounds=np.asarray(asset['bbox_world_zup'],dtype=float)
    factor=float(numeric.get('size_m',.30))*float(numeric.get('scale',1.))/float(np.max(bounds[1]-bounds[0]))
    center=np.mean(bounds,axis=0)
    # An AABB center equals transformed model bbox center for symmetric bounds;
    # derive the actual anchor from saved world joints for arbitrary rotations.
    from plate.prop_render import anchors_from_mhr70
    with np.load(Path(preview['init']).parents[2]/'fit/label_joints.npz',allow_pickle=False) as d:
        anchors=anchors_from_mhr70(d['kp'])
    from posev2.manual_pose import location_basis
    from types import SimpleNamespace
    offset=location_basis(SimpleNamespace(R=np.array(plate['R_world_from_cv'])))@np.array([numeric.get(k,0) for k in 'xyz'])
    anchor=anchors['char_pelvis' if numeric['anchor']=='world' else numeric['anchor']]
    world_pos=np.asarray(anchor)+offset-r@(factor*center)
    pos=dict(zip('xyz',(B@world_pos).tolist()))
    q=Rotation.from_matrix(B@r@B.T).as_quat()
    transform=dict(position=pos,quaternion=dict(zip('xyzw',q.tolist())),scale=dict.fromkeys('xyz',factor))
    euler=Rotation.from_quat(q).as_euler('XYZ')
    node_id=w['last_node_id']+1
    node=dict(id=node_id,type='Load3DAdvanced',title='Предмет · гизмо (фон без окклюзий)',pos=[430,1450],size=[760,760],
        flags={},order=0,mode=0,inputs=[],outputs=[dict(name=name,type=typ,links=[])
            for name,typ in [('model_3d','FILE_3D'),('model_3d_info','LOAD3D_MODEL_INFO'),('camera_info','LOAD3D_CAMERA'),('width','INT'),('height','INT')]],
        properties={'Node name for S&R':'Load3DAdvanced','Camera Config':dict(cameraType='perspective',fov=camera['fov'],state=camera,hasCustomUp=True,useCustomUp=True),
            'Scene Config':dict(showGrid=False,backgroundColor='#282828',backgroundImage=bg_name,backgroundRenderMode='tiled'),
            'Model Config':dict(upDirection='original',materialMode='original',showSkeleton=False,
                gizmo=dict(enabled=True,mode='translate',position=pos,rotation=dict(zip('xyz',euler.tolist())),scale=transform['scale']))},
        widgets_values=[model_name,None,1344,768],widgets_values_named=dict(model_file=model_name,
            viewport_state=dict(camera_info=camera,model_3d_info=[transform]),width=1344,height=768))
    node['widgets_values'][1]=node['widgets_values_named']['viewport_state']
    lid=w['last_link_id']+1
    slot=len(prop['inputs'])
    prop['inputs'].append(dict(name='model_3d_info',type='LOAD3D_MODEL_INFO',link=lid))
    node['outputs'][1]['links']=[lid]
    w['links'].append([lid,node_id,1,prop['id'],slot,'LOAD3D_MODEL_INFO'])
    w['nodes'].append(node);w['last_node_id']=node_id;w['last_link_id']=lid;w['id']=str(uuid.uuid4())
    place=by_title['3DA Place Pose'];remote=by_title['3DA H3 Remote']
    if len(place['outputs'])==7:
        place['outputs'] += [dict(name='prop_repair',type='IMAGE',links=[]),dict(name='prop_repair_mask',type='MASK',links=[])]
    if len(place['outputs'])==9:place['outputs'].append(dict(name='body_guides',type='3DA_BODY_GUIDES',links=[]))
    graph=wfkit.G(w,w,top=True)
    graph.connect(node,0,prop,'model_3d',type_='FILE_3D')
    graph.connect(place,7,remote,'prop_repair',type_='IMAGE')
    graph.connect(place,8,remote,'prop_repair_mask',type_='MASK')
    graph.connect(place,9,remote,'body_guides',type_='3DA_BODY_GUIDES')
    for title,path in [('Плита комнаты',plate_dir/'plate_rgb.png'),('Референс позы · SFW',Path(request['pose_ref']))]:
        n=by_title[title];name=stage(path,'3da_image');n['widgets_values'][0]=name;n['widgets_values_named']['image']=name
    for n in w['nodes']:
        if n['type']=='LoadImage':
            path=Path(cfg['comfy'])/'input'/n['widgets_values_named']['image']
            name=stage(path,'3da_image');n['widgets_values'][0]=name;n['widgets_values_named']['image']=name
    notes=by_title['README']
    for group in w.get('groups',[]):
        if group.get('title','').startswith('2 '):group['bounding']=[400,-30,820,2300]
    notes['widgets_values']=['Предпросмотр безопасен: Remote и его потребители muted.\nГизмо: Translate / Rotate / Scale; затем Queue для нового плейбласта. SAM берётся из кэша, H3 не запускается.\nПодключённый model_3d_info полностью заменяет числовые TRS/anchor/size_m Prop. Для числового режима отсоедините этот вход. Nonuniform scale поддерживается матрицей.\nНЕ нажимайте Fit to viewer: он меняет scale и origin. Камеру не двигать.\nФон 2D: он не скрывает предмет за телом/комнатой. Проверять окклюзии в основном плейбласте.\nПосле смены комнаты/тела подготовьте фон и камеру заново этим сборщиком.\nSeed установлен 101 отдельно в Place Pose и Remote. При изменении выставьте одинаковое число в обоих.\nФинал: включите Remote, финальное сохранение, raw preview и статус Remote. Режим rendered_separately защищает персонажа: H3 обрабатывает только тело, предмет вклеивается точным рендером. Референсы предмета отправляются только в режиме transfer. Общая очередь проверяется перед отправкой.']
    out.mkdir(parents=True,exist_ok=True)
    (out/'3DA_manual_gizmo.preview.json').write_text(json.dumps(w,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'3DA_manual_gizmo.preview.api.json').write_text(json.dumps(wfkit.to_api(w),ensure_ascii=False,indent=2),encoding='utf-8')
    full=copy.deepcopy(w)
    for n in full['nodes']:
        if n['mode']==2:n['mode']=0
    (out/'3DA_manual_gizmo.final.api.json').write_text(json.dumps(wfkit.to_api(full),ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'gizmo_setup.json').write_text(json.dumps(dict(camera=camera,model=model_name,background=bg_name,transform=transform,
        asset_bounds=asset,old_preview_factor=facts.get('normalization_factor'),current_factor=factor),indent=2),encoding='utf-8')
    return w


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ('preset','config','request','preview','background','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();import yaml
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    build(load(a.preset),yaml.safe_load(a.config.read_text(encoding='utf-8')),load(a.request),load(a.preview),a.background,a.out)
