"""09e rigid manual transforms and exact placement cache. No bone edits."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def placement_key(ref_path, profile, config, spec, height, seed, body):
    h=hashlib.sha256()
    for p in [Path(ref_path), profile.path, Path(config['repo'])/'config/placement.yaml']:
        h.update(p.read_bytes())
    root=Path(__file__).resolve().parent
    for name in ['place.py','assembly.py','pose_fit.py','orient.py','refine.py','top_region.py','profiles.py','body.py','auto_surfaces.py',
                 'manual_pose.py','shot_spec.py','zone.py','support_geometry.py','placement_config.py','source_vertical.py']:
        h.update((root/name).read_bytes())
    for obj in profile.data['objects']:
        for key in ('top_grid_npz','pts_npz','backrest_npz','mask_png'):
            if obj.get(key):
                h.update((profile.dir/obj[key]).read_bytes())
    assets=Path(config.get('location_asset_dirs',{}).get(profile.data['loc_id'],profile.dir))
    for name in ('plate.json','plate_mesh.obj','depth_loc.png'):
        h.update((assets/name).read_bytes())
    h.update(json.dumps({'spec':spec.to_dict(),'height':height,'seed':np.asarray(seed).tolist(),
                         'body':body},sort_keys=True).encode())
    return h.hexdigest()


def save_placement(directory,result,ref,plate,spec,key):
    import place
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    place.write(directory,result,ref,plate,spec,key)
    if 'X' not in result:
        return
    state=result['state']
    np.savez_compressed(directory/'manual_base.npz',X=result['X'],KP=result['KP'],reg=result['reg_lbl'],
                        groups=np.array([f'{g[0]}:{g[2]}:{g[3]}' for g in state['groups']]),
                        **{'g_'+g[0]:g[1] for g in state['groups']})
    (directory/'complete.json').write_text(json.dumps({'key':key}),encoding='utf-8')


def load_placement(directory,key):
    directory=Path(directory)
    if not (directory/'complete.json').is_file():
        return None
    if json.loads((directory/'complete.json').read_text())['key']!=key:
        return None
    with np.load(directory/'manual_base.npz',allow_pickle=False) as d:
        groups=[]
        for item in d['groups']:
            name,surf,role=str(item).split(':')
            groups.append((name,d['g_'+name].copy(),surf,role))
        return {'X':d['X'].copy(),'KP':d['KP'].copy(),'reg':d['reg'].copy(),'groups':groups,
                'report':json.loads((directory/'fit.json').read_text(encoding='utf-8'))}


def location_basis(plate):
    """Camera-horizontal right/depth/up basis; never apply CV downward Y to offsets."""
    x=np.array([plate.R[0,0],plate.R[1,0],0.],float)
    x/=np.linalg.norm(x)
    up=np.array([0.,0.,1.])
    y=np.cross(up,x)
    if y@plate.R[:,2]<0:
        y=-y
    return np.column_stack([x,y,up])


def transform(X,KP,plate,values):
    import orient as ori
    if values.get('body_matrix') is not None:
        from scene_edit import body_matrix
        m = body_matrix(values['body_matrix'])
        scale = float(np.cbrt(np.linalg.det(m[:3,:3])))
        return X @ m[:3,:3].T + m[:3,3], KP @ m[:3,:3].T + m[:3,3], {'body_matrix': m.tolist(), 'body_scale': scale}
    values={k:float(values.get(k,0.)) for k in ('body_dx','body_dy','body_dz','body_dyaw','body_dpitch','body_droll')}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('body transforms must be finite')
    basis=location_basis(plate)
    R=(ori.rot_axis(basis[:,2],math.radians(values['body_dyaw'])) @
       ori.rot_axis(basis[:,0],math.radians(values['body_dpitch'])) @
       ori.rot_axis(basis[:,1],math.radians(values['body_droll'])))
    pivot=(KP[9]+KP[10])/2
    shift=basis@np.array([values['body_dx'],values['body_dy'],values['body_dz']])
    return ori.rot_about(X,pivot,R)+shift,ori.rot_about(KP,pivot,R)+shift,values


def revalidate(base,ref,plate,profile,spec,seed,height,values):
    """Recompute diagnostics on transformed geometry, without fit/refine/clamp/lift."""
    import place
    import assembly
    import top_region as tr
    X,K,values=transform(base['X'],base['KP'],plate,values)
    rep=json.loads(json.dumps(base['report']))
    scale=values.get('body_scale',1.)
    rep['body_dims']={k:float(v)*scale for k,v in rep['body_dims'].items()}
    H=rep['body_dims']['H']
    treg,freg,_,oinfo=place.resolve_regions(plate,spec,seed,H,profile)
    if treg is None:
        raise ValueError('cached target disappeared')
    treg=tr.pose_region(treg,spec.pose_class)
    treg.object_id=oinfo.get('object_id',spec.target_object)
    treg.wall_planes=profile.walls()
    treg.diagnostic_surfaces=[{'id':o['id'],'soft':bool(o.get('soft',False)),'region':profile.region(o)}
                              for o in profile.data['objects'] if 'top_grid_npz' in o]
    if hasattr(treg,'auto_support_surfaces'):
        treg.diagnostic_surfaces=[{'id':r.object_id,'region':r,
            'soft':r.object_id in treg.auto_soft_ids or
                   (r.object_id==treg.object_id and bool(oinfo.get('soft')))}
            for r in treg.auto_support_surfaces]
    wall=None
    if spec.secondary_on('wall') and treg.wall_planes:
        i=place.zn.pick_wall(treg.wall_planes,seed[:2])
        wall=treg.wall_planes[i] if i is not None else None
    if treg.backrest is not None:
        hard=np.zeros(len(treg.backrest),bool)
        for w in treg.wall_planes:
            hard |= np.abs(treg.backrest@w['n']+w['d'])<=place.CFG['wall_backrest_m']
        treg.backrest_material=np.where(hard,'wall','pillows')
    S=place.build_surfaces(plate,treg,freg,wall)
    B={'reg':base['reg']}
    soft=bool(oinfo.get('soft'))
    ev=place.evaluate(X,K,B,base['groups'],S,plate,spec,treg,freg,H,soft)
    side={'front':1,'back':-1,'side':0}.get(rep.get('orient',{}).get('side_ref'),0)
    window=rep.get('orient',{}).get('yaw',{}).get('window_deg')
    limits_ok=rep.get('refine',{}).get('limits_ok',True) if rep.get('refine') else True
    val=assembly.validate(K,ev,spec,plate,side,window,limits_ok,X=X,treg=treg,soft=soft,height_info={})
    rep.update(status=val['status'],validation=val,penetration=ev['penetration'],footprint=ev['footprint'],
               frame=ev['frame'],contacts=ev['contacts'],support_object=ev['support_object'],
               support_parts_on=ev['support_parts_on'],support_gap_cm=ev['support_gap_cm'],
               back_contact_gap_m=ev['back_contact_gap_m'],back_gap_cm=ev['back_gap_cm'],wall_gap_cm=ev['wall_gap_cm'],
               cells=ev['cells'],cells_ok=ev['cells_ok'],balance_ok=ev['balance_ok'],
               max_penetration_cm=ev['penetration']['max_cm'],
               body_transform=values,transform_applied=True)
    # Old failure flags must not masquerade as current transformed diagnostics.
    diagnostic={'support_gap','height_conflict','no_back_contact','orient_fail','penetration_fail',
                'footprint_fail','cells_fail','refine_limits_fail','out_of_frame','body_out_of_frame',
                'orient_conflict','side_conflict','side_flip'}
    rep['flags']=sorted(set([f for f in rep.get('flags',[]) if f not in diagnostic]+val.get('flags',[])))
    rep['zone_diagnostics'].update(place.geometry_diagnostics(X,K,B,spec,treg,base['groups'],S,plate,soft))
    rep['orient'].update(tilt_final_deg=ev['tilt_deg'],roll_final_deg=ev['roll_deg'],check=ev['orient'])
    rep['support_contact']=ev.get('support_contact',{})
    rep.update(support_gap_parts=ev['support_contact']['gap_parts'],
               support_part_gaps_m=ev['support_part_gaps_m'],max_support_gap_m=ev['max_support_gap_m'])
    return X,K,rep


def write_transformed(directory,X,K,faces,report):
    p=Path(directory)
    p.mkdir(parents=True,exist_ok=True)
    # Full precision cached base is authoritative; OBJ used only by Blender.
    np.savez_compressed(p/'label_joints.npz',kp=K)
    with (p/'body.obj').open('w',encoding='utf-8') as out:
        for v in X:
            out.write('v '+' '.join(format(float(t),'.10g') for t in v)+'\n')
        for f in faces:
            out.write('f '+' '.join(str(int(i)+1) for i in f)+'\n')
    (p/'fit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
