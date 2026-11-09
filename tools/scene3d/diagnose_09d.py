"""One bounded Q01 cProfile / saved Q11 read-only diagnostics. No geometry edits."""
from pathlib import Path
import argparse
import cProfile
import hashlib
import json
import os
import pstats
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
STATE = Path('C:/work/scene3d_work/farm/diagnostics_09d')
CONFIG = Path('C:/work/ComfyUI-next/custom_nodes/ComfyUI-3DA/config.yaml')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def bootstrap():
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    os.environ['FARM_ROOT'] = cfg['farm_root']
    os.environ['FARM_COMFY'] = cfg['comfy']
    for p in (REPO / 'tools/scene3d/plate', REPO / 'tools/scene3d/posev2', REPO):
        sys.path.insert(0, str(p))
    return cfg


def sources():
    paths = list((REPO / 'tools/scene3d/posev2').glob('*.py')) + [
        REPO / 'tools/scene3d/pack_cli.py', REPO / 'config/placement.yaml', CONFIG,
        REPO / 'tools/scene3d/plate/pose_fit.py']
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def profile_worker():
    cfg = bootstrap()
    import numpy as np
    import run_v2 as runner
    import place
    import profiles
    import shot_spec
    import refine
    STATE.mkdir(parents=True, exist_ok=True)
    initial = sources()
    save(STATE / 'Q01_sources_start.json', initial)
    root = Path(cfg['farm_root'])
    p = next(p for p in runner.pr.prompts() if p['id'] == 'Q01_stand_hand_face')
    rf = root.parent / 'refs_v2_npz' / (p['id'] + '.ref.npz')
    ref = runner.pf.load_ref(rf)
    ref.update(path=str(rf), body_cache_dir=str(root / 'characters/stacy/geometry_cache'))
    spec = shot_spec.ShotSpec.load(root / 'pose_round/V2' / p['id'] / 'shot_spec.json')
    body = profiles.load_body('stacy', root)
    height = float(body['height_m']) if body else runner.DEFAULT_HEIGHT
    plate = runner.attach_geometry_assets(runner.pf.Plate(cfg['location_asset_dirs']['mainroom1']))
    profile = profiles.LocationProfile.load('mainroom1', root)
    seed = runner.table_seed(p, plate)
    np.random.seed(101)
    result = None
    prof = cProfile.Profile()
    t0 = time.monotonic()
    prof.enable()
    try:
        ctx = refine.load_ctx(cfg['comfy'])
        result = place.place(ref, plate, spec, seed, height, profile=profile,
                             body_src='profile' if body else 'proxy',
                             dims_override=body['body_dims'] if body else None,
                             refine_ctx=ctx, log=lambda text: print(text, flush=True))
    finally:
        prof.disable()
        prof.dump_stats(str(STATE / 'Q01_before_SDF.pstats'))
    elapsed = time.monotonic() - t0
    stats = pstats.Stats(prof)
    rows = []
    for (filename, line, function), (primitive, calls, self_s, cum_s, callers) in stats.stats.items():
        rows.append(dict(file=filename, line=line, function=function, primitive_calls=primitive,
                         calls=calls, self_s=self_s, cum_s=cum_s))
    rows.sort(key=lambda r: r['cum_s'], reverse=True)
    rep = result['rep']
    baseline = dict(id=p['id'], status=rep['status'], penetration=rep.get('penetration'),
                    target_instance=rep.get('target_instance'), support_object=rep.get('support_object'),
                    fit_source='single fresh placement before SDF; run_v2 inputs, no Blender/H3',
                    reference=str(rf), spec=spec.to_dict(), seed=seed.tolist(), height_m=height,
                    elapsed_s=elapsed, profiler_total_s=stats.total_tt,
                    pstats=str(STATE / 'Q01_before_SDF.pstats'), top10_cumtime=rows[:10],
                    distance_penetration_functions=[r for r in rows if any(
                        text in r['function'].lower() for text in ('penetration', 'distance', 'closest_point', 'proximity'))],
                    sources_start=initial, sources_end=sources(),
                    head=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
    save(STATE / 'Q01_profile.json', baseline)
    print(json.dumps({k:baseline[k] for k in ('status','elapsed_s','pstats','top10_cumtime')}, ensure_ascii=False), flush=True)


def supervise():
    STATE.mkdir(parents=True, exist_ok=True)
    command = ['C:/work/ComfyUI-next/venv/Scripts/python.exe', '-I', str(Path(__file__).resolve()), 'profile-worker']
    with (STATE / 'Q01_profile.log').open('w', encoding='utf-8') as out:
        proc = subprocess.Popen(command, cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        save(STATE / 'Q01_process.json', {'pid': proc.pid, 'timeout_s': 600, 'command': command})
        try:
            code = proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True, timeout=15)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            code = 124
    save(STATE / 'Q01_exit.json', {'exit_code': code, 'timeout_s': 600})
    print('Q01 profiling finished; exit_code=' + str(code) + '; state=' + str(STATE), flush=True)
    return code


def q11_worker():
    cfg = bootstrap()
    import numpy as np
    sys.path.insert(0, str(REPO / 'tools/scene3d'))
    import pack_cli
    import pose_fit as pf
    import profiles
    import place
    import body
    import shot_spec
    from scipy.spatial import cKDTree
    root = Path(cfg['farm_root'])
    req_path = root / 'pack_manual_09c/Q11_bench_side_sit_request.json'
    req = json.loads(req_path.read_text(encoding='utf-8'))
    outcome_path = root / 'pack_manual_09c/Q11_place_result_after_fixes.json'
    outcome = json.loads(outcome_path.read_text(encoding='utf-8'))
    job = Path(outcome['playblast']).parents[2]
    fit_dir = job / 'fit'
    watched = [req_path, outcome_path, *fit_dir.glob('*'), Path(req['location'])]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched if p.is_file()}
    rep = json.loads((fit_dir / 'fit.json').read_text(encoding='utf-8'))
    profile = profiles.LocationProfile(req['location'])
    plate_dir = Path(cfg['location_asset_dirs'][profile.data['loc_id']])
    plate = pf.Plate(plate_dir)
    X, F = pf.load_obj_mesh(fit_dir / 'body.obj')
    with np.load(fit_dir / 'label_joints.npz', allow_pickle=False) as d:
        KP = d['kp'].copy()
    rf = root / 'cache/pose_ref' / hashlib.sha256(Path(req['pose_ref']).read_bytes()).hexdigest() / 'pose.ref.npz'
    ref = pf.load_ref(rf)
    assert np.array_equal(F, ref['F']) and len(X) == len(ref['V'])
    spec = shot_spec.ShotSpec.load(fit_dir / 'shot_spec.json')
    seed = pack_cli.seed_for(profile, spec, cfg)
    selected, kind = profile.find_object(spec.target_object, spec.instance_hint, seed[:2], spec.pose_class)
    objects = []
    for obj in profile.data['objects']:
        row = {k: obj.get(k) for k in ('id', 'type', 'h0', 'soft')}
        if 'top_grid_npz' in obj:
            reg = profile.region(obj)
            P = reg.points3d()
            row.update(center_world_m=P.mean(0).tolist(), top_bounds_world_m=[P.min(0).tolist(), P.max(0).tolist()])
        else:
            with np.load(profile.dir / obj['pts_npz'], allow_pickle=False) as d:
                P = d['P']
            row['center_world_m'] = P.mean(0).tolist()
        objects.append(row)
    groups = []
    with np.load(fit_dir / 'fit_state.npz', allow_pickle=False) as state:
        for desc in state['groups']:
            name, surf, role = str(desc).split(':')
            groups.append((name, state['g_' + name].copy(), surf, role))
    reg = profile.region(selected)
    reg.wall_planes = profile.walls()
    reg.diagnostic_surfaces = [{'id': o['id'], 'soft': bool(o.get('soft')), 'region': profile.region(o)}
                               for o in profile.data['objects'] if 'top_grid_npz' in o]
    floor = profile.region(next(o for o in profile.data['objects'] if o['type'] == 'floor'))
    S = place.build_surfaces(plate, reg, floor)
    labels = body.regions(ref['V'], ref['kp'])
    geom = place.geometry_diagnostics(X, KP, {'reg': labels}, spec, reg, groups, S, plate, False)
    heights = {}
    for name in ('pelvis', 'foot.L', 'foot.R'):
        P = X[labels == name]
        heights[name] = {'vertex_count': len(P), 'z_min_m': float(P[:,2].min()),
                         'z_p50_m': float(np.median(P[:,2])), 'z_max_m': float(P[:,2].max()),
                         'relative_target_min_m': float(P[:,2].min() - reg.h0),
                         'relative_target_p50_m': float(np.median(P[:,2]) - reg.h0)}
    joint_heights = {}
    points = {'pelvis_hip_midpoint': (KP[body.K['l_hip']] + KP[body.K['r_hip']]) / 2}
    for side in ('l', 'r'):
        for part in ('an', 'heel', 'btoe', 'stoe'):
            points[side+'_'+part] = KP[body.K[side+'_'+part]]
    for name, P in points.items():
        joint_heights[name] = {'world_xyz_m': P.tolist(), 'relative_target_m': float(P[2] - reg.h0)}
    idx = np.unique(np.concatenate([g[1] for g in groups if g[3] == 'support']))
    # Saved support vertices are passed to Blender for exact vertical ray/mesh intersection.
    support = X[idx]
    save(STATE / 'Q11_support_vertices.json', support.tolist())
    result = {'source': 'existing fresh generic CLI09c, read-only; NOT V2/Q11',
              'request_path': str(req_path), 'outcome_path': str(outcome_path), 'job': str(job),
              'fit_path': str(fit_dir / 'fit.json'), 'reference': str(rf),
              'plate_dir': str(plate_dir), 'status': rep['status'], 'timings_saved': outcome['timings'],
              'objects': objects, 'bench_instances': [o for o in objects if o['type'] == 'bench'],
              'pink_sofa': {'registered_instances': [o for o in objects if o['type'] in ('sofa','armchair')],
                            'type': None, 'id': None, 'h0_m': None, 'center_world_m': None,
                            'reason': 'pink sofa visible in original plate is not an object in this location profile; no sofa/armchair/other instance registered'},
              'target_instance': selected['id'], 'resolved_type': kind,
              'find_object_rule': 'profiles.py:86-93: first kind bench has exactly one candidate; no instance_hint; return candidates[0], kind. Nearest-anchor branch 94-100 NOT reached.',
              'seed_world_m': seed.tolist(), 'target_region_h0_m': float(reg.h0),
              'saved_support_object': rep['support_object'], 'saved_support_parts_on': rep['support_parts_on'],
              'saved_support_info': rep['support_info'], 'saved_zone': rep['zone'],
              'geometry_diagnostics': geom, 'part_vertex_heights': heights, 'joint_heights': joint_heights,
              'support_world_bbox_m': [support.min(0).tolist(),support.max(0).tolist()],
              'input_sha256_before': before,
              'input_sha256_after': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched if p.is_file()}}
    save(STATE / 'Q11_diagnostics.json', result)
    print('Q11 read-only diagnostics saved ' + str(STATE / 'Q11_diagnostics.json'))


def q11_blender():
    # Existing renderer imports the exact saved plate/body meshes. Only camera/materials change.
    import bpy
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    sys.path.insert(0, str(REPO / 'tools/scene3d/plate'))
    import render_plate_scene as renderer
    data = json.loads((STATE / 'Q11_diagnostics.json').read_text(encoding='utf-8'))
    pd = Path(data['plate_dir'])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    plate = renderer.import_plate(str(pd / 'plate_mesh.obj'))
    info = json.loads((pd / 'plate.json').read_text(encoding='utf-8'))
    cam = renderer.setup_plate_camera(sc, info)
    bpy.context.view_layer.update()
    uv = plate.data.uv_layers.new(name='OriginalPhotoProjection')
    for loop in plate.data.loops:
        P = plate.matrix_world @ plate.data.vertices[loop.vertex_index].co
        p = world_to_camera_view(sc, cam, P)
        uv.data[loop.index].uv = (p.x, p.y)
    mat = bpy.data.materials.new('Exact photo mapped onto existing mesh')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    em = nodes.new('ShaderNodeEmission')
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(str(pd / 'plate_rgb.png'))
    mat.node_tree.links.new(tex.outputs['Color'], em.inputs['Color'])
    mat.node_tree.links.new(em.outputs[0], output.inputs['Surface'])
    plate.data.materials.clear()
    plate.data.materials.append(mat)
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(Path(data['job']) / 'fit/body.obj'), forward_axis='Y', up_axis='Z')
    body_objs = [o for o in bpy.data.objects if o not in before]
    clay = bpy.data.materials.new('Saved Q11 body')
    clay.use_nodes = True
    bs = clay.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (0.65, 0.65, 0.65, 1)
    bs.inputs['Emission Color'].default_value = (0.19, 0.19, 0.19, 1)
    bs.inputs['Emission Strength'].default_value = 1
    for ob in body_objs:
        ob.data.materials.clear()
        ob.data.materials.append(clay)
    # Exact downward rays against saved plate, no fabricated object surface.
    support = json.loads((STATE / 'Q11_support_vertices.json').read_text(encoding='utf-8'))
    hits = []
    inv = plate.matrix_world.inverted()
    for p in support:
        origin = Vector((p[0],p[1],p[2]+0.0001))
        ok, pos, normal, face = plate.ray_cast(inv @ origin, inv.to_3x3() @ Vector((0,0,-1)))
        if ok:
            pos = plate.matrix_world @ pos
            hits.append({'support_world_m':p,'hit_world_m':list(pos),'gap_m':p[2]-pos.z,'mesh_face':face})
    save(STATE / 'Q11_vertical_mesh_hits.json', {'n_support':len(support),'n_hits':len(hits),'hits':hits})
    cam.location = (1.6, 5.8, 12)
    cam.rotation_euler = (0,0,0)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 7.4
    # Top cutaway: camera clips only geometry above z=1.8m; saved meshes unchanged.
    cam.data.clip_start = 10.2
    cam.data.clip_end = 30
    sc.camera = cam
    sc.render.engine = 'BLENDER_EEVEE'
    sc.render.resolution_x = 1400
    sc.render.resolution_y = 1100
    sc.render.resolution_percentage = 100
    sc.world = bpy.data.worlds.new('Diagnostic world')
    sc.world.color = (0.65,0.65,0.65)
    sc.view_settings.view_transform = 'Standard'
    sc.render.image_settings.file_format = 'PNG'
    sc.render.filepath = str(STATE / 'Q11_top_raw.png')
    bpy.ops.render.render(write_still=True)
    # Projection metadata lets Pillow label exact world centers without guessing.
    bpy.context.view_layer.update()
    labels = []
    for o in data['objects']:
        p = world_to_camera_view(sc,cam,Vector(o['center_world_m']))
        labels.append({'id':o['id'],'pixel':[p.x*1400,(1-p.y)*1100]})
    save(STATE / 'Q11_top_projection.json', {'labels':labels,'camera_world_m':list(cam.location),
                                           'ortho_scale_m':cam.data.ortho_scale,'cutaway_above_z_m':1.8})


def finish_report():
    import statistics
    import shutil
    import datetime
    from PIL import Image, ImageDraw, ImageFont
    q01 = json.loads((STATE / 'Q01_profile.json').read_text(encoding='utf-8'))
    q11 = json.loads((STATE / 'Q11_diagnostics.json').read_text(encoding='utf-8'))
    proj = json.loads((STATE / 'Q11_top_projection.json').read_text(encoding='utf-8'))
    ray = json.loads((STATE / 'Q11_vertical_mesh_hits.json').read_text(encoding='utf-8'))
    # Labels in neutral ink, exact projected centers, no categorical palette.
    im = Image.open(STATE / 'Q11_top_raw.png').convert('RGB')
    canvas = Image.new('RGB',(1800,1200),'white')
    canvas.paste(im,(0,100))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    draw.text((24,12),'Q11 / existing fresh generic CLI09c / TOP -Z, +Y up',fill='black',font=font)
    draw.text((24,44),'Exact saved meshes; original photo projected; cutaway above z=1.8 m; NOT corrected geometry',fill='black',font=small)
    positions = {'bench_1':(970,705),'floor_1':(960,880),'wall_1':(30,920),'wall_2':(1130,500),
                 'wall_3':(1100,190),'wall_4':(240,735),'wall_5':(630,180)}
    for row in proj['labels']:
        xy = (row['pixel'][0],row['pixel'][1]+100)
        text = row['id'] + (' / selected target' if row['id']=='bench_1' else '')
        pos = positions[row['id']]
        draw.line([xy,(pos[0],pos[1]+14)],fill='black',width=2)
        draw.ellipse((xy[0]-4,xy[1]-4,xy[0]+4,xy[1]+4),fill='white',outline='black',width=2)
        box = draw.textbbox(pos,text,font=font)
        draw.rectangle((box[0]-6,box[1]-4,box[2]+6,box[3]+4),fill='white',outline='black',width=1)
        draw.text(pos,text,fill='black',font=font)
    draw.rectangle((1400,100,1800,1200),fill='white')
    for i,line in enumerate(['PROFILE INSTANCES','bench_1: bench, hard','h0 stored: 0.350 m','region h0: 0.349760 m','center XY: 1.6879,5.5153','','Pink sofa visible in photo:','NO instance in profile','type/id/h0/center: unknown','Not substituted into bench.','','Saved status: wrong_surface','saved support_object: other','','Measured contact: bench_1','63 / 179 support vertices','min target distance: 2.08 cm','target gap p50: 4.36 cm','','Gray figure = saved body','Wall labels = profile centers','No full object masks.']):
        draw.text((1420,120+i*34),line,fill='black',font=small)
    # A supplementary source-photo strip identifies sofa visually, without inventing its mesh center.
    source = Image.open(Path(q11['plate_dir']) / 'plate_rgb.png').convert('RGB')
    crop = source.crop((1100,510,1360,740))
    crop.thumbnail((260,220))
    canvas.paste(crop,(1425,935))
    draw.text((1420,900),'Pink sofa / source-photo crop',fill='black',font=small)
    image_path = REPO / 'docs/experiments/img/Q11_top_09d.jpg'
    canvas.save(image_path,quality=93)
    stats = pstats.Stats(q01['pstats'])
    def fn(k,v):
        return {'file':k[0],'line':k[1],'function':k[2],'calls':v[1],'self_s':v[2],'cum_s':v[3]}
    geometry = sorted([fn(k,v) for k,v in stats.stats.items() if 'ComfyUI-Farm' in k[0]],key=lambda r:r['cum_s'],reverse=True)
    # penetration_values encloses 5 penetration calls; add only other callers to avoid double count.
    pen = next(v for k,v in stats.stats.items() if k[2]=='penetration')
    outside = sum(v[3] for k,v in pen[4].items() if k[2]!='penetration_values')
    total_pen = next(v[3] for k,v in stats.stats.items() if k[2]=='penetration_values') + outside
    place_s = next(v[3] for k,v in stats.stats.items() if k[2]=='place')
    q01.update({'geometry_top10_cumtime':geometry[:10],
                'penetration_disjoint_cum_s':total_pen,'penetration_fraction_of_place':total_pen/place_s,
                'penetration_fraction_of_profiled_wall':total_pen/q01['elapsed_s'],
                'refine_ctx_available':True,'refine_mem_calls':0,
                'refine_evidence':'SAM3DBody_Loader.execute in pstats; load_ctx returned without exception; first rigid fit ok -> break before refine_mem',
                'body_dims_source':'C:/work/scene3d_work/farm/characters/stacy/body.json',
                'scope_note':'runner cached reference, not fresh generic SHA256 reference; excludes MoGe/SAM ref extraction, Plate construction, Blender, maps and H3',
                'cprofile_note':'Raw cProfile total_tt23.769s > perf wall21.219s; nested import cProfile timings not suitable for summing. Outer load_ctx/load_head frames absent in stats despite successful loader.execute, so import/runtime registration may affect enclosing frames; exact place cumtime and raw pstats preserved.'})
    baselines = []
    root = STATE.parent
    backup = root / 'pose_round/V2_before_09d_20261109_144847'
    for pid in ('Q03_bed_kneel','T02_kneel_lunge'):
        p = backup / pid / 'fit.json'
        if not p.is_file():
            p = root / 'pose_round/V2' / pid / 'fit.json'
        r = json.loads(p.read_text(encoding='utf-8'))
        baselines.append({'id':pid,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                          'fit_hash':r.get('fit_hash'),'status':r['status'],'penetration':r['penetration'],
                          'mtime_utc':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),
                          'freshness':'Q03 latest completed09c 2ae; T02 stale/not rerun09c2ae',
                          'current_baseline_eligible':pid.startswith('Q03')})
    hits = ray['hits']
    bench_hits = [h for h in hits if .30 <= h['hit_world_m'][2] <= .40]
    floor_hits = [h for h in hits if abs(h['hit_world_m'][2]) < .04]
    q11['vertical_mesh_rays'] = {'support_vertices':ray['n_support'],'hits':ray['n_hits'],
                                'bench_height_hits':len(bench_hits),'floor_height_hits':len(floor_hits),
                                'hit_height_min_p50_max_m':[min(h['hit_world_m'][2] for h in hits),statistics.median(h['hit_world_m'][2] for h in hits),max(h['hit_world_m'][2] for h in hits)],
                                'vertical_gap_min_p50_max_m':[min(h['gap_m'] for h in hits),statistics.median(h['gap_m'] for h in hits),max(h['gap_m'] for h in hits)],
                                'interpretation':'Actual mesh under selected support is mostly bench-height, remaining rays floor-height; does not establish all support groups in valid contact. No sofa instance exists.'}
    q11['inputs_unchanged'] = q11['input_sha256_before']==q11['input_sha256_after']
    report = {'head_at_profile':'e16da79914c18d966896d404625ae0324050e59c','scope':'facts only2,3; no source geometry/pack/workflow changes; no server; no commit/push',
              'q01':q01,'existing_baselines':baselines,'q11':q11,'top_view':{'path':str(image_path),**proj}}
    save(REPO / 'docs/experiments/3DA_PROFILE_2026-11-09d.json',report)
    dest=Path('C:/Users/user/Downloads/3DA_PROFILE_2026-11-09d')
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(image_path,dest/image_path.name)
    print(json.dumps({'penetration_total_s':total_pen,'place_fraction':total_pen/place_s,'rays_bench':len(bench_hits),'rays_floor':len(floor_hits),'image':str(image_path)},indent=1))


if __name__ == '__main__':
    if '--' in sys.argv:
        q11_blender()
        raise SystemExit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['profile', 'profile-worker', 'q11', 'report'])
    args = parser.parse_args()
    if args.mode == 'profile':
        raise SystemExit(supervise())
    if args.mode == 'q11':
        q11_worker()
    elif args.mode == 'report':
        finish_report()
    else:
        profile_worker()
