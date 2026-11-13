"""Regression checks for scene selection, fail-closed native H3 and AIO wiring."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/scene3d'),str(ROOT/'tools/scene3d/posev2')]
import scene_edit
import wfkit

spec=importlib.util.spec_from_file_location('aio_test_nodes',ROOT/'comfyui/ComfyUI-3DA/nodes.py')
nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)


def guides(size=128):
    rgb=torch.zeros((1,size,size,3));mask=torch.zeros((1,size,size));mask[:,size//2,size//2]=1
    return dict(validation={'ok':True,'reasons':[]},plate=rgb,init=rgb.clone(),playblast=rgb.clone(),strength=mask,
                prompt='accepted body prompt',prop_rgb=rgb.clone(),prop_alpha=torch.zeros_like(mask))


def test_native_gate_never_raises_and_falls_back():
    # 0.3.0: битые/пустые карты — запасной путь (mode=1, обычное фото), без исключения.
    for data in ({}, None, {'fatal':'комната не подготовлена'}, {'validation':{'ok':True}}):
        out=nodes.NativeGuides().execute(data)
        assert out[7] is False and out[8]==1 and out[9].startswith('3DA не сработал')
    assert 'шаг 1' in nodes.NativeGuides().execute({})[9]
    assert 'комната не подготовлена' in nodes.NativeGuides().execute({'fatal':'комната не подготовлена'})[9]
    g=guides();g['strength'][0,0,0]=float('nan')
    assert nodes.NativeGuides().execute(g)[8]==1
    ok=nodes.NativeGuides().execute(guides())
    assert ok[7] is True and ok[8]==2 and ok[9]=='3DA: сцена готова, H3 запущен'
    # Проблема геометрии — только предупреждение, H3 запускается.
    g=guides();g['validation']={'ok':False,'reasons':['Персонаж: опора выходит за край поверхности (не хватает 35 % площади)'],
                                'warnings':['body_partly_out_of_frame']}
    out=nodes.NativeGuides().execute(g)
    assert out[7] is True and out[8]==2
    assert 'Предупреждение (MiniMax исправит): Персонаж: опора' in out[9] and 'тело частично вне кадра' in out[9]
    # Режим 1 пользователя: сцена не запрашивается (ленивый вход) и не нужна.
    assert nodes.NativeGuides().check_lazy_status(body_guides=None,mode=1)==[]
    assert nodes.NativeGuides().check_lazy_status(body_guides=None,mode=2)==['body_guides']
    assert nodes.NativeGuides().execute(None,mode=1)[9].startswith('Режим 1')


def test_rejected_geometry_with_mesh_can_open_editor_but_cannot_run_h3():
    from types import SimpleNamespace
    g = guides()
    g['validation'] = {'ok': False, 'reasons': ['support_gap']}
    g['scene_setup'] = {'body_obj': 'real-body.obj', 'placement_key': 'fresh'}
    calls = []
    def export(setup, cfg):
        calls.append(setup)
        return {'model_file': 'scene.glb', 'manifest': 'scene.json'}
    with patch.object(nodes, 'config', return_value={}), patch.object(nodes, 'scene_helpers', return_value=SimpleNamespace(export=export)):
        result = nodes.SceneExport().execute(g)
    assert result['result'] == ('scene.glb', 'scene.json')
    assert calls == [g['scene_setup']]
    assert nodes.NativeGuides().execute(g)[8]==2  # 0.3.0: не блокирует


def test_scene_edits_new_room_camera_and_fingerprint(tmp_path):
    from types import SimpleNamespace
    directory = tmp_path / 'locations' / 'fresh'
    directory.mkdir(parents=True)
    profile = directory / 'profile.json'
    profile.write_text(json.dumps({'loc_id': 'fresh'}))
    plate = directory / 'plate.json'
    plate.write_text('{}')
    cfg = dict(repo=str(ROOT), data_root=str(tmp_path))
    helper = SimpleNamespace(read_manifest=lambda *a: {'location_id': 'fresh'}, check_file=lambda *a: None,
                             edits=lambda *a: ('body', 'props'))
    cameras = []
    camera = SimpleNamespace(camera_config=lambda p, size: (p, size), validate_camera=lambda *a: cameras.append(a))
    with patch.object(nodes, 'config', return_value=cfg), patch.object(nodes, 'scene_helpers', return_value=helper), patch.object(nodes, 'gizmo_helpers', return_value=camera):
        first = nodes.SceneEdits.IS_CHANGED(location=str(profile))
        assert nodes.SceneEdits().execute(None, [], 'actual-camera', str(profile), '') == ('body', 'props')
        plate.write_text('{"new": true}')
        assert nodes.SceneEdits.IS_CHANGED(location=str(profile)) != first
    assert cameras[0][0] == 'actual-camera'


def test_final_composition_restores_room_after_facepass_and_source_prop():
    g=guides(512);g['plate'][:]=.125;g['prop_rgb'][:,10:15,10:15]=.75;g['prop_alpha'][:,10:15,10:15]=1
    facepass=torch.ones_like(g['plate']);repair=g['plate'].clone();repair_mask=torch.zeros_like(g['strength']);repair_mask[:,:40,:40]=1
    with patch.object(nodes,'config',return_value={'repo':str(ROOT)}):
        final,support,metrics=nodes.NativeComposite().execute(facepass,g,repair,repair_mask)
    outside=~support.bool()
    assert outside.any() and torch.equal(final[outside],g['plate'][outside])
    assert torch.all(final[:,10:15,10:15]==.75)
    assert torch.all(final[:,256,256]==1)
    assert json.loads(metrics)['outside_changed_pixels']==0


def test_spatial_ssim_is_measured_and_can_fail_despite_zero_changed_pixels():
    g=guides();g['plate'][:]=.125;g['prop_rgb'][:,10:15,10:15]=.75;g['prop_alpha'][:,10:15,10:15]=1
    repair_mask=torch.zeros_like(g['strength']);repair_mask[:,6:20,6:20]=1
    with patch.object(nodes,'config',return_value={'repo':str(ROOT)}):
        metrics=json.loads(nodes.NativeComposite().execute(torch.ones_like(g['plate']),g,g['plate'],repair_mask)[2])
        assert metrics['outside_passed'] is False and 'warning' in metrics


def test_final_is_in_h3_raster_not_1mp_plate_rgb():
    # 0.3.4: МП пульта = МП итога. Старый original_plate (plate_rgb 1344×768) больше не задаёт размер.
    g=guides(256);g['original_plate']=torch.rand((1,96,168,3));g['plate']=torch.rand_like(g['plate'])
    image=torch.ones_like(g['plate'])
    with patch.object(nodes,'config',return_value={'repo':str(ROOT)}):
        final,support,metrics=nodes.NativeComposite().execute(image,g,g['plate'],torch.zeros_like(g['strength']))
    assert final.shape==image.shape
    assert torch.equal(final[~support.bool()],g['plate'][~support.bool()])
    assert json.loads(metrics)['final_raster_hw']==[256,256]


def test_prop_by_h3_keeps_h3_pixels_and_never_pastes_render_or_shadow():
    # 0.3.6: «предмет рисует H3» — в области силы пиксели H3, вне её чистая комната; без вклейки рендера и тени.
    g=guides();g['plate'][:]=.125;g['strength'][:]=0;g['strength'][:,40:60,40:60]=1
    g['prop_rgb'][:,45:50,45:50]=.9;g['prop_alpha'][:,45:50,45:50]=1;g['prop_shadow']=torch.ones_like(g['strength'])
    g['prop_mode']='h3';image=torch.full_like(g['plate'],.5)
    with patch.object(nodes,'config',return_value={'repo':str(ROOT)}):
        final,support,metrics=nodes.NativeComposite().execute(image,g,g['plate'],torch.zeros_like(g['strength']))
    assert torch.all(final[:,45:50,45:50]==.5) and torch.all(final[:,0:10,0:10]==.125)
    assert json.loads(metrics)['prop_mode']=='h3'


def test_shadow_rule_is_one_for_body_and_prop():
    # 0.3.7: «тени от H3» — тень рендера не вклеивается; «от рендера» — тень (тело+предмет) на комнате вне ядра тела.
    g=guides();g['plate'][:]=.5;g['strength'][:]=0;g['strength'][:,40:60,40:60]=1;g['prop_shadow']=torch.ones_like(g['strength'])
    image=torch.full_like(g['plate'],.5)
    with patch.object(nodes,'config',return_value={'repo':str(ROOT)}):
        g['shadow_mode']='h3'
        final=nodes.NativeComposite().execute(image,g,g['plate'],torch.zeros_like(g['strength']))[0]
        assert torch.all(final==.5)
        # 0.3.8: «от рендера» — тень уже в init (подсказка), H3 рисует её в зоне; после H3 второй раз не кладём.
        g['shadow_mode']='render';g['prop_mode']='h3'
        shadow=torch.zeros_like(g['strength']);shadow[:,70:80,70:80]=.5;g['room_shadow']=shadow
        final=nodes.NativeComposite().execute(image,g,g['plate'],torch.zeros_like(g['strength']))[0]
        assert torch.all(final==.5)


def test_render_shadow_skips_body_core(tmp_path):
    from PIL import Image
    Image.fromarray(np.full((32,32),255,np.uint8)).save(tmp_path/'s.png')
    strength=torch.zeros((1,32,32));strength[:,10:20,10:20]=1
    shadow=nodes.render_shadow({'body_shadow':str(tmp_path/'s.png')},strength)
    assert float(shadow[0,15,15])==0 and float(shadow[0,2,2])==1


def transform(entity,x=0,y=0,z=0):
    return dict(entity_id=entity,position=dict(x=x,y=y,z=z),quaternion=dict(x=0,y=0,z=0,w=1),scale=dict(x=1,y=1,z=1))


def test_selected_entity_does_not_move_other_entity():
    prop=dict(prop_id='fixture',anchor='world',x=0,y=0,z=0,yaw=0,pitch=0,roll=0,scale=1,size_m=.3,prop_strength=.35)
    manifest=dict(placement_key='pose-a',entities=[dict(id='body',kind='body',matrix_world=np.eye(4).tolist()),dict(id='prop_0',kind='prop',prop=prop)])
    body,props=scene_edit.edits(manifest,[transform('body',x=.2),transform('prop_0',x=.7)])
    np.testing.assert_allclose(np.array(json.loads(body)['matrix'])[:3,3],[.2,0,0])
    np.testing.assert_allclose(np.array(json.loads(props)[0]['absolute_matrix_world'])[:3,3],[.7,0,0])
    with pytest.raises(ValueError,match='другой позы'):scene_edit.body_delta(body,'pose-b')
    with pytest.raises(ValueError,match='отсутствуют'):scene_edit.edits(manifest,[transform('body')])
    scaled=transform('body');scaled['scale']['x']=2
    with pytest.raises(ValueError,match='равномерным'):scene_edit.edits(manifest,[scaled,transform('prop_0')])


def test_body_scale_and_rotation_match_viewport_without_changing_prop():
    from posev2.manual_pose import transform as body_transform
    # A rotated, scaled body root around its original pelvis, independently
    # of a prop with its own uniform scale/rotation (props keep proportions).
    initial=np.eye(4);initial[:3,3]=[.2,1.5,.6]
    manifest=dict(placement_key='same',entities=[dict(id='body',kind='body',matrix_world=initial.tolist()),
        dict(id='prop_0',kind='prop',prop={'prop_id':'fixture'})])
    body=transform('body',x=.3,y=.6,z=-1.5);body['scale']=dict(x=1.2,y=1.2,z=1.2)
    body['quaternion']=dict(x=0,y=np.sin(.2),z=0,w=np.cos(.2))
    prop=transform('prop_0',x=.7);prop['scale']=dict(x=1.1,y=1.1,z=1.1)
    prop['quaternion']=dict(x=np.sin(.1),y=0,z=0,w=np.cos(.1))
    delta,props=scene_edit.edits(manifest,[body,prop]);matrix=np.asarray(json.loads(delta)['matrix'])
    x=np.array([[.2,1.5,.6],[1.2,1.5,.6],[.2,2.5,.6]])
    actual,kp,metadata=body_transform(x,x,None,{'body_matrix':scene_edit.body_delta(delta,'same')})
    np.testing.assert_allclose(actual,x@matrix[:3,:3].T+matrix[:3,3])
    np.testing.assert_allclose(kp,actual)
    np.testing.assert_allclose(np.linalg.norm(actual[1]-actual[0]),1.2)
    assert metadata['body_scale']==pytest.approx(1.2)
    np.testing.assert_allclose(json.loads(props)[0]['absolute_matrix_world'],scene_edit.model_matrix([prop]))
    for bad in [np.diag([-1.,1.,1.,1.]),np.zeros((4,4)),np.full((4,4),np.nan)]:
        with pytest.raises(ValueError):scene_edit.body_matrix(bad)
    shear=np.eye(4);shear[0,1]=.2
    with pytest.raises(ValueError,match='равномерным'):scene_edit.body_matrix(shear)
    stretched=dict(prop,scale=dict(x=.8,y=1.1,z=1.3))
    with pytest.raises(ValueError,match='предмета должен быть равномерным'):scene_edit.edits(manifest,[body,stretched])


def test_body_rigid_transform_preserves_pose_and_dimensions():
    from posev2.manual_pose import transform as body_transform
    x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,1.]])
    matrix=np.eye(4);matrix[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];matrix[:3,3]=[.2,.4,.1]
    actual,kp,metadata=body_transform(x,x,None,{'body_matrix':matrix.tolist(),'body_dx':999})
    np.testing.assert_allclose(actual,x@matrix[:3,:3].T+matrix[:3,3])
    np.testing.assert_allclose(np.linalg.norm(actual[:,None]-actual[None],axis=-1),np.linalg.norm(x[:,None]-x[None],axis=-1))


def test_generated_aio_preserves_legacy_modules_and_mod_conditioning():
    w=json.loads((ROOT/'workflows/3da_aio/3DA_AIO_v1.0.ui.json').read_text(encoding='utf-8'))
    assert not wfkit.check_ui(w)
    assert w['extra']['3DA AIO']['source_sha256']=='5b6e88d9f743e516d9f059e50081513c788a9bfd1a57f21aa22f8e775e7cd1b6'
    graphs=[w]+w['definitions']['subgraphs'];allnodes=[n for g in graphs for n in g['nodes']]
    assert not any(n['type']=='3DAH3Remote' for n in allnodes)
    assert sum(n['type']=='MiniMaxH3RefModsLoader' for n in allnodes)==2
    assert sum(n['type']=='MiniMaxH3RefModApply' for n in allnodes)==3
    assert sum(n['type']=='VHS_VideoCombine' for n in allnodes)==2
    qwen=[n for n in allnodes if n['type']=='TextEncodeQwenImageEditPlus']
    assert len(qwen)==2
    for n in qwen:
        inputs={i['name']:i for i in n['inputs']}
        assert inputs['image1']['link'] and inputs['image2']['link'] and inputs['image1']['link']!=inputs['image2']['link']
    switch=next(n for n in allnodes if n.get('title','').startswith('3DA · pose_source'))
    assert switch['widgets_values_named']['sel_mode'] is True
    for g in w['definitions']['subgraphs']:
        if g['id']=='8c87aaa9-03cc-4a78-b986-9dbd7df00baf':assert len(g['nodes'])==70
        if g['id']=='eeafa102-d4f0-4f51-a26e-715929de67b7':assert len(g['nodes'])==39


@pytest.mark.parametrize('count',[1,2,8])
def test_refmod_uses_distinct_optional_inputs(count):
    w=json.loads((ROOT/f'workflows/3da_aio/3DA_RefMod_1-8.{count}.ui.json').read_text(encoding='utf-8'))
    assert not wfkit.check_ui(w)
    loads=[n for n in w['nodes'] if n['type']=='LoadImage'];assert len(loads)==8
    extract=next(n for n in w['nodes'] if n['type']=='MiniMaxH3RefModExtract')
    links={wfkit._norm(l)['id']:wfkit._norm(l) for l in w['links']}
    refs=[links[i['link']]['origin_id'] for i in extract['inputs'] if i['name'].startswith('refs_image.')]
    assert len(refs)==len(set(refs))==count
    assert extract['widgets_values_named']['save'] is False
    assert extract['widgets_values_named']['budget_policy']=='error'
    assert extract['widgets_values_named']['merge'] is False


def test_manager_web_copy_matches_standalone():
    for p in (ROOT/'comfyui/ComfyUI-3DA/web').glob('*.js'):
        assert p.read_bytes()==(ROOT/'web/3da'/p.name).read_bytes()
