"""Regressions from real core Fit-to-viewer 5m model failure; no live H3."""
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
sys.path.insert(0,str(ROOT/'tools/scene3d'))
from prop_validation import include_props,blocking_reasons
from prop_gizmo import camera_config,validate_camera
spec=importlib.util.spec_from_file_location('guard_nodes',ROOT/'comfyui/ComfyUI-3DA/nodes.py')
nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)


def report(flags):
    return dict(status='ok',validation=dict(ok=True,checks=[],first_failure=None),props=[dict(flags=flags)])


def test_five_metre_out_of_frame_prop_cannot_inherit_body_ok():
    r=include_props(report(dict(out_of_frame=True,prop_out_of_frame=True)))
    assert r['status']=='out_of_frame' and r['body_status']=='ok'
    assert r['body_validation']['ok'] and not r['validation']['ok']
    assert not r['h3_eligibility']['ok'] and blocking_reasons(r)
    assert r['validation']['first_failure']['check']=='prop[0]:frame'
    assert 'prop_out_of_frame' in r['validation']['flags']


@pytest.mark.parametrize('flag',['prop_hits_body','prop_hits_room'])
def test_collision_blocks_final_without_weakening_body_checks(flag):
    r=include_props(report({flag:True}));assert r['status']=='contact_fail'
    assert r['body_validation']['ok'] and not r['h3_eligibility']['ok']


def test_valid_floating_prop_retains_accepted_status():
    r=include_props(report({}));assert r['status']=='ok' and r['validation']['ok']
    assert r['h3_eligibility']['ok'] and not blocking_reasons(r)


def camera():
    p=dict(R_world_from_cv=np.eye(3).tolist(),cam_height_m=1.4,sensor_width_mm=36,lens_mm=24)
    return camera_config(p,[1344,768])

LIMITS=dict(position_m=.001,direction_deg=.05,fov_deg=.01,zoom=.0001)


def test_original_plate_camera_is_accepted_with_float_noise():
    expected=camera();actual=copy.deepcopy(expected);actual['position']['y']+=1e-7
    validate_camera(actual,expected,LIMITS)


@pytest.mark.parametrize('change',['fit','orbit','pan','fov','zoom'])
def test_fit_or_orbit_camera_cannot_silently_drive_plate_gizmo(change):
    expected=camera();actual=copy.deepcopy(expected)
    if change=='fit':actual['position']=dict(x=22.5,y=17.2,z=22.5);actual['target']=dict(x=0,y=2.5,z=0)
    if change=='orbit':actual['target']['x']+=.1
    if change=='pan':actual['position']['x']+=.01;actual['target']['x']+=.01
    if change=='fov':actual['fov']+=1
    if change=='zoom':actual['zoom']=1.1
    with pytest.raises(ValueError,match='Камера гизмо'):validate_camera(actual,expected,LIMITS)


def test_rejected_scene_is_visible_execution_error_before_transport_even_with_prompt():
    image=torch.zeros(1,64,64,3);mask=torch.zeros(1,64,64)
    guides=dict(prompt='body',render_prompt='render',validation=dict(ok=False,reasons=['prop_out_of_frame']))
    with patch.object(nodes,'config',return_value={'repo':str(ROOT)}),patch.object(nodes,'remote_run') as remote:
        with pytest.raises(nodes.GeometryRejected,match='prop_out_of_frame'):
            nodes.H3Remote().execute(image,image,image,image,mask,'render',101,
                body_guides=guides,prop_handling='rendered_separately')
        remote.assert_not_called()


def test_numeric_mode_ignores_unused_viewport_camera(tmp_path):
    directory=tmp_path/'props/p';directory.mkdir(parents=True)
    (directory/'prop.json').write_text(json.dumps({'name':'synthetic'}),encoding='utf-8')
    with patch.object(nodes,'config',return_value={'farm_root':str(tmp_path)}):
        result=nodes.Prop().execute('p','world',0.,0.,0.,0.,0.,0.,1.,
            camera_info={'invalid':'unused'},location='unused/nonexistent.json')
    assert 'absolute_matrix_world' not in json.loads(result[0])[0]
