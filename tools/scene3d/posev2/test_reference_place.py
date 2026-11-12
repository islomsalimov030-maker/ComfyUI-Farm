"""Reference pose transport without a named floor or a manual pose class."""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import reference_place as rp
import shot_spec
from test_step9c import skeleton
from test_support_09f2 import region


def scene():
    reg = region(np.ones((100, 100), bool), .5, (-1., -1.), 'surf_1')
    obj = dict(id='surf_1', name='other', type='other', top_grid_npz='observed.npz', seed_world=[0., 0., .5])
    profile = SimpleNamespace(data={'objects': [obj]}, region=lambda o: reg, walls=lambda: [])
    plate = SimpleNamespace(Wp=100, Hp=100, project=lambda X: (50+X[:, 0]*10, 70-X[:, 2]*10, np.ones(len(X))))
    points = np.array([[-.1,-.1,.5],[.1,-.1,.5],[.1,.1,.5],[-.1,.1,.5],[0.,0.,1.3]])
    ref = dict(V=points, kp=skeleton(), F=np.array([[0,1,2],[0,2,3]]))
    spec = shot_spec.ShotSpec('reference', 'auto', 'auto', ['reference'])
    return profile, plate, ref, spec


def clear_penetration(*args):
    return dict(ok=True, max_cm=0., excess_m=0., surfaces={'hard': dict(ok=True, p95_m=0., threshold_m=.03)})


def rigid_fit(ref, plate, target, height, **kw):
    assert 'forced' not in kw and np.array_equal(kw['target_world'], [0.,0.,.5])
    state = dict(M=np.eye(3), b=np.zeros(3), support=['hand.L'], contacts={'hand.L': (np.arange(4), 'h')}, sup_idx=np.arange(4))
    return ref['V'].copy(), dict(_state=state, contacts={}), np.array(['hand.L']*4+['torso'])


def test_arbitrary_reference_contacts_on_unclassified_surface_without_floor():
    profile, plate, ref, spec = scene()
    with patch.object(rp.pf, 'fit', side_effect=rigid_fit), patch.object(rp.legacy, 'penetration_stats', side_effect=clear_penetration):
        result = rp.place(ref, plate, profile, spec, 1.62)
    assert result['status'] == 'ok'
    assert result['rep']['no_floor_required'] and not result['rep']['pose_class_inferred']
    assert result['rep']['support_parts_on'] == ['hand.L']  # no foot/knee replacement
    np.testing.assert_array_equal(result['X'], ref['V'])
    np.testing.assert_array_equal(result['KP'], ref['kp'])


def test_editor_revalidation_preserves_pose_and_rejects_lost_contact():
    profile, plate, ref, spec = scene()
    plate.R = np.eye(3)
    groups = [('hand.L', np.arange(4), 'h', 'support')]
    base = dict(X=ref['V'], KP=ref['kp'], groups=groups, report={})
    matrix = np.eye(4);matrix[2,3] = .1
    with patch.object(rp.legacy, 'penetration_stats', side_effect=clear_penetration):
        X, KP, report = rp.revalidate(base, plate, profile, spec, {'body_matrix':matrix})
    assert report['status'] == 'contact_fail' and not report['validation']['ok']
    np.testing.assert_allclose(X[:,None]-X, ref['V'][:,None]-ref['V'])
    np.testing.assert_allclose(KP[:,None]-KP, ref['kp'][:,None]-ref['kp'])


def test_no_visible_support_still_has_diagnostic_mesh_and_rejects_h3():
    profile, plate, ref, spec = scene()
    profile.data['objects'] = []
    plate.R = np.eye(3)
    plate.pixel_point = lambda *a: np.array([0.,0.,.5])
    with patch.object(rp, 'height_est', return_value=1.62), patch.object(rp, 'regions', return_value=np.array(['hand.L']*5)), patch.object(rp.legacy, 'penetration_stats', side_effect=clear_penetration):
        result = rp.place(ref, plate, profile, spec, 1.62)
    assert result['X'].shape == ref['V'].shape and result['KP'].shape == ref['kp'].shape
    assert not result['rep']['validation']['ok'] and result['status'] == 'contact_fail'


def test_source_pose_and_penetration_failures_are_not_accepted():
    profile, plate, ref, spec = scene()
    groups = [('hand.L', np.arange(4), 'h', 'support')]
    pen = clear_penetration();pen['ok']=False;pen['surfaces']['hard'].update(ok=False,p95_m=.1)
    with patch.object(rp.legacy, 'penetration_stats', return_value=pen):
        report=rp.evaluate(ref['V'],ref['kp'],groups,plate,profile,spec,{'reference_pose_error_m':.02})
    assert not report['validation']['ok']
    assert {'reference_pose','penetration_surface:hard'} <= set(report['flags'])
