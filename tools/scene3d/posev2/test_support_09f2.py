"""09f2 4a/b/c: CPU-only geometry; synthetic SFW bodies and locations."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly
import auto_surfaces
import orient as ori
import place
import shot_spec as ss
import top_region as tr
from test_step9c import evaluation, skeleton


def region(mask, z, origin=(0., 0.), oid='legacy'):
    mask = np.asarray(mask, bool)
    reg = tr.TopRegion('bed', 'synthetic', np.array(origin), mask.shape,
        np.where(mask, z, np.nan), mask,
        ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * tr.CELL,
        z, tuple(np.argwhere(mask)[0]), np.array([-2., -2.]))
    reg.object_id = oid
    return reg


@pytest.mark.parametrize('cls', ['stand', 'crouch'])
@pytest.mark.parametrize('raised_side', ['L', 'R'])
def test_lowest_foot_only_not_hand_or_raised_leg(cls, raised_side):
    names = np.repeat(['foot.L', 'foot.R', 'hand.L'], 8)
    V = np.column_stack([np.zeros((24, 2)), np.r_[np.linspace(0., .2, 8), np.linspace(.35, .5, 8), np.full(8, -.2)]])
    if raised_side == 'L':
        V[:8, 2], V[8:16, 2] = V[8:16, 2].copy(), V[:8, 2].copy()
    spec = ss.ShotSpec('synthetic', cls, 'floor', ['foot.' + raised_side])
    B = dict(V=V, kp=skeleton(), reg=names, ref_c={})
    groups, _, _ = place.build_groups(B, spec, None, 1.62, False)
    assert len(groups) == 1
    S = {'target': place._surf(np.array([[0., 0., 0.]]))}
    contact = place.support_contact(V, groups, S, spec, False, reg=names)
    assert contact['ok'] and not contact['flags']
    assert contact['lowest_gap_m'] == 0.
    assert list(contact['part_gaps_m']) == ['foot.' + ('R' if raised_side == 'L' else 'L')]
    with patch.object(place, 'penetration_stats', return_value={'ok': True, 'excess_m': 0.}):
        fitted, _, info = assembly.height_fit(V + [0, 0, .3], skeleton(), groups, S, None, spec, None, False, 1.62)
    assert fitted[groups[0][1][0], 2] == pytest.approx(0.)
    assert fitted[-1, 2] == pytest.approx(-.2)  # hand never becomes the height pivot
    ev = evaluation()
    ev.update(support_object='floor', support_part_gaps_m=contact['part_gaps_m'], support_contact=contact)
    with patch.object(assembly, 'orientation_checks', return_value=[]):
        checked = assembly.validate(None, ev, spec, None, 0, None, True, height_info=info)
    assert checked['status'] == 'ok' and 'support_gap' not in checked['flags']
    floor = region(np.ones((20, 20)), 0., origin=(-.2, -.2), oid='floor')
    camera = SimpleNamespace(Wp=100, Hp=100, project=lambda x: (np.full(len(x), 50.), np.full(len(x), 50.), np.ones(len(x))))
    with patch.object(place, 'penetration_stats', return_value={'ok': True, 'max_cm': 0.}):
        measured = place.evaluate(V, skeleton(), B, groups, S, camera, spec, floor, floor, 1.62, False)
    assert measured['support_object'] == 'floor'
    assert measured['support_gap_cm'] == 0.
    assert measured['support_contact']['flags'] == []


@pytest.mark.parametrize('cls', ['stand', 'crouch'])
def test_missing_feet_cannot_claim_support(cls):
    spec = ss.ShotSpec('synthetic', cls, 'floor', ['knee.L'])
    contact = place.support_contact(np.zeros((3, 3)), [('knee.L', np.arange(3), 'target', 'support')],
        {'target': place._surf(np.zeros((1, 3)))}, spec, False, reg=np.full(3, 'knee.L'))
    assert not contact['ok']


def test_match_fraction_denominator_boundaries_origins_and_largest_area():
    legacy = region(np.ones((3, 10)), .5, origin=(.2, .4))
    candidates = [region(np.ones((10, 10)), .7, origin=(.2, .4), oid='surf_2'),
                  region(np.ones((3, 15)), .5, origin=(.2, .4), oid='surf_1'),
                  region(np.ones((11, 10)), .9, origin=(.2, .4), oid='surf_3')]
    with patch.object(auto_surfaces, 'discover_surfaces', return_value=(candidates, {})):
        selected, info = place.match_auto_support(None, legacy, 1.62)
    assert selected.object_id == 'surf_2'  # 100 cells > 45, although higher
    assert [r['id'] for r in info['surface_candidates']] == ['surf_2', 'surf_1']
    assert info['surface_candidates'][0]['overlap_fraction'] == pytest.approx(.30)
    assert not info['soft_surfaces']


@pytest.mark.parametrize('height,sink', [(1.62, .1134), (2.30, .161)])
def test_largest_support_and_soft_sink_from_body_dims(height, sink):
    legacy = region(np.ones((30, 30)), .5)
    mattress = region(np.ones((30, 30)), .5, oid='surf_1')
    pillow = region(np.ones((20, 20)), .7, oid='surf_2')
    edge = region(np.ones((10, 10)), .58, origin=(.4, .4), oid='surf_3')
    with patch.object(auto_surfaces, 'discover_surfaces', return_value=([pillow, edge, mattress], {})):
        selected, info = place.match_auto_support(None, legacy, height)
    assert selected.object_id == 'surf_1'
    assert info['soft_surfaces'] == ['surf_2']  # exactly 8cm is NOT soft
    assert selected.soft_surface_limits['surf_2'] == pytest.approx(sink)
    selected.diagnostic_surfaces = [dict(id=r.object_id, region=r, soft=r is pillow) for r in (mattress, pillow, edge)]
    P = pillow.points3d()[:2]
    plate = SimpleNamespace(trees={'any': place._surf(P)})
    with patch.object(place.pf, 'penetration', return_value=np.array([sink, sink])):
        pen = place.penetration_stats(plate, P, selected, False, 'lie')
    assert pen['ok']
    assert pen['surfaces']['soft:surf_2']['threshold_m'] == pytest.approx(sink)
    with patch.object(place.pf, 'penetration', return_value=np.array([sink + 1e-6] * 2)):
        assert not place.penetration_stats(plate, P, selected, False, 'lie')['ok']
    with patch.object(place, 'penetration_values', return_value=(np.array([.031]), np.array(['hard:wall_1']))):
        assert not place.penetration_stats(None, None, selected, False, 'lie')['ok']
    S = {'target': place._surf(mattress.points3d()), 'soft_support_region': selected}
    spec = ss.ShotSpec('synthetic', 'lie', 'bed', ['pelvis'])
    X = P - [0, 0, sink]
    contact = place.support_contact(X, [('pelvis', np.arange(2), 'target', 'support')], S, spec, False)
    assert contact['ok'] and not contact['flags']
    camera = SimpleNamespace(Wp=100, Hp=100, project=lambda x: (np.full(len(x), 50.), np.full(len(x), 50.), np.ones(len(x))))
    assert place.part_label(X, spec, selected, None, camera, False) == 'surf_1'
    X[:, 2] = .7 + .001  # above, not embedded: no free gap waiver
    assert place.support_contact(X, [('pelvis', np.arange(2), 'target', 'support')], S, spec, False)['flags']


@pytest.mark.parametrize('cls', ['sit_edge', 'sit_tucked', 'recline', 'lie'])
def test_resolve_no_match_keeps_09e_geometry_and_flags(cls):
    legacy = region(np.ones((10, 10)), .5)
    floor = region(np.ones((10, 10)), 0., origin=(2., 2.), oid='floor_1')
    obj = dict(id='bed_1', name='bed', type='bed', soft=True)
    profile = SimpleNamespace(find_object=lambda name, *args: (None, None) if name == 'floor' else (obj, 'bed'),
        region=lambda _: legacy, anchor=lambda *args: None)
    with patch.object(tr, 'top_region', return_value=floor), \
         patch.object(auto_surfaces, 'discover_surfaces', return_value=([floor], {})):
        treg, freg, source, info = place.resolve_regions(None, ss.ShotSpec('synthetic', cls, 'bed', ['pelvis']), np.zeros(3), 1.62, profile)
    assert treg is legacy and freg is floor and source == 'profile'
    assert info['surface_match'] == 'no_surface_match'
    assert not hasattr(treg, 'soft_surface_limits')


def test_no_surface_flag_and_body_dims_height_reach_early_report():
    legacy = region(np.ones((4, 4)), .5)
    B = dict(V=skeleton(), kp=skeleton(), reg=np.full(70, 'pelvis'), ref_c={}, dims={'H': 1.62})
    spec = ss.ShotSpec('synthetic', 'lie', 'bed', ['pelvis'])
    seen = []
    def resolve(_plate, _spec, _seed, H, _profile):
        seen.append(H)
        return legacy, None, 'profile', {'soft': True, 'surface_match': 'no_surface_match'}
    with patch.object(place, 'prepare_body', return_value=B), patch.object(place, 'resolve_regions', side_effect=resolve), \
         patch.object(place, 'orient_body'), patch.object(place, 'zone_too_small', return_value=True), \
         patch.object(tr, 'walls', return_value=[]):
        result = place.place({}, None, spec, np.zeros(3), 1.62, dims_override={'H': 2.30}, log=lambda _: None)
    assert seen == [2.30]
    assert 'no_surface_match' in result['rep']['flags']
    assert result['rep']['surface_match']['status'] == 'no_surface_match'
    assert result['rep']['surface_match']['soft_surface_ids'] == []


@pytest.mark.parametrize('cls', ['stand', 'kneel', 'crouch'])
@pytest.mark.parametrize('lean', [-40., 40.])
def test_root_clamp_uses_shoulder_midpoint_not_bent_neck(cls, lean):
    K = skeleton(0, lean)
    K[69] = (K[9] + K[10]) / 2 + [0., 0., .7]  # misleading vertical neck
    assert ori.tilt_deg(K, cls) == pytest.approx(abs(lean))
    assert abs(ori.roll_deg(K, cls)) == pytest.approx(abs(lean))
    X, J, _ = ori.assemble_tilt_roll(K.copy(), K, cls)
    assert abs(ori.roll_deg(J, cls)) <= 15. + 1e-8
    assert not any(not c['ok'] for c in assembly.orientation_checks(J, cls, SimpleNamespace(C=np.array([0., -3., 2.])), 0))
    np.testing.assert_allclose(X, J)
    np.testing.assert_allclose((J[9] + J[10]) / 2, (K[9] + K[10]) / 2)
    import refine
    _, refined, info = refine.clamp_refine_roll(K.copy(), K, {}, cls)
    assert abs(ori.roll_deg(refined, cls)) <= 15. + 1e-8
    assert info['root_clamped']
    K2 = K.copy(); K2[69] += [3., -2., 1.]
    _, J2, _ = ori.assemble_tilt_roll(K2.copy(), K2, cls)
    np.testing.assert_allclose(J[[5, 6, 9, 10]], J2[[5, 6, 9, 10]])
