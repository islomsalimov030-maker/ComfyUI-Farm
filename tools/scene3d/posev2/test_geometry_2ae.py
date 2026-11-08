"""09.11b coordinator rules 2a–2e, CPU synthetic regression tests."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import assembly
import body
import place
import profiles
import shot_spec as ss
import top_region as tr
from test_step9c import skeleton, evaluation
from test_step9d import region


class InstanceTests(unittest.TestCase):
    def profile(self, root):
        data = dict(loc_id='test', version=2, reviewed=False, scale={}, substitutes={}, anchors=[],
                    objects=[dict(id='bed_1', name='left', type='bed'),
                             dict(id='bed_2', name='right', type='bed')])
        p = Path(root) / 'profile.json'
        p.write_text(json.dumps(data))
        return profiles.LocationProfile(p)

    def test_type_hint_and_nearest_not_name_or_list_order(self):
        with tempfile.TemporaryDirectory() as root:
            p = self.profile(root)
            self.assertEqual(p.find_object('bed', 'bed_2')[0]['id'], 'bed_2')
            self.assertIsNone(p.find_object('bed', 'sofa_1')[0])
            self.assertIsNone(p.find_object('left')[0])
            self.assertIsNone(p.find_object('bed')[0])
            with patch.object(p, 'region', side_effect=lambda o: SimpleNamespace(
                    points3d=lambda: np.array([[0., 0., .4]]) if o['id'] == 'bed_1' else np.array([[5., 0., .4]]))):
                self.assertEqual(p.find_object('bed', anchor_xy=[4., 0.])[0]['id'], 'bed_2')
            p.data['anchors'] = [dict(id='a', object_id='bed_1', pose_classes=['kneel']),
                                 dict(id='b', object_id='bed_2', pose_classes=['kneel'])]
            with patch.object(p, 'zone_metrics', side_effect=lambda a: {'xy': np.array([[0., 0.]]) if
                              a['id'] == 'a' else np.array([[4., 0.]])}):
                self.assertEqual(p.find_object('bed', anchor_xy=[3., 0.], pose_class='kneel')[0]['id'], 'bed_2')

    def test_floor_hint_missing_is_no_target(self):
        with tempfile.TemporaryDirectory() as root:
            p = self.profile(root)
            spec = ss.ShotSpec('F', 'stand', 'floor', ['foot.L'], instance_hint='floor_missing')
            reg, _, source, info = place.resolve_regions(None, spec, np.zeros(3), 1.62, p)
            self.assertIsNone(reg)
            self.assertEqual(source, 'no_target')
            self.assertIn('hint', info['reason'])

    def test_wrong_surface_compares_instance_even_with_partial_flags(self):
        spec = ss.ShotSpec('T', 'recline', 'bed', ['pelvis'])
        ev = evaluation()
        ev.update(target_instance='bed_1', support_object='bed_2', back_contact_gap_m=None)
        with patch.object(assembly, 'orientation_checks', return_value=[]):
            val = assembly.validate(None, ev, spec, None, 0, None, True)
        self.assertEqual(val['status'], 'wrong_surface')
        self.assertEqual(val['first_failure']['check'], 'support_target')

    def test_contact_labels_actual_instance_not_selected_type(self):
        r1, r2 = region(shape=(10, 10)), region(shape=(10, 10), origin=(1., 0.))
        r1.object_id = 'bed_1'
        r1.diagnostic_surfaces = [dict(id='bed_1', soft=True, region=r1), dict(id='bed_2', soft=True, region=r2)]
        plate = SimpleNamespace(Wp=100, Hp=100, project=lambda X: (np.ones(len(X)), np.ones(len(X)), np.ones(len(X))))
        spec = ss.ShotSpec('T', 'kneel', 'bed', ['knee.L'])
        self.assertEqual(place.part_label(np.array([[1.05, .05, .45]]), spec, r1, None, plate, True), 'bed_2')


class SupportTests(unittest.TestCase):
    def test_kneel_lowest_support_not_non_support_body_bottom(self):
        V = np.array([[0., 0., 0.]] + [[0., 0., .3]] * 6 + [[0., 0., .32]] * 6)
        B = dict(V=V, kp=skeleton(), reg=np.array(['hand.L'] + ['knee.L'] * 6 + ['shin.L'] * 6), ref_c={})
        spec = ss.ShotSpec('Q03', 'kneel', 'bed', ['knee.L', 'shin.L'])
        groups, _, _ = place.build_groups(B, spec, None, 1.62, False)
        self.assertEqual({g[0] for g in groups}, {'knee.L', 'shin.L'})
        for pc in ('stand', 'crouch'):
            spec = ss.ShotSpec(pc, pc, 'floor', ['knee.L', 'shin.L'])
            self.assertFalse(place.build_groups(B, spec, None, 1.62, False)[0])

    def test_other_classes_share_support_low_band(self):
        V = np.array([[0., 0., 0.], [0., 0., .3], [0., 0., .34], [0., 0., .5]])
        B = dict(V=V, kp=skeleton(), reg=np.array(['hand.L', 'pelvis', 'torso', 'torso']), ref_c={})
        spec = ss.ShotSpec('L', 'lie', 'bed', ['pelvis', 'torso'])
        groups, _, _ = place.build_groups(B, spec, None, 1.62, False)
        np.testing.assert_array_equal(next(g[1] for g in groups if g[0] == 'torso'), [2])


class FootprintTests(unittest.TestCase):
    def test_bbox_polygon_and_dominant_top_profile(self):
        r = region(shape=(20, 20))
        bbox = [[0., 0.], [.2, .4]]
        polygon = [[0., 0.], [.2, 0.], [.2, .4], [0., .4]]
        mask = tr.footprint_mask(r, bbox=bbox)
        self.assertEqual(mask.sum(), 200)
        np.testing.assert_array_equal(mask, tr.footprint_mask(r, polygon=polygon))
        r.height[0, :] = 3.  # outside narrow polygon in the next assertion
        self.assertLess(tr.footprint_mask(r, polygon=[[.1, .1], [.3, .1], [.2, .3]]).sum(), 100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = region(shape=(20, 20)); r.height[10:, :] = 3.
            tr.save_npz(r, root / 'top.npz')
            obj = dict(id='bed_1', name='bed', type='bed', top_grid_npz='top.npz', footprint_bbox=bbox)
            (root / 'profile.json').write_text(json.dumps(dict(loc_id='T', version=2, reviewed=False,
                scale={}, objects=[obj], anchors=[], substitutes={})))
            p = profiles.LocationProfile(root / 'profile.json')
            loaded = p.region(obj)
            self.assertEqual(loaded.h0, .45)
            self.assertEqual(loaded.object_mask.sum(), 200)
            self.assertNotIn('dominant_top_unavailable', loaded.flags)
            del obj['footprint_bbox']
            legacy = p.region(obj)
            self.assertIn('stop2c', legacy.top_diagnostics['reason'])


class ThresholdTests(unittest.TestCase):
    def test_class_soft_thresholds_and_hard_unchanged(self):
        plate = SimpleNamespace()
        X = np.ones((8, 3))
        for pc in ss.POSE_CLASSES:
            expected = .08 if pc in ('lie', 'recline') else .06
            with patch.object(place, 'penetration_values', return_value=(np.full(8, .07), np.repeat('soft:bed_1', 8))):
                pen = place.penetration_stats(plate, X, None, True, pc)
            self.assertEqual(pen['surfaces']['soft:bed_1']['threshold_m'], expected)
            self.assertEqual(pen['ok'], pc in ('lie', 'recline'))
            with patch.object(place, 'penetration_values', return_value=(np.full(8, .031), np.repeat('hard:wall', 8))):
                self.assertFalse(place.penetration_stats(plate, X, None, True, pc)['ok'])

    def test_torso_lateral_boundary_and_cache_threshold_invalidation(self):
        kp = skeleton()
        pelvis = (kp[9] + kp[10]) / 2
        shoulders = (kp[5] + kp[6]) / 2
        half = np.linalg.norm(kp[5] - kp[6]) / 2
        V = np.array([pelvis + [half * .8, -.1, .1], pelvis + [half * .81, -.1, .1]])
        N = np.tile([0., -1., 0.], (2, 1))
        np.testing.assert_array_equal(body.torso_back_indices(V, N, kp), [0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            V = np.array([[-.1, -.1, 1.], [.1, -.1, 1.], [0., -.1, 1.3]])
            np.savez(root / 'body.npz', V=V, F=np.array([[0, 1, 2]]), kp=kp)
            body.cached_torso_back(root / 'body.npz', root / 'cache')
            with patch.dict(body.section('place'), torso_back_halfwidth_frac=.7):
                body.cached_torso_back(root / 'body.npz', root / 'cache')
            self.assertEqual(len(list((root / 'cache').glob('*.npz'))), 2)
