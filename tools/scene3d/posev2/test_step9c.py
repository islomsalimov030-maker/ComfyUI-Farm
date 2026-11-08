"""Step 9c: literal height/contact, shared p95, roll and missing-data outcomes."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly
import orient as ori
import place
import shot_spec as ss


def skeleton(tilt=0, roll=0):
    K = np.zeros((70, 3))
    K[9], K[10] = [-.1, 0, .9], [.1, 0, .9]
    K[5], K[6], K[69] = [-.18, 0, 1.4], [.18, 0, 1.4], [0, 0, 1.45]
    c = (K[9] + K[10]) / 2
    R = ori.rot_axis([1, 0, 0], np.radians(tilt)) @ ori.rot_axis([0, 1, 0], np.radians(roll))
    return ori.rot_about(K, c, R)


def evaluation():
    return dict(support_object='bed', support_part_gaps_m={'pelvis': 0.0},
                penetration={'soft_p95_m': 0., 'hard_p95_m': 0.},
                support_ok=True, footprint={'ok': True}, cells_ok=True, frame={'ok': True})


class SharedP95(unittest.TestCase):
    def test_same_geometry_same_value_in_place_and_validator(self):
        P = np.column_stack([np.arange(102), np.zeros(102), np.zeros(102)])
        plate = SimpleNamespace(trees={'any': place._surf(P)})
        depths = np.r_[np.full(100, .02991), .2, .3]
        spec = ss.ShotSpec('p95', 'sit_edge', 'bed', ['pelvis'])
        with patch.object(place.pf, 'penetration', return_value=depths), \
             patch.object(assembly, 'orientation_checks', return_value=[]):
            placed = place.penetration_stats(plate, P, None, False)
            validated = assembly.validate(None, evaluation(), spec, plate, 0, None, True,
                                          X=P, soft=False)
        self.assertEqual(placed, validated['penetration'])
        self.assertEqual(placed['surfaces']['hard']['p95_m'], .02991)
        self.assertTrue(validated['ok'])  # max is .3, rounded cm is 3.0

    def test_per_surface_not_pooled_material(self):
        depths = np.r_[np.full(100, .01), .04]
        material = np.array(['hard:wall0'] * 100 + ['hard:wall1'])
        with patch.object(place, 'penetration_values', return_value=(depths, material)):
            result = place.penetration_stats(None, None, None, False)
        self.assertAlmostEqual(result['hard_p95_m'], .01)
        self.assertFalse(result['ok'])
        self.assertAlmostEqual(result['surfaces']['hard:wall1']['p95_m'], .04)


class HeightContact(unittest.TestCase):
    def setUp(self):
        self.K = skeleton()
        self.S = {'target': place._surf(np.array([[0., 0., 0.], [1., 0., 0.]]))}
        self.groups = [('pelvis', np.array([0, 1]), 'target', 'support')]
        self.spec = ss.ShotSpec('h', 'sit_edge', 'bed', ['pelvis'])

    def fit(self, X, penetration):
        # A single orientation isolates stages (a)->(b), with no new numerical rule.
        with patch.object(place, 'penetration_stats', side_effect=penetration), \
             patch.object(ori, 'forward', return_value=None):
            return assembly.height_fit(X, self.K, self.groups, self.S, None,
                                       self.spec, None, False, 1.62)

    def test_min_not_median_and_both_vertical_directions(self):
        for z in (.2, -.2):
            X = np.array([[0., 0., z], [1., 0., z + .08]])
            seen = []
            def metric(*args):
                seen.append(args[1].copy())
                return {'ok': True, 'excess_m': 0.}
            V, _, info = self.fit(X, metric)
            self.assertAlmostEqual(V[0, 2], 0.)
            self.assertAlmostEqual(V[1, 2], .08)
            self.assertAlmostEqual(seen[0][0, 2], 0.)
            self.assertAlmostEqual(info['contact_min_m'], 0.)
            self.assertAlmostEqual(info['shift_z_m'], -z)

    def test_exact_5mm_lift_and_2cm_ceiling(self):
        X = np.array([[0., 0., .4], [1., 0., .4]])
        visited = []
        def metric(_plate, V, *_):
            z = float(V[0, 2]); visited.append(z)
            return {'ok': z >= .015 - 1e-12, 'excess_m': max(.015 - z, 0.)}
        V, _, info = self.fit(X, metric)
        np.testing.assert_allclose(visited, [0., .005, .010, .015])
        self.assertAlmostEqual(V[0, 2], .015)
        self.assertFalse(info['height_conflict'])
        self.assertAlmostEqual(info['lift_ceiling_m'], .02)

    def test_conflict_is_partial_and_keeps_failed_checks(self):
        X = np.array([[0., 0., .4], [1., 0., .4]])
        visited = []
        def metric(_plate, V, *_):
            visited.append(V[0, 2])
            return {'ok': False, 'excess_m': .032}
        V, _, info = self.fit(X, metric)
        np.testing.assert_allclose(visited, [0., .005, .010, .015, .020])
        self.assertAlmostEqual(V[0, 2], .02)
        self.assertTrue(info['height_conflict'])
        ev = evaluation()
        ev['penetration']['hard_p95_m'] = .062
        with patch.object(assembly, 'orientation_checks', return_value=[]):
            val = assembly.validate(None, ev, self.spec, None, 0, None, True, height_info=info)
        self.assertEqual(val['status'], 'partial')
        self.assertIn('height_conflict', val['flags'])
        self.assertFalse(val['ok'])

    def test_empty_support_not_height_success(self):
        with patch.object(ori, 'forward', return_value=None):
            _, _, info = assembly.height_fit(np.zeros((2, 3)), self.K, [], self.S, None,
                                             self.spec, None, False, 1.62)
        self.assertFalse(info['penetration_ok'])
        self.assertEqual(info['reason'], 'no_support_vertices')


class Roll(unittest.TestCase):
    def test_assembly_clamps_both_signs_and_all_classes(self):
        for cls in ori.TILT_RANGE:
            for sign in (-1, 1):
                with self.subTest(cls=cls, sign=sign):
                    K = skeleton(10, sign * 35)
                    X, J, _ = ori.assemble_tilt_roll(K.copy(), K, cls)
                    self.assertLessEqual(abs(ori.roll_deg(J)), ori.ROLL_MAX + 1e-8)
                    lo, hi = ori.TILT_RANGE[cls]
                    self.assertGreaterEqual(ori.tilt_deg(J), lo - .05)
                    self.assertLessEqual(ori.tilt_deg(J), hi + .05)
                    np.testing.assert_allclose(X, J)

    def test_post_refine_root_clamp_preserves_tilt_and_pelvis(self):
        import refine
        K = skeleton(45, 40)
        tilt = ori.tilt_deg(K)
        X, J, info = refine.clamp_refine_roll(K.copy(), K, {})
        self.assertTrue(info['root_clamped'])
        self.assertAlmostEqual(info['roll_final_deg'], ori.roll_deg(J))
        self.assertLessEqual(abs(ori.roll_deg(J)), ori.ROLL_MAX + 1e-8)
        self.assertAlmostEqual(ori.tilt_deg(J), tilt)
        np.testing.assert_allclose((J[9] + J[10]) / 2, (K[9] + K[10]) / 2)


class Recline(unittest.TestCase):
    def test_missing_zone_no_heuristic_substitute(self):
        K = skeleton(40)
        B = dict(V=K, kp=K, reg=np.full(70, 'torso'), ref_c={})
        spec = ss.ShotSpec('r', 'recline', 'bed', ['pelvis'])
        reg = SimpleNamespace(backrest=np.array([[0., 1., 1.]]))
        groups, _, info = place.build_groups(B, spec, reg, 1.62, False)
        self.assertFalse(any(g[0] == 'torso_back' for g in groups))
        self.assertIn('missing torso_back', info['back_contact_unavailable'])
        V, J, sw = place.recline_sweep(dict(X=K, KP=K), groups, {}, None, 1.62, 40)
        self.assertFalse(sw['touch'])
        np.testing.assert_array_equal(J, K)

    def test_sweep_after_height_about_pelvis_and_hard_limit(self):
        K = skeleton(40)
        X = np.array([[0., -.12, 1.2]])
        target = X.copy()
        S = {'back': place._surf(target)}
        groups = [('torso_back', np.array([0]), 'back', 'back')]
        for hard_ok in (True, False):
            pen = {'surfaces': {'hard:wall0': {'ok': hard_ok}}}
            with patch.object(place, 'penetration_stats', return_value=pen):
                V, J, info = place.recline_sweep(dict(X=X, KP=K), groups, S, None, 1.62, 40)
            self.assertEqual(info['touch'], hard_ok)
            np.testing.assert_allclose((J[9] + J[10]) / 2, (K[9] + K[10]) / 2)
            self.assertLessEqual(abs(ori.roll_deg(J)), 15 + 1e-8)
            if hard_ok:
                self.assertAlmostEqual(info['back_contact_gap_m'], 0., places=7)
                self.assertAlmostEqual(ori.tilt_deg(J), 40.)
            else:
                np.testing.assert_array_equal(V, X)


class ZoneDiagnostics(unittest.TestCase):
    def test_mapped_counts_not_lost_when_kneel_selects_none(self):
        names = ['knee.L', 'shin.L', 'knee.R', 'shin.R', 'foot.L']
        V = np.column_stack([np.zeros(50), np.zeros(50), np.repeat([.3, .3, .3, .3, 0.], 10)])
        B = dict(V=V, kp=skeleton(), reg=np.repeat(names, 10), ref_c={})
        spec = ss.ShotSpec('Q03', 'kneel', 'bed', names[:4], instance_hint='bed_1')
        groups, _, _ = place.build_groups(B, spec, None, 1.62, False)
        d = place.zone_diagnostics(B, spec, SimpleNamespace(mask=np.ones((10, 20), bool)), groups)
        self.assertEqual(d['target_object'], 'bed')
        self.assertEqual(d['instance_hint'], 'bed_1')
        self.assertAlmostEqual(d['zone_mask_area_m2'], .08)
        self.assertEqual(d['support_vertices_after_mapping'], dict.fromkeys(names[:4], 10))
        self.assertEqual(d['support_vertex_count'], 40)
        self.assertEqual(d['body_footprint_area_m2'], 0.)  # collinear XY -> degenerate approved hull
        self.assertTrue(d['support_hull']['degenerate'])
        self.assertIsNone(d['zero_support_cause'])
        self.assertFalse(place.zone_too_small(d))
        self.assertTrue(place.zone_too_small(dict(d, zone_mask_area_m2=0.)))
        self.assertTrue(place.zone_too_small(dict(d, body_footprint_area_m2=.09)))

    def test_empty_mask_place_returns_no_target_with_diagnostic(self):
        K = skeleton()
        B = dict(V=K, kp=K, reg=np.full(70, 'pelvis'), ref_c={}, dims={'H': 1.62})
        spec = ss.ShotSpec('Q03', 'kneel', 'bed', ['knee.L'])
        treg = SimpleNamespace(mask=np.zeros((2, 2), bool), backrest=None, flags=[])
        with patch.object(place, 'prepare_body', return_value=B), \
             patch.object(place, 'resolve_regions', return_value=(treg, None, 'profile', {'soft': True})), \
             patch.object(place, 'orient_body'), patch.object(place.tr, 'walls', return_value=[]):
            result = place.place({}, None, spec, np.zeros(3), 1.62, log=lambda _: None)
        self.assertEqual(result['status'], 'no_target')
        self.assertIn('zone_too_small', result['rep']['flags'])
        self.assertEqual(result['rep']['zone_diagnostics']['zone_mask_area_m2'], 0.)


if __name__ == '__main__':
    unittest.main()
