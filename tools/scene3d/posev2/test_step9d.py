"""D1–D5/E CPU-only synthetic and failure regression tests. No render/model runtime."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly
import body
import place
import profiles
import shot_spec as ss
import support_geometry as sg
import top_region as tr
from test_step9c import skeleton, evaluation


def region(height=.45, shape=(60, 60), origin=(0, 0)):
    mask = np.ones(shape, bool)
    dt = tr.ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * tr.CELL
    return tr.TopRegion('bed', 'synthetic', np.array(origin, float), shape,
                        np.full(shape, height), mask, dt, height, (0, 0), np.array([0., -2.]))


class HullTests(unittest.TestCase):
    def test_square_known_area_and_exact_partial_cells(self):
        P = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [.4, .3]])
        r = region(shape=(50, 50))
        m = sg.hull_metrics(P, r)
        self.assertAlmostEqual(m['area_m2'], 1.)
        self.assertAlmostEqual(m['in_mask_fraction'], 1.)
        r.mask[:5] = False
        self.assertAlmostEqual(sg.hull_metrics(P, r)['in_mask_fraction'], .9)
        # Fractional edges: no cell-center or vertex-count approximation.
        P = np.array([[.005, .005], [.035, .005], [.035, .035], [.005, .035]])
        r = region(shape=(2, 2))
        r.mask[0, 0] = False
        m = sg.hull_metrics(P, r)
        self.assertAlmostEqual(m['area_m2'], .0009)
        self.assertAlmostEqual(m['in_mask_fraction'], .75)

    def test_hull_captures_hole_and_degenerate_has_no_fraction(self):
        r = region(shape=(50, 50))
        r.mask[10:40, 10:40] = False
        P = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertAlmostEqual(sg.hull_metrics(P, r)['in_mask_fraction'], .64)
        for p in (np.empty((0, 3)), np.ones((3, 3)), np.array([[0, 0], [1, 0], [2, 0]])):
            m = sg.hull_metrics(p, r)
            self.assertEqual(m['area_m2'], 0)
            self.assertIsNone(m['in_mask_fraction'])

    def test_ninety_percent_boundary_passes(self):
        r = region(shape=(50, 50))
        r.mask[:5] = False
        X = np.array([[0., 0., .45], [1., 0., .45], [1., 1., .45], [0., 1., .45]])
        s = ss.ShotSpec('ninety', 'lie', 'bed', ['torso'])
        self.assertTrue(place.footprint(X, r, s, 1.62, np.arange(4))['ok'])
        r.mask[5] = False
        self.assertFalse(place.footprint(X, r, s, 1.62, np.arange(4))['ok'])

    def test_threshold_is_area_not_support_vertex_fraction(self):
        r = region(shape=(50, 50))
        X = np.array([[.01, .01, .45], [.99, .01, .45], [.99, .99, .45], [.01, .99, .45]])
        s = ss.ShotSpec('hull', 'lie', 'bed', ['torso'])
        r.mask[10:40, 10:40] = False
        # Every support vertex is in mask, but only .625 of hull lies inside.
        self.assertTrue(r.in_mask(X).all())
        m = place.footprint(X, r, s, 1.62, np.arange(4))
        self.assertFalse(m['ok'])
        self.assertLess(m['support_hull']['in_mask_fraction'], .9)


class TorsoTests(unittest.TestCase):
    def setUp(self):
        self.K = skeleton()
        self.V = np.array([[0., -.1, 1.1], [0., .1, 1.1], [0., -.1, .8], [0., -.1, 1.5],
                           [0., -.1, 1.2], [0., -.1, 1.2]])
        # skeleton left is -X, its forward is +Y.
        self.N = np.array([[0., -1, 0], [0., 1, 0], [0., -1, 0], [0., -1, 0],
                           [0., -1, 0], [0., -.5, .875]])

    def test_named_mhr_parts_nonempty_all_backward_strict_dot(self):
        self.V[4, 0] = .3  # outside the approved central lateral band
        self.N[5] = [1., -.5, 0]  # normalized dot above threshold, excluded
        idx = body.torso_back_indices(self.V, self.N, self.K)
        np.testing.assert_array_equal(idx, [0])
        self.assertTrue(np.all(self.N[idx] @ body.facing(self.K, np.array([0., 0, 1])) < -.5))

    def test_geometric_rule_does_not_require_anatomy_labels(self):
        expected = body.torso_back_indices(self.V, self.N, self.K)
        self.assertGreater(len(expected), 0)
        idx = body.torso_back_indices(self.V, self.N, self.K, part_labels=np.repeat('arm', 6))
        np.testing.assert_array_equal(idx, expected)
        # Rotation and translation preserve the body-frame selection.
        R = place.pf.rot_xyz(.4, .7, -.3)
        idx = body.torso_back_indices(self.V @ R.T + 3, self.N @ R.T, self.K @ R.T + 3)
        np.testing.assert_array_equal(idx, expected)

    def test_cache_per_asset_and_topology_no_gpu(self):
        V = np.array([[-.1, -.1, 1.0], [.1, -.1, 1.0], [0., -.1, 1.3]])
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / 'model.npz'
            np.savez(asset, V=V, F=np.array([[0, 1, 2]]), kp=self.K, part_labels=np.repeat('torso', 3))
            a = body.cached_torso_back(asset, Path(tmp) / 'cache')
            self.assertGreater(len(a), 0)
            with patch.object(body, 'torso_back_indices', side_effect=AssertionError('recomputed')):
                b = body.cached_torso_back(asset, Path(tmp) / 'cache')
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(list((Path(tmp) / 'cache').glob('*.npz'))), 1)

    def test_separate_back_indices_do_not_erase_torso_region(self):
        B = dict(V=self.V, kp=self.K, reg=np.repeat('torso', 6), ref_c={}, torso_back_indices=np.array([0]))
        s = ss.ShotSpec('T03', 'recline', 'bed', ['torso'])
        groups, _, info = place.build_groups(B, s, SimpleNamespace(backrest=self.V[:1]), 1.62, False)
        self.assertEqual(info['torso_back_vertices'], 1)
        self.assertTrue(any(g[0] == 'torso' and g[3] == 'support' for g in groups))
        np.testing.assert_array_equal(next(g[1] for g in groups if g[0] == 'torso_back'), [0])


class DominantTopTests(unittest.TestCase):
    def test_bed_mattress_pillows_wall_margin_roundtrip(self):
        r = region(shape=(50, 100))
        r.height[:, 80:] = .60
        wall = {'n': np.array([0., -1, 0]), 'd': 2.}
        tr.apply_dominant_top(r, np.ones(r.shape, bool), [wall])
        self.assertAlmostEqual(r.h0, .45)
        self.assertEqual(r.raised_mask.sum(), 1000)
        self.assertEqual(r.soft_backrest_mask.sum(), 1000)
        self.assertFalse(r.mask[:, 80:].any())
        self.assertIsNotNone(r.backrest)
        for cls in ('kneel', 'lie', 'sit_tucked'):
            z = tr.pose_region(r, cls)
            self.assertFalse(z.mask[:, 78:].any())  # .04m and .02m strip excluded.
            self.assertTrue(z.mask[:, 77].all())
        self.assertIs(tr.pose_region(r, 'recline'), r)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'top.npz'
            tr.save_npz(r, path)
            copy = tr.load_npz(path)
            np.testing.assert_array_equal(copy.soft_backrest_mask, r.soft_backrest_mask)
            np.testing.assert_array_equal(copy.object_mask, r.object_mask)
            self.assertEqual(copy.top_diagnostics, r.top_diagnostics)

    def test_profile_roundtrip_preserves_class_trim_for_zone_metrics(self):
        import zone
        r = region(shape=(50, 100))
        r.height[:, 80:] = .60
        tr.apply_dominant_top(r, np.ones(r.shape, bool), [{'n': np.array([0., -1, 0]), 'd': 2.}])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tr.save_npz(r, root / 'bed.npz')
            data = dict(loc_id='synth', version=2, reviewed=False, scale={},
                        objects=[dict(id='bed_1', name='bed', type='bed', top_grid_npz='bed.npz')],
                        anchors=[], substitutes={})
            (root / 'profile.json').write_text(json.dumps(data))
            profile = profiles.LocationProfile(root / 'profile.json')
            restored = profile.region(data['objects'][0])
            self.assertNotIn('dominant_top_unavailable', restored.flags)
            trimmed = tr.pose_region(restored, 'kneel')
            m = zone.metrics(trimmed)
            self.assertTrue(trimmed.in_mask(m['xy']).all())
            self.assertLess(m['xy'][:, 1].max(), 1.56)
            self.assertTrue(np.allclose(m['h'], .45))

    def test_raised_is_not_backrest_without_adjoining_wall(self):
        r = region(shape=(20, 20))
        r.height[10:15, 10:15] = .60
        tr.apply_dominant_top(r, np.ones(r.shape, bool), [])
        self.assertTrue(r.raised_mask.any())
        self.assertFalse(r.soft_backrest_mask.any())
        self.assertEqual(r.mask.sum(), 375)

    def test_histogram_counts_cell_area_not_max_and_ignores_outside_object(self):
        r = region(shape=(20, 20))
        r.height[0] = 3.
        r.height[1] = .60
        obj = np.ones(r.shape, bool); obj[0] = False
        h, mask, _, high, _ = tr.dominant_top(r.height, obj, r.origin, [])
        self.assertAlmostEqual(h, .45)
        self.assertFalse(mask[0].any())
        self.assertEqual(high.sum(), 20)
        with self.assertRaisesRegex(ValueError, 'empty object mask'):
            tr.dominant_top(r.height, np.zeros(r.shape, bool), r.origin, [])


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.S = {'target': place._surf(np.array([[0., 0, 0.], [1., 0, 0.]]))}
        self.groups = [('foot.L', np.array([0]), 'target', 'support'),
                       ('foot.R', np.array([1]), 'target', 'support'),
                       ('shin.L', np.array([2]), 'target', 'support')]
        self.X = np.array([[0., 0, 0.], [1., 0, .055], [0., 0, .05]])

    def test_q05_one_foot_and_other_gaps_are_flags_not_failure(self):
        for cls in ('stand', 'crouch'):
            spec = ss.ShotSpec('Q05', cls, 'floor', ['foot.L', 'foot.R'])
            c = place.support_contact(self.X, self.groups, self.S, spec, False)
            self.assertTrue(c['ok'])
            self.assertEqual(c['foot_contact_parts'], ['foot.L'])
            self.assertEqual(c['flags'], ['support_gap'])
            ev = evaluation()
            ev.update(support_object='floor', support_contact=c, support_ok=c['ok'],
                      support_part_gaps_m=c['part_gaps_m'])
            with patch.object(assembly, 'orientation_checks', return_value=[]):
                val = assembly.validate(None, ev, spec, None, 0, None, True)
            self.assertTrue(val['ok'])
            self.assertEqual(val['status'], 'ok')
            self.assertIn('support_gap', val['flags'])

    def test_lowest_without_foot_and_foot_without_body_lowest_fail(self):
        spec = ss.ShotSpec('Q05', 'crouch', 'floor', ['foot.L'])
        X = self.X.copy(); X[0, 2] = .04; X[2, 2] = 0
        self.assertFalse(place.support_contact(X, self.groups, self.S, spec, False)['ok'])
        X = self.X.copy(); X[2, 2] = -.011
        self.assertFalse(place.support_contact(X, self.groups, self.S, spec, False)['ok'])

    def test_other_classes_keep_all_group_gap_requirement(self):
        for cls in ('kneel', 'sit_edge', 'lie'):
            spec = ss.ShotSpec('old', cls, 'bed', ['foot.L', 'foot.R'])
            self.assertFalse(place.support_contact(self.X, self.groups, self.S, spec, False)['ok'])

    def test_height_does_not_sweep_to_force_other_parts_down(self):
        spec = ss.ShotSpec('Q05', 'crouch', 'floor', ['foot.L'])
        K = skeleton()
        with patch.object(place, 'penetration_stats', return_value={'ok': True, 'excess_m': 0}), \
             patch.object(assembly.ori, 'rot_axis', side_effect=AssertionError('unneeded sweep')):
            X, _, info = assembly.height_fit(self.X + [0, 0, .3], K, self.groups, self.S,
                                             None, spec, None, False, 1.62)
        self.assertAlmostEqual(X[0, 2], 0.)
        self.assertTrue(info['support_rule_ok'])


class DiagnosticTests(unittest.TestCase):
    def test_actual_surface_not_target_and_q04_penetration_parts(self):
        target, floor = region(), region(height=0.)
        target.diagnostic_surfaces = [{'id': 'bed_1', 'region': target, 'soft': True},
                                       {'id': 'floor_1', 'region': floor, 'soft': False}]
        X = np.array([[.2, .2, 0], [.4, .2, 0], [.2, .4, 0], [.4, .4, .03]])
        B = {'reg': np.array(['knee.L', 'knee.L', 'shin.L', 'torso'])}
        spec = ss.ShotSpec('Q04', 'kneel', 'bed', ['knee.L', 'shin.L'])
        groups = [('knee.L', np.array([0, 1]), 'target', 'support'),
                  ('shin.L', np.array([2]), 'target', 'support')]
        S = {'target': place._surf(target.points3d())}
        with patch.object(place, 'penetration_values', return_value=(np.array([.01, .02, .0, .08]),
                                                                   np.array(['hard:floor_1'] * 3 + ['soft:bed_1']))):
            d = place.geometry_diagnostics(X, skeleton(), B, spec, target, groups, S, None, True)
        self.assertEqual(d['actual_surface']['id'], 'floor_1')
        self.assertEqual(d['actual_surface']['h0_m'], 0.)
        self.assertAlmostEqual(d['target_gap_min_m'], -.45)
        self.assertAlmostEqual(d['target_gap_p50_m'], -.45)
        self.assertIn('torso', d['penetration_parts_per_surface']['soft:bed_1'])
        self.assertAlmostEqual(d['penetration_per_surface']['soft:bed_1']['p95_m'], .08)
        self.assertEqual(set(d['angles_deg']), {'yaw', 'pitch', 'roll'})
        json.dumps(d)  # diagnostics can be stored directly in fit.json.

    def test_q03_zero_support_keeps_cause_and_mapped_hull(self):
        V = np.array([[0., 0., .3], [.2, 0., .3], [0., .2, .4], [.2, .2, .4], [0, 0, 0.]])
        B = dict(V=V, kp=skeleton(), reg=np.array(['knee.L', 'shin.L', 'knee.R', 'shin.R', 'foot.L']),
                 dims={'H': 1.62}, ref_c={})
        spec = ss.ShotSpec('Q03', 'kneel', 'bed', ['knee.L', 'shin.L', 'knee.R', 'shin.R'])
        groups, _, _ = place.build_groups(B, spec, None, 1.62, False)
        d = place.zone_diagnostics(B, spec, region(), groups)
        self.assertEqual(d['support_vertex_count'], 2)
        self.assertIsNone(d['zero_support_cause'])
        self.assertAlmostEqual(d['body_footprint_area_m2'], .04)
        self.assertEqual([k['part'] for k in d['kneel_selection'] if k['selected']], ['knee.L'])


if __name__ == '__main__':
    unittest.main()
