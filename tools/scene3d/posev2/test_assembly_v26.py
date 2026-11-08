"""Шаг 9: kneel, материалы стен, корневой откат refine и полный валидатор."""
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


class Kneel(unittest.TestCase):
    def test_knees_not_feet(self):
        names = ['knee.L', 'shin.L', 'knee.R', 'shin.R', 'foot.L', 'foot.R']
        V = np.vstack([np.column_stack([np.full(10, i), np.zeros(10), np.full(10, z)])
                       for i, z in enumerate([.03, .035, .03, .035, 0., 0.])])
        K = np.zeros((70, 3)); K[9:11, 2] = 1
        B = dict(V=V, kp=K, reg=np.repeat(names, 10), ref_c={})
        spec = ss.ShotSpec('kneel', 'kneel', 'bed', ['knee.L', 'knee.R'])
        groups, _, _ = place.build_groups(B, spec, None, 1.62, False)
        support = {g[0] for g in groups if g[3] == 'support'}
        self.assertEqual(support, set(names[:4]))

    def test_lunge_other_foot(self):
        names = ['knee.L', 'shin.L', 'knee.R', 'shin.R', 'foot.L', 'foot.R']
        V = np.vstack([np.column_stack([np.full(10, i), np.zeros(10), np.full(10, z)])
                       for i, z in enumerate([.03, .04, .5, .3, .1, 0.])])
        K = np.zeros((70, 3)); K[9:11, 2] = 1
        B = dict(V=V, kp=K, reg=np.repeat(names, 10), ref_c={})
        spec = ss.ShotSpec('lunge', 'kneel', 'floor', ['knee.L', 'foot.R'])
        groups, _, _ = place.build_groups(B, spec, None, 1.62, False)
        self.assertEqual({g[0] for g in groups if g[3] == 'support'}, {'knee.L', 'shin.L', 'foot.R'})


class Materials(unittest.TestCase):
    def test_wall_hard_even_above_soft_target(self):
        P = np.array([[0, 1, 1.], [0, .5, .5]])
        plate = SimpleNamespace(trees={'any': (cKDTree(P), P, np.tile([0, -1, 0], (2, 1)))})
        reg = SimpleNamespace(mask=np.ones((1, 1), bool), in_mask=lambda X: np.ones(len(X), bool),
                              backrest=None, wall_planes=[dict(n=np.array([0., -1., 0.]), d=1.)])
        with patch.object(place.pf, 'penetration', return_value=np.array([.045, .045])):
            rep = place.penetration_stats(plate, P, reg, True)
        self.assertAlmostEqual(rep['hard_p95_m'], .045)
        self.assertAlmostEqual(rep['soft_p95_m'], .045)
        self.assertFalse(rep['ok'])


class RefineRoot(unittest.TestCase):
    def test_clamp_roll_does_not_hide_validation_failure(self):
        K = np.zeros((70, 3))
        K[9], K[10] = [-.1, 0, .9], [.1, 0, .9]
        K[5], K[6], K[69] = [-.18, 0, 1.4], [.18, 0, 1.4], [0, 0, 1.45]
        C = SimpleNamespace(C=np.array([0., 3., 1.2]))
        center = (K[9] + K[10]) / 2
        K = ori.rot_about(K, center, ori.rot_axis(np.array([0., 1., 0.]), np.deg2rad(30)))
        before = assembly.orientation_checks(K, 'stand', C, 1)
        self.assertFalse(next(c['ok'] for c in before if c['check'] == 'roll'))
        _, K2 = assembly.clamp_root(K.copy(), K, 'stand', C, 1)
        self.assertLessEqual(abs(ori.roll_deg(K2)), ori.ROLL_MAX + .05)
        self.assertTrue(all(c['ok'] for c in assembly.orientation_checks(K2, 'stand', C, 1)))


class BackContact(unittest.TestCase):
    def test_recline_requires_torso_back_not_secondary_wall(self):
        ev = dict(support_object='bed', support_part_gaps_m={'pelvis': 0.0},
                  penetration={'soft_p95_m': 0.0, 'hard_p95_m': 0.0},
                  support_ok=True, footprint={'ok': True}, cells_ok=True, frame={'ok': True},
                  back_gap_m=.50, back_contact_gap_m=.50, wall_gap_m=0.0)
        spec = ss.ShotSpec('recline', 'recline', 'bed', ['pelvis'])
        with patch.object(assembly, 'orientation_checks', side_effect=lambda *args: []):
            bad = assembly.validate(None, ev, spec, None, None, None, True)
            self.assertFalse(bad['ok'])
            self.assertEqual(bad['first_failure']['check'], 'back_contact')
            self.assertAlmostEqual(bad['first_failure']['excess'], .49)
            self.assertEqual(bad['status'], 'partial')
            self.assertIn('no_back_contact', bad['flags'])
            ev['back_contact_gap_m'] = None
            self.assertFalse(assembly.validate(None, ev, spec, None, None, None, True)['ok'])
            ev['back_contact_gap_m'] = .009
            self.assertTrue(assembly.validate(None, ev, spec, None, None, None, True)['ok'])
            ev['back_contact_gap_m'] = .01001
            self.assertFalse(assembly.validate(None, ev, spec, None, None, None, True)['ok'])


if __name__ == '__main__':
    unittest.main()
