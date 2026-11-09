"""09d: lie roll is preserved through place and post-refine, not clamped."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly
import orient as ori
import place
import refine
from shot_spec import ShotSpec
from test_step9d import region


def lying(roll):
    K = np.zeros((70, 3))
    # supine: longitudinal axis +Y, anatomical left +X, chest +Z
    K[9], K[10] = [.1, 0, .6], [-.1, 0, .6]
    K[5], K[6], K[69] = [.18, .5, .6], [-.18, .5, .6], [0, .55, .6]
    K[:5] = [0, .7, .6]
    return ori.rot_about(K, np.array([0, 0, .6]), ori.rot_axis([0, 1, 0], -np.radians(roll)))


class LieRoll(unittest.TestCase):
    def test_full_signed_roll(self):
        for angle in (0, -90, 90, 180):
            self.assertAlmostEqual(ori.wrap(ori.roll_deg(lying(angle), 'lie') - angle), 0)

    def test_assemble_preserves_roll_and_axis_plane(self):
        for angle in (-90, 0, 90, 180):
            K = lying(angle)
            K = ori.rot_about(K, (K[9] + K[10]) / 2, ori.rot_axis([1, 0, 0], np.radians(25)))
            X, J, info = ori.assemble_tilt_roll(K.copy(), K, 'lie')
            self.assertLessEqual(abs(ori.tilt_deg(J) - 90), 15.01)
            self.assertLess(abs(ori.wrap(ori.roll_deg(J, 'lie') - angle)), .01)
            self.assertEqual(info['clamped_deg'], 0)

    def test_post_refine_does_not_clamp_lie(self):
        for angle in (-90, 0, 90, 180):
            K = lying(angle)
            X, J, info = refine.clamp_refine_roll(K.copy(), K.copy(), {}, 'lie')
            np.testing.assert_allclose(J, K)
            self.assertFalse(info['root_clamped'])

    def test_root_restores_reference_roll(self):
        K = lying(65)
        C = SimpleNamespace(C=np.array([0., -3, 2]))
        _, J = assembly.clamp_root(K.copy(), K, 'lie', C, 0, lie_roll_ref=90)
        self.assertAlmostEqual(ori.roll_deg(J, 'lie'), 90)

    def test_place_preserves_side_and_back(self):
        # Synthetic rigid body + flat bed: real place/fit/height/validator run;
        # only asset/model preparation and image diagnostics are substituted.
        for angle in (-90, 90, 0):
            with self.subTest(roll=angle):
                K = lying(angle) + [0, 2., 0]
                V = np.array([[-.08, -.1, 0], [.08, -.1, 0], [-.08, .1, 0], [.08, .1, 0.]])
                B = dict(V=V, kp=K, V0=V.copy(), kp0=K.copy(), reg=np.full(4, 'pelvis'),
                         ref_c={}, dims={'H':1.62,'L_thigh':.4,'L_torso':.55},
                         side_ref=0, cam_axis=np.array([0, -1., 0]), s=1., up_src='synthetic',
                         n_planes=0, elev_ref=0.)
                R = region(shape=(150, 150), origin=(-1.5, -1.5))
                P = R.points3d()
                surf = place._surf(P)
                C = SimpleNamespace(C=np.array([0., -3, 2.]), h=2., Wp=1000, Hp=1000,
                                    trees={n:surf for n in ('any','h','v')},
                                    project=lambda X:(X[:,0],X[:,1],np.ones(len(X))),
                                    depth_at=lambda u,v:np.ones(len(u)))
                spec = ShotSpec('synthetic-lie', 'lie', 'bed', ['pelvis'])
                with patch.object(place, 'prepare_body', return_value=B), \
                     patch.object(place, 'resolve_regions', return_value=(R,None,'proxy',{'soft':True})), \
                     patch.object(place.tr, 'walls', return_value=[]), \
                     patch.object(place.zn, 'zone_cells', return_value=([{'xy':[0.,0.],'h':.45,'anchor':'support'}],{'flags':[]})), \
                     patch.object(place, 'framing', return_value={'ok':True}), \
                     patch.object(place, 'geometry_diagnostics', return_value={}):
                    result = place.place({}, C, spec, np.array([0.,0.,.45]), 1.62, log=lambda _:None)
                self.assertLessEqual(abs(ori.wrap(ori.roll_deg(result['KP'], 'lie') - angle)), 10.)
                self.assertLessEqual(abs(ori.tilt_deg(result['KP']) - 90), 15.01)
                self.assertFalse(any(f.startswith('orient_clamped') for f in result['rep']['flags']))


if __name__ == '__main__':
    unittest.main()
