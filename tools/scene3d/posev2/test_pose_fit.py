"""Офлайн-тест посадки: синтетическая комната + меши SAM 3D Body из testdata/pose_v2.
  python -m unittest tools/scene3d/posev2/test_pose_fit.py"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose_fit as pf  # noqa: E402
import test_synth_plate as tsp  # noqa: E402

TD = HERE.parent / "testdata" / "pose_v2"


class PoseFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        tsp.make(os.path.join(cls.tmp, "plate"))
        cls.plate = pf.Plate(os.path.join(cls.tmp, "plate"))

    def run_fit(self, name, target, contacts):
        ref = pf.load_ref(TD / f"{name}.person.npz")
        return pf.fit(ref, self.plate, target, 1.62, contacts, log=lambda m: None)[1]

    def test_floor_sit(self):
        r = self.run_fit("T04_floor_sit_knees_up", (0.5, 0.9), ["pelvis:h:floor", "foot.L:h:floor", "foot.R:h:floor"])
        self.assertEqual(r["status"], "ok")
        for k in ("pelvis", "foot.L", "foot.R"):
            self.assertLess(abs(r["contacts"][k]["gap_cm"]), 2.0, k)
        self.assertLess(abs(r["angle_to_camera_deg"]["ref"] - r["angle_to_camera_deg"]["plate"]), 10)

    def test_standing_lowest(self):
        r = self.run_fit("T05_barre_standing_split", (0.3, 0.9), None)
        self.assertEqual(r["status"], "ok")
        self.assertLess(r["max_penetration_cm"], 3.0)

    def test_lying_reports_mismatch_honestly(self):
        r = self.run_fit("Q08_bed_recline_elbows", (0.5, 0.5), ["torso:h:bed", "pelvis:h:bed", "forearm.L:h:bed"])
        self.assertLess(r["support_gap_cm"], 3.0)
        self.assertIn(r["status"], ("ok", "pose_mismatch"))


if __name__ == "__main__":
    unittest.main()


def synth_ref_scene(name, up, rng_seed=0):
    """Реф с синтетической сценой: пол под самой низкой точкой тела (нормаль up), стена за телом и «остатки тела»
    в точках MoGe (вершины + шум 1 см) — как на краях силуэта у настоящего MoGe."""
    import numpy as np
    ref = pf.load_ref(TD / f"{name}.person.npz")
    V = ref["V"]
    rng = np.random.default_rng(rng_seed)
    up = up / np.linalg.norm(up)
    a = np.cross(up, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    low = V[np.argmin(V @ up)]
    g = rng.uniform(-1.5, 1.5, (4000, 2))
    floor = low + g[:, :1] * a + g[:, 1:] * b
    c = V.mean(0)
    back = np.array([c[0], 0, c[2]])
    back /= np.linalg.norm(back)
    back -= (back @ up) * up
    back /= np.linalg.norm(back)
    w0 = c + back * 1.0
    side = np.cross(up, back)
    g2 = rng.uniform(-1.5, 1.5, (3000, 2))
    wall = w0 + g2[:, :1] * side + (g2[:, 1:] + 1.5) * up
    noise = V[rng.choice(len(V), 1500)] + rng.normal(0, 0.01, (1500, 3))
    ref["S"] = np.concatenate([floor, wall, noise])
    ref["N"] = None
    return ref


class RefScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        tsp.make(os.path.join(cls.tmp, "plate"))
        cls.plate = pf.Plate(os.path.join(cls.tmp, "plate"))

    def test_standing_support_is_foot(self):
        import numpy as np
        ref = pf.load_ref(TD / "T05_barre_standing_split.person.npz")
        kp = ref["kp"]
        lo = 13 if kp[13] @ np.array([0, -1.0, 0]) < kp[14] @ np.array([0, -1.0, 0]) else 14
        hip = kp[lo - 4]
        r = pf.fit(synth_ref_scene("T05_barre_standing_split", hip - kp[lo]), self.plate, (0.3, 0.9), 1.62, None,
                   log=lambda m: None)[1]
        self.assertEqual(r["up_from"], "plane")
        self.assertTrue(all(k.startswith("foot") for k in r["support"]), r["support"])
        self.assertLess(r["max_penetration_cm"], 3.0)

    def test_floor_sit_support_is_pelvis(self):
        import numpy as np
        ref = pf.load_ref(TD / "T04_floor_sit_knees_up.person.npz")
        reg = pf.regions(ref["V"], ref["kp"])
        forced = pf.ref_contacts(ref, reg, ["pelvis:h:f", "foot.L:h:f", "foot.R:h:f"])
        up, _ = pf.plane_up(ref["V"], forced, np.array([0, -1.0, 0]))
        r = pf.fit(synth_ref_scene("T04_floor_sit_knees_up", up), self.plate, (0.5, 0.9), 1.62, None,
                   log=lambda m: None)[1]
        self.assertIn("pelvis", r["support"])
        self.assertLess(r["support_gap_cm"], 2.0)


class Seat(unittest.TestCase):
    def test_seat_pixel_on_front_face_goes_to_bed_top(self):
        import numpy as np
        tmp = tempfile.mkdtemp()
        tsp.make(os.path.join(tmp, "plate"))
        plate = pf.Plate(os.path.join(tmp, "plate"))
        P, v = plate.seat_point(0.5, 0.65)                                   # передняя стенка кровати
        self.assertAlmostEqual(float(P[2]), 0.5, delta=0.03)
        P, v = plate.seat_point(0.5, 0.45)                                   # уже верх кровати — без сдвига
        self.assertAlmostEqual(float(P[2]), 0.5, delta=0.03)
        self.assertEqual(v, 0.45)
        self.assertTrue(np.isfinite(P).all())
