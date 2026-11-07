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
