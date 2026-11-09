"""Офлайн-тесты постановки v2.4 (07.11f): shot_spec, body_dims ×1.15 (зоны и пороги масштабируются, код не падает),
загрузка profile.json / body.json и фолбэк на height_proxy.
  python -m unittest tools/scene3d/posev2/test_v24.py"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orient as ori  # noqa: E402
import place  # noqa: E402
import pose_fit as pf  # noqa: E402
import profiles as prof  # noqa: E402
import shot_spec as ss  # noqa: E402
import test_synth_plate as tsp  # noqa: E402
import top_region as tr  # noqa: E402
import zone as zn  # noqa: E402

TD = HERE.parent / "testdata" / "pose_v2"
REPO = HERE.parents[2]
K = 1.15


def ref_of(name):
    r = pf.load_ref(TD / f"{name}.person.npz")
    r["path"] = str(TD / f"{name}.person.npz")
    return r


class Spec(unittest.TestCase):
    def test_table_matches_gold(self):
        rows = json.loads((HERE.parent / "plate" / "pose_round_prompts.json").read_text(encoding="utf-8"))["prompts"]
        gold = json.loads((REPO / "eval" / "shot_spec_gold.json").read_text(encoding="utf-8"))["specs"]
        self.assertEqual(len(rows), 17)
        self.assertEqual([ss.from_table(r).to_dict() for r in rows], gold)
        with self.assertRaises(ValueError):
            ss.ShotSpec("x", "fly", "bed", ["pelvis"])

    def test_save_load(self):
        s = ss.ShotSpec("Q08", "recline", "bed", ["pelvis", "elbow.L"])
        f = Path(tempfile.mkdtemp()) / "shot_spec.json"
        s.save(f)
        self.assertEqual(ss.ShotSpec.load(f), s)
        self.assertEqual(s.support_regions(), ["pelvis", "forearm.L"])


class Scale(unittest.TestCase):
    """body_dims ×1.15 по росту → зоны и телесные пороги ×1.15; посадка не падает."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        tsp.make(os.path.join(cls.tmp, "plate"), bed_y1=4.8)        # стена в 0.2 м за кроватью — «спинка»
        cls.plate = pf.Plate(os.path.join(cls.tmp, "plate"))
        cls.seed = cls.plate.seat_point(0.5, 0.65)[0]

    def dims(self, k):
        ref = ref_of("Q08_bed_recline_elbows")
        s = 1.62 * k / pf.height_est(ref["kp"])
        return prof.body_dims_from_kp(ref["kp"] * s)

    def test_body_dims(self):
        d1, d2 = self.dims(1.0), self.dims(K)
        for k in ("H", "L_thigh", "L_torso"):
            self.assertAlmostEqual(d2[k] / d1[k], K, delta=0.005, msg=k)
        self.assertAlmostEqual(d1["H"], 1.62, delta=0.01)

    def test_zones_and_thresholds_scale(self):
        d1, d2 = self.dims(1.0), self.dims(K)
        reg1 = tr.top_region(self.plate, "bed", self.seed, d1["H"])
        reg2 = tr.top_region(self.plate, "bed", self.seed, d2["H"])
        self.assertAlmostEqual(reg1.h0, 0.5, delta=0.03)
        self.assertNotIn("scale_suspect", reg1.flags)
        self.assertIsNotNone(reg1.backrest)
        self.assertIsNotNone(reg2.backrest)
        # спинка: верх полосы h0 + 0.55·H
        self.assertLessEqual(reg1.backrest[:, 2].max(), reg1.h0 + 0.55 * d1["H"] + 1e-6)
        self.assertGreater(reg2.backrest[:, 2].max(), reg1.backrest[:, 2].max() + 0.05)
        m = zn.metrics(reg1)
        c1, i1 = zn.zone_cells(m, "sit_edge", d1, self.seed[:2])
        c2, i2 = zn.zone_cells(m, "sit_edge", d2, self.seed[:2])
        self.assertAlmostEqual(i2["band_m"][1] / i1["band_m"][1], K, delta=0.01)
        self.assertGreater(i2["n_ok"], i1["n_ok"])
        self.assertTrue(all(c["near"] <= i1["band_m"][1] + 1e-6 for c in c1))
        r1, ri1 = zn.zone_cells(m, "recline", d1, self.seed[:2], has_backrest=True)
        r2, ri2 = zn.zone_cells(m, "recline", d2, self.seed[:2], has_backrest=True)
        self.assertAlmostEqual(ri2["band_m"][0] / ri1["band_m"][0], K, delta=0.01)
        self.assertTrue(all(ri1["band_m"][0] - 1e-6 <= c["far"] <= ri1["band_m"][1] + 1e-6 for c in r1))
        _, rn = zn.zone_cells(m, "recline", d1, self.seed[:2], has_backrest=False)
        self.assertIn("no_backrest", rn["flags"])
        # v0.16: lie uses pelvis→neck axis [75,105]°, independent of head height/H.
        KP = np.zeros((70, 3))
        KP[9], KP[10], KP[69] = [0.1, 0, 0.5], [-0.1, 0, 0.5], [0, 0.5, 0.5]
        KP[0:5] = [0, 0.7, 0.5 + 0.31]
        for H in (d1["H"], d2["H"]):
            self.assertTrue(ori.check(KP, "lie", H)["ok"])
            for tilt, expected in ((74., False), (76., True), (104., True), (106., False)):
                K2 = KP.copy()
                a = np.radians(tilt)
                K2[69] = [0., .5 * np.sin(a), .5 + .5 * np.cos(a)]
                self.assertEqual(ori.check(K2, "lie", H)["ok"], expected)

    def test_place_scaled_no_crash(self):
        for k in (1.0, K):
            spec = ss.ShotSpec("Q08", "recline", "bed", ["pelvis", "elbow.L", "elbow.R"])
            res = place.place_pose(ref_of("Q08_bed_recline_elbows"), self.plate, spec, self.seed, 1.62 * k,
                                   log=lambda m: None)
            rep = res["rep"]
            self.assertIn(rep["status"], place.STATUS_RANK)
            self.assertAlmostEqual(rep["body_dims"]["H"], 1.62 * k, delta=0.02)
            self.assertEqual(rep["zone"]["source"], "proxy")
            self.assertIn("orient", rep)
        spec = ss.ShotSpec("T04", "sit_floor", "floor", ["pelvis", "foot.L", "foot.R"])
        seed = self.plate.pixel_point(0.5, 0.9)
        rep = place.place_pose(ref_of("T04_floor_sit_knees_up"), self.plate, spec, seed, 1.62 * K,
                               log=lambda m: None)["rep"]
        self.assertIn(rep["status"], place.STATUS_RANK)
        self.assertEqual(rep["target_instance"], "floor")
        self.assertIn(rep["support_object"], ("floor", "other", "none"))
        if rep["support_object"] != "floor":
            self.assertEqual(rep["status"], "wrong_surface")


class Profiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        tsp.make(os.path.join(cls.tmp, "plate"), bed_y1=4.8)
        cls.plate = pf.Plate(os.path.join(cls.tmp, "plate"))
        cls.seed = cls.plate.seat_point(0.5, 0.65)[0]
        cls.root = Path(cls.tmp) / "profiles"

    def test_location_profile_and_fallback(self):
        prof.write_draft("synth", self.plate, {"bed": self.seed}, 1.62, root=self.root)
        lp = prof.LocationProfile.load("synth", root=self.root)
        self.assertIsNotNone(lp)
        self.assertFalse(lp.data["reviewed"])
        self.assertEqual(lp.find_object("bed")[1], "bed")
        self.assertEqual(lp.find_object("armchair")[1], "bed")             # замена из substitutes
        self.assertIsNone(lp.find_object("piano")[0])
        self.assertIsNotNone(lp.find_object("floor")[0])
        self.assertTrue(any("recline" in a["pose_classes"] for a in lp.data["anchors"]))
        self.assertIsNone(prof.LocationProfile.load("nope", root=self.root))
        ref = ref_of("Q08_bed_recline_elbows")
        spec = ss.ShotSpec("Q08", "recline", "bed", ["pelvis", "elbow.L", "elbow.R"])
        rep = place.place_pose(ref, self.plate, spec, self.seed, 1.62, profile=lp, log=lambda m: None)["rep"]
        self.assertEqual(rep["zone"]["source"], "profile")
        self.assertEqual(rep["zone"]["anchor_id"], "bed_1_recline")
        spec = ss.ShotSpec("X", "sit_edge", "piano", ["pelvis"])
        res = place.place_pose(ref, self.plate, spec, self.seed, 1.62, profile=lp, log=lambda m: None)
        self.assertEqual(res["status"], "no_target")
        spec = ss.ShotSpec("X", "sit_edge", "armchair", ["pelvis"])
        rep = place.place_pose(ref, self.plate, spec, self.seed, 1.62, profile=lp, log=lambda m: None)["rep"]
        self.assertEqual(rep["zone"]["via"], "bed")

    def test_body_json(self):
        self.assertIsNone(prof.load_body("nobody", root=self.root))
        dims = {"H": 1.70, "L_thigh": 0.43, "L_torso": 0.55}
        prof.write_body("tchar", 1.70, dims, root=self.root, shape_params=[0.1] * 45)
        b = prof.load_body("tchar", root=self.root)
        self.assertEqual(b["body_dims"], dims)
        spec = ss.ShotSpec("T04", "sit_floor", "floor", ["pelvis", "foot.L", "foot.R"])
        rep = place.place_pose(ref_of("T04_floor_sit_knees_up"), self.plate, spec, self.plate.pixel_point(0.5, 0.9),
                               b["height_m"], body_src="profile", dims_override=b["body_dims"], log=lambda m: None)["rep"]
        self.assertEqual(rep["body_source"], "profile")
        self.assertEqual(rep["body_dims"], dims)


if __name__ == "__main__":
    unittest.main()
