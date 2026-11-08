"""v2.6: YAML от корня репо без fallback; длинная доступная сторона sit_edge, включая авто-якоря."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orient as ori
import placement_config as pc
import pose_fit as pf
import profiles
import refine
import top_region as tr
import zone as zn


def rectangle():
    """Сетка 2:1; камера справа, то есть ближайшая yaw-нормаль у короткой стороны."""
    mask = np.ones((100, 50), bool)
    return tr.TopRegion("bench", "synthetic", np.zeros(2), mask.shape, np.full(mask.shape, 0.5), mask,
                        ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * tr.CELL,
                        0.5, (50, 25), np.array([3.0, 0.5]))


def accessible_sides():
    return [dict(s, accessible=True, access_frac=1.0) for s in zn.rect_sides(rectangle())]


class Config(unittest.TestCase):
    def test_yaml_reads_repo_root_from_other_cwd(self):
        old_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                pc._load.cache_clear()
                self.assertEqual(pc.CONFIG_PATH, HERE.parents[2] / "config" / "placement.yaml")
                self.assertEqual(pc.section("place")["attempts"], 5)
                self.assertEqual(pc.section("orient")["tilt_range"]["recline"], [25, 70])
                self.assertEqual(pc.section("zone")["long_side_frac"], 0.8)
                self.assertEqual(pc.section("top_region")["scale_range"]["bed"], [0.2, 0.75])
                self.assertEqual(pc.section("refine")["limits_deg"]["legs"], 20)
                self.assertEqual(pc.section("pose_fit")["penetration_depth_max_m"], 0.35)
                os.chdir(old_cwd)  # Windows не удаляет текущую рабочую директорию.
        finally:
            os.chdir(old_cwd)
            pc._load.cache_clear()

    def test_missing_section_or_key_has_no_fallback(self):
        with self.assertRaises(KeyError):
            pc.section("absent_section")
        with self.assertRaises(KeyError):
            pc.section("place")["absent_threshold"]

    def test_missing_file_has_no_fallback(self):
        pc._load.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(pc, "CONFIG_PATH", Path(tmp) / "absent.yaml"):
                with self.assertRaises(FileNotFoundError):
                    pc.section("place")
        finally:
            pc._load.cache_clear()

    def test_non_mapping_section_rejected(self):
        with patch.object(pc, "_load", return_value={"bad": []}):
            with self.assertRaises(ValueError):
                pc.section("bad")

    def test_modules_use_yaml(self):
        for module, name in [(ori, "orient"), (zn, "zone"), (tr, "top_region"),
                             (pf, "pose_fit"), (refine, "refine")]:
            self.assertEqual(module.CFG, pc.section(name))
        self.assertEqual(ori.TILT_RANGE, pc.section("orient")["tilt_range"])
        self.assertEqual(tr.CELL, pc.section("top_region")["cell"])
        self.assertEqual(refine.LIMITS_DEG, pc.section("refine")["limits_deg"])


class LongSide(unittest.TestCase):
    def test_rectangle_two_to_one_prioritizes_long_over_yaw(self):
        sides = accessible_sides()
        self.assertAlmostEqual(max(s["length"] for s in sides) / min(s["length"] for s in sides), 2.0)
        short = min(sides, key=lambda s: s["length"])
        chosen = zn.pick_side(sides, short["yaw_deg"])
        self.assertGreaterEqual(chosen["length"], 0.8 * max(s["length"] for s in sides))
        self.assertFalse(chosen["edge_short"])
        self.assertTrue(all("edge_short" not in s for s in sides))

    def test_yaw_breaks_tie_between_long_sides(self):
        sides = accessible_sides()
        long = max(sides, key=lambda s: s["length"])
        self.assertEqual(zn.pick_side(sides, long["yaw_deg"])["yaw_deg"], long["yaw_deg"])

    def test_only_short_accessible_flags_edge_short(self):
        sides = accessible_sides()
        threshold = 0.8 * max(s["length"] for s in sides)
        for side in sides:
            side["accessible"] = side["length"] < threshold
        chosen = zn.pick_side(sides, 0)
        self.assertTrue(chosen["edge_short"])
        self.assertLess(chosen["length"], threshold)

    def test_threshold_uses_available_max(self):
        sides = [{"length": 2.0, "yaw_deg": 0, "accessible": False},
                 {"length": 1.25, "yaw_deg": 90, "accessible": True},
                 {"length": 1.0, "yaw_deg": 0, "accessible": True},
                 {"length": 0.9, "yaw_deg": 0, "accessible": True}]
        # 1.0 is exactly 80% of 1.25, despite the inaccessible 2.0 side.
        chosen = zn.pick_side(sides, 0)
        self.assertEqual(chosen["length"], 1.0)
        self.assertTrue(chosen["edge_short"])

    def test_no_available_side(self):
        self.assertIsNone(zn.pick_side([], 0))
        self.assertIsNone(zn.pick_side([dict(s, accessible=False) for s in accessible_sides()], 0))

    def test_auto_anchor_uses_same_long_side_rule(self):
        reg, sides = rectangle(), accessible_sides()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(tr, "top_region", return_value=reg), \
                patch.object(tr, "save_npz"), patch.object(tr, "mask_png"), \
                patch.object(tr, "walls", return_value=[]), \
                patch.object(zn, "edge_sides", return_value=sides):
            path = profiles.write_draft("sfw_rectangle", None, {"bench": [1, 0.5, 0.5]}, 1.62, root=tmp)
            draft = json.loads(path.read_text(encoding="utf-8"))
        anchor = next(a for a in draft["anchors"] if "sit_edge" in a["pose_classes"])
        chosen = min(sides, key=lambda s: abs(ori.wrap(s["yaw_deg"] - anchor["facing_yaw_deg"])))
        self.assertGreaterEqual(chosen["length"], 0.8 * max(s["length"] for s in sides))


if __name__ == "__main__":
    unittest.main()
