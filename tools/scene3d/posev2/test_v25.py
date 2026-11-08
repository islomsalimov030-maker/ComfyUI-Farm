"""3D-A v2.5 (08.11a, п.12): стена на синтетической плоскости; recline делит опору таз/спинка; след lie поворачивает
тело; p95 проникновения; кадр; сторона к камере сохраняется.
  python -m unittest test_v25        (из tools/scene3d/posev2, python ComfyUI-next)
"""
import math
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
import shot_spec as ss  # noqa: E402
import test_synth_plate as tsp  # noqa: E402
import top_region as tr  # noqa: E402

TD = HERE.parent / "testdata" / "pose_v2"


def ref_of(name):
    r = pf.load_ref(TD / f"{name}.person.npz")
    r["path"] = str(TD / f"{name}.person.npz")
    return r


def plate_of(**kw):
    d = os.path.join(tempfile.mkdtemp(), "plate")
    tsp.make(d, **kw)
    return pf.Plate(d)


class Walls(unittest.TestCase):
    def test_wall_plane(self):
        p = plate_of(bed_y1=4.5)                       # стена y = 5, фасад кровати y = 2.5 (0.5 м высотой)
        W = tr.walls(p)
        self.assertGreaterEqual(len(W), 1)
        w = W[0]
        self.assertAlmostEqual(abs(w["n"][1]), 1.0, delta=0.02)
        self.assertAlmostEqual(abs(w["n"][2]), 0.0, delta=1e-6)
        self.assertAlmostEqual(-w["d"] / w["n"][1], 5.0, delta=0.03)   # плоскость y = 5
        self.assertGreater(w["n"] @ (p.C - [0, 5, 0]), 0)                # нормаль к камере
        self.assertGreaterEqual(w["area_m2"], tr.WALL_MIN_AREA)
        self.assertGreater(w["z"][1], 1.0)
        low = [x for x in W if x["z"][1] < 1.0]                          # фасад кровати — низкая плоскость
        for x in low:
            self.assertLess(x["z"][1], 0.6)


class Recline(unittest.TestCase):
    def test_support_split_pelvis_backrest(self):
        p = plate_of(bed_y1=4.8)                       # стена в 0.2 м за кроватью — спинка
        seed = p.seat_point(0.5, 0.65)[0]
        spec = ss.ShotSpec("Q08", "recline", "bed", ["pelvis"],
                           secondary_contacts=[{"part": "elbow.L", "surface": "bed"},
                                               {"part": "elbow.R", "surface": "bed"}])
        B = place.prepare_body(ref_of("Q08_bed_recline_elbows"), spec, 1.62)
        place.orient_body(B, spec)
        H = B["dims"]["H"]
        treg = tr.top_region(p, "bed", seed, H)
        self.assertIsNotNone(treg.backrest)
        groups, _, info = place.build_groups(B, spec, treg, H, False)
        sup = {g[0]: g[2] for g in groups if g[3] == "support"}
        back = {g[0]: g[2] for g in groups if g[3] == "back"}
        self.assertIn("pelvis", sup)
        self.assertEqual(set(sup), {"pelvis"})  # 2b: only requested support_parts
        self.assertTrue(all(s == "target" for s in sup.values()))         # опора — только верх
        self.assertIn("torso_back", back)  # 2d: geometry from SAM keypoints, no labels required
        self.assertGreater(info['torso_back_vertices'], 0)
        self.assertTrue(all(s == "back" for s in back.values()))
        pel_z = ((B["kp"][9] + B["kp"][10]) / 2)[2]
        for n, idx, surf, role in groups:                                 # на спинке — только выше таза
            if role == "back" and n not in ("back", "torso_back"):
                self.assertGreater(B["V"][idx, 2].mean(), pel_z + place.RECLINE_UP_H * H)
        self.assertTrue(all(g[0] in place.BACK_REGIONS + ("back", "torso_back") for g in groups if g[3] == "back"))


class LieFootprint(unittest.TestCase):
    def test_lie_rotates(self):
        p = plate_of(bed_y1=4.8, bed_x=0.45)           # кровать 0.9 × 2.3 м: лёжа поперёк луча не влезает
        seed = p.seat_point(0.5, 0.75)[0]
        spec = ss.ShotSpec("L", "lie", "bed", ["pelvis"])
        rep = place.place_pose(ref_of("Q08_bed_recline_elbows"), p, spec, seed, 1.62, log=lambda m: None)["rep"]
        tried = [a["lie_yaw_add"] for a in rep["attempts"]]
        self.assertTrue(rep["footprint"]["checked"])
        if rep["footprint"]["ok"]:
            # D1: footprint is now the support_parts hull (pelvis here), not the whole low body.
            # It may fit at zero yaw; acceptance must use measured area fraction.
            self.assertGreaterEqual(rep["footprint"]["support_hull"]["in_mask_fraction"],
                                    place.CFG["hull_in_mask_min"])
        else:
            self.assertTrue(any(t != 0 for t in tried) or rep["footprint"]["n_out"] > 0)
        self.assertLessEqual(abs(rep["orient"]["lie_yaw_add_deg"]), 90)
        self.assertEqual(rep["orient"]["lie_yaw_add_deg"] % 15, 0)


class Penetration(unittest.TestCase):
    def test_p95(self):
        p = plate_of()
        base = p.pixel_point(0.5, 0.9)                                    # пол в кадре
        rng = np.random.default_rng(0)
        X = base + np.column_stack([rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.2, 0.2, 200), np.zeros(200)])
        X[:, 2] = 0.05                                                     # над полом — не проникают
        X[:100, 2] = -0.01                                                 # 100 вершин на 1 см в полу
        X[100:103, 2] = -0.10                                              # 3 вершины на 10 см
        st = place.penetration_stats(p, X, None, soft=False)
        self.assertAlmostEqual(st["max_cm"], 10.0, delta=0.5)
        self.assertLess(st["hard_p95_cm"], 3.0)                            # p95 не равен максимуму
        self.assertTrue(st["ok"])
        X[:100, 2] = -0.04
        self.assertFalse(place.penetration_stats(p, X, None, soft=False)["ok"])   # жёсткое > 3 см
        bed = tr.top_region(p, "bed", p.seat_point(0.5, 0.65)[0], 1.62)
        Y = bed.points3d()[:50].copy()
        Y[:, 2] -= 0.045                                                   # 4.5 см в матрас — мягкое, ≤ 6 см
        st2 = place.penetration_stats(p, Y, bed, soft=True)
        self.assertGreater(st2["soft_n"], 0)
        self.assertTrue(st2["ok"])


class Frame(unittest.TestCase):
    def test_framing(self):
        p = plate_of()
        P = np.array([p.pixel_point(0.5, 0.9), p.pixel_point(0.4, 0.85)])
        self.assertTrue(place.framing(P, p)["ok"])
        Q = np.vstack([P, p.pixel_point(0.995, 0.9)])                      # за полем 2 %
        self.assertFalse(place.framing(Q, p)["ok"])


class Side(unittest.TestCase):
    def test_solve_yaw_keeps_side(self):
        KP = np.zeros((70, 3))                                             # стоит, грудь +y
        KP[9], KP[10] = [-0.1, 0, 0.9], [0.1, 0, 0.9]
        KP[5], KP[6] = [-0.18, 0, 1.4], [0.18, 0, 1.4]
        KP[69] = [0, 0, 1.45]
        self.assertGreater(ori.chest(KP)[1], 0.9)
        cam = np.array([0, 3.0, 1.2])                                      # камера спереди
        self.assertEqual(ori.side_to_camera(KP, cam), 1)
        to = cam - (KP[9] + KP[10]) / 2
        fn = lambda a: ori.side_for_yaws(KP, a, to)                        # noqa: E731
        y, conflict, inw = ori.solve_yaw(90.0, 1, window=-90.0, side_fn=fn)   # окно смотрит от камеры
        self.assertTrue(conflict)
        self.assertEqual(int(fn(np.array([y]))[0]), 1)                     # сторона сохранена
        y2, c2, _ = ori.solve_yaw(150.0, 1, window=None, side_fn=fn)
        self.assertFalse(c2)
        self.assertEqual(int(fn(np.array([y2]))[0]), 1)
        self.assertLess(abs(ori.wrap(y2 - 150.0)), 61)

    def test_place_keeps_side(self):
        p = plate_of()
        spec = ss.ShotSpec("T04", "sit_floor", "floor", ["pelvis", "foot.L", "foot.R"])
        rep = place.place_pose(ref_of("T04_floor_sit_knees_up"), p, spec, p.pixel_point(0.5, 0.9), 1.62,
                               log=lambda m: None)["rep"]
        # Integration: actual place() and final validator invoke the same p95
        # on identical final geometry, including unrounded per-surface values.
        self.assertEqual(rep['penetration'], rep['validation']['penetration'])
        self.assertEqual(rep['penetration']['hard_p95_m'], rep['validation']['penetration']['hard_p95_m'])
        o = rep["orient"]
        if o["side_ref"] != "side":
            self.assertIn(o["side_final"], (o["side_ref"], "side"))
        self.assertNotIn("side_conflict", rep["flags"])


if __name__ == "__main__":
    unittest.main()
