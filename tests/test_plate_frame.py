"""plate_geometry 0.2.0: floor / several levels / no floor / top-down, on synthetic MoGe-like point maps."""
import importlib.util
import os
import sys

import numpy as np

PG = os.environ.get("PLATE_GEOMETRY", os.path.join(os.path.dirname(__file__), "../tools/scene3d/plate/plate_geometry.py"))
spec = importlib.util.spec_from_file_location("plate_geometry", PG)
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

W, H, FX = 336, 192, 260.0


def render(h, pitch, rects, roll=0.0):
    """Point map (OpenCV camera) of world rectangles (Z up, camera at (0,0,h) looking along +Y, pitched down)."""
    t, r = np.radians(pitch), np.radians(roll)
    up = pg._up(pitch, roll)
    fwd = np.array([0, np.sin(r) * 0 + 0, 1.0])
    # camera axes in world: forward pitched down, roll about forward
    F = np.array([0, np.cos(t), -np.sin(t)])
    U0 = np.array([0, np.sin(t), np.cos(t)])
    X0 = np.cross(F, U0)
    X = np.cos(r) * X0 + np.sin(r) * U0
    U = -np.sin(r) * X0 + np.cos(r) * U0
    j, i = np.mgrid[0:H, 0:W]
    d_cv = np.stack([(i + .5 - W / 2) / FX, (j + .5 - H / 2) / FX, np.ones((H, W))], -1)
    d_w = d_cv[..., :1] * X + d_cv[..., 1:2] * (-U) + d_cv[..., 2:] * F
    C = np.array([0, 0, h])
    best = np.full((H, W), np.inf)
    for o, a, b in rects:
        o, a, b = map(np.asarray, (o, a, b))
        n = np.cross(a, b)
        den = d_w @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = ((o - C) @ n) / den
        P = C + tt[..., None] * d_w - o
        ua = P @ a / (a @ a)
        ub = P @ b / (b @ b)
        hit = (tt > 0.05) & (ua >= 0) & (ua <= 1) & (ub >= 0) & (ub <= 1) & (tt < best)
        best[hit] = tt[hit]
    valid = np.isfinite(best)
    pts = d_cv * np.where(valid, best, 0)[..., None]
    rng = np.random.default_rng(1)
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(rng.normal(0, 1, (H, W)), 6)
    pts = pts * (1 + 0.004 * smooth / smooth.std())[..., None]   # MoGe-like: smooth, not per-pixel noise
    pts = pts * (1 + rng.normal(0, 0.0005, (H, W, 1)))
    assert np.allclose(up, np.array([X @ [0, 0, 1], -(U @ [0, 0, 1]), F @ [0, 0, 1]]), atol=1e-9)
    return pts.astype(np.float32), valid


FLOOR = ([-6, 0.3, 0], [12, 0, 0], [0, 12, 0])
WALL = ([-6, 7, 0], [12, 0, 0], [0, 0, 3])


def frame(pts, valid, normals=None):
    n, d, rep = pg.choose_floor(pts, normals, valid, H, np.random.default_rng(0))
    return n, d, rep, pg.world_rotation(n)


def angle(a, b):
    return np.degrees(np.arccos(np.clip(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def test_room_with_floor_keeps_old_ransac():
    pts, valid = render(1.35, 20, [FLOOR, WALL])
    n, d, rep, R = frame(pts, valid)
    assert rep["source"] == "ransac_lower" and rep["visible"], rep
    assert abs(d - 1.35) < 0.03 and angle(n, pg._up(20, 0)) < 1.0


def test_bed_over_floor_strip_is_not_the_floor():
    bed = ([-2.5, 1.8, 0.55], [5, 0, 0], [0, 3, 0])
    bed_front = ([-2.5, 1.8, 0], [5, 0, 0], [0, 0, 0.55])
    pts, valid = render(1.35, 35, [FLOOR, WALL, bed, bed_front])
    n, d, rep, R = frame(pts, valid)
    assert rep["visible"] and abs(d - 1.35) < 0.04, rep
    assert any(abs(lv["z"] - 0.55) < 0.05 for lv in rep["levels"]), rep["levels"]


def test_bed_filling_the_frame_is_not_the_floor():
    # Red room 13.11 (real plate: camera ~9 deg down, bed top fills the lower frame, floor is a sliver):
    # 0.1.10 took the bed top for the floor (camera 0.58 m above it). A level with a large part of the
    # scene visible below it is not the floor -> no visible floor, prior camera height.
    wall = ([-6, 4.5, 0], [12, 0, 0], [0, 0, 3])
    for front, floor_seen in [(2.32, False), (2.9, True)]:
        bed = ([-3, front, 0.6], [6, 0, 0], [0, 4.5 - front, 0])
        bed_front = ([-3, front, 0], [6, 0, 0], [0, 0, 0.6])
        pts, valid = render(1.27, 9, [FLOOR, wall, bed, bed_front])
        n, d, rep, _ = frame(pts, valid)
        assert angle(n, pg._up(9, 0)) < 2.0, rep
        if floor_seen:   # a strip of floor in front of the bed: the floor, camera height right
            assert rep["visible"] and abs(d - 1.27) < 0.04, (d, rep)
        else:
            assert rep["visible"] is False and rep["source"] == "prior_height" and d >= 1.27, (d, rep)
        assert any(abs(lv["z"] - (d - 1.27 + 0.6)) < 0.06 for lv in rep["levels"]), (d, rep["levels"])


def test_wall_close_up_without_floor_does_not_crash():
    wall = ([-3, 1.2, 0.0], [6, 0, 0], [0, 0, 4])
    pts, valid = render(1.4, 5, [wall])
    assert valid.all()
    n, d, rep, R = frame(pts, valid)
    assert rep["visible"] is False and rep["source"] == "prior_height", rep
    assert angle(n, pg._up(5, 0)) < 2.0 and d >= 1.5


def test_top_down_floor():
    pts, valid = render(2.4, 80, [([-4, -3, 0], [8, 0, 0], [0, 8, 0])])
    n, d, rep, R = frame(pts, valid)
    assert rep["visible"] and abs(d - 2.4) < 0.05, rep
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_roll_and_steps():
    step = ([-6, 3.0, 0.18], [12, 0, 0], [0, 4, 0])
    step_front = ([-6, 3.0, 0], [12, 0, 0], [0, 0, 0.18])
    pts, valid = render(1.5, 25, [FLOOR, step, step_front, WALL], roll=5)
    n, d, rep, R = frame(pts, valid)
    assert abs(d - 1.5) < 0.04 and angle(n, pg._up(25, 5)) < 1.0, (d, rep)
    assert sorted(round(lv["z"], 1) for lv in rep["levels"])[:2] == [0.0, 0.2], rep["levels"]


def test_level_camera_at_a_wall_is_not_top_down():
    pts, valid = render(1.5, 0, [([-3, 1.0, -1], [6, 0, 0], [0, 0, 5])])
    n, d, rep, R = frame(pts, valid)
    assert rep["visible"] is False and angle(n, pg._up(0, 0)) < 2.0, rep


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
