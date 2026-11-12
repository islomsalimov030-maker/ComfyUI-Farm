"""pose_fit.penetration 0.2.0: normal-depth cap and full depth below the visible floor (= roomGeometry.inspect)."""
import importlib
import os
import sys
import types

import numpy as np
from scipy.spatial import cKDTree

POSEV2 = os.environ.get("POSEV2_DIR", os.path.join(os.path.dirname(__file__), "../tools/scene3d/posev2"))
sys.path.insert(0, os.path.abspath(POSEV2))
pf = importlib.import_module("pose_fit")


def fake_plate(h=1.35, pitch_deg=30.0, floor_visible=True, table=None, W=320, H=180, fov_x=60.0):
    """Floor z=0 (and optionally a table top) seen by a pinhole camera at (0,0,h) looking along +Y, pitched down."""
    t = np.radians(pitch_deg)
    fwd = np.array([0, np.cos(t), -np.sin(t)])
    right = np.array([1.0, 0, 0])
    down = np.cross(fwd, right)                      # cv y axis (image down)
    R = np.stack([right, down, fwd], axis=1)         # world <- cv
    C = np.array([0, 0, h])
    fx = (W / 2) / np.tan(np.radians(fov_x) / 2)
    xs, ys = np.meshgrid(np.linspace(-4, 4, 161), np.linspace(0.3, 9, 175))
    P = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    if table:                                        # (x0, x1, y0, y1, z)
        x0, x1, y0, y1, zt = table
        keep = ~((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1))
        tx, ty = np.meshgrid(np.linspace(x0, x1, 21), np.linspace(y0, y1, 21))
        P = np.vstack([P[keep], np.stack([tx.ravel(), ty.ravel(), np.full(tx.size, zt)], axis=1)])
    N = np.tile([0, 0, 1.0], (len(P), 1))

    def project(X):
        c = (np.asarray(X, float) - C) @ R
        return fx * c[:, 0] / c[:, 2] + W / 2, fx * c[:, 1] / c[:, 2] + H / 2, c[:, 2]

    # depth map: nearest visible surface along each pixel ray (analytic: floor, table top)
    def depth_at(u, v):
        d = (np.stack([(u - W / 2) / fx, (v - H / 2) / fx, np.ones_like(u)], axis=1)) @ R.T
        zs = np.full(len(u), np.inf)
        down_rays = d[:, 2] < 0
        s = np.where(down_rays, -h / np.where(down_rays, d[:, 2], -1), np.inf)
        zs = np.minimum(zs, s)                       # cv z of a ray with unit z component = distance factor
        if table:
            x0, x1, y0, y1, zt = table
            st = np.where(d[:, 2] < 0, (zt - h) / np.where(d[:, 2] < 0, d[:, 2], -1), np.inf)
            hit = C + st[:, None] * d
            on = (hit[:, 0] >= x0) & (hit[:, 0] <= x1) & (hit[:, 1] >= y0) & (hit[:, 1] <= y1)
            zs = np.where(on & (st < zs), st, zs)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return np.where(inside, zs, np.inf)

    plate = types.SimpleNamespace(info={"floor": {"visible": floor_visible}}, P=P, Nrm=N,
                                  trees={"any": (cKDTree(P), P, N)}, project=project, depth_at=depth_at)
    return plate


def column(x, y, z0, z1, n=40):
    return np.stack([np.full(n, x), np.full(n, y), np.linspace(z0, z1, n)], axis=1)


def p95(pen):
    pen = np.sort(pen[pen > pf.PEN_MIN_M])
    return pen[min(len(pen) - 1, int(0.95 * len(pen)))] if len(pen) else 0.0


def test_standing_on_floor_is_clean():
    assert p95(pf.penetration(fake_plate(), column(0, 3, 0.0, 1.6))) == 0.0


def test_deep_below_floor_counts_full_depth():
    # yoga 645ee190: body 49 cm under the floor -> before 0.2.0 the 35 cm cap along the ray reported ~17 cm
    pen = pf.penetration(fake_plate(), column(0, 3, -0.49, 1.1))
    assert 0.45 <= p95(pen) <= 0.50


def test_buried_prop_is_detected():
    pen = pf.penetration(fake_plate(), column(1.0, 3, -0.52, -0.34, 10))
    assert p95(pen) >= 0.45


def test_grazing_ray_uses_normal_cap():
    plate = fake_plate(h=0.5, pitch_deg=5.0)
    pen = pf.penetration(plate, column(0, 5, -0.2, 1.4))
    assert 0.18 <= p95(pen) <= 0.21


def test_no_visible_floor_no_floor_rule():
    pen = pf.penetration(fake_plate(floor_visible=False), column(0, 3, -0.6, -0.5, 5))
    assert p95(pen) == 0.0          # deeper than the cap and no floor: nothing to compare with


def test_under_table_is_not_inside_floor():
    plate = fake_plate(table=(-0.6, 0.6, 2.6, 3.4, 0.75))
    pen = pf.penetration(plate, column(0, 3, 0.0, 0.3))
    assert p95(pen) == 0.0
