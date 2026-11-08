"""Approved step 9d geometry. Exact convex-polygon/grid-cell intersection; no raster estimate."""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from placement_config import section


def polygon_area(poly):
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.0
    # Translate first: avoid cancellation for far-away world origins.
    p = p - p[0]
    return abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) -
                            p[:, 1] * np.roll(p[:, 0], -1)))) / 2


def convex_polygon(points):
    p = np.unique(np.asarray(points, float)[:, :2], axis=0)
    if len(p) < 3:
        return np.empty((0, 2))
    try:
        return p[ConvexHull(p).vertices]
    except QhullError:
        return np.empty((0, 2))


def clip_rect(poly, lo, hi):
    p = np.asarray(poly, float)
    for axis in (0, 1):
        for bound, sign in ((lo[axis], 1), (hi[axis], -1)):
            if not len(p):
                return p
            out = []
            prev = p[-1]
            pin = sign * (prev[axis] - bound) >= 0
            for cur in p:
                cin = sign * (cur[axis] - bound) >= 0
                if cin != pin:
                    out.append(prev + (cur - prev) * ((bound - prev[axis]) / (cur[axis] - prev[axis])))
                if cin:
                    out.append(cur)
                prev, pin = cur, cin
            p = np.asarray(out, float).reshape(-1, 2)
    return p


def hull_metrics(points, region=None, margin=0.0):
    hull = convex_polygon(points)
    area = polygon_area(hull)
    out = {"area_m2": area, "in_mask_area_m2": None, "in_mask_fraction": None,
           "hull_xy": hull.tolist(), "degenerate": area == 0.0}
    if region is None or not area:
        return out
    cell = section("top_region")["cell"]
    mask = region.mask if margin == 0 else region.mask & (region.dt >= margin)
    ij = np.argwhere(mask)
    lo = region.origin + ij * cell
    hi = lo + cell
    keep = np.all(hi >= hull.min(0), axis=1) & np.all(lo <= hull.max(0), axis=1)
    lo, hi = lo[keep], hi[keep]
    # Whole cells strictly inside the convex polygon have exact area CELL².
    # Clip only boundary cells; placement evaluates this repeatedly during XY search.
    full = np.ones(len(lo), bool)
    possible = np.ones(len(lo), bool)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        edge = b - a
        # Minimum/maximum oriented edge-cross over the cell's four corners.
        dx0, dx1 = lo[:, 0] - a[0], hi[:, 0] - a[0]
        dy0, dy1 = lo[:, 1] - a[1], hi[:, 1] - a[1]
        cross = np.stack([edge[0] * y - edge[1] * x for x, y in
                          ((dx0, dy0), (dx0, dy1), (dx1, dy0), (dx1, dy1))])
        full &= cross.min(axis=0) >= 0
        possible &= cross.max(axis=0) >= 0
    boundary = possible & ~full
    inside = int(full.sum()) * cell ** 2 + sum(
        polygon_area(clip_rect(hull, a, b)) for a, b in zip(lo[boundary], hi[boundary]))
    out.update(in_mask_area_m2=float(inside), in_mask_fraction=float(np.clip(inside / area, 0, 1)))
    return out
