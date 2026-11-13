"""Серверная проверка предметов (0.2.0 «е»), те же правила, что у 3D-окна (web/roomGeometry.js).

- Проникновение: pose_fit.penetration (по нормали, кэп 35 см; ниже видимого пола — полная глубина);
  p95 проникших точек > допуска жёсткой поверхности (3 см) → prop_penetration. Простое касание
  поверхности (BVH-пересечение в Blender без глубины) больше не отказ: prop_hits_room = пересечение И глубина.
- Опора: предмет не касается тела (≤ 3 см) и висит над видимой горизонтальной поверхностью > 5 см
  → prop_floating. Опоры под предметом в кадре не видно — не проверяем (как в окне).
Мир posev2: z — вверх, пол z = 0.
"""
import numpy as np

PEN_TOL_M = 0.03        # = place.PEN_HARD_M (жёсткая поверхность)
PEN_MIN_M = 0.002       # = pose_fit.PEN_MIN_M
FLOAT_TOL_M = 0.05      # «висит > 5 см»
BODY_TOUCH_M = 0.03     # предмет у тела — держат/надет, опору не проверяем
CELL_M = 0.10           # = roomGeometry.js сетка горизонталей (пол над точкой)
BOTTOM_BAND_M = 0.02
HORIZONTAL_NZ = 0.8


def surface_name(plate, point):
    tree, P, N = plate.trees["any"]
    _, j = tree.query(point)
    if N[j][2] > HORIZONTAL_NZ:
        return "hard:floor" if abs(P[j][2]) < 0.03 else "hard:surface"
    if abs(N[j][2]) < 0.3:
        return "hard:wall"
    return "hard:any"


def penetration(plate, points, depth_fn=None):
    if depth_fn is None:
        from pose_fit import penetration as depth_fn
    depth = np.asarray(depth_fn(plate, points), float)
    hit = depth > PEN_MIN_M
    if not hit.any():
        return dict(depth_m=0.0, max_m=0.0, n=0, ok=True, surface=None)
    p95 = float(np.percentile(depth[hit], 95))
    deepest = points[int(np.argmax(depth))]
    return dict(depth_m=p95, max_m=float(depth.max()), n=int(hit.sum()), ok=p95 <= PEN_TOL_M,
                surface=surface_name(plate, deepest))


def support(plate, points, body=None):
    points = np.asarray(points, float)
    if body is not None and len(body):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(np.asarray(body, float)).query(points)
        if float(d.min()) <= BODY_TOUCH_M:
            return dict(checked=False, reason="touches_body", gap_m=None, ok=True)
    tree, P, _ = plate.trees["h"]
    if not len(P):
        return dict(checked=False, reason="no_visible_support", gap_m=None, ok=True)
    zmin = float(points[:, 2].min())
    bottom = points[points[:, 2] <= zmin + BOTTOM_BAND_M]
    from scipy.spatial import cKDTree
    xy = cKDTree(P[:, :2])
    heights = []
    for group in xy.query_ball_point(bottom[:, :2], CELL_M / 2):
        below = [P[k][2] for k in group if P[k][2] <= zmin + BOTTOM_BAND_M]
        if below:
            heights.append(max(below))
    if not heights:
        return dict(checked=False, reason="no_visible_support", gap_m=None, ok=True)
    gap = zmin - max(heights)
    return dict(checked=True, reason=None, gap_m=float(gap), ok=gap <= FLOAT_TOL_M)


def check(plate, points, body=None, overlap_pairs=0, depth_fn=None):
    """→ (flags, checks) предмета. flags дополняют flags из prop_render."""
    points = np.asarray(points, float)
    pen = penetration(plate, points, depth_fn)
    sup = support(plate, points, body)
    flags = {"prop_penetration": not pen["ok"], "prop_floating": not sup["ok"],
             "prop_hits_room": bool(overlap_pairs) and not pen["ok"]}
    return flags, dict(penetration=pen, support=sup, room_overlap_pairs=int(overlap_pairs),
                       room_depth_m=pen["depth_m"])
