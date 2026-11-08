"""Постановка v2.4 (07.11f, п.6 Р2): зона посадки на верхе предмета (target ≠ floor).

Метрики ячеек (body-независимые): near/far — глубинное расстояние до ближнего/дальнего к камере края маски в том же
поперечном столбце, dt — до любого края. Правило класса применяется с body_dims персонажа:
  sit_edge               — таз: near ≤ 0.6·L_thigh;
  kneel, lie, sit_tucked — центр опорных частей: ячейка с максимумом dt;
  recline                — таз: 0.6·L_torso ≤ far ≤ 0.9·L_torso (есть спинка); нет спинки → как kneel + no_backrest.
Возвращает упорядоченный список кандидатов (≤ N_CELLS, разнесены ≥ MIN_SEP м) — попытки п.8 идут по нему.
"""
import numpy as np

from placement_config import section
from top_region import CELL

CFG = section("zone")
EDGE_MIN = CFG["edge_min"]
MIN_SEP = CFG["min_sep"]
N_CELLS = CFG["n_cells"]
CENTER_CLASSES = ("kneel", "lie", "sit_tucked")


def metrics(reg):
    """метрики всех ячеек маски: xy (M,2), h, near, far, dt."""
    i, j = reg.cells()
    xy = reg.origin + (np.stack([i, j], 1) + 0.5) * CELL
    return {"xy": xy, "h": reg.height[i, j], "near": reg.edge_dist(xy, "near"), "far": reg.edge_dist(xy, "far"),
            "dt": reg.dt[i, j]}


def _spread(order, xy, n=N_CELLS, sep=MIN_SEP):
    out = []
    for k in order:
        if all(np.linalg.norm(xy[k] - xy[m]) >= sep for m in out):
            out.append(k)
        if len(out) >= n:
            break
    return out


def zone_cells(m, pose_class, dims, seed_xy=None, has_backrest=True):
    """m — metrics(); dims — {H, L_thigh, L_torso}. -> (cands, info). cands: [{xy, h, near, far, dt, anchor}]."""
    xy, dt = m["xy"], m["dt"]
    seed_d = np.zeros(len(xy)) if seed_xy is None else np.linalg.norm(xy - np.asarray(seed_xy)[:2], axis=1)
    flags = []
    rule = pose_class
    if pose_class == "recline" and not has_backrest:
        flags.append("no_backrest")
        rule = "kneel"
    if rule == "sit_edge":
        lim = EDGE_PELVIS_THIGH * dims["L_thigh"]
        ok = (m["near"] <= lim) & (dt >= EDGE_MIN)
        order = np.lexsort((-dt, seed_d))                         # ближе к точке таблицы по поперёк, затем глубже
        anchor = "pelvis"
        band = [0.0, round(lim, 3)]
    elif rule in CENTER_CLASSES:
        ok = dt >= EDGE_MIN
        order = np.lexsort((seed_d, -np.round(dt, CFG["rank_decimals"])))           # максимум dt; при равенстве — ближе к точке таблицы
        anchor = "support"
        band = None
    elif rule == "recline":
        lo, hi = CFG["recline_torso"][0] * dims["L_torso"], CFG["recline_torso"][1] * dims["L_torso"]
        ok = (m["far"] >= lo) & (m["far"] <= hi) & (dt >= EDGE_MIN)
        order = np.lexsort((seed_d, -np.round(dt, CFG["rank_decimals"])))
        anchor = "pelvis"
        band = [round(lo, 3), round(hi, 3)]
    else:
        raise ValueError(f"зона не определена для {pose_class} (target ≠ floor)")
    order = [k for k in order if ok[k]]
    pick = _spread(order, xy)
    cands = [{"xy": xy[k].tolist(), "h": float(m["h"][k]), "near": float(m["near"][k]), "far": float(m["far"][k]),
              "dt": float(dt[k]), "anchor": anchor} for k in pick]
    info = {"rule": rule, "band_m": band, "n_ok": int(len(order)), "flags": flags,
            "_all": (xy[order], np.asarray(m["h"])[order])}           # все ячейки зоны (сдвиг к центру кадра, п.8)
    return cands, info


# ------------------------------------------------------------------------------------------------ 08.11a п.4: край sit_edge
ACCESS_W_H = CFG["access_w_h"]
ACCESS_FRAC = CFG["access_frac"]
ACCESS_GAP = CFG["access_gap"]
SIDE_STEP = CFG["side_step"]
EDGE_PELVIS_THIGH = CFG["edge_pelvis_thigh"]


def rect_sides(reg):
    """стороны минимального описанного прямоугольника маски (перебор угла 0.5°). -> [{p0, p1, n (наружу), length,
    yaw_deg (нормали)}]."""
    P = reg.points3d()[:, :2]
    best = None
    for a in np.radians(np.arange(0, 90, CFG["rect_step_deg"])):
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        Q = P @ R                                                   # координаты в повёрнутых осях
        lo, hi = Q.min(0) - CELL / 2, Q.max(0) + CELL / 2                   # + половина ячейки
        ar = np.prod(hi - lo)
        if best is None or ar < best[0]:
            best = (ar, R, lo, hi)
    _, R, lo, hi = best
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]) @ R.T
    c = corners.mean(0)
    sides = []
    for k in range(4):
        p0, p1 = corners[k], corners[(k + 1) % 4]
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([t[1], -t[0]])
        if n @ ((p0 + p1) / 2 - c) < 0:
            n = -n
        sides.append({"p0": p0, "p1": p1, "n": n, "length": float(np.linalg.norm(p1 - p0)),
                      "yaw_deg": float(np.degrees(np.arctan2(n[1], n[0])))})
    return sides


def side_access(side, freg, H, plate=None):
    """доля длины стороны, перед которой свободный пол (маска floor) шириной ≥ ACCESS_W_H·H.
    plate задан: точка полосы вне кадра камеры плиты считается свободной (пол за краем кадра не виден, но и
    видимых препятствий там нет); в кадре — только маска floor."""
    if freg is None or not freg.mask.any():
        return 0.0
    L = side["length"]
    ts = np.arange(SIDE_STEP / 2, L, SIDE_STEP) / L
    ds = np.arange(ACCESS_GAP, ACCESS_GAP + ACCESS_W_H * H + CFG["access_endpoint_eps"], CELL)
    good = 0
    for t in ts:
        s = side["p0"] + t * (side["p1"] - side["p0"])
        pts = s[None] + ds[:, None] * side["n"][None]
        free = freg.in_mask(pts)
        if plate is not None:
            P3 = np.column_stack([pts, np.full(len(pts), freg.h0)])
            u, v, z = plate.project(P3)
            free |= (u < 0) | (u >= plate.Wp) | (v < 0) | (v >= plate.Hp) | (z <= 0)
        good += bool(np.all(free))
    return good / max(len(ts), 1)


def edge_sides(reg, freg, H, plate=None):
    """все стороны + доступность (п.4). -> список dict с accessible/access_frac."""
    out = []
    for s in rect_sides(reg):
        f = side_access(s, freg, H, plate)
        out.append(dict(s, access_frac=round(f, 2), accessible=bool(f >= ACCESS_FRAC)))
    return out


def pick_side(sides, want_yaw_deg):
    """Сначала длинные ≥ long_side_frac·max(доступных), затем ближайшая к yaw нормаль.
    edge_short — выбранная сторона короткая относительно всех: длинные стороны недоступны.
    Входные стороны не изменяются; нет доступных -> None.
    """
    acc = [s for s in sides if s["accessible"]]
    if not acc:
        return None
    min_long = CFG["long_side_frac"] * max(s["length"] for s in acc)
    long = [s for s in acc if s["length"] >= min_long]
    best = min(long, key=lambda s: abs((s["yaw_deg"] - want_yaw_deg + 180) % 360 - 180))
    return dict(best, edge_short=bool(best["length"] < CFG["long_side_frac"] * max(s["length"] for s in sides)))


def edge_cells(m, side, dims, seed_xy=None):
    """ячейки sit_edge у выбранной стороны: таз не дальше 0.6·L_thigh от неё, dt ≥ EDGE_MIN, проекция на сторону
    внутри отрезка. Порядок: ближе к точке таблицы вдоль стороны, затем глубже (dt)."""
    xy, dt = m["xy"], m["dt"]
    p0, n = side["p0"], side["n"]
    t = (side["p1"] - p0) / side["length"]
    dist = (p0 - xy) @ n                                            # внутрь от стороны (+)
    along = (xy - p0) @ t
    lim = EDGE_PELVIS_THIGH * dims["L_thigh"]
    ok = (dist >= 0) & (dist <= lim) & (dt >= EDGE_MIN) & (along >= 0) & (along <= side["length"])
    seed_a = 0.5 * side["length"] if seed_xy is None else float((np.asarray(seed_xy)[:2] - p0) @ t)
    order = np.lexsort((-dt, np.abs(along - seed_a)))
    order = [k for k in order if ok[k]]
    pick = _spread(order, xy)
    cands = [{"xy": xy[k].tolist(), "h": float(m["h"][k]), "near": float(dist[k]), "far": float(m["far"][k]),
              "dt": float(dt[k]), "anchor": "pelvis"} for k in pick]
    return cands, {"rule": "sit_edge_side", "band_m": [0.0, round(lim, 3)], "n_ok": int(len(order)), "flags": [],
                   "_all": (xy[order], np.asarray(m["h"])[order]),
                   "side_yaw_deg": round(side["yaw_deg"], 1), "side_len_m": round(side["length"], 2),
                   "side_access": side.get("access_frac")}


# ------------------------------------------------------------------------------------------------ 08.11a п.6: у стены
WALL_BAND_H = CFG["wall_band_h"]
WALL_MIN_TOP = CFG["wall_min_top"]
WALL_LAT_MARGIN = CFG["wall_lat_margin"]


def pick_wall(walls, seed_xy):
    """ближайшая к точке таблицы стена, у которой проекция точки внутри стены и верх ≥ WALL_MIN_TOP."""
    s = np.asarray(seed_xy, float)[:2]
    best = None
    for k, w in enumerate(walls):
        if w["z"][1] < WALL_MIN_TOP:
            continue
        lat = float(np.r_[s, 0] @ w["lat_dir"])
        if not (w["lat"][0] - CFG["wall_pick_lat_margin"] <= lat <= w["lat"][1] + CFG["wall_pick_lat_margin"]):
            continue
        D = float(w["n"][:2] @ s + w["d"])
        if D < 0:
            continue
        if best is None or D < best[0]:
            best = (D, k)
    return None if best is None else best[1]


def wall_cells(freg, wall, dims, reach_m, seed_xy=None):
    """ячейки пола: расстояние таза до плоскости стены WALL_BAND_H·H, проекция внутри стены; порядок — ближе к
    |D − горизонтальный вынос рук| (руки достают), затем к точке таблицы."""
    i, j = freg.cells()
    xy = freg.origin + (np.stack([i, j], 1) + 0.5) * CELL
    D = xy @ wall["n"][:2] + wall["d"]
    lat = xy @ wall["lat_dir"][:2]
    H = dims["H"]
    ok = (D >= WALL_BAND_H[0] * H) & (D <= WALL_BAND_H[1] * H) & (lat >= wall["lat"][0] + WALL_LAT_MARGIN) & \
         (lat <= wall["lat"][1] - WALL_LAT_MARGIN) & (freg.dt[i, j] >= EDGE_MIN)
    seed_d = np.zeros(len(xy)) if seed_xy is None else np.linalg.norm(xy - np.asarray(seed_xy)[:2], axis=1)
    order = np.lexsort((seed_d, np.round(np.abs(D - reach_m), CFG["rank_decimals"])))
    order = [k for k in order if ok[k]]
    pick = _spread(order, xy)
    cands = [{"xy": xy[k].tolist(), "h": float(freg.height[i[k], j[k]]), "near": float(D[k]), "far": 0.0,
              "dt": float(freg.dt[i[k], j[k]]), "anchor": "pelvis_xy"} for k in pick]
    return cands, {"rule": "wall", "band_m": [round(WALL_BAND_H[0] * H, 3), round(WALL_BAND_H[1] * H, 3)],
                   "_all": (xy[order], freg.height[i[order], j[order]]),
                   "n_ok": int(len(order)), "flags": [], "reach_m": round(reach_m, 3)}
