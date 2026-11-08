"""Постановка v2.5 (08.11a, шаг 8): посадка по ShotSpec. Читает только shot_spec (+ профили / height_proxy).

  п.2  опора: весонесущие части → target; stand/crouch/kneel — опорные части по телу после ориентации
       (части в пределах 0.04·H от нижней точки), сторона ≠ таблице → table_side_mismatch.
  п.3  ориентация собирается: yaw = ракурс рефа (сторона к камере обязательна; sit_edge/recline — в facing ± 60°),
       наклон корпуса с рефа зажат в диапазон класса, крен ≤ ±15°; recline — таз/бёдра на верх, части
       выше таза > 0.05·H — на спинку, наклон 25–70° подбирается до касания спинки. orient_fail нет → orient_clamped.
  п.4  sit_edge: сторона минимального прямоугольника маски с доступным полом, нормаль ближе к yaw рефа.
  п.5  след (kneel, lie, sit_tucked, recline): вершины ниже h0 + 0.15·H — внутри маски верха с отступом 3 см;
       не влезает → поворот yaw (lie, шаг 15° до ±90°) → следующая ячейка (≤ 5) → contact_fail.
  п.6  стена из secondary_contacts → зона на полу 0.15–0.45·H от плоскости, yaw руками к стене; нет стены → partial.
  п.7  проникновение = p95 проникших вершин; мягкое ≤ 6 см, жёсткое ≤ 3 см.
  п.8  кадр: тело в кадре плиты с полем 2 %; иначе сдвиг к центру кадра (≤ 5), иначе out_of_frame.
  п.10 статус ok | partial | wrong_surface | contact_fail | out_of_frame | no_target + флаги.
Телесные пороги — только через body_dims (H, L_thigh, L_torso); пороги локации — метры.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import Delaunay, cKDTree

import orient as ori
import pose_fit as pf
import profiles as prof
import shot_spec as ssm
import top_region as tr
import zone as zn
from body import height_est, regions
from placement_config import section
from support_geometry import hull_metrics

CFG = section("place")

SUP_GAP_CM = CFG["sup_gap_cm"]
PEN_SOFT_M = CFG["pen_soft_m"]
PEN_HARD_M = CFG["pen_hard_m"]
PEN_MIN_M = CFG["pen_min_m"]
ON_SURF_M = CFG["on_surf_m"]
SHIFT_STEP = CFG["shift_step"]
SHIFT_MAX_H = CFG["shift_max_h"]
ATTEMPTS = CFG["attempts"]
FRAME_TRIES = CFG["frame_tries"]
FRAME_PAD = CFG["frame_pad"]
BACK_FRAC = CFG["back_frac"]
BODY_LOW_H = CFG["body_low_h"]
FOOT_LOW_H = CFG["foot_low_h"]
FOOT_MARGIN = CFG["foot_margin"]
FOOT_CLASSES = ("kneel", "lie", "sit_tucked", "recline")
BACK_ZONE_M = CFG["back_zone_m"]
RECLINE_UP_H = CFG["recline_up_h"]
RECLINE_TILTS = np.array(CFG["recline_tilts"])
TOUCH_M = CFG["touch_m"]
LIE_YAW = CFG["lie_yaw"]
CONFLICT_DEG = CFG["conflict_deg"]
BODY_SUPPORT_REGIONS = ("foot.L", "foot.R", "knee.L", "knee.R", "shin.L", "shin.R")
BACK_REGIONS = ("torso", "arm.L", "arm.R", "forearm.L", "forearm.R")   # п.3г: спина, лопатки, локти
STATUS_RANK = {"ok": 0, "partial": 1, "contact_fail": 2, "out_of_frame": 3, "wrong_surface": 4, "no_target": 5}
B25_BODY = pf.MHR_TO_B25
SIDE_NAME = {1: "front", -1: "back", 0: "side"}


# ------------------------------------------------------------------------------------------------ поверхности
def _surf(P, N=None):
    if N is None:
        N = np.tile([0, 0, 1.0], (len(P), 1))
    return cKDTree(P), P, N


def build_surfaces(plate, treg, freg, wall=None):
    """target — только маска верха; спинка — отдельно (back); пол; стена (wall) — точки плоскости."""
    S = {k: plate.trees[k] for k in ("h", "v", "any")}
    S["target"] = _surf(treg.points3d())
    if treg.backrest is not None:
        S["back"] = _surf(treg.backrest, treg.backrest_n)
    if freg is not None and freg.mask.any():
        S["floor"] = _surf(freg.points3d())
    if wall is not None:
        S["wall"] = _surf(wall["pts"], np.tile(wall["n"], (len(wall["pts"]), 1)))
    return S


# ------------------------------------------------------------------------------------------------ тело
def prepare_body(ref, spec, height):
    V0, kp0 = ref["V"], ref["kp"]
    reg = regions(V0, kp0)
    torso_idx, torso_reason = np.empty(0, dtype=int), None
    try:
        if "body_model_path" in ref:
            from body import cached_torso_back
            torso_idx = cached_torso_back(ref["body_model_path"], ref["body_cache_dir"])
            with np.load(ref["body_model_path"], allow_pickle=False) as model:
                if model["V"].shape != V0.shape or not np.array_equal(model["F"], ref["F"]):
                    raise ValueError("torso_back model topology differs from reference")
        else:
            from body import torso_back_indices
            N, _ = pf.vertex_normals(V0, ref["F"])
            torso_idx = torso_back_indices(V0, N, kp0, **{k: ref[k] for k in
                ("part_labels", "skin_weights", "joint_names") if k in ref})
    except ValueError as exc:
        torso_reason = str(exc)
    planes = pf.ref_planes(ref["S"], body=cKDTree(V0)) if len(ref["S"]) else []
    names = list(dict.fromkeys(spec.support_regions()))
    sup_c = pf.ref_contacts(ref, reg, [f"{n}:h:target" for n in names])
    up, up_src = pf.ref_up(ref, planes, sup_c)
    ref_c = pf.ref_contacts(ref, reg, None, up, planes) if planes else {}
    f = np.array([0, 0, 1.0]) - up[2] * up
    f /= np.linalg.norm(f)
    Rref = np.stack([np.cross(f, up), f, up])
    s = height / height_est(kp0)
    V = (V0 @ Rref.T) * s
    kp = (kp0 @ Rref.T) * s
    pel = (kp[9] + kp[10]) / 2
    side_ref = ori.side_to_camera(kp, np.zeros(3))               # камера рефа в начале координат
    return {"torso_back_indices": torso_idx, "torso_back_unavailable": torso_reason,
            "V0": V0, "kp0": kp0, "reg": reg, "up_src": up_src, "Rref": Rref, "s": s, "V": V, "kp": kp,
            "ref_c": ref_c, "dims": prof.body_dims_from_kp(kp), "n_planes": len(planes), "side_ref": side_ref,
            "cam_axis": Rref @ np.array([0, 0, 1.0]),
            "elev_ref": math.degrees(math.atan2(-pel[2], math.hypot(pel[0], pel[1])))}


def orient_body(B, spec):
    """п.3б/в: знак наклона (кандидаты ×180°), наклон в диапазон класса, крен ≤ 15°. Мир рефа, Z вверх."""
    name, R = ori.tilt_sign_candidate(B["kp"], B["cam_axis"], spec.pose_class)
    c = (B["kp"][9] + B["kp"][10]) / 2
    V, kp = ori.rot_about(B["V"], c, R), ori.rot_about(B["kp"], c, R)
    V, kp, info = ori.assemble_tilt_roll(V, kp, spec.pose_class)
    info["sign_candidate"] = name
    B["V"], B["kp"], B["orient"] = V, kp, info
    pel = (kp[9] + kp[10]) / 2
    fwd = ori.forward(kp)
    to_cam = -pel.copy()
    to_cam[2] = 0
    to_cam /= np.linalg.norm(to_cam)
    B["rel_deg"] = math.degrees(pf.signed_angle(to_cam, fwd)) if fwd is not None else 0.0
    return B


def _lowest(V, idx, q=CFG["support_percentile"]):
    h = V[idx, 2]
    return idx[h <= np.percentile(h, q)]


def build_groups(B, spec, treg, H, has_wall):
    """(имя, индексы, поверхность, роль); роли: support | back | wall | secondary. -> (groups, flags, info)."""
    V, kp, reg = B["V"], B["kp"], B["reg"]
    groups, flags, info = [], [], {}
    zmin = float(V[:, 2].min())
    if spec.body_support():
        chosen = []
        if spec.pose_class == "kneel":
            # Колено определяет весонесущую ногу; её ступня не становится опорой.
            knee_sides = []
            for side in ("L", "R"):
                ki = np.where(reg == f"knee.{side}")[0]
                if len(ki) and V[ki, 2].min() <= zmin + CFG["kneel_low_h"] * H:
                    knee_sides.append(side)
                    for part in ("knee", "shin"):
                        name = f"{part}.{side}"
                        idx = np.where(reg == name)[0]
                        if len(idx):
                            groups.append((name, _lowest(V, idx), "target", "support"))
                            chosen.append(name)
            if len(knee_sides) == 1:
                other = "R" if knee_sides[0] == "L" else "L"
                name = f"foot.{other}"
                idx = np.where(reg == name)[0]
                low = idx[V[idx, 2] <= zmin + BODY_LOW_H * H]
                if len(low) >= CFG["support_min_vertices"]:
                    groups.append((name, low, "target", "support"))
                    chosen.append(name)
        else:
            for r in BODY_SUPPORT_REGIONS:
                idx = np.where(reg == r)[0]
                low = idx[V[idx, 2] <= zmin + BODY_LOW_H * H]
                if len(low) >= CFG["support_min_vertices"]:
                    groups.append((r, low, "target", "support"))
                    chosen.append(r)
        sides_t = {p.split(".")[1] for p in spec.support_parts if "." in p and p.split(".")[0] in ("foot", "knee", "toes")}
        sides_b = {r.split(".")[1] for r in chosen if r.split(".")[0] in ("foot", "knee")}
        if sides_t and sides_b and sides_t != sides_b:
            flags.append("table_side_mismatch")
        info = {"body_support": chosen, "table_support": spec.support_parts}
    else:
        names = list(dict.fromkeys(spec.support_regions()))
        if spec.pose_class == "recline":
            names += [n for n in ("thigh.L", "thigh.R") if n not in names]
        for n in names:
            idx = np.where(reg == n)[0]
            if len(idx):
                groups.append((n, _lowest(V, idx), "target", "support"))
    have = {g[0] for g in groups}
    pel_z = float(((kp[9] + kp[10]) / 2)[2])
    back_ok = spec.pose_class == "recline" and treg is not None and treg.backrest is not None
    for c in spec.secondary_contacts:                           # таблица
        r = ssm.PART_REGION.get(c["part"], c["part"])
        idx = np.where(reg == r)[0]
        if not len(idx) or r in have:
            continue
        if c["surface"] == "wall":
            if has_wall:
                groups.append((r, idx, "wall", "wall"))
                have.add(r)
            continue
        if back_ok and r in BACK_REGIONS and V[idx, 2].mean() > pel_z + RECLINE_UP_H * H:
            groups.append((r, idx, "back", "back"))
        else:
            groups.append((r, _lowest(V, idx), "target" if c["surface"] == spec.target_object else
                           ("floor" if c["surface"] == "floor" else "any"), "secondary"))
        have.add(r)
    for n, c in B["ref_c"].items():                             # контакты рефа (сцена MoGe)
        if n in have:
            continue
        if back_ok and n in BACK_REGIONS and V[c["idx"], 2].mean() > pel_z + RECLINE_UP_H * H:
            groups.append((n, c["idx"], "back", "back"))
        else:
            groups.append((n, c["idx"], c["kind"] if c["kind"] in ("h", "v") else "any", "secondary"))
        have.add(n)
    if spec.pose_class == "recline":
        ti = B.get("torso_back_indices", np.where(reg == "torso_back")[0])
        info["torso_back_vertices"] = int(len(ti))
        if back_ok and len(ti):
            groups.append(("torso_back", ti, "back", "back"))
        elif not len(ti):
            info["back_contact_unavailable"] = B.get("torso_back_unavailable") or "missing torso_back vertex zone; no approved mapping"
    return groups, flags, info


def similarity(V0, X):
    A = np.hstack([V0, np.ones((len(V0), 1))])
    Bm = np.linalg.lstsq(A, X, rcond=None)[0]
    return Bm[:3].T, Bm[3]


# ------------------------------------------------------------------------------------------------ yaw
def _ang(v):
    return math.degrees(math.atan2(v[1], v[0]))


def yaw_target(B, plate, T, window=None, desired=None):
    """п.3а: yaw «вперёд» тела в мире плиты. want = ракурс рефа (или desired — стена); сторона к камере (3D, нормаль
    груди) = как в рефе; window — центр окна ±60° (якорь sit_edge/recline или ракурс рефа для стены). -> (yaw, info)."""
    tc = plate.C - np.asarray(T, float)
    tc[2] = 0
    tc_deg = _ang(tc)
    ref_deg = ori.wrap(tc_deg + B["rel_deg"])
    want = ref_deg if desired is None else desired
    kp = B["kp"]
    pel_up = float(((kp[9] + kp[10]) / 2)[2] - B["V"][:, 2].min())
    to_cam = plate.C - (np.asarray(T, float) + [0, 0, pel_up])
    yaw, side_conflict, in_w = ori.solve_yaw(want, B["side_ref"], window,
                                             side_fn=lambda a: ori.side_for_yaws(kp, a, to_cam))
    return yaw, {"ref_yaw_deg": round(ref_deg, 1), "want_deg": round(want, 1), "yaw_deg": round(yaw, 1),
                 "window_deg": None if window is None else round(window, 1), "side_conflict": side_conflict,
                 "in_window": in_w, "tc_deg": round(tc_deg, 1)}


# ------------------------------------------------------------------------------------------------ жёсткая посадка
def rigid_fit(B, groups, plate, T, anchor, S, yaw_deg, yaw_free_deg=CFG["rigid_yaw_free_deg"], rot_free_deg=CFG["rigid_rot_free_deg"]):
    """yaw задан (сборка п.3); оптимизатор — сдвиг + yaw в ±yaw_free + наклон/крен в ±rot_free (границы)."""
    V, kp, reg = B["V"], B["kp"], B["reg"]
    fwd = ori.forward(kp)
    want = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
    yaw0 = pf.signed_angle(fwd, want) if fwd is not None else 0.0
    R0 = pf.rot_z(yaw0)
    V, kp = V @ R0.T, kp @ R0.T
    sup = [g for g in groups if g[3] == "support"]
    sup_idx = np.concatenate([g[1] for g in sup]) if sup else np.argsort(V[:, 2])[: max(CFG["fallback_support_min"], len(V) // CFG["fallback_support_divisor"])]
    if anchor == "pelvis":
        pi_ = np.where(reg == "pelvis")[0]
        pivot = V[pi_[V[pi_, 2] <= np.percentile(V[pi_, 2], CFG["support_percentile"])]].mean(0)
    elif anchor == "pelvis_xy":                                  # стена: xy таза, z опоры
        pel = (kp[9] + kp[10]) / 2
        pivot = np.array([pel[0], pel[1], V[sup_idx, 2].mean()])
    else:
        pivot = V[sup_idx].mean(0)
    V, kp = V - pivot + T, kp - pivot + T
    pivot = np.asarray(T, float).copy()
    rng = np.random.default_rng(0)
    pen_idx = rng.choice(len(V), min(CFG["rigid_pen_sample_points"], len(V)), replace=False)
    gs = []
    for name, idx, surf, role in groups:
        if surf not in S:
            continue
        gs.append((name, idx if len(idx) <= CFG["rigid_group_sample_points"] else rng.choice(idx, CFG["rigid_group_sample_points"], replace=False), surf, role))

    def transform(p, X):
        return (X - pivot) @ pf.rot_xyz(p[0], p[4], p[5]).T + pivot + p[1:4]

    def residuals(p):
        X = transform(p, V)
        res = []
        for _, idx, surf, role in gs:
            tree, P, N = S[surf]
            d, j = tree.query(X[idx])
            sd = np.einsum("ij,ij->i", X[idx] - P[j], N[j])
            k_n = 1.0 / math.sqrt(len(idx))
            if role == "support":
                res.append(CFG["rigid_support_weight"] * k_n * sd / CFG["rigid_support_sigma_m"])
            elif role in ("back", "wall"):          # мягко: не заваливать корпус ради спинки/стены (дожимает refine)
                res.append(k_n * np.where(d < CFG["rigid_back_near_m"], sd / CFG["rigid_back_sigma_m"], 0.0))
            else:
                res.append(k_n * np.where(d < CFG["rigid_secondary_near_m"], sd / CFG["rigid_secondary_sigma_m"], 0.0))
        pen = np.maximum(pf.penetration(plate, X[pen_idx]) - CFG["rigid_pen_slack_m"], 0.0)
        res.append(CFG["rigid_pen_weight"] / math.sqrt(len(pen_idx)) * pen / CFG["rigid_pen_sigma_m"])
        res.append(np.array([p[0] / CFG["rigid_yaw_prior_rad"], p[4] / CFG["rigid_tilt_prior_rad"], p[5] / CFG["rigid_roll_prior_rad"], np.hypot(p[1], p[2]) / CFG["rigid_shift_prior_m"]]))
        return np.concatenate(res)

    yb, rb = math.radians(max(yaw_free_deg, CFG["rigid_yaw_min_deg"])), math.radians(rot_free_deg)
    lo = [-yb, *([-CFG["rigid_shift_bound_m"]] * 3), -rb, -rb]
    hi = [yb, *([CFG["rigid_shift_bound_m"]] * 3), rb, rb]
    sol = least_squares(residuals, np.zeros(6), bounds=(lo, hi), loss="soft_l1", f_scale=CFG["rigid_f_scale"], diff_step=CFG["rigid_diff_step"],
                        max_nfev=CFG["rigid_max_nfev"])
    X, KP = transform(sol.x, V), transform(sol.x, kp)
    return {"X": X, "KP": KP, "pivot": pivot, "params": [round(float(x), 4) for x in sol.x],
            "cost": round(float(sol.cost), 3), "nfev": int(sol.nfev), "yaw_set_deg": round(yaw_deg, 1)}


def recline_sweep(r, groups, S, plate, H, tilt_ref, treg=None, soft=False):
    """п.3г: наклон корпуса 25–70° (поворот всего тела вокруг опорной точки таза), пока верх спины не коснётся
    спинки; из касающихся — ближе к наклону рефа, иначе минимальный зазор. -> (X, KP, info)."""
    back = [g for g in groups if g[0] == "torso_back"]
    if not back or "back" not in S:
        return r["X"], r["KP"], {"touch": False, "reason": "missing torso_back or backrest", "tilts_tried": 0}
    idx = back[0][1]
    X, KP = r["X"], r["KP"]
    pi_ = (KP[9] + KP[10]) / 2
    piv = pi_  # вращение строго вокруг таза после высоты
    f = ori.forward(KP)
    if f is None:
        return X, KP, None
    lat = np.cross(ori.UP, f)
    t0 = ori.tilt_deg(KP)
    tree, P, N = S["back"]
    rows = []
    lo, hi = ori.TILT_RANGE["recline"]
    for t in RECLINE_TILTS:
        if not lo <= t <= hi:
            continue
        best = None
        for sgn in (1, -1):
            R = ori.rot_axis(lat, sgn * math.radians(t - t0))
            err = abs(ori.tilt_deg(ori.rot_about(KP, piv, R)) - t)
            if best is None or err < best[0]:
                best = (err, R)
        if best[0] > CFG["recline_match_deg"]:
            continue
        best = best[1]
        Xb, Kb = ori.rot_about(X, piv, best), ori.rot_about(KP, piv, best)
        Xb, Kb = ori.clamp_roll(Xb, Kb)
        d, _ = tree.query(Xb[idx])
        gap = float(d.min())
        pen = penetration_stats(plate, Xb, treg, soft)
        hard_ok = all(s["ok"] for n, s in pen["surfaces"].items() if n.startswith("hard"))
        rows.append((t, gap, Xb, Kb, hard_ok))
    touch = [rw for rw in rows if rw[1] <= CFG["contact_tol"] and rw[4]]
    if not touch:
        return X, KP, {"touch": False, "reason": "no contact within class tilt range and hard limit",
                       "tilts_tried": len(rows)}
    pick = min(touch, key=lambda rw: abs(rw[0] - tilt_ref))
    return pick[2], pick[3], {"tilt_deg": float(pick[0]), "back_contact_gap_m": pick[1],
                              "touch": True, "tilts_tried": len(rows)}


# ------------------------------------------------------------------------------------------------ оценка
def _sd(S, surf, Xi):
    tree, P, N = S[surf]
    d, j = tree.query(Xi)
    return np.einsum("ij,ij->i", Xi - P[j], N[j]), d


def _on(reg, Xi, plate, sink):
    """вершина на верхе предмета: ячейка маски и −sink−1 см ≤ z − h ≤ ON_SURF_M; вне кадра — только по h0."""
    if reg is None or not reg.mask.any():
        return np.zeros(len(Xi), bool)
    u, v, z = plate.project(Xi)
    vis = (u >= 0) & (u < plate.Wp) & (v >= 0) & (v < plate.Hp) & (z > 0)
    i, j = reg.cell_of(Xi)
    ok = reg.inside(i, j)
    out = np.zeros(len(Xi), bool)
    dz = Xi[ok, 2] - reg.height[i[ok], j[ok]]
    out[ok] = reg.mask[i[ok], j[ok]] & (dz <= ON_SURF_M) & (dz >= -sink - CFG["surface_sink_slack_m"])
    dz0 = Xi[~vis, 2] - reg.h0
    out[~vis] = (dz0 <= ON_SURF_M) & (dz0 >= -sink - CFG["surface_sink_slack_m"])
    return out


def _near_back(treg, Xi, r=ON_SURF_M):
    if treg is None or treg.backrest is None:
        return np.zeros(len(Xi), bool)
    d, _ = cKDTree(treg.backrest).query(Xi)
    return d <= r


def part_label(Xi, spec, treg, freg, plate, soft, recline=False):
    sink = PEN_SOFT_M if soft else PEN_HARD_M
    on = _on(treg, Xi, plate, sink)
    if recline:
        on = on | _near_back(treg, Xi)
    if on.mean() >= CFG["surface_majority_frac"]:
        return spec.target_object
    if freg is not None and _on(freg, Xi, plate, PEN_HARD_M).mean() >= CFG["surface_majority_frac"]:
        return "floor"
    return "other"


def penetration_values(plate, X, treg, soft):
    """Материал принадлежит проникнутой поверхности, не проекции вершины тела.

    Точки спинки, совпавшие с RANSAC-стеной, всегда hard, даже над bed.
    """
    pen = pf.penetration(plate, X)
    _, j = plate.trees["any"][0].query(X)
    P = plate.trees["any"][1][j]
    material = np.full(len(X), "hard", dtype="U32")
    if soft and treg is not None and treg.mask.any():
        material[treg.in_mask(P)] = "soft:target"
        if treg.backrest is not None:
            material[_near_back(treg, P, BACK_ZONE_M)] = "soft:pillows"
    candidates = getattr(treg, "diagnostic_surfaces", ())
    nearest = np.full(len(P), np.inf)
    for surface in candidates:
        reg = surface["region"]
        ii, jj = reg.cell_of(P)
        inside = reg.inside(ii, jj)
        sel = np.zeros(len(P), bool)
        sel[inside] = reg.mask[ii[inside], jj[inside]]
        distance = np.full(len(P), np.inf)
        distance[sel] = np.abs(P[sel, 2] - reg.height[ii[sel], jj[sel]])
        pick = sel & (distance < nearest) & (material != "soft:pillows")
        material[pick] = ("soft:" if surface["soft"] else "hard:") + surface["id"]
        nearest[pick] = distance[pick]
    for k, wall in enumerate(getattr(treg, "wall_planes", ())):
        near = np.abs(P @ wall["n"] + wall["d"]) <= CFG["wall_backrest_m"]
        material[near] = "hard:" + wall.get("id", f"wall{k}")
    return pen, material


def penetration_stats(plate, X, treg, soft):
    pen, material = penetration_values(plate, X, treg, soft)
    hit = pen > PEN_MIN_M
    out = {"max_cm": round(float(pen.max()) * 100, 1), "n_pen": int(hit.sum())}
    ok, excess = True, 0.0
    for name, lim in (("soft", PEN_SOFT_M), ("hard", PEN_HARD_M)):
        sel = hit & np.char.startswith(material, name)
        p95 = float(np.percentile(pen[sel], CFG["penetration_percentile"])) if sel.any() else 0.0
        out[f"{name}_p95_cm"] = round(p95 * 100, 1)
        out[f"{name}_max_cm"] = round(float(pen[sel].max()) * 100, 1) if sel.any() else 0.0
        out[f"{name}_n"] = int(sel.sum())
        out[f"{name}_p95_m"] = p95  # валидатор не сравнивает округлённые см
        # Сводка по материалу — диагностика. Решение только по каждой поверхности.
    out["p95_cm"] = round(float(np.percentile(pen[hit], CFG["penetration_percentile"])) * 100, 1) if hit.any() else 0.0
    surfaces = {}
    for surface in np.unique(material):
        values = pen[hit & (material == surface)]
        p95 = float(np.percentile(values, CFG["penetration_percentile"])) if len(values) else 0.0
        lim = PEN_SOFT_M if surface.startswith("soft") else PEN_HARD_M
        surfaces[str(surface)] = {"p95_m": p95, "max_m": float(values.max()) if len(values) else 0.0,
                                  "threshold_m": lim, "ok": bool(p95 <= lim), "n": len(values)}
        ok &= p95 <= lim
        excess = max(excess, p95 - lim)
    out["surfaces"] = surfaces
    out["excess_m"] = max(0.0, excess)
    out["ok"] = bool(ok)
    return out


def back_zone(treg, X):
    """зона спинки сверху: в плане ближе BACK_ZONE_M к точкам спинки или глубже дальнего края маски."""
    if treg is None or treg.backrest is None:
        return np.zeros(len(X), bool)
    d, _ = cKDTree(treg.backrest[:, :2]).query(X[:, :2])
    far = treg.edge_dist(X[:, :2], "far")
    return (d <= BACK_ZONE_M) | (np.nan_to_num(far, nan=1.0) < 0)


def body_lowest_gap(X, S):
    return float(_sd(S, "target", X[[int(np.argmin(X[:, 2]))]])[0][0])


def support_contact(X, groups, S, spec, soft):
    """D5: stand/crouch need body-lowest + one foot; other gaps are diagnostic flags."""
    gaps = {n: float(np.median(_sd(S, s, X[idx])[0])) for n, idx, s, role in groups
            if role == "support" and s in S and len(idx)}
    if spec.pose_class not in ("stand", "crouch"):
        sink = PEN_SOFT_M if soft else PEN_HARD_M
        ok = bool(gaps) and all(-sink <= g <= SUP_GAP_CM / 100 for g in gaps.values())
        return {"ok": ok, "part_gaps_m": gaps, "flags": []}
    lowest = body_lowest_gap(X, S)
    foot_gaps = {n: float(np.min(np.abs(_sd(S, s, X[idx])[0]))) for n, idx, s, role in groups
                 if role == "support" and n.startswith("foot.") and s in S and len(idx)}
    feet = [n for n, gap in foot_gaps.items() if gap <= CFG["contact_tol"]]
    flagged = [n for n, g in gaps.items() if g > CFG["contact_tol"] and n not in feet]
    return {"ok": bool(abs(lowest) <= CFG["contact_tol"] and feet), "lowest_gap_m": lowest,
            "foot_contact_parts": feet, "foot_min_abs_gaps_m": foot_gaps, "part_gaps_m": gaps,
            "gap_parts": flagged, "flags": ["support_gap"] if flagged else []}


def footprint(X, treg, spec, H, support_idx=None):
    """D1: support_parts hull >=90% inside top mask. Legacy low-body counts are diagnostic only."""
    points = X[support_idx] if support_idx is not None else np.empty((0, 3))
    hull = hull_metrics(points, treg)
    hull_ok = hull["in_mask_fraction"] is not None and hull["in_mask_fraction"] >= CFG["hull_in_mask_min"]
    if spec.pose_class not in FOOT_CLASSES or spec.target_object == "floor" or treg is None:
        return {"checked": False, "ok": True, "support_hull": hull}
    low = X[:, 2] < treg.h0 + FOOT_LOW_H * H
    if spec.pose_class == "recline":
        low &= ~back_zone(treg, X)
    P = X[low]
    inside = treg.in_mask(P) & (treg.dt_at(P) >= FOOT_MARGIN)
    return {"checked": True, "ok": bool(hull_ok), "support_hull": hull,
            "n_low": int(low.sum()), "n_out": int((~inside).sum())}


def framing(X, plate):
    """п.8: проекция всего тела внутри кадра с полем 2 %."""
    u, v, z = plate.project(X)
    W, Hh = plate.Wp, plate.Hp
    pu, pv = FRAME_PAD * W, FRAME_PAD * Hh
    ok = bool((z > 0).all() and u.min() >= pu and u.max() <= W - pu and v.min() >= pv and v.max() <= Hh - pv)
    return {"ok": ok, "bbox_px": [round(float(u.min())), round(float(v.min())), round(float(u.max())),
                                  round(float(v.max()))], "frame_px": [W, Hh]}


def evaluate(X, KP, B, groups, S, plate, spec, treg, freg, H, soft):
    rep_c, sup_sd = {}, []
    labels, extra = {}, {}
    for name, idx, surf, role in groups:
        if surf not in S:
            continue
        sd, d = _sd(S, surf, X[idx])
        rep_c[name] = {"role": role, "surface": surf, "gap_cm": round(float(np.median(sd)) * 100, 1),
                       "min_cm": round(float(sd.min()) * 100, 1)}
        if role == "support":
            sup_sd.append(sd)
            contact_vertices = idx
            if spec.pose_class in ("stand", "crouch") and name.startswith("foot."):
                contact_vertices = idx[np.abs(sd) <= CFG["contact_tol"]]
            labels[name] = part_label(X[contact_vertices], spec, treg, freg, plate, soft,
                                      recline=spec.pose_class == "recline") if len(contact_vertices) else "other"
        elif role in ("back", "wall"):
            extra.setdefault(role, []).append(sd)
    g = np.concatenate(sup_sd) if sup_sd else np.zeros(1)
    sup_gap = float(np.median(g[g <= np.percentile(g, CFG["gap_percentile"])]))
    sink = PEN_SOFT_M if soft else PEN_HARD_M
    contact_groups = list(groups)
    if spec.pose_class in ("stand", "crouch"):
        present = {g[0] for g in groups if g[3] == "support"}
        for name in spec.support_regions():
            idx = np.flatnonzero(B["reg"] == name)
            if name not in present and len(idx):
                contact_groups.append((name, _lowest(X, idx), "target", "support"))
    contact = support_contact(X, contact_groups, S, spec, soft)
    part_gaps = contact["part_gaps_m"]
    max_gap = max(part_gaps.values(), default=float("inf"))
    sup_ok = contact["ok"]
    gaps, gaps_m = {}, {}
    for role, lst in extra.items():
        gb = np.concatenate(lst)
        gaps_m[role] = float(np.median(gb[gb <= np.percentile(gb, CFG["gap_percentile"])]))
        gaps[role] = round(gaps_m[role] * 100, 1)
    pen = penetration_stats(plate, X, treg, soft)
    vals = [v for n, v in labels.items() if spec.pose_class not in ("stand", "crouch") or
            n in contact.get("foot_contact_parts", [])]
    if vals and all(v == spec.target_object for v in vals):
        sup_obj = spec.target_object
    else:
        rest = [v for v in vals if v != spec.target_object] or ["none"]
        sup_obj = max(set(rest), key=rest.count)
    cells_ok, cells = True, {}
    if spec.target_object != "floor" and spec.pose_class == "sit_edge":
        for name, idx, surf, role in groups:
            if role != "support":
                continue
            c = X[idx].mean(0)[None]
            dt = float(treg.dt_at(c)[0])
            inm = bool(treg.in_mask(c)[0])
            cells[name] = {"in_mask": inm, "dt_m": round(dt, 3)}
            cells_ok &= inm and dt >= zn.EDGE_MIN
    support_idx = np.flatnonzero(np.isin(B["reg"], spec.support_regions()))
    fp = footprint(X, treg, spec, H, support_idx)
    fr = framing(X, plate)
    reg = B["reg"]
    pts = [X[idx, :2] for name, idx, surf, role in groups
           if role == "support" and name in rep_c and abs(rep_c[name]["gap_cm"]) < CFG["balance_contact_max_cm"]]
    com = sum(pf.MASS.get(r, 0.0) * X[reg == r].mean(0) for r in np.unique(reg)) / \
        sum(pf.MASS.get(r, 0.0) for r in np.unique(reg))
    balance = None
    if pts:
        P2 = np.concatenate(pts)
        inside = False
        try:
            inside = bool(Delaunay(P2).find_simplex(com[:2]) >= 0)
        except Exception:
            pass
        balance = inside or bool(np.min(np.linalg.norm(P2 - com[:2], axis=1)) < CFG["balance_margin_m"])
    back_idx = [idx for name, idx, surf, role in groups if name == "torso_back" and surf == "back"]
    back_contact_gap = float(S["back"][0].query(X[np.concatenate(back_idx)])[0].min()) if back_idx and "back" in S else None
    return {"support_contact": contact, "back_contact_gap_m": back_contact_gap,
            "support_gap_cm": round(sup_gap * 100, 1), "support_part_gaps_m": part_gaps,
            "max_support_gap_m": max_gap, "support_ok": bool(sup_ok), "back_gap_cm": gaps.get("back"),
            "wall_gap_cm": gaps.get("wall"), "back_gap_m": gaps_m.get("back"),
            "wall_gap_m": gaps_m.get("wall"), "penetration": pen, "max_penetration_cm": pen["max_cm"],
            "contacts": rep_c, "support_parts_on": labels, "support_object": sup_obj, "cells": cells,
            "cells_ok": bool(cells_ok), "footprint": fp, "frame": fr, "balance_ok": balance,
            "tilt_deg": round(ori.tilt_deg(KP), 1), "roll_deg": round(ori.roll_deg(KP), 1),
            "orient": ori.check(KP, spec.pose_class, H)}


def status_of(ev, spec, limits_ok=True, no_secondary=False):
    if ev["support_object"] != spec.target_object:
        return "wrong_surface"
    if not (ev["support_ok"] and ev["penetration"]["ok"] and ev["footprint"]["ok"] and ev["cells_ok"] and limits_ok):
        return "contact_fail"
    if not ev["frame"]["ok"]:
        return "out_of_frame"
    return "partial" if no_secondary else "ok"


# ------------------------------------------------------------------------------------------------ сдвиг в зону (Р2)
def _foot_out(X, treg, spec, H):
    """вершины следа (п.5) вне маски с отступом: (маска low, маска out)."""
    low = X[:, 2] < treg.h0 + FOOT_LOW_H * H
    if spec.pose_class == "recline":
        low &= ~back_zone(treg, X)
    P = X[low]
    out = ~(treg.in_mask(P) & (treg.dt_at(P) >= FOOT_MARGIN))
    return P, out


def shift_into(X, KP, groups, S, treg, H, spec=None, support_idx=None):
    """Р2: опорные части в ячейки dt ≥ 5 см; 08.11a: для классов следа (п.5) — и след внутри маски с отступом.
    Сдвиг шагом 1 см до 0.5·H: к максимуму dt, а если мешает только след — от вершин, вылезших за край."""
    sup = [(n, idx) for n, idx, surf, role in groups if role == "support"]
    if not sup or not treg.mask.any():
        return X, KP, 0.0, False
    use_fp = spec is not None and spec.pose_class in FOOT_CLASSES

    def sup_ok(Xc):
        cs = np.array([Xc[idx].mean(0) for _, idx in sup])
        return bool(np.all(treg.in_mask(cs) & (treg.dt_at(cs) >= zn.EDGE_MIN)))

    def ok(Xc):
        return sup_ok(Xc) and (not use_fp or footprint(Xc, treg, spec, H, support_idx)["ok"])
    if ok(X):
        return X, KP, 0.0, True
    c0 = np.mean([X[idx].mean(0)[:2] for _, idx in sup], axis=0)
    if sup_ok(X) and use_fp:                                   # только след: от вылезших вершин внутрь
        P, out = _foot_out(X, treg, spec, H)
        tgt = c0 + (P[~out, :2].mean(0) - P[out, :2].mean(0)) if (~out).any() else c0
    else:
        i, j = np.unravel_index(int(np.argmax(treg.dt)), treg.dt.shape)
        tgt = treg.center(i, j)
    d = tgt - c0
    n = np.linalg.norm(d)
    if n < 1e-6:
        return X, KP, 0.0, False
    d /= n
    allidx = np.concatenate([idx for _, idx in sup])
    sd0, _ = _sd(S, "target", X[allidx])
    for k in range(1, int(SHIFT_MAX_H * H / SHIFT_STEP) + 1):
        off = np.array([d[0], d[1], 0.0]) * k * SHIFT_STEP
        Xc = X + off
        if ok(Xc):
            sd1, _ = _sd(S, "target", Xc[allidx])
            dz = float(np.median(sd0) - np.median(sd1))
            off2 = off + [0, 0, dz]
            return X + off2, KP + off2, round(k * SHIFT_STEP, 2), True
    return X, KP, None, False


# ------------------------------------------------------------------------------------------------ зона
def resolve_regions(plate, spec, seed, H, profile):
    """-> (treg, freg, zone_src, obj_info) | (None, …, 'no_target')."""
    freg, info = None, {}
    if profile is not None:
        fo, _ = profile.find_object("floor")
        freg = profile.region(fo) if fo is not None else None
    if freg is None:
        freg = tr.top_region(plate, "floor", None, H)
    if spec.target_object == "floor":
        return freg, freg, "floor", {"object_id": "floor", "type": "floor", "soft": False}
    if profile is not None:
        obj, via = profile.find_object(spec.target_object)
        if obj is None or obj.get("type") == "wall":
            return None, freg, "no_target", {"reason": f"{spec.target_object} нет в profile.objects и substitutes"}
        info = {"object_id": obj["id"], "via": via, "type": obj.get("type", tr.object_type(obj["name"])),
                "soft": bool(obj.get("soft", tr.is_soft(obj["name"]))), "name": obj["name"],
                "backrest_name": obj.get("backrest_name")}
        a = profile.anchor(obj["id"], spec.pose_class)
        if a is not None:
            info["anchor_id"] = a["id"]
            return profile.region(obj), freg, "profile", info
    treg = tr.top_region(plate, spec.target_object, seed, H)
    if not treg.mask.any():
        info["reason"] = "height_proxy: пустая маска"
    info.setdefault("type", tr.object_type(spec.target_object))
    info.setdefault("soft", tr.is_soft(spec.target_object))
    info.setdefault("name", spec.target_object)
    info.setdefault("backrest_name", tr.backrest_name(spec.target_object, treg.backrest is not None))
    return treg, freg, "proxy", info


def hand_vec(B, groups):
    """горизонтальный вектор таз → центр контактных с стеной частей (мир рефа после ориентации)."""
    idx = [g[1] for g in groups if g[3] == "wall"]
    kp = B["kp"]
    pel = (kp[9] + kp[10]) / 2
    if not idx:
        return None
    h = B["V"][np.concatenate(idx)].mean(0) - pel
    h[2] = 0
    return h


def frame_target(plate, T, bbox, zone_xy, zone_h):
    """п.8: ячейка зоны, куда сдвинуть опорную точку, чтобы рамка тела ушла к центру кадра: проекция ячейки
    ближе всего к proj(T) + (центр кадра − центр рамки). -> 3D-точка | None."""
    if zone_xy is None or not len(zone_xy):
        return None
    u0, v0, _ = plate.project(np.asarray(T, float)[None])
    want = np.array([u0[0] + plate.Wp / 2 - (bbox[0] + bbox[2]) / 2, v0[0] + plate.Hp / 2 - (bbox[1] + bbox[3]) / 2])
    P = np.column_stack([zone_xy, zone_h])
    u, v, z = plate.project(P)
    d = np.hypot(u - want[0], v - want[1])
    d[z <= 0] = np.inf
    k = int(np.argmin(d))
    return P[k] if np.isfinite(d[k]) else None


def zone_diagnostics(B, spec, treg, groups):
    counts = {name: int(len(idx)) for name, idx, _, role in groups if role == "support"}
    mapped = {name: int(np.count_nonzero(B["reg"] == name)) for name in dict.fromkeys(spec.support_regions())}
    idx = np.flatnonzero(np.isin(B["reg"], list(mapped)))
    hull = hull_metrics(B["V"][idx])
    # This is pre-placement area only; mask fraction is measured only after world placement.
    cause = []
    if spec.pose_class == "kneel":
        zmin = float(B["V"][:, 2].min())
        H = B.get("dims", {}).get("H", height_est(B["kp"]))
        for side in ("L", "R"):
            ki = np.flatnonzero(B["reg"] == "knee." + side)
            gap = float(B["V"][ki, 2].min() - zmin) if len(ki) else None
            cause.append({"part": "knee." + side, "min_above_body_lowest_m": gap,
                          "threshold_m": CFG["kneel_low_h"] * H,
                          "selected": gap is not None and gap <= CFG["kneel_low_h"] * H})
    return {"target_object": spec.target_object, "instance_hint": spec.instance_hint,
            "zone_mask_area_m2": float(treg.mask.sum()) * tr.CELL ** 2 if treg is not None else None,
            "body_footprint_area_m2": hull["area_m2"], "support_hull": hull,
            "support_vertices_after_mapping": mapped, "support_group_vertices": counts,
            "support_vertex_count": sum(counts.values()), "kneel_selection": cause,
            "zero_support_cause": "kneel_low_h rejected both knees; no knee/shin groups" if
                spec.pose_class == "kneel" and not counts and cause and not any(c["selected"] for c in cause) else None}


def geometry_diagnostics(X, KP, B, spec, treg, groups, S, plate, soft):
    """E: measured world geometry, not target renamed as the actual surface."""
    idx = np.unique(np.concatenate([g[1] for g in groups if g[3] == "support"])) if any(
        g[3] == "support" for g in groups) else np.empty(0, int)
    # Preserve evidence even if the kneel selector rejected every mapped vertex.
    source = "selected_support"
    if not len(idx):
        idx = np.flatnonzero(np.isin(B["reg"], spec.support_regions()))
        source = "mapped_support_before_selection"
    P = X[idx]
    hull_idx = np.flatnonzero(np.isin(B["reg"], spec.support_regions()))
    hull = hull_metrics(X[hull_idx], treg)
    distances = _sd(S, "target", P)[0] if len(P) else np.empty(0)
    actual = []
    for item in getattr(treg, "diagnostic_surfaces", ()):
        reg = item["region"]
        if not len(P) or not reg.mask.any():
            continue
        tree = cKDTree(reg.points3d())
        d, _ = tree.query(P)
        on = reg.in_mask(P) & (d <= ON_SURF_M)
        if on.any():
            actual.append({"id": item["id"], "h0_m": float(reg.h0), "n_contact": int(on.sum()),
                           "min_distance_m": float(d[on].min())})
    actual.sort(key=lambda a: (-a["n_contact"], a["min_distance_m"]))
    pen, materials = penetration_values(plate, X, treg, soft)
    by_surface = {}
    for name in np.unique(materials):
        parts = {}
        for part in np.unique(B["reg"]):
            v = pen[(materials == name) & (B["reg"] == part) & (pen > PEN_MIN_M)]
            if len(v):
                parts[str(part)] = {"n": len(v), "p95_m": float(np.percentile(v, CFG["penetration_percentile"])),
                                    "max_m": float(v.max())}
        by_surface[str(name)] = parts
    f = ori.forward(KP)
    return {"measurement_vertices": source, "target_h0_m": float(treg.h0),
            "actual_support_surfaces": actual, "actual_surface": actual[0] if actual else None,
            "actual_surface_reason": None if actual else "no measured contact on known surface",
            "target_gap_min_m": float(distances.min()) if len(distances) else None,
            "target_gap_p50_m": float(np.median(distances)) if len(distances) else None,
            "support_hull": hull, "hull_parts": spec.support_regions(),
            "penetration_per_surface": penetration_stats(plate, X, treg, soft)["surfaces"],
            "penetration_parts_per_surface": by_surface,
            "angles_deg": {"yaw": _ang(f) if f is not None else None,
                           "pitch": float(ori.tilt_deg(KP)), "roll": float(ori.roll_deg(KP))},
            "top_definition": getattr(treg, "top_diagnostics", {})}


def zone_too_small(diagnostics):
    area, body = diagnostics["zone_mask_area_m2"], diagnostics["body_footprint_area_m2"]
    return area is not None and (area == 0 or (body is not None and area < body))


# ------------------------------------------------------------------------------------------------ поза целиком
def place(ref, plate, spec, seed, height, *, profile=None, body_src="proxy", dims_override=None, refine_ctx=None,
               log=print):
    """seed — 3D-точка из таблицы (поднятая точка сиденья / точка стоп). -> dict результата (X, KP, rep, …)."""
    B = prepare_body(ref, spec, height)
    if dims_override:
        B["dims"] = dict(dims_override)
    H = B["dims"]["H"]
    treg, freg, zsrc, oinfo = resolve_regions(plate, spec, seed, H, profile)
    base = {"shot_spec": spec.to_dict(), "body_dims": B["dims"], "body_source": body_src, "height_m": height}
    if treg is None:
        diagnostics = zone_diagnostics(B, spec, treg, [])
        return {"status": "no_target", "rep": dict(base, status="no_target", zone={"source": zsrc, **oinfo},
                                                   zone_diagnostics=diagnostics)}
    treg = tr.pose_region(treg, spec.pose_class)
    treg.object_id = oinfo.get("object_id", spec.target_object)
    if freg is not None:
        freg.object_id = getattr(freg, "object_id", "floor")
    soft = bool(oinfo.get("soft"))
    treg.diagnostic_surfaces = ([{"id": o["id"], "soft": bool(o.get("soft", False)),
                                 "region": profile.region(o)} for o in profile.data["objects"]
                                if "top_grid_npz" in o] if profile is not None else
                               [{"id": treg.object_id, "soft": soft, "region": treg}])
    treg.wall_planes = profile.walls() if profile is not None else tr.walls(plate)
    if treg.backrest is not None:
        hard_back = np.zeros(len(treg.backrest), bool)
        for w in treg.wall_planes:
            hard_back |= np.abs(treg.backrest @ w["n"] + w["d"]) <= CFG["wall_backrest_m"]
        treg.backrest_material = np.where(hard_back, "wall", "pillows")
    orient_body(B, spec)
    flags = [f for f in treg.flags]
    # стена (п.6)
    wall, wall_info, no_secondary = None, None, False
    if spec.secondary_on("wall"):
        W = profile.walls() if profile is not None else tr.walls(plate)
        k = zn.pick_wall(W, seed[:2]) if W else None
        if k is None:
            no_secondary = True
            flags.append("no_secondary")
        else:
            wall = W[k]
            wall_info = {"wall": k, "area_m2": wall["area_m2"], "n": [round(float(x), 3) for x in wall["n"][:2]],
                         "d": round(wall["d"], 3)}
    groups, gflags, ginfo = build_groups(B, spec, treg, H, wall is not None)
    flags += gflags
    diagnostics = zone_diagnostics(B, spec, treg, groups)
    if zone_too_small(diagnostics):
        return {"status": "no_target", "rep": dict(base, status="no_target",
                    flags=sorted(set(flags + ["zone_too_small"])), support_info=ginfo,
                    zone_diagnostics=diagnostics, zone={"source": zsrc, **oinfo})}
    S = build_surfaces(plate, treg, freg, wall)
    # зона
    window, desired_fn = None, None
    if spec.target_object == "floor":
        if wall is not None:
            hv = hand_vec(B, groups)
            reach = float(np.linalg.norm(hv)) if hv is not None else CFG["wall_reach_h"] * H
            cells, zinfo = zn.wall_cells(freg, wall, B["dims"], reach, seed[:2])
            zinfo.update(wall_info)
            if not cells:
                cells = [{"xy": list(seed[:2]), "h": float(seed[2]), "anchor": "support"}]
                zinfo["flags"] = ["wall_zone_empty"]
        else:
            cells = [{"xy": list(seed[:2]), "h": float(seed[2]), "anchor": "support"}]
            zinfo = {"rule": "floor", "band_m": None, "flags": []}
        zreg = freg
        if "_all" not in zinfo:
            fi_, fj_ = freg.cells()
            okf = freg.dt[fi_, fj_] >= zn.EDGE_MIN
            zinfo["_all"] = (freg.origin + (np.stack([fi_, fj_], 1)[okf] + 0.5) * tr.CELL, freg.height[fi_, fj_][okf])
    else:
        zreg = treg
        if zsrc == "profile" and getattr(treg, "soft_backrest_mask", None) is None:
            m = profile.zone_metrics(profile.anchor(oinfo["object_id"], spec.pose_class))
        else:
            m = zn.metrics(treg)  # class-specific D4 mask must not reuse the untrimmed anchor cells
        if spec.pose_class == "sit_edge":
            sides = zn.edge_sides(treg, freg, H, plate)
            _, yi = yaw_target(B, plate, seed)
            side = zn.pick_side(sides, yi["ref_yaw_deg"])
            if side is None:
                flags.append("no_access_side")
                cells, zinfo = zn.zone_cells(m, spec.pose_class, B["dims"], seed_xy=seed[:2],
                                             has_backrest=treg.backrest is not None)
            else:
                cells, zinfo = zn.edge_cells(m, side, B["dims"], seed[:2])
                window = side["yaw_deg"]
                if side.get("edge_short"):
                    flags.append("edge_short")
            zinfo["sides"] = [{"yaw_deg": round(s["yaw_deg"], 1), "len_m": round(s["length"], 2),
                               "access": s["access_frac"]} for s in sides]
        else:
            cells, zinfo = zn.zone_cells(m, spec.pose_class, B["dims"], seed_xy=seed[:2],
                                         has_backrest=treg.backrest is not None)
            if spec.pose_class == "recline":
                if zsrc == "profile":
                    window = profile.anchor(oinfo["object_id"], spec.pose_class)["facing_yaw_deg"]
                else:
                    window = _ang(-treg.cam_axes()[0])
        flags += zinfo["flags"]
        if not cells:
            return {"status": "contact_fail",
                    "rep": dict(base, status="contact_fail", flags=sorted(set(flags + ["zone_empty"])),
                                zone_diagnostics=diagnostics, support_info=ginfo,
                                zone={"source": zsrc, **oinfo, **{k: v for k, v in zinfo.items() if k != "_all"},
                                      "region": treg.summary()})}
    if spec.pose_class == "recline" and treg.backrest is None:
        flags.append("no_backrest")
    zall = zinfo.pop("_all", (None, None))

    def yaw_for(T, extra=0.0):
        if wall is not None:
            hv = hand_vec(B, groups)
            _, yi0 = yaw_target(B, plate, T)
            if hv is not None:
                fwd = ori.forward(B["kp"])
                dlt = math.degrees(pf.signed_angle(fwd, hv)) if fwd is not None else 0.0
                des = ori.wrap(_ang(-wall["n"]) - dlt)
                return yaw_target(B, plate, T, window=yi0["ref_yaw_deg"], desired=des)
            return yaw_target(B, plate, T)
        y, yi = yaw_target(B, plate, T, window=window)
        if extra:
            y2, yi2 = yaw_target(B, plate, T, window=window, desired=ori.wrap(yi["want_deg"] + extra))
            return y2, dict(yi2, lie_yaw_add=extra)
        return y, yi

    def fit_at(T, anchor):
        """одна посадка в точку T: поворот следа (lie) → recline-спинка → сдвиг в зону → оценка."""
        tries = LIE_YAW if spec.pose_class == "lie" and spec.target_object != "floor" else [0]
        best = None
        for add in tries:
            y, yi = yaw_for(T, add)
            r = rigid_fit(B, groups, plate, T, anchor, S, y)
            X, KP = r["X"], r["KP"]
            # orient_conflict (п.3а): ракурс после посадки уехал > 20° — сравнить с посадкой без свободы yaw
            fwd = ori.forward(KP)
            got = _ang(fwd) if fwd is not None else y
            conflict = None
            if abs(ori.wrap(got - yi["ref_yaw_deg"])) > CONFLICT_DEG and not yi["side_conflict"]:
                r2 = rigid_fit(B, groups, plate, T, anchor, S, y, yaw_free_deg=CFG["locked_yaw_free_deg"])
                e1 = ori.reproj_plate(KP, plate, ref, B25_BODY)
                e2 = ori.reproj_plate(r2["KP"], plate, ref, B25_BODY)
                conflict = {"plate_deg": round(got, 1), "reproj_plate": round(e1, 3), "reproj_ref": round(e2, 3),
                            "picked": "plate" if e1 <= e2 else "ref"}
                if e2 < e1:
                    r, X, KP = r2, r2["X"], r2["KP"]
            if B["side_ref"] and ori.side_to_camera(KP, plate.C) == -B["side_ref"]:   # п.3а: сторона не меняется
                r = rigid_fit(B, groups, plate, T, anchor, S, y, yaw_free_deg=CFG["locked_yaw_free_deg"])
                X, KP = r["X"], r["KP"]
            import assembly
            X, KP = assembly.clamp_root(X, KP, spec.pose_class, plate, B["side_ref"], window)
            shift = 0.0
            if spec.target_object != "floor":
                X, KP, shift, _ = shift_into(X, KP, groups, S, treg, H, spec,
                                                np.flatnonzero(np.isin(B["reg"], spec.support_regions())))
            # После финального XY: (а) min contact → (б) bounded p95 lift.
            X, KP, height_info = assembly.height_fit(X, KP, groups, S, plate, spec, treg, soft, H)
            sw = None
            if spec.pose_class == "recline":
                X, KP, sw = recline_sweep(dict(r, X=X, KP=KP), groups, S, plate, H,
                                            B["orient"]["tilt_deg"], treg, soft)
            ev = evaluate(X, KP, B, groups, S, plate, spec, treg, freg, H, soft)
            rec = {"fit": r, "X": X, "KP": KP, "ev": ev, "yaw": yi, "shift_m": shift, "conflict": conflict,
                   "recline": sw, "lie_yaw_add": add, "height": height_info,
                   "side_flip": bool(B["side_ref"] and ori.side_to_camera(KP, plate.C) == -B["side_ref"])}
            key = (not ev["footprint"]["ok"], ev["footprint"].get("n_out", 0), abs(add))
            if best is None or key < best[0]:
                best = (key, rec)
            if ev["footprint"]["ok"]:
                break
        return best[1]

    attempts, final = [], None
    for k, cell in enumerate(cells[:ATTEMPTS]):
        T0 = np.array([cell["xy"][0], cell["xy"][1], cell["h"]])
        T, fi, best = T0, 0, None
        frame_hist = []
        while True:                                           # п.8: тело вне кадра → сдвиг внутри зоны к центру
            r_ = fit_at(T, cell["anchor"])
            import assembly
            val = assembly.validate(r_["KP"], r_["ev"], spec, plate, B["side_ref"], window, True,
                                    X=r_["X"], treg=treg, soft=soft, height_info=r_["height"])
            r_["validation"] = val
            st_ = ("partial" if no_secondary else "ok") if val["ok"] else val["status"]
            frame_hist.append({"T": [round(float(x), 3) for x in T], "frame_ok": r_["ev"]["frame"]["ok"],
                               "status": st_})
            key = (not r_["ev"]["frame"]["ok"], STATUS_RANK[st_])
            if best is None or key < best[0]:
                best = (key, r_, st_, T)
            if r_["ev"]["frame"]["ok"] or fi >= FRAME_TRIES:
                break
            pc = frame_target(plate, T0, r_["ev"]["frame"]["bbox_px"], *zall)
            if pc is None:
                break
            fi += 1
            T = T0 + (pc - T0) * (fi / FRAME_TRIES)
        _, rec, st, T = best
        rec.update(attempt=k, cell=cell, stage="fit", status=st, frame_tries=frame_hist, T=T)
        attempts.append(rec)
        ev = rec["ev"]
        log(f"[{spec.id}] попытка {k} fit: {st}; опора {ev['support_object']} {ev['support_gap_cm']} см, "
            f"прон. p95 {ev['penetration']['p95_cm']} max {ev['penetration']['max_cm']} см, след {ev['footprint']}, "
            f"кадр {ev['frame']['ok']}, наклон {ev['tilt_deg']} крен {ev['roll_deg']}, yaw {rec['yaw']['yaw_deg']}")
        if st in ("ok", "partial") and not rec["side_flip"]:
            final = rec
            break
        if refine_ctx is not None:
            import refine as rf
            B["groups"] = groups
            rr = rf.refine_mem(refine_ctx, ref, B, rec, S, plate, spec, H, log=lambda m: log(f"[{spec.id}] {m}"))
            if rr is not None:
                X2, K2, rinfo = rr
                orient_bad = not all(c["ok"] for c in assembly.orientation_checks(
                    K2, spec.pose_class, plate, B["side_ref"], window))
                if orient_bad:
                    X2, K2 = assembly.clamp_root(X2, K2, spec.pose_class, plate, B["side_ref"], window)
                    rinfo["root_clamped"] = True
                shift2 = 0.0
                if spec.target_object != "floor":
                    X2, K2, shift2, _ = shift_into(X2, K2, groups, S, treg, H, spec,
                                                np.flatnonzero(np.isin(B["reg"], spec.support_regions())))
                X2, K2, hi2 = assembly.height_fit(X2, K2, groups, S, plate, spec, treg, soft, H)
                sw2 = None
                if spec.pose_class == "recline":
                    X2, K2, sw2 = recline_sweep(dict(rec["fit"], X=X2, KP=K2), groups, S, plate, H,
                                              B["orient"]["tilt_deg"], treg, soft)
                rinfo["height_after_refine"] = hi2
                ev2 = evaluate(X2, K2, B, groups, S, plate, spec, treg, freg, H, soft)
                val2 = assembly.validate(K2, ev2, spec, plate, B["side_ref"], window, rinfo["limits_ok"],
                                         X=X2, treg=treg, soft=soft, height_info=hi2)
                st2 = ("partial" if no_secondary else "ok") if val2["ok"] else val2["status"]
                flip2 = bool(B["side_ref"] and ori.side_to_camera(K2, plate.C) == -B["side_ref"])
                rec2 = dict(rec, stage="refine", status=st2, X=X2, KP=K2, ev=ev2, refine=rinfo, rigid=rec,
                            shift_m=shift2, side_flip=flip2, validation=val2, height=hi2, recline=sw2)
                if flip2:
                    log(f"[{spec.id}] попытка {k} refine развернул тело другой стороной к камере — в конец выбора")
                attempts.append(rec2)
                log(f"[{spec.id}] попытка {k} refine: {st2}; опора {ev2['support_object']} {ev2['support_gap_cm']} см, "
                    f"прон. p95 {ev2['penetration']['p95_cm']} см, след {ev2['footprint']}, кадр {ev2['frame']['ok']}, "
                    f"группы {rinfo['groups_deg']} лимиты {rinfo['limits_ok']}")
                if st2 in ("ok", "partial") and not flip2:
                    final = rec2
                    break
    if final is None:
        final = min(attempts, key=lambda a: (a["side_flip"], STATUS_RANK[a["status"]], a["ev"]["footprint"].get("n_out", 0),
                                            abs(a["ev"]["support_gap_cm"]) + a["ev"]["penetration"]["p95_cm"]))
    ev = final["ev"]
    fl = list(flags) + final["validation"].get("flags", [])
    diagnostics.update(geometry_diagnostics(final["X"], final["KP"], B, spec, treg, groups, S, plate, soft))
    oi = B["orient"]
    tilt_final, roll_final = ev["tilt_deg"], ev["roll_deg"]
    clamped = round(abs(tilt_final - oi["tilt_ref_deg"]) + (0.0 if spec.pose_class == "lie" else
                                                            abs(roll_final - oi["roll_ref_deg"])), 1)
    if oi["clamped_deg"] > CFG["clamped_flag_deg"]:
        fl.append(f"orient_clamped:{oi['clamped_deg']}")
    if final["yaw"]["side_conflict"]:
        fl.append("side_conflict")
    if final["conflict"]:
        fl.append("orient_conflict")
    if final["side_flip"]:
        fl.append("side_flip")
    pel = (final["KP"][9] + final["KP"][10]) / 2
    fwd = ori.forward(final["KP"])
    tc = plate.C - pel
    tc[2] = 0
    tc /= np.linalg.norm(tc)
    ang_p = math.degrees(pf.signed_angle(tc, fwd)) if fwd is not None else 0.0
    side_final = ori.side_to_camera(final["KP"], plate.C)
    elev_p = math.degrees(math.atan2(plate.h - pel[2], math.hypot(*(plate.C - pel)[:2])))
    rep = dict(base, status=final["status"], support=[g[0] for g in groups if g[3] == "support"],
               support_info=ginfo, validation=final["validation"], height_adjust=final["height"],
               refine_root_clamps=sum(bool(a.get("refine", {}).get("root_clamped")) for a in attempts), secondary=[{"part": g[0], "surface": g[2], "role": g[3]} for g in groups
                                               if g[3] != "support"],
               support_object=ev["support_object"], support_parts_on=ev["support_parts_on"],
               support_gap_cm=ev["support_gap_cm"], back_gap_cm=ev["back_gap_cm"], wall_gap_cm=ev["wall_gap_cm"],
               back_contact_gap_m=ev["back_contact_gap_m"], zone_diagnostics=diagnostics,
               max_penetration_cm=ev["penetration"]["max_cm"], penetration=ev["penetration"],
               footprint=ev["footprint"], frame=ev["frame"], frame_tries=final["frame_tries"],
               balance_ok=ev["balance_ok"], contacts=ev["contacts"], cells=ev["cells"], cells_ok=ev["cells_ok"],
               target_info=oinfo,
               zone={"source": zsrc, **oinfo, **{k: v for k, v in zinfo.items() if k != "flags"},
                     "cell": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in final["cell"].items()},
                     "attempt": final["attempt"], "shift_m": final["shift_m"], "region": treg.summary()},
               orient={"sign_candidate": oi["sign_candidate"], "tilt_ref_deg": oi["tilt_ref_deg"],
                       "tilt_set_deg": oi["tilt_deg"], "tilt_final_deg": tilt_final, "roll_ref_deg": oi["roll_ref_deg"],
                       "roll_final_deg": roll_final, "clamped_deg": oi["clamped_deg"], "total_change_deg": clamped,
                       "yaw": final["yaw"], "side_ref": SIDE_NAME[B["side_ref"]], "side_final": SIDE_NAME[side_final], "conflict": final["conflict"],
                       "recline": final["recline"], "lie_yaw_add_deg": final["lie_yaw_add"],
                       "check": ev["orient"]},
               angle_to_camera_deg={"ref": round(B["rel_deg"], 1), "plate": round(ang_p, 1)},
               camera_elevation_deg={"ref": round(B["elev_ref"], 1), "plate": round(elev_p, 1)},
               up_from=B["up_src"], ref_planes=B["n_planes"], height_scale=round(B["s"], 3),
               refine=final.get("refine"), stage=final["stage"],
               attempts=[{"attempt": a["attempt"], "stage": a["stage"], "status": a["status"],
                          "cell_xy": [round(x, 3) for x in a["cell"]["xy"]], "shift_m": a["shift_m"],
                          "support_object": a["ev"]["support_object"], "gap_cm": a["ev"]["support_gap_cm"],
                          "pen_p95_cm": a["ev"]["penetration"]["p95_cm"], "footprint_out": a["ev"]["footprint"].get("n_out"),
                          "frame_ok": a["ev"]["frame"]["ok"], "frame_tries": len(a["frame_tries"]) - 1,
                          "lie_yaw_add": a["lie_yaw_add"], "validation": a["validation"], "height": a["height"],
                          **({"groups_deg": a["refine"]["groups_deg"], "limits_ok": a["refine"]["limits_ok"]}
                             if a["stage"] == "refine" else {})} for a in attempts],
               flags=sorted(set(fl)), fit={k: final["fit"][k] for k in ("params", "cost", "nfev", "yaw_set_deg")})
    ku, kv, kz = plate.project(final["KP"][pf.MHR_TO_B25])
    kzp = plate.depth_at(ku, kv)
    rep["keypoints_body25_px"] = [[round(float(a), 1), round(float(b), 1),
                                   int(0 <= a < plate.Wp and 0 <= b < plate.Hp and z > 0 and z <= zp + CFG["visibility_depth_m"])]
                                  for a, b, z, zp in zip(ku, kv, kz, kzp)] + [[0.0, 0.0, 0]] * 30
    rep["px_size"] = [plate.Wp, plate.Hp]
    M, b = similarity(B["V0"], final["X"] if final["stage"] == "fit" else final["rigid"]["X"])
    state = {"M": M, "b": b, "target": final["fit"]["pivot"], "groups": groups}
    return {"status": final["status"], "rep": rep, "X": final["X"], "KP": final["KP"], "reg_lbl": B["reg"],
            "state": state, "final": final, "regions": (treg, freg)}


place_pose = place  # совместимость CLI шага 7; сборка существует только одна


def write(out, res, ref, plate, spec, fit_hash):
    """файлы позы: body.obj, fit.json, fit_state.npz, overlay.jpg, shot_spec.json; при refine — body_rigid.obj,
    fit_rigid.json, fit_hash.txt (всегда заново)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if "KP" in res:
        np.savez_compressed(out / "label_joints.npz", kp=res["KP"])
    for fn in ("body.obj", "fit.json", "fit_state.npz", "overlay.jpg", "body_rigid.obj", "fit_rigid.json",
               "fit_hash.txt", "shot_spec.json"):
        (out / fn).unlink(missing_ok=True)
    spec.save(out / "shot_spec.json")
    rep = dict(res["rep"], fit_hash=fit_hash)
    if "X" not in res:
        (out / "fit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        return
    st = res["state"]
    np.savez_compressed(out / "fit_state.npz", M=st["M"], b=st["b"], target=st["target"],
                        groups=np.array([f"{g[0]}:{g[2]}:{g[3]}" for g in st["groups"]]),
                        **{f"g_{g[0]}": g[1] for g in st["groups"]})
    fin = res["final"]
    rep = json.loads(json.dumps(rep, default=float))
    if fin["stage"] == "refine":
        rig = fin["rigid"]
        rrep = dict(rep, status=rig["status"], stage="fit", support_gap_cm=rig["ev"]["support_gap_cm"],
                    max_penetration_cm=rig["ev"]["penetration"]["max_cm"], refine=None)
        pf.write_outputs(out, rig["X"], ref["F"], rrep, plate, res["reg_lbl"], rep["contacts"].keys())
        (out / "body.obj").replace(out / "body_rigid.obj")
        (out / "fit.json").replace(out / "fit_rigid.json")
        (out / "fit_hash.txt").write_text(fit_hash, encoding="utf-8")
    pf.write_outputs(out, res["X"], ref["F"], rep, plate, res["reg_lbl"], rep["contacts"].keys())
