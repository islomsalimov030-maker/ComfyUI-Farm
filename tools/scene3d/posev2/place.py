"""Постановка v2.4 (07.11f): посадка по ShotSpec. Читает только shot_spec (+ профили / height_proxy).

  Р1 (п.5): опорные части касаются только маски target_object (+ спинки для recline); пол — только если target = floor.
  Р2 (п.6): зона на верхе предмета (zone.py) — ячейка-попытка; опорные части в ячейках с dt ≥ 5 см, иначе сдвиг
            к максимуму dt шагом 1 см до 0.5·H.
  Р3 (п.7): ориентация оси таз→шея после fit и после refine; при нарушении — кандидаты поворота корня (orient.py).
  Р4 (п.8): refine с лимитами по группам (корпус/шея/голова ≤ 15°, руки ≤ 15°, ноги ≤ 20°); контакт не достигнут —
            следующая ячейка зоны (≤ 5 попыток), иначе contact_fail.
  Р5 (п.9): статус ok | wrong_surface | orient_fail | contact_fail | no_target.
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
import top_region as tr
import zone as zn
from body import facing, height_est, regions

SUP_GAP_CM = 3.0          # прежние геометрические условия (контакт, см)
PEN_CM = 3.0
ON_SURF_M = 0.04          # вершина «на поверхности» (м)
SHIFT_STEP = 0.01         # м
SHIFT_MAX_H = 0.5         # × H
ATTEMPTS = 5
BACK_FRAC = 0.2           # доля вершин корпуса со стороны спины
STATUS_RANK = {"ok": 0, "contact_fail": 1, "wrong_surface": 2, "orient_fail": 3, "no_target": 4}
B25_BODY = pf.MHR_TO_B25


# ------------------------------------------------------------------------------------------------ поверхности (Р1)
def _surf(P, N=None):
    if N is None:
        N = np.tile([0, 0, 1.0], (len(P), 1))
    return cKDTree(P), P, N


def build_surfaces(plate, treg, freg):
    S = {k: plate.trees[k] for k in ("h", "v", "any")}
    S["target"] = _surf(treg.points3d())
    if treg.backrest is not None:
        S["back"] = _surf(treg.backrest, treg.backrest_n)
    if freg is not None and freg.mask.any():
        S["floor"] = _surf(freg.points3d())
    return S


# ------------------------------------------------------------------------------------------------ тело
def prepare_body(ref, spec, height, with_back):
    V0, kp0 = ref["V"], ref["kp"]
    reg = regions(V0, kp0)
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
    groups = []                                                    # (имя, индексы, поверхность, роль)
    for n in names:
        if n not in sup_c:
            continue
        idx = sup_c[n]["idx"]
        h = V[idx, 2]
        groups.append((n, idx[h <= np.percentile(h, 15)], "target", "support"))
    for n, c in ref_c.items():
        if n in sup_c:
            continue
        groups.append((n, c["idx"], c["kind"] if c["kind"] in ("h", "v") else "any", "secondary"))
    if with_back:
        ti = np.where(reg == "torso")[0]
        left = kp[5] - kp[6]
        spine = kp[69] - (kp[9] + kp[10]) / 2
        chest = np.cross(left, spine)
        chest /= np.linalg.norm(chest)
        pr_ = (V[ti] - V[ti].mean(0)) @ chest
        groups.append(("back", ti[pr_ <= np.percentile(pr_, 100 * BACK_FRAC)], "back", "back"))
    return {"V0": V0, "kp0": kp0, "reg": reg, "up_src": up_src, "Rref": Rref, "s": s, "V": V, "kp": kp,
            "groups": groups, "dims": prof.body_dims_from_kp(kp), "n_planes": len(planes),
            "missing_support": [n for n in names if n not in sup_c]}


def similarity(V0, X):
    A = np.hstack([V0, np.ones((len(V0), 1))])
    Bm = np.linalg.lstsq(A, X, rcond=None)[0]
    return Bm[:3].T, Bm[3]


# ------------------------------------------------------------------------------------------------ жёсткая посадка
def rigid_fit(B, plate, T, anchor, S, cand_R=None):
    V, kp, reg = B["V"].copy(), B["kp"].copy(), B["reg"]
    Zup = np.array([0, 0, 1.0])
    pel = (kp[9] + kp[10]) / 2
    fwd = facing(kp, Zup)
    to_cam = -pel.copy()
    to_cam[2] = 0
    to_cam /= np.linalg.norm(to_cam)
    rel = pf.signed_angle(to_cam, fwd) if fwd is not None else 0.0
    elev_ref = math.degrees(math.atan2(-pel[2], math.hypot(pel[0], pel[1])))
    tc = plate.C - T
    tc[2] = 0
    tc /= np.linalg.norm(tc)
    want = pf.rot_z(rel) @ tc
    yaw0 = pf.signed_angle(fwd, want) if fwd is not None else 0.0
    R0 = pf.rot_z(yaw0)
    V, kp = V @ R0.T, kp @ R0.T
    cam_axis = R0 @ B["Rref"] @ np.array([0, 0, 1.0])
    if cand_R is not None:
        c = (kp[9] + kp[10]) / 2
        V, kp = (V - c) @ cand_R.T + c, (kp - c) @ cand_R.T + c
    sup = [g for g in B["groups"] if g[3] == "support"]
    sup_idx = np.concatenate([g[1] for g in sup]) if sup else np.argsort(V[:, 2])[: max(20, len(V) // 50)]
    if anchor == "pelvis":
        pi_ = np.where(reg == "pelvis")[0]
        piv_idx = pi_[V[pi_, 2] <= np.percentile(V[pi_, 2], 15)]
    else:
        piv_idx = sup_idx
    pivot = V[piv_idx].mean(0)
    V, kp = V - pivot + T, kp - pivot + T
    pivot = np.asarray(T, float).copy()
    tilt = 0.05 if B["up_src"] in ("plane", "contact_plane") else 0.3
    rng = np.random.default_rng(0)
    pen_idx = rng.choice(len(V), min(2500, len(V)), replace=False)
    groups = []
    for name, idx, surf, role in B["groups"]:
        if surf not in S:
            continue
        groups.append((name, idx if len(idx) <= 60 else rng.choice(idx, 60, replace=False), surf, role))

    def transform(p, X):
        return (X - pivot) @ pf.rot_xyz(p[0], p[4], p[5]).T + pivot + p[1:4]

    def residuals(p):
        X = transform(p, V)
        res = []
        for _, idx, surf, role in groups:
            tree, P, N = S[surf]
            d, j = tree.query(X[idx])
            sd = np.einsum("ij,ij->i", X[idx] - P[j], N[j])
            k_n = 1.0 / math.sqrt(len(idx))
            if role in ("support", "back"):
                res.append(3.0 * k_n * sd / 0.01)
            else:
                res.append(k_n * np.where(d < 0.4, sd / 0.03, 0.0))
        pen = np.maximum(pf.penetration(plate, X[pen_idx]) - 0.005, 0.0)
        res.append(10.0 / math.sqrt(len(pen_idx)) * pen / 0.01)
        res.append(np.array([p[0] / 0.35, p[4] / tilt, p[5] / tilt, np.hypot(p[1], p[2]) / 0.25]))
        return np.concatenate(res)

    sol = least_squares(residuals, np.zeros(6), loss="soft_l1", f_scale=1.0, diff_step=1e-3, max_nfev=400)
    X, KP = transform(sol.x, V), transform(sol.x, kp)
    return {"X": X, "KP": KP, "pivot": pivot, "cam_axis": cam_axis, "rel_ref": rel, "elev_ref": elev_ref,
            "params": [round(float(x), 4) for x in sol.x], "cost": round(float(sol.cost), 3), "nfev": int(sol.nfev)}


# ------------------------------------------------------------------------------------------------ оценка
def _sd(S, surf, Xi):
    tree, P, N = S[surf]
    d, j = tree.query(Xi)
    return np.einsum("ij,ij->i", Xi - P[j], N[j]), d


def _on(reg, Xi, plate):
    """вершина на верхе предмета: ячейка маски и |z − h| ≤ ON_SURF_M; вне кадра фото (сетки там нет) — только высота h0."""
    if reg is None or not reg.mask.any():
        return np.zeros(len(Xi), bool)
    u, v, z = plate.project(Xi)
    vis = (u >= 0) & (u < plate.Wp) & (v >= 0) & (v < plate.Hp) & (z > 0)
    i, j = reg.cell_of(Xi)
    ok = reg.inside(i, j)
    out = np.zeros(len(Xi), bool)
    out[ok] = reg.mask[i[ok], j[ok]] & (np.abs(Xi[ok, 2] - reg.height[i[ok], j[ok]]) <= ON_SURF_M)
    out[~vis] = np.abs(Xi[~vis, 2] - reg.h0) <= ON_SURF_M
    return out


def part_label(Xi, spec, treg, freg, plate):
    if _on(treg, Xi, plate).mean() >= 0.5:
        return spec.target_object
    if freg is not None and _on(freg, Xi, plate).mean() >= 0.5:
        return "floor"
    return "other"


def evaluate(X, KP, B, S, plate, spec, treg, freg, H):
    rep_c, sup_sd, back_sd = {}, [], []
    labels = {}
    for name, idx, surf, role in B["groups"]:
        if surf not in S:
            continue
        sd, d = _sd(S, surf, X[idx])
        rep_c[name] = {"role": role, "surface": surf, "gap_cm": round(float(np.median(sd)) * 100, 1),
                       "min_cm": round(float(sd.min()) * 100, 1)}
        if role == "support":
            sup_sd.append(sd)
            labels[name] = part_label(X[idx], spec, treg, freg, plate)
        elif role == "back":
            back_sd.append(sd)
    g = np.concatenate(sup_sd) if sup_sd else np.zeros(1)
    sup_gap = abs(float(np.median(g[g <= np.percentile(g, 30)]))) * 100
    back_gap = None
    if back_sd:
        gb = np.concatenate(back_sd)
        back_gap = abs(float(np.median(gb[gb <= np.percentile(gb, 30)]))) * 100
    pen = pf.penetration(plate, X)
    max_pen = float(pen.max()) * 100
    vals = list(labels.values())
    if vals and all(v == spec.target_object for v in vals):
        sup_obj = spec.target_object
    else:
        rest = [v for v in vals if v != spec.target_object] or ["none"]
        sup_obj = max(set(rest), key=rest.count)
    cells_ok, cells = True, {}
    if spec.target_object != "floor":
        for name, idx, surf, role in B["groups"]:
            if role != "support":
                continue
            c = X[idx].mean(0)[None]
            dt = float(treg.dt_at(c)[0])
            inm = bool(treg.in_mask(c)[0])
            cells[name] = {"in_mask": inm, "dt_m": round(dt, 3)}
            cells_ok &= inm and dt >= zn.EDGE_MIN
    geom_ok = sup_gap <= SUP_GAP_CM and max_pen <= PEN_CM and (back_gap is None or back_gap <= SUP_GAP_CM)
    # равновесие (как в pose_fit): центр масс над опорным многоугольником
    reg = B["reg"]
    pts = [X[idx, :2] for name, idx, surf, role in B["groups"]
           if role == "support" and name in rep_c and abs(rep_c[name]["gap_cm"]) < 3]
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
        balance = inside or bool(np.min(np.linalg.norm(P2 - com[:2], axis=1)) < 0.04)
    return {"support_gap_cm": round(sup_gap, 1), "back_gap_cm": None if back_gap is None else round(back_gap, 1),
            "max_penetration_cm": round(max_pen, 1), "pen_verts_over_2cm": int((pen > 0.02).sum()),
            "contacts": rep_c, "support_parts_on": labels, "support_object": sup_obj, "cells": cells,
            "cells_ok": bool(cells_ok), "geom_ok": bool(geom_ok), "balance_ok": balance,
            "orient": ori.check(KP, spec.pose_class, H)}


def status_of(ev, spec, limits_ok=True):
    if not ev["orient"]["ok"]:
        return "orient_fail"
    if ev["support_object"] != spec.target_object:
        return "wrong_surface"
    if not (ev["geom_ok"] and ev["cells_ok"] and limits_ok):
        return "contact_fail"
    return "ok"


# ------------------------------------------------------------------------------------------------ сдвиг в зону (Р2)
def shift_into(X, KP, B, S, treg, H):
    sup = [(n, idx) for n, idx, surf, role in B["groups"] if role == "support"]
    if not sup or not treg.mask.any():
        return X, KP, 0.0, False

    def ok(Xc):
        cs = np.array([Xc[idx].mean(0) for _, idx in sup])
        return bool(np.all(treg.in_mask(cs) & (treg.dt_at(cs) >= zn.EDGE_MIN)))
    if ok(X):
        return X, KP, 0.0, True
    i, j = np.unravel_index(int(np.argmax(treg.dt)), treg.dt.shape)
    tgt = treg.center(i, j)
    c0 = np.mean([X[idx].mean(0)[:2] for _, idx in sup], axis=0)
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


# ------------------------------------------------------------------------------------------------ ориентация (Р3)
def fit_oriented(B, plate, T, anchor, S, spec, treg, freg, ref, H, log):
    r0 = rigid_fit(B, plate, T, anchor, S)
    G0 = similarity(B["V0"], r0["X"])
    o0 = ori.check(r0["KP"], spec.pose_class, H)
    cand_rep = [{"name": "identity", "ok": o0["ok"], "reproj": round(ori.reproj_error(r0["KP"], G0, ref, B25_BODY), 3)}]
    if o0["ok"]:
        r0["cand"] = "identity"
        return r0, cand_rep, True
    best = None
    for name, R in ori.candidates(r0["KP"], r0["cam_axis"])[1:]:
        r = rigid_fit(B, plate, T, anchor, S, cand_R=R)
        o = ori.check(r["KP"], spec.pose_class, H)
        e = ori.reproj_error(r["KP"], G0, ref, B25_BODY)
        cand_rep.append({"name": name, "ok": o["ok"], "reproj": round(e, 3)})
        if o["ok"] and (best is None or e < best[0]):
            r["cand"] = name
            best = (e, r)
    if best is None:
        r0["cand"] = "identity"
        log(f"ориентация: ни один кандидат не прошёл {[(c['name'], c['reproj']) for c in cand_rep]}")
        return r0, cand_rep, False
    return best[1], cand_rep, True


# ------------------------------------------------------------------------------------------------ поза целиком
def resolve_regions(plate, spec, seed, H, profile):
    """-> (treg, freg, zone_src, obj_info) | (None, …, 'no_target')."""
    freg, info = None, {}
    if profile is not None:
        fo, _ = profile.find_object("floor")
        freg = profile.region(fo) if fo is not None else None
    if freg is None:
        freg = tr.top_region(plate, "floor", None, H)
    if spec.target_object == "floor":
        return freg, freg, "floor", {"object_id": "floor"}
    if profile is not None:
        obj, via = profile.find_object(spec.target_object)
        if obj is None:
            return None, freg, "no_target", {"reason": f"{spec.target_object} нет в profile.objects и substitutes"}
        info = {"object_id": obj["id"], "via": via}
        a = profile.anchor(obj["id"], spec.pose_class)
        if a is not None:
            info["anchor_id"] = a["id"]
            return profile.region(obj), freg, "profile", info
    treg = tr.top_region(plate, spec.target_object, seed, H)
    if not treg.mask.any():
        return None, freg, "no_target", {"reason": "height_proxy: пустая маска"}
    return treg, freg, "proxy", info


def place_pose(ref, plate, spec, seed, height, *, profile=None, body_src="proxy", dims_override=None, refine_ctx=None,
               log=print):
    """seed — 3D-точка из таблицы (поднятая точка сиденья / точка стоп). -> dict результата (X, KP, rep, …)."""
    has_back_guess = spec.pose_class == "recline"
    B = prepare_body(ref, spec, height, with_back=False)
    if dims_override:
        B["dims"] = dict(dims_override)
    H = B["dims"]["H"]
    treg, freg, zsrc, oinfo = resolve_regions(plate, spec, seed, H, profile)
    base = {"shot_spec": spec.to_dict(), "body_dims": B["dims"], "body_source": body_src, "height_m": height}
    if treg is None:
        return {"status": "no_target", "rep": dict(base, status="no_target", zone={"source": zsrc, **oinfo})}
    with_back = has_back_guess and treg.backrest is not None
    if with_back:
        B = dict(prepare_body(ref, spec, height, with_back=True), dims=B["dims"])
    S = build_surfaces(plate, treg, freg)
    flags = list(treg.flags)
    if B["missing_support"]:
        flags.append("support_part_missing:" + ",".join(B["missing_support"]))
    if spec.target_object == "floor":
        cells = [{"xy": list(seed[:2]), "h": float(seed[2]), "anchor": "support"}]
        zinfo = {"rule": "floor", "band_m": None, "flags": []}
    else:
        if zsrc == "profile":
            m = profile.zone_metrics(profile.anchor(oinfo["object_id"], spec.pose_class))
        else:
            m = zn.metrics(treg)
        cells, zinfo = zn.zone_cells(m, spec.pose_class, B["dims"], seed_xy=seed[:2],
                                     has_backrest=treg.backrest is not None)
        flags += zinfo["flags"]
        if not cells:
            return {"status": "contact_fail",
                    "rep": dict(base, status="contact_fail", flags=flags + ["zone_empty"],
                                zone={"source": zsrc, **oinfo, **zinfo, "region": treg.summary()})}
    attempts, final = [], None
    for k, cell in enumerate(cells[:ATTEMPTS]):
        T = np.array([cell["xy"][0], cell["xy"][1], cell["h"]])
        r, cand_rep, o_ok = fit_oriented(B, plate, T, cell["anchor"], S, spec, treg, freg, ref, H,
                                         lambda m: log(f"[{spec.id}] попытка {k}: {m}"))
        X, KP, shift = r["X"], r["KP"], 0.0
        if spec.target_object != "floor":
            X, KP, shift, _ = shift_into(X, KP, B, S, treg, H)
        ev = evaluate(X, KP, B, S, plate, spec, treg, freg, H)
        st = status_of(ev, spec)
        rec = {"attempt": k, "cell": cell, "stage": "fit", "status": st, "shift_m": shift, "fit": r,
               "X": X, "KP": KP, "ev": ev, "cand": r["cand"], "cand_rep": cand_rep, "orient_fit": ev["orient"]}
        attempts.append(rec)
        log(f"[{spec.id}] попытка {k} fit: {st}; опора {ev['support_object']} {ev['support_gap_cm']} см, "
            f"прон. {ev['max_penetration_cm']} см, ячейки {ev['cells_ok']}, ориент. {r['cand']} {ev['orient']['ok']}, "
            f"сдвиг {shift}")
        if st == "ok":
            final = rec
            break
        if st == "orient_fail":                       # от ячейки не зависит — дальше не пробуем
            final = rec
            break
        if refine_ctx is not None:
            import refine as rf
            rr = rf.refine_mem(refine_ctx, ref, B, rec, S, plate, spec, H, log=lambda m: log(f"[{spec.id}] {m}"))
            if rr is not None:
                X2, K2, rinfo = rr
                shift2 = 0.0
                if spec.target_object != "floor":
                    X2, K2, shift2, _ = shift_into(X2, K2, B, S, treg, H)
                ev2 = evaluate(X2, K2, B, S, plate, spec, treg, freg, H)
                st2 = status_of(ev2, spec, rinfo["limits_ok"])
                rec2 = dict(rec, stage="refine", status=st2, X=X2, KP=K2, ev=ev2, refine=rinfo, rigid=rec,
                            shift_m=shift2)
                attempts.append(rec2)
                log(f"[{spec.id}] попытка {k} refine: {st2}; опора {ev2['support_object']} {ev2['support_gap_cm']} см, "
                    f"прон. {ev2['max_penetration_cm']} см, группы {rinfo['groups_deg']} лимиты {rinfo['limits_ok']}")
                if st2 == "ok":
                    final = rec2
                    break
    if final is None:
        final = min(attempts, key=lambda a: (STATUS_RANK[a["status"]],
                                            a["ev"]["support_gap_cm"] + a["ev"]["max_penetration_cm"]))
    ev = final["ev"]
    fl = flags + [f for f in zinfo.get("flags", []) if f not in flags]
    ang_ref = math.degrees(final["fit"]["rel_ref"])
    pel = (final["KP"][9] + final["KP"][10]) / 2
    fwd = facing(final["KP"], np.array([0, 0, 1.0]))
    tc = plate.C - pel
    tc[2] = 0
    tc /= np.linalg.norm(tc)
    ang_p = math.degrees(pf.signed_angle(tc, fwd)) if fwd is not None else 0.0
    conflict = abs((ang_ref - ang_p + 180) % 360 - 180) > ori.CONFLICT_DEG
    if conflict:
        fl.append("orient_conflict")
    elev_p = math.degrees(math.atan2(plate.h - pel[2], math.hypot(*(plate.C - pel)[:2])))
    rep = dict(base, status=final["status"], support=[g[0] for g in B["groups"] if g[3] == "support"],
               support_object=ev["support_object"], support_parts_on=ev["support_parts_on"],
               support_gap_cm=ev["support_gap_cm"], back_gap_cm=ev["back_gap_cm"],
               max_penetration_cm=ev["max_penetration_cm"], pen_verts_over_2cm=ev["pen_verts_over_2cm"],
               balance_ok=ev["balance_ok"], contacts=ev["contacts"], cells=ev["cells"], cells_ok=ev["cells_ok"],
               zone={"source": zsrc, **oinfo, **{k: v for k, v in zinfo.items() if k != "flags"},
                     "cell": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in final["cell"].items()},
                     "attempt": final["attempt"], "shift_m": final["shift_m"], "region": treg.summary()},
               orient_check={"after_fit": final["orient_fit"],
                             "after_refine": ev["orient"] if final["stage"] == "refine" else None,
                             "candidate": final["cand"], "candidates": final["cand_rep"],
                             "reproj_err": next((c["reproj"] for c in final["cand_rep"] if c["name"] == final["cand"]),
                                                None),
                             "conflict": bool(conflict)},
               angle_to_camera_deg={"ref": round(ang_ref, 1), "plate": round(ang_p, 1)},
               camera_elevation_deg={"ref": round(final["fit"]["elev_ref"], 1), "plate": round(elev_p, 1)},
               up_from=B["up_src"], ref_planes=B["n_planes"], height_scale=round(B["s"], 3),
               refine=final.get("refine"), stage=final["stage"],
               attempts=[{"attempt": a["attempt"], "stage": a["stage"], "status": a["status"],
                          "cell_xy": [round(x, 3) for x in a["cell"]["xy"]], "shift_m": a["shift_m"],
                          "support_object": a["ev"]["support_object"], "gap_cm": a["ev"]["support_gap_cm"],
                          "pen_cm": a["ev"]["max_penetration_cm"], "cand": a["cand"],
                          **({"groups_deg": a["refine"]["groups_deg"], "limits_ok": a["refine"]["limits_ok"]}
                             if a["stage"] == "refine" else {})} for a in attempts],
               flags=sorted(set(fl)), fit={k: final["fit"][k] for k in ("params", "cost", "nfev")})
    ku, kv, kz = plate.project(final["KP"][pf.MHR_TO_B25])
    kzp = plate.depth_at(ku, kv)
    rep["keypoints_body25_px"] = [[round(float(a), 1), round(float(b), 1),
                                   int(0 <= a < plate.Wp and 0 <= b < plate.Hp and z > 0 and z <= zp + 0.02)]
                                  for a, b, z, zp in zip(ku, kv, kz, kzp)] + [[0.0, 0.0, 0]] * 30
    rep["px_size"] = [plate.Wp, plate.Hp]
    M, b = similarity(B["V0"], final["X"] if final["stage"] == "fit" else final["rigid"]["X"])
    state = {"M": M, "b": b, "target": final["fit"]["pivot"], "groups": B["groups"]}
    return {"status": final["status"], "rep": rep, "X": final["X"], "KP": final["KP"], "reg_lbl": B["reg"],
            "state": state, "final": final, "regions": (treg, freg)}


def write(out, res, ref, plate, spec, fit_hash):
    """файлы позы: body.obj, fit.json, fit_state.npz, overlay.jpg, shot_spec.json; при refine — body_rigid.obj,
    fit_rigid.json, fit_hash.txt (всегда заново)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in ("body.obj", "fit.json", "fit_state.npz", "overlay.jpg", "body_rigid.obj", "fit_rigid.json",
               "fit_hash.txt", "shot_spec.json"):
        (out / fn).unlink(missing_ok=True)
    spec.save(out / "shot_spec.json")
    rep = dict(res["rep"], fit_hash=fit_hash)
    if "X" not in res:
        (out / "fit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
        return
    st = res["state"]
    np.savez_compressed(out / "fit_state.npz", M=st["M"], b=st["b"], target=st["target"],
                        groups=np.array([f"{g[0]}:{g[2]}:{g[3]}" for g in st["groups"]]),
                        **{f"g_{g[0]}": g[1] for g in st["groups"]})
    fin = res["final"]
    if fin["stage"] == "refine":
        rig = fin["rigid"]
        rrep = dict(rep, status=rig["status"], stage="fit", support_gap_cm=rig["ev"]["support_gap_cm"],
                    max_penetration_cm=rig["ev"]["max_penetration_cm"], refine=None)
        pf.write_outputs(out, rig["X"], ref["F"], rrep, plate, res["reg_lbl"], rep["contacts"].keys())
        (out / "body.obj").replace(out / "body_rigid.obj")
        (out / "fit.json").replace(out / "fit_rigid.json")
        (out / "fit_hash.txt").write_text(fit_hash, encoding="utf-8")
    pf.write_outputs(out, res["X"], ref["F"], rep, plate, res["reg_lbl"], rep["contacts"].keys())
