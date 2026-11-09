"""Шаг 9: этап высоты и финальная проверка единой place(). Не оптимизатор позы."""
import math

import numpy as np

import orient as ori
from placement_config import section


def orientation_checks(K, pose_class, plate, side_ref, window=None):
    c = section("place")
    eps = c["orientation_tolerance_deg"]
    tilt, roll = ori.tilt_deg(K), ori.roll_deg(K)
    lo, hi = ori.TILT_RANGE[pose_class]
    checks = [("tilt", max(lo - tilt, tilt - hi, 0.0), "deg")]
    if pose_class != "lie":
        checks.append(("roll", max(abs(roll) - ori.ROLL_MAX, 0.0), "deg"))
    got = ori.side_to_camera(K, plate.C)
    checks.append(("camera_side", float(bool(side_ref) and got != side_ref), "side"))
    if window is not None:
        f = ori.forward(K)
        yaw = math.degrees(math.atan2(f[1], f[0])) if f is not None else None
        checks.append(("yaw_window", float("inf") if yaw is None else
                       max(abs(ori.wrap(yaw - window)) - ori.YAW_WINDOW, 0.0), "deg"))
    return [{"check": n, "excess": float(e), "unit": u, "ok": bool(e <= (eps if u == "deg" else 0))}
            for n, e, u in checks]


def clamp_root(X, K, pose_class, plate, side_ref, window=None, lie_roll_ref=None):
    """Только корневой rigid-поворот; суставы refine остаются неизменными."""
    for _ in range(section("place")["frame_tries"]):
        X, K, _ = ori.assemble_tilt_roll(X, K, pose_class)
        if all(v["ok"] for v in orientation_checks(K, pose_class, plate, side_ref, window)):
            break
    f = ori.forward(K)
    if f is not None:
        yaw = math.degrees(math.atan2(f[1], f[0]))
        pel = (K[9] + K[10]) / 2
        to_cam = plate.C - pel
        y, _, _ = ori.solve_yaw(yaw, side_ref, window, lambda a: ori.side_for_yaws(K, a, to_cam))
        R = ori.rot_axis(ori.UP, math.radians(ori.wrap(y - yaw)))
        X, K = ori.rot_about(X, pel, R), ori.rot_about(K, pel, R)
    if pose_class == "lie" and lie_roll_ref is not None:
        pel = (K[9] + K[10]) / 2
        axis = K[69] - pel
        delta = ori.wrap(lie_roll_ref - ori.roll_deg(K, "lie"))
        # roll_deg grows opposite to the right-hand rotation about pelvis→neck.
        R = ori.rot_axis(axis, -math.radians(delta))
        X, K = ori.rot_about(X, pel, R), ori.rot_about(K, pel, R)
    return X, K


def height_fit(X, K, groups, S, plate, spec, treg, soft, H):
    """Сначала min signed support contact=0, затем подъём 5 мм до p95.

    Потолок подъёма от контакта: contact_tol + 1 см. Каждый кандидат
    наклона/крена проходит те же этапы (а)→(б), без изменения порогов.
    """
    import place as pl
    c = section("place")
    origin = (K[9] + K[10]) / 2
    support = [(idx, surf) for _, idx, surf, role in groups if role == "support" and surf in S]

    def settle(V, J):
        if not support:
            return V, J, {"shift_z_m": 0.0, "max_gap_m": float("inf"), "penetration_ok": False,
                          "height_conflict": False, "reason": "no_support_vertices"}
        # target — горизонтальная поверхность верха, signed distance сохраняет
        # сторону: при исходном проникновении тело поднимается, не опускается.
        standing = spec.pose_class in ("stand", "crouch")
        dz = -pl.body_lowest_gap(V, S) if standing else -min(
            float(pl._sd(S, surf, V[idx])[0].min()) for idx, surf in support)
        V, J = V + [0, 0, dz], J + [0, 0, dz]
        contact_min = pl.body_lowest_gap(V, S) if standing else min(
            float(pl._sd(S, surf, V[idx])[0].min()) for idx, surf in support)
        contact_ok = abs(contact_min) <= c["contact_tol"]
        pen = pl.penetration_stats(plate, V, treg, soft, spec.pose_class)
        moved = 0.0
        ceiling = c["contact_tol"] + c["height_lift_extra_m"]
        step = c["height_lift_step_m"]
        while contact_ok and not pen["ok"] and moved + step <= ceiling + np.finfo(float).eps:
            V, J = V + [0, 0, step], J + [0, 0, step]
            dz += step
            moved += step
            pen = pl.penetration_stats(plate, V, treg, soft, spec.pose_class)
        gaps = [float(np.median(pl._sd(S, surf, V[idx])[0])) for idx, surf in support]
        contact = pl.support_contact(V, groups, S, spec, soft)
        return V, J, {"support_rule_ok": contact["ok"], "support_flags": contact["flags"],
                      "shift_z_m": float(dz), "contact_min_m": contact_min, "contact_ok": bool(contact_ok),
                      "lift_from_contact_m": moved, "lift_ceiling_m": ceiling,
                      "max_gap_m": max(gaps), "part_gaps_m": gaps,
                      "penetration_ok": pen["ok"], "penetration_excess_m": pen["excess_m"],
                      "height_conflict": bool(not contact_ok or not pen["ok"])}

    V, J, info = settle(X.copy(), K.copy())
    info.update(tilt_adjust_deg=0.0, roll_adjust_deg=0.0)
    def score(ii):
        return (not ii["penetration_ok"], not ii.get("support_rule_ok", False),
                0.0 if spec.pose_class in ("stand", "crouch") else
                max(abs(v) for v in ii.get("part_gaps_m", [float("inf")])))
    best = score(info)
    if info["penetration_ok"] and info.get("support_rule_ok", False):
        return V, J, info
    f = ori.forward(K)
    if f is None:
        return V, J, info
    lateral = np.cross(f, ori.UP)
    if spec.pose_class == "lie":
        long_axis = K[69] - origin
        lateral = np.cross(long_axis, ori.UP)
        lateral /= max(np.linalg.norm(lateral), ori.CFG["norm_eps"])
    vals = np.arange(-c["gap_sweep_deg"], c["gap_sweep_deg"] + c["gap_sweep_step_deg"] / 2,
                     c["gap_sweep_step_deg"])
    for tilt in vals:
        for roll in vals:
            R = ori.rot_axis(f, math.radians(float(roll))) @ ori.rot_axis(lateral, math.radians(float(tilt)))
            A, B = ori.rot_about(X, origin, R), ori.rot_about(K, origin, R)
            lo, hi = ori.TILT_RANGE[spec.pose_class]
            if not lo <= ori.tilt_deg(B) <= hi:
                continue
            if spec.pose_class == "lie":
                # Lie не имеет глобального roll-clamp. Высотная подстройка
                # не вращает тело вокруг длинной оси, сохраняя крен рефа.
                if roll != 0:
                    continue
            elif abs(ori.roll_deg(B)) > ori.ROLL_MAX:
                continue
            A, B, ii = settle(A, B)
            key = score(ii)
            if key < best:
                best, V, J = key, A, B
                info = dict(ii, tilt_adjust_deg=float(tilt), roll_adjust_deg=float(roll))
    return V, J, info


def validate(K, ev, spec, plate, side_ref, window, limits_ok, *, X=None, treg=None, soft=False,
             height_info=None):
    import place as pl
    c = section("place")
    # В реальной place() всегда передаётся геометрия. Ровно тот же расчёт,
    # что у этапа высоты/evaluate, а не max или иной процентиль.
    pen = pl.penetration_stats(plate, X, treg, soft, spec.pose_class) if X is not None else ev["penetration"]
    checks = orientation_checks(K, spec.pose_class, plate, side_ref, window)
    def add(name, ok, excess=0.0, unit="m"):
        checks.append({"check": name, "ok": bool(ok), "excess": float(excess), "unit": unit})
    selected = ev.get("target_instance", spec.target_object)
    add("support_target", ev["support_object"] == selected,
        float(ev["support_object"] != selected), "surface")
    if spec.pose_class not in ("stand", "crouch"):
        for name, gap in ev["support_part_gaps_m"].items():
            add("support_gap:" + name, gap <= c["sup_gap_cm"] / 100,
                max(gap - c["sup_gap_cm"] / 100, 0.0))
    else:
        contact = ev.get("support_contact", {})
        lowest = contact.get("lowest_gap_m", float("inf"))
        add("body_lowest_contact", abs(lowest) <= c["contact_tol"], max(abs(lowest) - c["contact_tol"], 0.0))
        add("one_foot_contact", bool(contact.get("foot_contact_parts")),
            float(not contact.get("foot_contact_parts")), "foot")
    for name, lim in (("soft", pl.soft_limit(spec.pose_class)), ("hard", c["pen_hard_m"])):
        value = pen[name + "_p95_m"]
        if not pen.get("surfaces"):
            add("penetration:" + name, value <= lim, max(value - lim, 0.0))
    for name, s in pen.get("surfaces", {}).items():
        add("penetration_surface:" + name, s["ok"], max(s["p95_m"] - s["threshold_m"], 0.0))
    if spec.pose_class == "recline":
        # Только torso_back→спинка. Контакт руки/стены не заменяет спину.
        gap = ev.get("back_contact_gap_m")
        hard_ok = all(s["ok"] for n, s in pen.get("surfaces", {}).items() if n.startswith("hard"))
        hard_ok &= pen["hard_p95_m"] <= c["pen_hard_m"] if not pen.get("surfaces") else True
        add("back_contact", gap is not None and gap <= c["contact_tol"] and hard_ok,
            max(gap - c["contact_tol"], 0.0) if gap is not None else float("inf"))
    add("height_contact", not (height_info or {}).get("height_conflict", False),
        (height_info or {}).get("penetration_excess_m", 0.0))
    add("support_contact", ev["support_ok"], 0.0 if ev["support_ok"] else
        max(abs(v) for v in ev["support_part_gaps_m"].values()) if ev["support_part_gaps_m"] else float("inf"))
    hull_fraction = ev["footprint"].get("support_hull", {}).get("in_mask_fraction")
    add("footprint", ev["footprint"]["ok"],
        max(c["hull_in_mask_min"] - hull_fraction, 0.0) if hull_fraction is not None else
        (0.0 if ev["footprint"]["ok"] else float("inf")), "area_fraction")
    add("cells", ev["cells_ok"], float(not ev["cells_ok"]), "cells")
    add("frame", ev["frame"]["ok"], float(not ev["frame"]["ok"]), "frame")
    add("refine_limits", limits_ok, float(not limits_ok), "limit")
    fail = next((x for x in checks if not x["ok"]), None)
    status = "ok" if fail is None else ("wrong_surface" if fail["check"] == "support_target" else
                                        "out_of_frame" if fail["check"] == "frame" else "contact_fail")
    flags = list(ev.get("support_contact", {}).get("flags", []))
    partial_flags = []
    if (height_info or {}).get("height_conflict"):
        partial_flags.append("height_conflict")
    if spec.pose_class == "recline" and not next(x["ok"] for x in checks if x["check"] == "back_contact"):
        partial_flags.append("no_back_contact")
    flags += partial_flags
    if partial_flags and ev["support_object"] == selected:
        status = "partial"  # явные исходы п.4в / п.5б; проверки остаются красными
    return {"ok": fail is None, "first_failure": fail, "checks": checks, "status": status,
            "flags": flags, "penetration": pen}
