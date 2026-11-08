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
    checks.append(("roll", max(abs(roll) - ori.ROLL_MAX, 0.0) if pose_class != "lie" else 0.0, "deg"))
    got = ori.side_to_camera(K, plate.C)
    checks.append(("camera_side", float(bool(side_ref) and got != side_ref), "side"))
    if window is not None:
        f = ori.forward(K)
        yaw = math.degrees(math.atan2(f[1], f[0])) if f is not None else None
        checks.append(("yaw_window", float("inf") if yaw is None else
                       max(abs(ori.wrap(yaw - window)) - ori.YAW_WINDOW, 0.0), "deg"))
    return [{"check": n, "excess": float(e), "unit": u, "ok": bool(e <= (eps if u == "deg" else 0))}
            for n, e, u in checks]


def clamp_root(X, K, pose_class, plate, side_ref, window=None):
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
    return X, K


def height_fit(X, K, groups, S, plate, spec, treg, soft, H):
    """Подъём до p95 по материалам в пороге, затем контакт каждой опоры.

    При невозможном контакте — перебор наклона/крена ±10° шаг2°,
    без выхода из диапазона класса. Неудача не ослабляет валидатор.
    """
    import place as pl
    c = section("place")
    origin = (K[9] + K[10]) / 2
    support = [(idx, surf) for _, idx, surf, role in groups if role == "support" and surf in S]

    def settle(V, J):
        if not support:
            return V, J, {"shift_z_m": 0.0, "max_gap_m": float("inf"), "penetration_ok": False}
        # Сперва на контакт: опустить до самого высокого из требуемых сдвигов.
        dz = -min(float(np.median(pl._sd(S, surf, V[idx])[0])) for idx, surf in support)
        V, J = V + [0, 0, dz], J + [0, 0, dz]
        pen = pl.penetration_stats(plate, V, treg, soft)
        # Существующий линейный шаг высоты / предел H взяты из YAML.
        moved = 0.0
        while not pen["ok"] and moved < c["shift_max_h"] * H:
            step = min(max(c["shift_step"], pen["excess_m"]), c["shift_max_h"] * H - moved)
            previous = pen["excess_m"]
            V, J = V + [0, 0, step], J + [0, 0, step]
            dz += step
            moved += step
            pen = pl.penetration_stats(plate, V, treg, soft)
            if not pen["ok"] and pen["excess_m"] >= previous:
                break  # вертикальный подъём не устраняет проникновение в стену
        gaps = [float(np.median(pl._sd(S, surf, V[idx])[0])) for idx, surf in support]
        return V, J, {"shift_z_m": float(dz), "max_gap_m": max(gaps), "part_gaps_m": gaps,
                      "penetration_ok": pen["ok"], "penetration_excess_m": pen["excess_m"]}

    V, J, info = settle(X.copy(), K.copy())
    info.update(tilt_adjust_deg=0.0, roll_adjust_deg=0.0)
    best = (not info["penetration_ok"], max(abs(v) for v in info.get("part_gaps_m", [float("inf")])))
    if info["penetration_ok"] and info["max_gap_m"] <= c["sup_gap_cm"] / 100:
        return V, J, info
    f = ori.forward(K)
    if f is None:
        return V, J, info
    lateral = np.cross(f, ori.UP)
    vals = np.arange(-c["gap_sweep_deg"], c["gap_sweep_deg"] + c["gap_sweep_step_deg"] / 2,
                     c["gap_sweep_step_deg"])
    for tilt in vals:
        for roll in vals:
            R = ori.rot_axis(f, math.radians(float(roll))) @ ori.rot_axis(lateral, math.radians(float(tilt)))
            A, B = ori.rot_about(X, origin, R), ori.rot_about(K, origin, R)
            lo, hi = ori.TILT_RANGE[spec.pose_class]
            if not lo <= ori.tilt_deg(B) <= hi:
                continue
            if spec.pose_class != "lie" and abs(ori.roll_deg(B)) > ori.ROLL_MAX:
                continue
            A, B, ii = settle(A, B)
            key = (not ii["penetration_ok"], max(abs(v) for v in ii.get("part_gaps_m", [float("inf")])))
            if key < best:
                best, V, J = key, A, B
                info = dict(ii, tilt_adjust_deg=float(tilt), roll_adjust_deg=float(roll))
    return V, J, info


def validate(K, ev, spec, plate, side_ref, window, limits_ok):
    c = section("place")
    checks = orientation_checks(K, spec.pose_class, plate, side_ref, window)
    def add(name, ok, excess=0.0, unit="m"):
        checks.append({"check": name, "ok": bool(ok), "excess": float(excess), "unit": unit})
    add("support_target", ev["support_object"] == spec.target_object,
        float(ev["support_object"] != spec.target_object), "surface")
    for name, gap in ev["support_part_gaps_m"].items():
        add("support_gap:" + name, gap <= c["sup_gap_cm"] / 100,
            max(gap - c["sup_gap_cm"] / 100, 0.0))
    for name, lim in (("soft", c["pen_soft_m"]), ("hard", c["pen_hard_m"])):
        value = ev["penetration"][name + "_p95_m"]
        add("penetration:" + name, value <= lim, max(value - lim, 0.0))
    for name, s in ev["penetration"].get("surfaces", {}).items():
        add("penetration_surface:" + name, s["ok"], max(s["p95_m"] - s["threshold_m"], 0.0))
    if spec.pose_class == "recline":
        # Back is a required contact, not an optional table secondary. Use the
        # unrounded distance; penetration limits are checked per surface above.
        gap = ev.get("back_gap_m")
        if gap is None:
            gap = ev.get("wall_gap_m")
        add("back_contact", gap is not None and gap <= c["sup_gap_cm"] / 100,
            max(gap - c["sup_gap_cm"] / 100, 0.0) if gap is not None else float("inf"))
    add("support_contact", ev["support_ok"], 0.0 if ev["support_ok"] else
        max(abs(v) for v in ev["support_part_gaps_m"].values()) if ev["support_part_gaps_m"] else float("inf"))
    add("footprint", ev["footprint"]["ok"], ev["footprint"].get("n_out", 0), "vertices")
    add("cells", ev["cells_ok"], float(not ev["cells_ok"]), "cells")
    add("frame", ev["frame"]["ok"], float(not ev["frame"]["ok"]), "frame")
    add("refine_limits", limits_ok, float(not limits_ok), "limit")
    fail = next((x for x in checks if not x["ok"]), None)
    status = "ok" if fail is None else ("wrong_surface" if fail["check"] == "support_target" else
                                        "out_of_frame" if fail["check"] == "frame" else "contact_fail")
    return {"ok": fail is None, "first_failure": fail, "checks": checks, "status": status}
