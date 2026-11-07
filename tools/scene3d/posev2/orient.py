"""Постановка v2.4 (07.11f, п.7 Р3): проверка ориентации корпуса (ось таз→шея) и кандидаты поворота корня.
Пороги по телу — через body_dims (H); углы — градусы.
"""
import math

import numpy as np

UPRIGHT = ("stand", "kneel", "sit_edge", "sit_tucked", "sit_floor", "crouch")
FLAT = ("lie", "all_fours")
MAX_FROM_VERTICAL = 45.0
RECLINE_ELEV = (20.0, 70.0)
FLAT_HEAD_H = 0.18          # × H
CONFLICT_DEG = 20.0
# MHR70: 0 nose, 1–4 глаза/уши, 9/10 бёдра, 69 шея
HEAD = [0, 1, 2, 3, 4]


def axis(KP):
    pel = (KP[9] + KP[10]) / 2
    return pel, KP[69], KP[HEAD].mean(0)


def check(KP, pose_class, H):
    pel, neck, head = axis(KP)
    a = neck - pel
    from_v = math.degrees(math.acos(np.clip(a[2] / max(np.linalg.norm(a), 1e-9), -1, 1)))
    elev = 90.0 - from_v
    dh = float(head[2] - pel[2])
    if pose_class in UPRIGHT:
        ok = from_v <= MAX_FROM_VERTICAL and dh > 0
        rule = f"от вертикали ≤ {MAX_FROM_VERTICAL:.0f}°, голова выше таза"
    elif pose_class == "recline":
        ok = RECLINE_ELEV[0] <= elev <= RECLINE_ELEV[1] and dh > 0
        rule = f"над горизонталью {RECLINE_ELEV[0]:.0f}–{RECLINE_ELEV[1]:.0f}°, голова выше таза"
    elif pose_class in FLAT:
        ok = abs(dh) <= FLAT_HEAD_H * H
        rule = f"|h(голова) − h(таз)| ≤ {FLAT_HEAD_H}·H = {FLAT_HEAD_H * H:.2f} м"
    else:
        raise ValueError(pose_class)
    return {"ok": bool(ok), "from_vertical_deg": round(from_v, 1), "elev_deg": round(elev, 1),
            "head_minus_pelvis_m": round(dh, 3), "rule": rule}


def rot_axis(u, ang):
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def candidates(kp, cam_axis):
    """kp — точки тела после начального поворота (мир, Z вверх); cam_axis — ось камеры рефа в том же мире.
    -> [(имя, R 3×3)] — поворот корня вокруг центра таза."""
    lat = kp[9] - kp[10]
    return [("identity", np.eye(3)), ("cam180", rot_axis(cam_axis, math.pi)),
            ("up180", rot_axis([0, 0, 1.0], math.pi)), ("lat180", rot_axis(lat, math.pi))]


def reproj_error(KP_world, G0, ref, idx):
    """Ошибка репроекции на 2D-точки рефа: тело переводится в систему камеры рефа обратным к посадке «исходного»
    кандидата преобразованием G0 = (M, b) (X = M v + b), проецируется интринсиками рефа; центроиды совмещаются
    (сдвиг посадки не штрафуется). -> доля высоты рамки 2D-точек рефа."""
    M, b = G0
    Y = np.linalg.solve(M, (KP_world[idx] - b).T).T
    z = np.maximum(Y[:, 2], 1e-6)
    uv = np.stack([ref["f"] * Y[:, 0] / z + ref["cx"], ref["f"] * Y[:, 1] / z + ref["cy"]], 1)
    k2 = ref["kp2d"][idx]
    uv = uv - uv.mean(0) + k2.mean(0)
    bh = max(float(k2[:, 1].max() - k2[:, 1].min()), 1.0)
    return float(np.linalg.norm(uv - k2, axis=1).mean() / bh)
