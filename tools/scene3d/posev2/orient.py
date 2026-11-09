"""Постановка v2.4 (07.11f, п.7 Р3): проверка ориентации корпуса (ось таз→шея) и кандидаты поворота корня.
Пороги по телу — через body_dims (H); углы — градусы.
"""
import math

import numpy as np

from placement_config import section

CFG = section("orient")

UPRIGHT = ("stand", "kneel", "sit_edge", "sit_tucked", "sit_floor", "crouch")
FLAT = ("lie", "all_fours")
MAX_FROM_VERTICAL = CFG["max_from_vertical"]
RECLINE_ELEV = CFG["recline_elev"]
FLAT_HEAD_H = CFG["flat_head_h"]
CONFLICT_DEG = CFG["conflict_deg"]
# MHR70: 0 nose, 1–4 глаза/уши, 9/10 бёдра, 69 шея
HEAD = [0, 1, 2, 3, 4]


SHOULDER_AXIS_CLASSES = ("stand", "kneel", "crouch")


def torso_top(KP, pose_class=None):
    """MHR shoulders 5/6, not the head/neck estimate, for root clamps."""
    return (KP[5] + KP[6]) / 2 if pose_class in SHOULDER_AXIS_CLASSES else KP[69]


def axis(KP, pose_class=None):
    pel = (KP[9] + KP[10]) / 2
    return pel, torso_top(KP, pose_class), KP[HEAD].mean(0)


def check(KP, pose_class, H):
    pel, neck, head = axis(KP, pose_class)
    a = neck - pel
    from_v = math.degrees(math.acos(np.clip(a[2] / max(np.linalg.norm(a), CFG["norm_eps"]), -1, 1)))
    elev = 90.0 - from_v
    dh = float(head[2] - pel[2])
    if pose_class in UPRIGHT:
        ok = from_v <= MAX_FROM_VERTICAL and dh > 0
        rule = f"от вертикали ≤ {MAX_FROM_VERTICAL:.0f}°, голова выше таза"
    elif pose_class == "recline":
        ok = RECLINE_ELEV[0] <= elev <= RECLINE_ELEV[1] and dh > 0
        rule = f"над горизонталью {RECLINE_ELEV[0]:.0f}–{RECLINE_ELEV[1]:.0f}°, голова выше таза"
    elif pose_class == "lie":
        lo, hi = CFG["tilt_range"]["lie"]
        ok = lo <= from_v <= hi
        rule = f"ось таз→шея: от вертикали {lo}–{hi}°, крен с рефа без clamp"
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


# ------------------------------------------------------------------------------------------------ 08.11a п.3: сборка
TILT_RANGE = CFG["tilt_range"]
ROLL_MAX = CFG["roll_max"]
YAW_WINDOW = CFG["yaw_window"]
UP = np.array([0, 0, 1.0])
# куда наклонять, если реф почти вертикален, а класс требует больше: назад (recline, lie) / вперёд (прочие)
LEAN_BACK = ("recline", "lie")


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def left_vec(KP):
    return (KP[9] - KP[10]) + (KP[5] - KP[6])


def tilt_deg(KP, pose_class=None):
    """Torso axis from vertical; upright root clamps use shoulder midpoint."""
    a = torso_top(KP, pose_class) - (KP[9] + KP[10]) / 2
    return math.degrees(math.acos(np.clip(a[2] / max(np.linalg.norm(a), CFG["norm_eps"]), -1, 1)))


def roll_deg(KP, pose_class=None):
    """Для lie — полный крен вокруг длинной оси: спина=0, бок=±90, живот=±180.

    Stand/kneel/crouch: боковой наклон оси таз→середина плеч.
    Для остальных классов сохраняется прежний наклон поперечной оси.
    """
    lv = left_vec(KP)
    if pose_class in SHOULDER_AXIS_CLASSES:
        a = torso_top(KP, pose_class) - (KP[9] + KP[10]) / 2
        lat = lv.copy()
        lat[2] = 0
        lat /= max(np.linalg.norm(lat), CFG["norm_eps"])
        return math.degrees(math.atan2(-a @ lat, a[2]))
    if pose_class == "lie":
        a = KP[69] - (KP[9] + KP[10]) / 2
        a = a / max(np.linalg.norm(a), CFG["norm_eps"])
        lv = lv - a * (lv @ a)
        lv = lv / max(np.linalg.norm(lv), CFG["norm_eps"])
        normal = np.cross(lv, a)
        return math.degrees(math.atan2(lv @ UP, normal @ UP))
    return math.degrees(math.asin(np.clip(lv[2] / max(np.linalg.norm(lv), CFG["norm_eps"]), -1, 1)))


def chest(KP):
    """нормаль груди (3D): left × (таз→шея); не зависит от оценки «вверх»."""
    c = np.cross(left_vec(KP), KP[69] - (KP[9] + KP[10]) / 2)
    return c / max(np.linalg.norm(c), CFG["norm_eps"])


CHEST_H_MIN = CFG["chest_h_min"]        # |горизонтальная часть нормали груди| — ниже «перёд» по горизонтали не определён (лёжа на спине)


def forward(KP):
    """горизонтальное «вперёд»: горизонтальная часть нормали груди; если грудь смотрит вверх/вниз — left × up."""
    c = chest(KP)
    f = np.array([c[0], c[1], 0.0])
    if np.linalg.norm(f) >= CHEST_H_MIN:
        return f / np.linalg.norm(f)
    f = np.cross(left_vec(KP), UP)
    f[2] = 0
    n = np.linalg.norm(f)
    return f / n if n > CFG["direction_eps"] else None


SIDE_DEAD = CFG["side_dead"]           # |cos| нормали груди к направлению на камеру — ниже сторона не определена


def side_to_camera(KP, cam):
    """+1 — к камере передом (нормаль груди · направление на камеру > 0, 3D), −1 — спиной; 0 — почти боком
    (|cos| < SIDE_DEAD, ограничение не действует)."""
    to = np.asarray(cam, float) - (KP[9] + KP[10]) / 2
    c = chest(KP) @ (to / np.linalg.norm(to))
    return 0 if abs(c) < SIDE_DEAD else (1 if c > 0 else -1)


def side_for_yaws(KP, angles_deg, to_cam):
    """сторона к камере (3D) для тела, повёрнутого вокруг Z так, что его «вперёд» = angles_deg. -> массив ±1/0."""
    c = chest(KP)
    f = forward(KP)
    f0 = math.degrees(math.atan2(f[1], f[0])) if f is not None else 0.0
    d = np.radians(np.asarray(angles_deg) - f0)
    cx = c[0] * np.cos(d) - c[1] * np.sin(d)
    cy = c[0] * np.sin(d) + c[1] * np.cos(d)
    t = np.asarray(to_cam, float) / np.linalg.norm(to_cam)
    v = cx * t[0] + cy * t[1] + c[2] * t[2]
    return np.where(np.abs(v) < SIDE_DEAD, 0, np.sign(v)).astype(int)


def rot_about(X, c, R):
    return (X - c) @ R.T + c


def tilt_sign_candidate(kp, cam_axis, pose_class):
    """кандидаты ×180° шага 7 — только чтобы выбрать знак наклона рефа: identity / cam180 / lat180, чей наклон
    ближе к диапазону класса (identity при равенстве ±5°). -> (имя, R)."""
    lo, hi = TILT_RANGE[pose_class]
    best = None
    for name, R in candidates(kp, cam_axis):
        if name == "up180":                                   # меняет только yaw — к знаку наклона отношения не имеет
            continue
        c = (kp[9] + kp[10]) / 2
        t = tilt_deg(rot_about(kp, c, R), pose_class)
        dist = max(lo - t, 0.0, t - hi) + (0.0 if name == "identity" else CFG["identity_bias_deg"])
        if best is None or dist < best[0]:
            best = (dist, name, R)
    return best[1], best[2]


def clamp_roll(V, kp, pose_class=None):
    """Rigid root clamp; upright lateral lean comes from pelvis→shoulders."""
    r = roll_deg(kp, pose_class)
    if pose_class in SHOULDER_AXIS_CLASSES:
        if abs(r) <= ROLL_MAX:
            return V, kp
        c = (kp[9] + kp[10]) / 2
        lat = left_vec(kp).copy()
        lat[2] = 0
        if np.linalg.norm(lat) < CFG["direction_eps"]:
            return V, kp  # validator retains the measured failure
        R = rot_axis(np.cross(lat, UP), math.radians(math.copysign(ROLL_MAX, r) - r))
        return rot_about(V, c, R), rot_about(kp, c, R)
    if abs(r) <= ROLL_MAX:
        return V, kp
    c = (kp[9] + kp[10]) / 2
    axis_ = kp[69] - c
    axis_ /= max(np.linalg.norm(axis_), CFG["norm_eps"])
    lv = left_vec(kp)
    parallel = axis_ * (axis_ @ lv)
    a, b, offset = (lv - parallel)[2], np.cross(axis_, lv)[2], parallel[2]
    radius = math.hypot(a, b)
    if radius < CFG["direction_eps"]:
        return V, kp  # вырожденные данные: валидатор не объявляет успех
    target = math.sin(math.radians(math.copysign(ROLL_MAX, r))) * np.linalg.norm(lv)
    if abs(target - offset) > radius:
        return V, kp  # заданный крен недостижим без изменения наклона
    phase = math.atan2(b, a)
    delta = math.acos(np.clip((target - offset) / radius, -1, 1))
    angles = [math.radians(wrap(math.degrees(phase + sign * delta))) for sign in (-1, 1)]
    R = rot_axis(axis_, min(angles, key=abs))
    return rot_about(V, c, R), rot_about(kp, c, R)


def assemble_tilt_roll(V, kp, pose_class):
    """Наклон в диапазоне класса; крен ≤ ±15°, кроме lie (полный крен с рефа).
    Поворот вокруг центра таза.
    -> V, kp, info{tilt_ref, tilt, roll_ref, roll, clamped_deg}."""
    c = (kp[9] + kp[10]) / 2
    t0, r0 = tilt_deg(kp, pose_class), roll_deg(kp, pose_class)
    info = {"tilt_ref_deg": round(t0, 1), "roll_ref_deg": round(r0, 1)}
    if pose_class != "lie":
        V, kp = clamp_roll(V, kp, pose_class)
    lo, hi = TILT_RANGE[pose_class]
    t1 = tilt_deg(kp, pose_class)
    tt = min(max(t1, lo), hi)
    if abs(tt - t1) > CFG["tilt_change_deg"]:
        a = torso_top(kp, pose_class) - c
        k = np.cross(a, UP)
        vertical = t1 < CFG["vertical_deg"]                                   # почти вертикальная ось — сторону наклона задаёт класс
        f = forward(kp)
        if vertical or np.linalg.norm(k) < CFG["direction_eps"]:
            k = np.cross(f, UP) if f is not None else np.array([1.0, 0, 0])
        k = k / np.linalg.norm(k)
        for sgn in (1, -1):
            R = rot_axis(k, sgn * math.radians(tt - t1))
            k2 = rot_about(kp, c, R)
            if abs(tilt_deg(k2, pose_class) - tt) >= CFG["tilt_match_deg"]:
                continue
            if vertical and f is not None:
                lean = (torso_top(k2, pose_class) - c) @ f
                if (pose_class in LEAN_BACK) != (lean < 0):
                    continue
            V, kp = rot_about(V, c, R), k2
            break
    # Наклон выше мог снова внести крен. Финальный clamp сохраняет ось корпуса.
    if pose_class != "lie":
        V, kp = clamp_roll(V, kp, pose_class)
    info.update(tilt_deg=round(tilt_deg(kp, pose_class), 1), roll_deg=round(roll_deg(kp, pose_class), 1))
    info["clamped_deg"] = (0.0 if pose_class == "lie" else
                           round(abs(info["tilt_deg"] - t0) + abs(info["roll_deg"] - r0), 1))
    return V, kp, info


def solve_yaw(want_deg, side_ref, window=None, side_fn=None):
    """yaw «вперёд» тела (мир, градусы): ближайший к want при условиях: сторона к камере = side_ref
    (side_fn(углы) -> ±1/0, 3D по нормали груди; 0 у тела «боком» допускается), и |yaw − window| ≤ 60°.
    Невозможно и то и другое → side_conflict, приоритет у стороны к камере. -> (yaw, side_conflict, in_window)."""
    a = np.arange(-180.0, 180.0, CFG["yaw_step_deg"])
    if side_ref and side_fn is not None:
        side_ok = side_fn(a) == side_ref                    # поворот в «боком»/другую сторону — запрещён
    else:
        side_ok = np.ones_like(a, bool)
    if window is not None:
        dw = np.abs(wrap(a - window)) - YAW_WINDOW
        in_w = dw <= 0
    else:
        dw = np.zeros_like(a)
        in_w = np.ones_like(a, bool)
    dist = np.abs(wrap(a - want_deg))
    ok = side_ok & in_w
    if ok.any():
        k = np.flatnonzero(ok)[np.argmin(dist[ok])]
        return float(a[k]), False, True
    if not side_ok.any():
        return float(want_deg), True, False
    k = np.flatnonzero(side_ok)[np.lexsort((dist[side_ok], dw[side_ok]))[0]]
    return float(a[k]), True, False


def reproj_plate(KP, plate, ref, idx):
    """ошибка формы 2D: проекция точек тела камерой плиты против 2D-точек рефа (центр и высота рамки совмещены).
    -> доля высоты рамки рефа (для orient_conflict: ракурс рефа против ракурса посадки)."""
    u, v, z = plate.project(KP[idx])
    P = np.stack([u, v], 1)
    k2 = ref["kp2d"][idx]
    bh_r = max(float(k2[:, 1].max() - k2[:, 1].min()), CFG["reproj_min_height_px"])
    bh_p = max(float(P[:, 1].max() - P[:, 1].min()), CFG["reproj_min_height_px"])
    P = (P - P.mean(0)) / bh_p
    Q = (k2 - k2.mean(0)) / bh_r
    return float(np.linalg.norm(P - Q, axis=1).mean())


def reproj_error(KP_world, G0, ref, idx):
    """Ошибка репроекции на 2D-точки рефа: тело переводится в систему камеры рефа обратным к посадке «исходного»
    кандидата преобразованием G0 = (M, b) (X = M v + b), проецируется интринсиками рефа; центроиды совмещаются
    (сдвиг посадки не штрафуется). -> доля высоты рамки 2D-точек рефа."""
    M, b = G0
    Y = np.linalg.solve(M, (KP_world[idx] - b).T).T
    z = np.maximum(Y[:, 2], CFG["reproj_depth_eps"])
    uv = np.stack([ref["f"] * Y[:, 0] / z + ref["cx"], ref["f"] * Y[:, 1] / z + ref["cy"]], 1)
    k2 = ref["kp2d"][idx]
    uv = uv - uv.mean(0) + k2.mean(0)
    bh = max(float(k2[:, 1].max() - k2[:, 1].min()), CFG["reproj_min_height_px"])
    return float(np.linalg.norm(uv - k2, axis=1).mean() / bh)
