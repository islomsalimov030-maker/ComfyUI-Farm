"""Постановка v2, шаг 2: посадка меша тела с референса в фото локации (чистый numpy/scipy, CPU, без лицензий).

  python tools/scene3d/posev2/pose_fit.py --ref <stem>.ref.npz --plate C:/work/scene3d_work/plate/mainroom3 \
      --target 0.5,0.77 --out <dir>/<id> [--height 1.62] [--contacts pelvis:h,foot.L:h]

Что делает:
 1. контакты референса — вершины тела ближе 3 см к его поверхностям (MoGe), по регионам; тип опоры h/v по нормали;
    без сцены рефа (или с --contacts) — регионы из аргумента;
 2. «вверх» рефа — по нормалям горизонтальных контактов (иначе пол-подобная плоскость, иначе -y камеры);
 3. рост → --height; ракурс = угол «перёд тела ↔ направление на камеру» и угол камеры над тазом в рефе;
 4. начальная посадка: опорный регион (сильнейший по PRIORITY) в точку фото --target (доли кадра), тот же ракурс
    к камере фото; затем least_squares по [yaw, tx, ty, tz, pitch, roll]: контакты на поверхности своего типа,
    нет проникновения в комнату (по карте глубины фото), сильная привязка к гравитации и мягкая к ракурсу/точке;
 5. отчёт и файлы: body.obj (мир плиты: Z вверх, пол z=0), fit.json, overlay.jpg (тело поверх фото).
Поза тела не меняется (только жёсткая посадка). Если опора не садится < 3 см или проникновение > 3 см —
status "pose_mismatch" (нужна поправка позы — следующий шаг, MHR в ComfyUI).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial import cKDTree, Delaunay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body import PRIORITY, facing, height_est, regions  # noqa: E402
from placement_config import section  # noqa: E402

CFG = section("pose_fit")

CONTACT_M = CFG["contact_m"]
# MHR70 -> BODY-25 (первые 18 точек в порядке render_h3_maps.BODY25_NAMES)
MHR_TO_B25 = [0, 69, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3]
CONTACT_DETECT_M = CFG["contact_detect_m"]   # порог поиска контактов на рефе (поза SAM у стоп/кистей бывает на 2–4 см выше поверхности)
# доли массы сегментов (Dempster/Winter, округлённо) — центр масс по регионам, а не по вершинам (голова плотная)
MASS = {"head": 0.08, "torso": 0.30, "pelvis": 0.14, "thigh.L": 0.10, "thigh.R": 0.10, "knee.L": 0.01, "knee.R": 0.01,
        "shin.L": 0.045, "shin.R": 0.045, "foot.L": 0.014, "foot.R": 0.014, "arm.L": 0.027, "arm.R": 0.027,
        "forearm.L": 0.016, "forearm.R": 0.016, "hand.L": 0.006, "hand.R": 0.006}


# ------------------------------------------------------------------------------------------------ geometry helpers
def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_xyz(yaw, pitch, roll):
    cx, sx, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rot_z(yaw) @ Ry @ Rx


def signed_angle(a, b):
    """угол от a к b вокруг +Z (рад), a, b — горизонтальные."""
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def load_ref(path):
    d = np.load(path, allow_pickle=True)
    if "verts" in d.files:                                            # ref_scene.py
        sp = d["scene_pts"]
        sn = d["scene_nrm"] if d["scene_nrm"].size else None
        return dict(**{k: d[k] for k in ("part_labels", "skin_weights", "joint_names") if k in d},
                    V=d["verts"].astype(float), F=d["faces"], kp=d["kp3d"].astype(float), S=sp.astype(float),
                    N=None if sn is None else sn.astype(float), W=int(d["W"]), H=int(d["H"]),
                    f=float(d["f"]), cx=float(d["cx"]), cy=float(d["cy"]), kp2d=d["kp2d"].astype(float))
    t = d["pred_cam_t"]                                               # pose_v2_probe.py (без сцены)
    kp = d["pred_keypoints_3d"] + t
    k2 = d["pred_keypoints_2d"]
    x, y = kp[:, 0] / kp[:, 2], kp[:, 1] / kp[:, 2]
    A = np.zeros((2 * len(x), 3))
    A[:len(x), 0], A[:len(x), 1], A[len(x):, 0], A[len(x):, 2] = x, 1, y, 1
    f, cx, cy = np.linalg.lstsq(A, np.concatenate([k2[:, 0], k2[:, 1]]), rcond=None)[0]
    return dict(V=(d["pred_vertices"] + t).astype(float), F=d["top.faces"], kp=kp.astype(float),
                S=np.zeros((0, 3)), N=None, W=int(round(2 * cx)), H=int(round(2 * cy)), f=f, cx=cx, cy=cy,
                kp2d=k2.astype(float))


def fit_input_hash(ref_path, plate_dir, extra):
    """Хэш входа fit: содержимое ref.npz + plate.json + параметры (shot_spec, точка, ячейка зоны …)."""
    import hashlib
    h = hashlib.sha1()
    h.update(Path(ref_path).read_bytes())
    h.update((Path(plate_dir) / "plate.json").read_bytes())
    h.update(json.dumps(extra, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


def load_obj_mesh(path):
    vs, fs = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                q = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
                for k in range(1, len(q) - 1):
                    fs.append([q[0], q[k], q[k + 1]])
    return np.array(vs), np.array(fs, int)


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(n, CFG["vertex_normal_eps"]), n[:, 0] > CFG["vertex_normal_eps"]


class Plate:
    def __init__(self, d):
        d = Path(d)
        self.info = json.loads((d / "plate.json").read_text(encoding="utf-8"))
        self.R = np.array(self.info["R_world_from_cv"], float)
        self.h = float(self.info["cam_height_m"])
        self.C = np.array([0, 0, self.h])
        dep = np.asarray(Image.open(d / "depth_loc.png"), np.float64) / 65535.0
        lo, hi = self.info["inv_norm"]["lo"], self.info["inv_norm"]["hi"]
        inv = lo + dep * (hi - lo)
        self.valid = dep > CFG["depth_valid_min"]
        self.Z = np.where(self.valid, 1.0 / np.maximum(inv, CFG["inverse_depth_min"]), np.inf)
        self.Hp, self.Wp = dep.shape
        self.fx = (self.Wp / 2) / math.tan(math.radians(self.info["fov_x_deg"]) / 2)
        self.cx, self.cy = self.Wp / 2, self.Hp / 2
        V, F = load_obj_mesh(d / "plate_mesh.obj")
        self.V, self.F, self.dir = V, F, d
        n, ok = vertex_normals(V, F)
        to_cam = self.C - V
        n[np.einsum("ij,ij->i", n, to_cam) < 0] *= -1                      # нормали к камере
        self.P, self.Nrm = V[ok], n[ok]
        hor = np.abs(self.Nrm[:, 2]) > CFG["horizontal_normal_min"]
        ver = np.abs(self.Nrm[:, 2]) < CFG["vertical_normal_max"]
        self.trees = {"h": (cKDTree(self.P[hor]), self.P[hor], self.Nrm[hor]),
                      "v": (cKDTree(self.P[ver]), self.P[ver], self.Nrm[ver]),
                      "any": (cKDTree(self.P), self.P, self.Nrm)}
        rgb = d / "plate_rgb.png"
        self.rgb = Image.open(rgb).convert("RGB") if rgb.is_file() else None

    def project(self, X):
        c = (X - self.C) @ self.R                                          # world -> cv (R ортонормирована)
        z = c[:, 2]
        return self.fx * c[:, 0] / z + self.cx, self.fx * c[:, 1] / z + self.cy, z

    def depth_at(self, u, v):
        iu = np.clip(np.round(u).astype(int), 0, self.Wp - 1)
        iv = np.clip(np.round(v).astype(int), 0, self.Hp - 1)
        inside = (u >= 0) & (u < self.Wp) & (v >= 0) & (v < self.Hp)
        z = self.Z[iv, iu]
        return np.where(inside, z, np.inf)

    def seat_point(self, fu, fv):
        """Пиксель сиденья/кровати → верх сиденья. Разметка часто ложится на пол под сиденьем или на его переднюю
        стенку: тогда идём вверх по столбцу до горизонтальной поверхности 0.25–1.0 м. Нет такой — пол (сидя на полу/коврике)."""
        tree, Pp, N = self.trees["any"]
        horiz = lambda Q: N[tree.query(Q)[1]][2] > CFG["seat_normal_min"]  # noqa: E731
        P = self.pixel_point(fu, fv)
        if CFG["seat_height_m"][0] <= P[2] <= CFG["seat_height_m"][1] and horiz(P):
            return P, fv
        for i in range(1, CFG["seat_scan_steps"]):
            vv = fv - i * CFG["seat_scan_step_frac"]
            try:
                Q = self.pixel_point(fu, vv)
            except SystemExit:
                continue
            if CFG["seat_height_m"][0] <= Q[2] <= CFG["seat_height_m"][1] and horiz(Q):
                return Q, vv
            if Q[2] > CFG["seat_scan_max_h"]:                                                  # ушли выше мебели
                break
        return P, fv

    def pixel_point(self, fu, fv):
        u, v = fu * self.Wp, fv * self.Hp
        z = float(self.depth_at(np.array([u]), np.array([v]))[0])
        if not np.isfinite(z):
            raise SystemExit(f"точка {fu},{fv} вне глубины плиты")
        c = np.array([(u - self.cx) / self.fx * z, (v - self.cy) / self.fx * z, z])
        return self.R @ c + self.C


def penetration(plate, X):
    """Глубина проникновения (м, >0) по нормали поверхности: кандидат — точка за видимой поверхностью фото
    (по глубине, 0..35 см), величина — по нормали ближайшей точки поверхности (а не вдоль луча камеры)."""
    u, v, z = plate.project(X)
    zp = plate.depth_at(u, v)
    beh = z - zp
    cand = (beh > 0) & (beh < CFG["penetration_depth_max_m"])
    pen = np.zeros(len(X))
    if cand.any():
        tree, P, N = plate.trees["any"]
        _, j = tree.query(X[cand])
        sd = np.einsum("ij,ij->i", X[cand] - P[j], N[j])
        pen[cand] = np.maximum(-sd, 0.0)
    return pen


# ------------------------------------------------------------------------------------------------ reference analysis
def ref_planes(S, max_planes=CFG["ref_max_planes"], thr=CFG["ref_plane_thr"], seed=0, body=None):
    """Плоскости сцены рефа (RANSAC, последовательно): пол, кровать, сиденье, стены. Отдельные «летающие» точки
    на краю силуэта и остатки тела в MoGe в плоскости не собираются — поэтому контакты считаем только с плоскостями."""
    rng = np.random.default_rng(seed)
    pts = S.copy()
    planes = []
    min_in = max(CFG["ref_plane_min_points"], int(CFG["ref_plane_min_frac"] * len(S)))
    for _ in range(max_planes):
        if len(pts) < min_in:
            break
        best = None
        for _ in range(CFG["ref_plane_trials"]):
            q = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(q[1] - q[0], q[2] - q[0])
            if np.linalg.norm(n) < CFG["ref_plane_normal_eps"]:
                continue
            n /= np.linalg.norm(n)
            inl = np.abs((pts - q[0]) @ n) < thr
            if best is None or inl.sum() > best[1].sum():
                best = (n, inl)
        if best is None or best[1].sum() < min_in:
            break
        P = pts[best[1]]
        c = P.mean(0)
        n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
        if n @ (-c) < 0:                                                  # нормаль к камере (камера в начале)
            n = -n
        pts = pts[~best[1]]
        if body is not None and np.median(body.query(P)[0]) < CFG["ref_plane_body_exclude_m"]:     # «плоскость» из остатков тела — не сцена
            continue
        sub = P[rng.choice(len(P), min(len(P), CFG["ref_plane_sample_points"]), replace=False)]
        planes.append({"n": n, "d": float(-n @ c), "tree": cKDTree(sub), "n_in": int(len(P))})
    return planes


def ref_contacts(ref, reg, forced, up=None, planes=None):
    V = ref["V"]
    out = {}
    if forced:
        for item in forced:
            name, kind, surf = (item.split(":") + ["h", "s"])[:3] if item.count(":") >= 1 else (item, "h", "s")
            idx = np.where(reg == name)[0]
            if len(idx):
                out[name] = {"idx": idx, "kind": kind, "normal": None, "n": int(len(idx)), "surface": surf}
        return out
    if not planes:
        return out
    for pi, pl in enumerate(planes):
        h = V @ pl["n"] + pl["d"]                                         # высота над плоскостью (+ со стороны камеры)
        near = (h > -CFG["ref_contact_below_m"]) & (h < CONTACT_DETECT_M)
        if near.sum() == 0:
            continue
        dxy, _ = pl["tree"].query(V[near])                                # в пределах самой поверхности
        idx_all = np.where(near)[0][dxy < CFG["ref_contact_surface_m"]]
        nz = abs(pl["n"] @ up) if up is not None else 1.0
        kind = "h" if nz > CFG["ref_contact_horizontal_min"] else ("v" if nz < CFG["ref_contact_vertical_max"] else "any")
        for name in np.unique(reg[idx_all]):
            idx = idx_all[reg[idx_all] == name]
            if len(idx) < max(CFG["ref_contact_min_points"], int(CFG["ref_contact_min_frac"] * (reg == name).sum())):
                continue
            prev = out.get(str(name))
            if prev is None or len(idx) > prev["n"]:
                out[str(name)] = {"idx": idx, "kind": kind, "normal": pl["n"], "n": int(len(idx)),
                                  "surface": f"p{pi}"}
    return out


def plane_up(V, contacts, up):
    """«Вверх» по самым низким точкам горизонтальных контактов одной поверхности (≥ 3 точки — плоскость,
    2 — только наклон вдоль линии между ними). Две итерации: «низ» считается относительно текущего up."""
    groups = {}
    for k, c in contacts.items():
        if c.get("kind") == "h":
            groups.setdefault(c.get("surface", "s"), []).append(c["idx"])
    best = max(groups.values(), key=len, default=[])
    if len(best) < 2:
        return up, False
    for _ in range(CFG["plane_up_iters"]):
        cents = []
        for idx in best:
            h = V[idx] @ up
            low = idx[h <= np.percentile(h, CFG["plane_up_percentile"])]
            cents.append(V[low].mean(0))
        C = np.array(cents)
        if len(C) >= 3:
            n = np.linalg.svd(C - C.mean(0))[2][-1]
        else:
            d = C[1] - C[0]
            d /= np.linalg.norm(d)
            n = up - (up @ d) * d
        n /= np.linalg.norm(n)
        up = n if n @ up > 0 else -n
    return up, True


def ref_up(ref, planes, contacts):
    """«Вверх» рефа: самая большая плоскость с нормалью в пределах 45° от «вверх» камеры (пол, кровать);
    иначе — плоскость по низу заданных контактов; иначе — «вверх» камеры."""
    cam_up = np.array([0, -1.0, 0])
    cands = []
    for pl in planes or []:
        n = pl["n"] if pl["n"] @ cam_up > 0 else -pl["n"]
        if n @ cam_up > math.cos(math.radians(CFG["ref_up_max_deg"])):
            cands.append((pl["n_in"], n))
    if cands:
        return max(cands, key=lambda t: t[0])[1], "plane"
    up, ok = plane_up(ref["V"], contacts, cam_up)
    return (up, "contact_plane") if ok else (up, "camera")


# ------------------------------------------------------------------------------------------------ fit
def fit(ref, plate, target, height, forced=None, log=print, seat=False):
    V0, kp0 = ref["V"], ref["kp"]
    reg = regions(V0, kp0)
    planes = ref_planes(ref["S"], body=cKDTree(V0)) if (len(ref["S"]) and not forced) else []
    forced_c = ref_contacts(ref, reg, forced) if forced else {}
    up, up_src = ref_up(ref, planes, forced_c)
    contacts = forced_c if forced else ref_contacts(ref, reg, None, up, planes)
    f = np.array([0, 0, 1.0]) - up[2] * up
    f /= np.linalg.norm(f)
    Rref = np.stack([np.cross(f, up), f, up])
    s = height / height_est(kp0)
    V = (V0 @ Rref.T) * s
    kp = (kp0 @ Rref.T) * s
    Zup = np.array([0, 0, 1.0])
    for c in contacts.values():                       # заданные вручную контакты: только «низ» региона по вертикали
        if c["normal"] is None and c["kind"] == "h":
            h = V[c["idx"], 2]
            c["idx"] = c["idx"][h <= np.percentile(h, 15)]
    for c in contacts.values():
        if c["kind"] is None:
            nz = abs((Rref @ c["normal"])[2])
            c["kind"] = "h" if nz > 0.8 else ("v" if nz < 0.35 else "any")
    # ракурс рефа (камера в начале координат, масштаб s на углы не влияет)
    pel = (kp[9] + kp[10]) / 2
    fwd = facing(kp, Zup)
    to_cam = -pel.copy()
    to_cam[2] = 0
    to_cam /= np.linalg.norm(to_cam)
    rel = signed_angle(to_cam, fwd) if fwd is not None else 0.0
    elev_ref = math.degrees(math.atan2(-pel[2], math.hypot(pel[0], pel[1])))
    # опорный регион
    if seat and not any(PRIORITY.get(k, 9) <= 1 for k, c in contacts.items() if c["kind"] == "h"):
        # строка таблицы «сиденье»: опора — таз на сиденье, даже если на рефе его контакт не найден
        idx = np.where(reg == "pelvis")[0]
        h = V[idx, 2]
        contacts["pelvis"] = {"idx": idx[h <= np.percentile(h, 15)], "kind": "h", "normal": None, "n": int(len(idx)),
                              "surface": "seat"}
        log("[fit] сиденье по таблице: опора — таз")
    hor = {k: c for k, c in contacts.items() if c["kind"] == "h"}          # опора — только горизонтальные контакты
    pool = hor or contacts
    if pool:
        best = min(PRIORITY.get(k, 9) for k in pool)
        support = [k for k in pool if PRIORITY.get(k, 9) == best]
    else:
        support = []
    if support:
        sup_idx = np.concatenate([contacts[k]["idx"] for k in support])
    else:                                                               # без контактов: самые низкие 2 % вершин
        sup_idx = np.argsort(V[:, 2])[: max(20, len(V) // 50)]
        support = ["lowest"]
    if seat:
        T, v_used = plate.seat_point(*target)
        if v_used != target[1]:
            log(f"[fit] пиксель сиденья {target} не на верх сиденья — поднят до v {v_used:.3f} (z {T[2]:.2f} м)")
    else:
        T = plate.pixel_point(*target)
    # начальная посадка: ракурс к камере фото как в рефе
    to_cam_p = plate.C - T
    to_cam_p[2] = 0
    to_cam_p /= np.linalg.norm(to_cam_p)
    want = rot_z(rel) @ to_cam_p
    yaw0 = signed_angle(fwd, want) if fwd is not None else 0.0
    R0 = rot_z(yaw0)
    V = V @ R0.T
    kp = kp @ R0.T
    pivot = V[sup_idx].mean(0)
    V = V - pivot + T
    kp = kp - pivot + T
    pivot = T.copy()

    tilt = 0.05 if up_src in ("plane", "contact_plane") else 0.3     # насколько доверяем «вверх» рефа
    rng = np.random.default_rng(0)
    pen_idx = rng.choice(len(V), min(2500, len(V)), replace=False)
    groups = []
    for k, c in contacts.items():
        idx = c["idx"] if len(c["idx"]) <= 60 else rng.choice(c["idx"], 60, replace=False)
        groups.append((k, idx, c["kind"], k in support))
    if support == ["lowest"]:
        groups.append(("lowest", sup_idx[:60], "h", True))

    def transform(p, X):
        return (X - pivot) @ rot_xyz(p[0], p[4], p[5]).T + pivot + p[1:4]

    def residuals(p):
        X = transform(p, V)
        res = []
        for k, idx, kind, is_sup in groups:
            tree, P, N = plate.trees[kind if kind in ("h", "v") else "any"]
            d, j = tree.query(X[idx])
            sd = np.einsum("ij,ij->i", X[idx] - P[j], N[j])
            k_n = 1.0 / math.sqrt(len(idx))                                # каждый регион весит как одна невязка
            if is_sup:
                res.append(3.0 * k_n * sd / 0.01)
            else:
                res.append(k_n * np.where(d < 0.4, sd / 0.03, 0.0))      # вторичные: мягко, только рядом с поверхностью
        pen = np.maximum(penetration(plate, X[pen_idx]) - 0.005, 0.0)
        res.append(10.0 / math.sqrt(len(pen_idx)) * pen / 0.01)
        res.append(np.array([(p[0]) / 0.35, p[4] / tilt, p[5] / tilt, np.hypot(p[1], p[2]) / 0.25]))
        return np.concatenate(res)

    sol = least_squares(residuals, np.zeros(6), loss="soft_l1", f_scale=1.0, diff_step=1e-3, max_nfev=400)
    p = sol.x
    X = transform(p, V)
    KP = transform(p, kp)

    # ---- отчёт
    rep_c = {}
    for k, c in contacts.items():
        tree, P, N = plate.trees[c["kind"] if c["kind"] in ("h", "v") else "any"]
        d, j = tree.query(X[c["idx"]])
        sd = np.einsum("ij,ij->i", X[c["idx"]] - P[j], N[j])
        rep_c[k] = {"kind": c["kind"], "support": k in support, "gap_cm": round(float(np.median(sd)) * 100, 1),
                    "min_cm": round(float(sd.min()) * 100, 1), "nearest_surface_cm": round(float(d.min()) * 100, 1)}
    pen = penetration(plate, X)
    max_pen = float(pen.max()) * 100
    if support == ["lowest"]:
        sup_gap = float(np.median(X[sup_idx, 2] - T[2])) * 100
    else:
        gaps_all = []
        for k in support:                                 # зазор опоры — по самым низким 30 % вершин опорных регионов
            tree, P, N = plate.trees[contacts[k]["kind"] if contacts[k]["kind"] in ("h", "v") else "any"]
            Xi = X[contacts[k]["idx"]]
            d, j = tree.query(Xi)
            gaps_all.append(np.einsum("ij,ij->i", Xi - P[j], N[j]))
        g = np.concatenate(gaps_all)
        sup_gap = abs(float(np.median(g[g <= np.percentile(g, 30)]))) * 100
    # равновесие: центр масс над опорным многоугольником (горизонтальные контакты в пределах 3 см)
    pts = []
    for k, c in contacts.items():
        if c["kind"] == "h" and abs(rep_c[k]["gap_cm"]) < 3:
            pts.append(X[c["idx"], :2])
    com = sum(MASS.get(r, 0.0) * X[reg == r].mean(0) for r in np.unique(reg) if (reg == r).any()) / \
        sum(MASS.get(r, 0.0) for r in np.unique(reg))
    balance = None
    if pts:
        P2 = np.concatenate(pts)
        inside = False
        try:
            inside = bool(Delaunay(P2).find_simplex(com[:2]) >= 0)
        except Exception:
            pass
        balance = inside or bool(np.min(np.linalg.norm(P2 - com[:2], axis=1)) < 0.04)
    pel_p = (KP[9] + KP[10]) / 2
    fwd_p = facing(KP, Zup)
    tc = plate.C - pel_p
    tc[2] = 0
    tc /= np.linalg.norm(tc)
    rel_p = signed_angle(tc, fwd_p) if fwd_p is not None else 0.0
    elev_p = math.degrees(math.atan2(plate.h - pel_p[2], math.hypot(*(plate.C - pel_p)[:2])))
    xs, ys = ref["V"][:, 0] / ref["V"][:, 2] * ref["f"] + ref["cx"], ref["V"][:, 1] / ref["V"][:, 2] * ref["f"] + ref["cy"]
    status = "ok" if sup_gap <= 3 and max_pen <= 3 else "pose_mismatch"
    # BODY-25 (первые 18: тело, глаза, уши) в пикселях фото — для PCK, как у манекена; кисти нулями
    ku, kv, kz = plate.project(KP[MHR_TO_B25])
    kzp = plate.depth_at(ku, kv)
    kp25 = [[round(float(a), 1), round(float(b), 1),
             int(0 <= a < plate.Wp and 0 <= b < plate.Hp and z > 0 and z <= zp + 0.02)]
            for a, b, z, zp in zip(ku, kv, kz, kzp)] + [[0.0, 0.0, 0]] * 30
    rep = {"status": status, "support": support, "support_gap_cm": round(sup_gap, 1),
           "max_penetration_cm": round(max_pen, 1), "pen_verts_over_2cm": int((pen > 0.02).sum()),
           "balance_ok": balance, "contacts": rep_c, "up_from": up_src, "ref_planes": len(planes), "height_scale": round(s, 3),
           "angle_to_camera_deg": {"ref": round(math.degrees(rel), 1), "plate": round(math.degrees(rel_p), 1)},
           "camera_elevation_deg": {"ref": round(elev_ref, 1), "plate": round(elev_p, 1)},
           "ref_plan": {"body_h_frac": round(float((ys.max() - ys.min()) / ref["H"]), 3),
                        "body_w_frac": round(float((xs.max() - xs.min()) / ref["W"]), 3)},
           "fit": {"params": [round(float(x), 4) for x in p], "cost": round(float(sol.cost), 3), "nfev": int(sol.nfev)},
           "target_world": [round(float(x), 3) for x in T], "seat": bool(seat),
           "keypoints_body25_px": kp25, "px_size": [plate.Wp, plate.Hp]}
    # состояние для refine.py: аффинное преобразование камера рефа -> мир плиты (X = V0 @ M.T + b) и контакты
    A = np.hstack([V0, np.ones((len(V0), 1))])
    Bm = np.linalg.lstsq(A, X, rcond=None)[0]
    rep["_state"] = {"M": Bm[:3].T, "b": Bm[3], "support": support, "target": T,
                     "contacts": {k: (c["idx"], c["kind"]) for k, c in contacts.items()},
                     "sup_idx": sup_idx}
    log(f"[fit] {status}: опора {support} {sup_gap:.1f} см, проникновение {max_pen:.1f} см, "
        f"ракурс {rep['angle_to_camera_deg']}, камера {rep['camera_elevation_deg']}, равновесие {balance}")
    return X, rep, reg


def write_outputs(out, X, F, rep, plate, reg, contacts_names):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    st = rep.pop("_state", None)
    if st is not None:
        np.savez_compressed(out / "fit_state.npz", M=st["M"], b=st["b"], target=st["target"], sup_idx=st["sup_idx"],
                            support=np.array(st["support"]),
                            **{f"c_{k}": idx for k, (idx, kind) in st["contacts"].items()},
                            kinds=np.array([f"{k}:{kind}" for k, (idx, kind) in st["contacts"].items()]))
    with open(out / "body.obj", "w") as fh:
        fh.write("o body\n")
        fh.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in X)
        fh.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in np.asarray(F))
    (out / "fit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    if plate.rgb is not None:
        im = plate.rgb.resize((plate.Wp, plate.Hp)).copy()
        dr = ImageDraw.Draw(im)
        u, v, z = plate.project(X)
        zp = plate.depth_at(u, v)
        vis = z <= zp + 0.02
        isc = np.isin(reg, list(contacts_names))
        for k in range(0, len(X), 6):
            col = (255, 60, 60) if isc[k] else ((0, 210, 255) if vis[k] else (0, 70, 110))
            dr.point((float(u[k]), float(v[k])), fill=col)
        im.save(out / "overlay.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--plate", required=True)
    ap.add_argument("--target", required=True, help="u,v в долях кадра фото: куда встаёт опорный регион")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=float, default=1.62)
    ap.add_argument("--contacts", default="", help="переопределить контакты: pelvis:h,foot.L:h,hand.R:v")
    a = ap.parse_args()
    ref = load_ref(a.ref)
    plate = Plate(a.plate)
    forced = [x for x in a.contacts.split(",") if x] or None
    X, rep, reg = fit(ref, plate, tuple(float(x) for x in a.target.split(",")), a.height, forced)
    write_outputs(a.out, X, ref["F"], rep, plate, reg, rep["contacts"].keys())


if __name__ == "__main__":
    main()
