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

CONTACT_M = 0.03


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
        return dict(V=d["verts"].astype(float), F=d["faces"], kp=d["kp3d"].astype(float), S=sp.astype(float),
                    N=None if sn is None else sn.astype(float), W=int(d["W"]), H=int(d["H"]),
                    f=float(d["f"]), cx=float(d["cx"]), cy=float(d["cy"]))
    t = d["pred_cam_t"]                                               # pose_v2_probe.py (без сцены)
    kp = d["pred_keypoints_3d"] + t
    k2 = d["pred_keypoints_2d"]
    x, y = kp[:, 0] / kp[:, 2], kp[:, 1] / kp[:, 2]
    A = np.zeros((2 * len(x), 3))
    A[:len(x), 0], A[:len(x), 1], A[len(x):, 0], A[len(x):, 2] = x, 1, y, 1
    f, cx, cy = np.linalg.lstsq(A, np.concatenate([k2[:, 0], k2[:, 1]]), rcond=None)[0]
    return dict(V=(d["pred_vertices"] + t).astype(float), F=d["top.faces"], kp=kp.astype(float),
                S=np.zeros((0, 3)), N=None, W=int(round(2 * cx)), H=int(round(2 * cy)), f=f, cx=cx, cy=cy)


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
    return vn / np.maximum(n, 1e-12), n[:, 0] > 1e-12


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
        self.valid = dep > 5e-4
        self.Z = np.where(self.valid, 1.0 / np.maximum(inv, 1e-6), np.inf)
        self.Hp, self.Wp = dep.shape
        self.fx = (self.Wp / 2) / math.tan(math.radians(self.info["fov_x_deg"]) / 2)
        self.cx, self.cy = self.Wp / 2, self.Hp / 2
        V, F = load_obj_mesh(d / "plate_mesh.obj")
        n, ok = vertex_normals(V, F)
        to_cam = self.C - V
        n[np.einsum("ij,ij->i", n, to_cam) < 0] *= -1                      # нормали к камере
        self.P, self.Nrm = V[ok], n[ok]
        hor = np.abs(self.Nrm[:, 2]) > 0.8
        ver = np.abs(self.Nrm[:, 2]) < 0.35
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
    cand = (beh > 0) & (beh < 0.35)
    pen = np.zeros(len(X))
    if cand.any():
        tree, P, N = plate.trees["any"]
        _, j = tree.query(X[cand])
        sd = np.einsum("ij,ij->i", X[cand] - P[j], N[j])
        pen[cand] = np.maximum(-sd, 0.0)
    return pen


# ------------------------------------------------------------------------------------------------ reference analysis
def ref_contacts(ref, reg, forced):
    V = ref["V"]
    out = {}
    if forced:
        for item in forced:
            name, kind, surf = (item.split(":") + ["h", "s"])[:3] if item.count(":") >= 1 else (item, "h", "s")
            idx = np.where(reg == name)[0]
            if len(idx):
                out[name] = {"idx": idx, "kind": kind, "normal": None, "n": int(len(idx)), "surface": surf}
        return out
    if not len(ref["S"]):
        return out
    tree = cKDTree(ref["S"])
    d, j = tree.query(V, distance_upper_bound=0.2)
    near = d < CONTACT_M
    for name in np.unique(reg[near]):
        idx = np.where(near & (reg == name))[0]
        if len(idx) < max(8, int(0.02 * (reg == name).sum())):
            continue
        if ref["N"] is not None:
            n = ref["N"][j[idx]].mean(0)
        else:
            pts = ref["S"][tree.query_ball_point(V[idx].mean(0), 0.08)]
            n = np.linalg.svd(pts - pts.mean(0))[2][-1] if len(pts) > 10 else np.zeros(3)
        if np.linalg.norm(n) < 1e-6:
            continue
        n = n / np.linalg.norm(n)
        if n @ (V.mean(0) - V[idx].mean(0)) < 0:                          # нормаль к телу
            n = -n
        out[str(name)] = {"idx": idx, "kind": None, "normal": n, "n": int(len(idx))}
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
    for _ in range(2):
        cents = []
        for idx in best:
            h = V[idx] @ up
            low = idx[h <= np.percentile(h, 15)]
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


def ref_up(ref, contacts):
    cand = [c["normal"] * c["n"] for c in contacts.values()
            if c["normal"] is not None and c["normal"] @ np.array([0, -1.0, 0]) > 0.5]
    if cand:
        up = np.sum(cand, 0)
        return up / np.linalg.norm(up), "contacts"
    S, N = ref["S"], ref["N"]
    if len(S) > 500 and N is not None:
        Ns = np.where((N[:, 1] > 0)[:, None], -N, N)                  # знак нормалей MoGe не гарантирован
        m = Ns[:, 1] < -0.85
        if m.sum() > 300:
            up = Ns[m].mean(0)
            return up / np.linalg.norm(up), "scene"
    up, ok = plane_up(ref["V"], contacts, np.array([0, -1.0, 0]))
    return (up, "contact_plane") if ok else (up, "camera")


# ------------------------------------------------------------------------------------------------ fit
def fit(ref, plate, target, height, forced=None, log=print):
    V0, kp0 = ref["V"], ref["kp"]
    reg = regions(V0, kp0)
    contacts = ref_contacts(ref, reg, forced)
    up, up_src = ref_up(ref, contacts)
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
    if contacts:
        best = min(PRIORITY.get(k, 9) for k in contacts)
        support = [k for k in contacts if PRIORITY.get(k, 9) == best]
    else:
        support = []
    if support:
        sup_idx = np.concatenate([contacts[k]["idx"] for k in support])
    else:                                                               # без контактов: самые низкие 2 % вершин
        sup_idx = np.argsort(V[:, 2])[: max(20, len(V) // 50)]
        support = ["lowest"]
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

    tilt = 0.05 if up_src in ("contacts", "scene", "contact_plane") else 0.3     # насколько доверяем «вверх» рефа
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
        sup_gap = max(abs(rep_c[k]["gap_cm"]) for k in support)
    # равновесие: центр масс над опорным многоугольником (горизонтальные контакты в пределах 3 см)
    pts = []
    for k, c in contacts.items():
        if c["kind"] == "h" and abs(rep_c[k]["gap_cm"]) < 3:
            pts.append(X[c["idx"], :2])
    com = X.mean(0)
    balance = None
    if pts:
        P2 = np.concatenate(pts)
        try:
            balance = bool(Delaunay(P2).find_simplex(com[:2]) >= 0)
        except Exception:
            balance = bool(np.min(np.linalg.norm(P2 - com[:2], axis=1)) < 0.05)
    pel_p = (KP[9] + KP[10]) / 2
    fwd_p = facing(KP, Zup)
    tc = plate.C - pel_p
    tc[2] = 0
    tc /= np.linalg.norm(tc)
    rel_p = signed_angle(tc, fwd_p) if fwd_p is not None else 0.0
    elev_p = math.degrees(math.atan2(plate.h - pel_p[2], math.hypot(*(plate.C - pel_p)[:2])))
    xs, ys = ref["V"][:, 0] / ref["V"][:, 2] * ref["f"] + ref["cx"], ref["V"][:, 1] / ref["V"][:, 2] * ref["f"] + ref["cy"]
    status = "ok" if sup_gap <= 3 and max_pen <= 3 else "pose_mismatch"
    rep = {"status": status, "support": support, "support_gap_cm": round(sup_gap, 1),
           "max_penetration_cm": round(max_pen, 1), "pen_verts_over_2cm": int((pen > 0.02).sum()),
           "balance_ok": balance, "contacts": rep_c, "up_from": up_src, "height_scale": round(s, 3),
           "angle_to_camera_deg": {"ref": round(math.degrees(rel), 1), "plate": round(math.degrees(rel_p), 1)},
           "camera_elevation_deg": {"ref": round(elev_ref, 1), "plate": round(elev_p, 1)},
           "ref_plan": {"body_h_frac": round(float((ys.max() - ys.min()) / ref["H"]), 3),
                        "body_w_frac": round(float((xs.max() - xs.min()) / ref["W"]), 3)},
           "fit": {"params": [round(float(x), 4) for x in p], "cost": round(float(sol.cost), 3), "nfev": int(sol.nfev)},
           "target_world": [round(float(x), 3) for x in T]}
    log(f"[fit] {status}: опора {support} {sup_gap:.1f} см, проникновение {max_pen:.1f} см, "
        f"ракурс {rep['angle_to_camera_deg']}, камера {rep['camera_elevation_deg']}, равновесие {balance}")
    return X, rep, reg


def write_outputs(out, X, F, rep, plate, reg, contacts_names):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
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
