"""3DA 0.3.20: своё телосложение персонажа (MHR shape/scale из body.json) вместо телосложения человека с фото позы.

Профиль (один раз): ref_scene.py по фото персонажа -> медиана mhr_shape_params / mhr_scale_params -> body.json.
Постановка (каждый раз, кэш по позе и форме): поза и кисти — из ref.npz фото позы, форма — из body.json.
Сдвиг считается дельтой MHR (F(форма персонажа) - F(форма с фото)) в той же позе и добавляется к вершинам и
точкам SAM — так сохраняется точная привязка к камере фото позы.

  python body_shape.py --comfy <ComfyUI> --ref pose.ref.npz --body body.json --out shaped/pose.ref.npz
"""
import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SHAPE_KEYS = ("shape_params", "scale_params")


def body_shape(body):
    """(shape, scale) из body.json или None, если телосложение не снято."""
    if not body:
        return None
    shape, scale = body.get("shape_params") or [], body.get("scale_params") or []
    if not shape or not scale:
        return None
    shape, scale = np.asarray(shape, np.float32), np.asarray(scale, np.float32)
    if shape.ndim != 1 or scale.ndim != 1 or not np.isfinite(shape).all() or not np.isfinite(scale).all():
        raise ValueError("body_json: shape_params/scale_params must be finite number lists")
    return shape, scale


def shape_hash(shape, scale):
    h = hashlib.sha256()
    h.update(np.asarray(shape, np.float32).tobytes())
    h.update(np.asarray(scale, np.float32).tobytes())
    return h.hexdigest()[:16]


def aggregate(refs):
    """refs: [(name, npz-dict)] -> профиль формы. Медиана по фото, где SAM нашёл человека."""
    rows = [(name, np.asarray(d["mhr_shape_params"], np.float32).reshape(-1),
             np.asarray(d["mhr_scale_params"], np.float32).reshape(-1))
            for name, d in refs if "mhr_shape_params" in d and "mhr_scale_params" in d]
    if not rows:
        raise ValueError("body profile: на фото не найден человек (нет параметров MHR)")
    if len({r[1].size for r in rows}) != 1 or len({r[2].size for r in rows}) != 1:
        raise ValueError("body profile: разная длина параметров MHR между фото")
    shape = np.median(np.stack([r[1] for r in rows]), axis=0)
    scale = np.median(np.stack([r[2] for r in rows]), axis=0)
    spread = {name: round(float(np.linalg.norm(s - shape) + np.linalg.norm(c - scale)), 4) for name, s, c in rows}
    return dict(shape_params=shape, scale_params=scale, photos=len(rows), spread=spread)


def write_profile(body_json, prof, photo_hashes, now=None):
    """Дописать форму в body.json (остальные поля сохраняются), прежний файл — в резервную копию."""
    path = Path(body_json)
    body = json.loads(path.read_text(encoding="utf-8"))
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + ".bak_" + stamp)
    shutil.copy2(path, backup)
    body.update(shape_model="MHR", shape_params=[round(float(x), 6) for x in prof["shape_params"]],
                scale_params=[round(float(x), 6) for x in prof["scale_params"]],
                shape_source={"photos": prof["photos"], "photo_sha256": list(photo_hashes),
                              "method": "median SAM 3D Body MHR", "date": stamp})
    path.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
    return backup


def write_fit(body_json, fit, cm, now=None):
    """Форма, подогнанная под мерки, -> body.json (+ measurements_cm), прежний файл — в резервную копию."""
    path = Path(body_json)
    body = json.loads(path.read_text(encoding="utf-8"))
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + ".bak_" + stamp)
    shutil.copy2(path, backup)
    body.update(shape_model="MHR", shape_params=[round(float(x), 6) for x in fit["shape_params"]],
                scale_params=[round(float(x), 6) for x in fit["scale_params"]],
                measurements_cm={k: float(v) for k, v in cm.items()},
                shape_source={"method": "fit MHR to measurements (start: previous shape)", "date": stamp,
                              "got_cm": fit.get("got_cm"), "previous": body.get("shape_source")})
    path.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
    return backup


def shaped_ref(npz, forward, shape, scale):
    """npz (dict из ref.npz) -> dict с формой персонажа. forward(shape, scale) -> (verts, kp70) в осях MHR."""
    old_shape = np.asarray(npz["mhr_shape_params"], np.float32).reshape(-1)
    old_scale = np.asarray(npz["mhr_scale_params"], np.float32).reshape(-1)
    if old_shape.size != shape.size or old_scale.size != scale.size:
        raise ValueError(f"body_json: длина формы {shape.size}/{scale.size} не совпадает с MHR {old_shape.size}/{old_scale.size}")
    v0, k0 = forward(old_shape, old_scale)
    v1, k1 = forward(shape, scale)
    flip = np.array([1.0, -1.0, -1.0], np.float32)
    out = dict(npz)
    V = np.asarray(npz["verts"], np.float32)
    if V.shape != v0.shape:
        raise ValueError("ref.npz: число вершин не совпадает с MHR")
    cam_t = np.asarray(npz["mhr_pred_cam_t"], np.float32).reshape(3)
    residual = float(np.median(np.linalg.norm(v0 * flip + cam_t - V, axis=1)))
    out["verts"] = (V + (v1 - v0) * flip).astype(np.float32)
    kp3 = np.asarray(npz["kp3d"], np.float32)
    kp3 = kp3 + (k1[:len(kp3)] - k0[:len(kp3)]) * flip
    out["kp3d"] = kp3.astype(np.float32)
    f, cx, cy = float(npz["f"]), float(npz["cx"]), float(npz["cy"])
    out["kp2d"] = np.stack([f * kp3[:, 0] / kp3[:, 2] + cx, f * kp3[:, 1] / kp3[:, 2] + cy], 1).astype(np.float32)
    out["mhr_shape_params"], out["mhr_scale_params"] = shape.astype(np.float32), scale.astype(np.float32)
    out["body_shape_hash"] = np.array(shape_hash(shape, scale))
    return out, residual


def arm_mask(V, kp):
    """Вершины рук по ближайшим опорным точкам MHR70 (body.regions) — из срезов мерок исключаются."""
    import body
    labels = body.regions(np.asarray(V), np.asarray(kp))
    return np.array([str(x).split(".")[0] in ("arm", "forearm", "hand") for x in labels])


def fit_measurements(forward_rest, faces, target, init_shape, init_scale, reg=.02, max_nfev=900, log=print):
    """Подогнать форму MHR под мерки (доли роста, body_measure.KEYS). forward_rest(shape, scale) -> (V, kp70)
    в стойке без позы. Старт — форма SAM (init); регуляризация держит форму рядом со стартом."""
    import body_measure
    from scipy.optimize import least_squares
    keys = [k for k in body_measure.KEYS if k in target]
    if not keys:
        raise ValueError("body fit: нет мерок")
    ns = len(init_shape)
    x0 = np.concatenate([init_shape, init_scale]).astype(np.float64)
    V0, kp0 = forward_rest(init_shape, init_scale)
    arms = arm_mask(V0, kp0)
    edges = body_measure.prepare(faces, arms)
    weight = {"thigh": .5}
    cache = {}

    def measures(x):
        key = x.tobytes()
        if key not in cache:
            V, _ = forward_rest(x[:ns].astype(np.float32), x[ns:].astype(np.float32))
            cache[key] = body_measure.measure(V, faces, edges=edges)[0]
        return cache[key]

    def residual(x):
        m = measures(x)
        r = [(m[k] - target[k]) / target[k] * weight.get(k, 1.) for k in keys]
        return np.concatenate([r, reg * (x - x0)])
    start = measures(x0)
    def jac(x, step=.05):  # абсолютный шаг: у scipy diff_step относительный и у нулевых параметров ~1e-8
        r0 = residual(x)
        J = np.empty((len(r0), len(x)))
        for i in range(len(x)):
            xi = x.copy()
            xi[i] += step
            J[:, i] = (residual(xi) - r0) / step
        return J
    res = least_squares(residual, x0, jac=jac, max_nfev=max_nfev, x_scale=1.)
    got = measures(res.x)
    log(f"[body_fit] {res.nfev} оценок, start {start}, got {got}, target {target}")
    return dict(shape_params=res.x[:ns].astype(np.float32), scale_params=res.x[ns:].astype(np.float32),
                start=start, got=got, nfev=int(res.nfev))


def rest_forward(head, torch, dev):
    """MHR в стойке без позы (T-поза), метры."""
    z = lambda n: torch.zeros(1, n, device=dev)  # noqa: E731

    def forward(shape, scale):
        T = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)[None]  # noqa: E731
        with torch.no_grad():
            v, kp = head.mhr_forward(global_trans=z(3), global_rot=z(3), body_pose_params=z(130),
                                     hand_pose_params=z(head.num_hand_comps * 2), scale_params=T(scale),
                                     shape_params=T(shape), expr_params=z(72), return_keypoints=True)
        return v[0].float().cpu().numpy(), kp[0, :70].float().cpu().numpy()
    return forward


def mhr_forward(head, torch, dev, npz):
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)[None]  # noqa: E731
    gr, bp, hp = T(npz["mhr_global_rot"]), T(npz["mhr_body_pose_params"]), T(npz["mhr_hand_pose_params"])
    zeros_t, zeros_e = torch.zeros(1, 3, device=dev), torch.zeros(1, 72, device=dev)

    def forward(shape, scale):
        with torch.no_grad():
            v, kp = head.mhr_forward(global_trans=zeros_t, global_rot=gr, body_pose_params=bp, hand_pose_params=hp,
                                     scale_params=T(scale), shape_params=T(shape), expr_params=zeros_e,
                                     return_keypoints=True)
        return v[0].float().cpu().numpy(), kp[0, :70].float().cpu().numpy()
    return forward


def main_fit(a, ctx):
    """--fit: мерки (см, JSON) -> подогнанная форма в JSON-результат (body.json пишет pack_cli)."""
    import body_measure
    body = json.loads(Path(a.body).read_text(encoding="utf-8"))
    cm = json.loads(Path(a.fit).read_text(encoding="utf-8"))
    target = body_measure.ratios(cm, float(body["height_m"]))
    head, torch, dev = ctx["head"], ctx["torch"], ctx["dev"]
    init = body_shape(body)
    if init is None:
        init = (np.zeros(head.num_shape_comps, np.float32), np.zeros(head.num_scale_comps, np.float32))
    faces = head.faces_np() if hasattr(head, "faces_np") else head.faces.cpu().numpy()
    r = fit_measurements(rest_forward(head, torch, dev), faces, target, *init)
    h = float(body["height_m"])
    out = dict(shape_params=[float(x) for x in r["shape_params"]], scale_params=[float(x) for x in r["scale_params"]],
               start_cm=body_measure.in_cm(r["start"], h), got_cm=body_measure.in_cm(r["got"], h),
               target_cm={k: float(v) for k, v in cm.items() if k in target}, nfev=r["nfev"])
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit", help="JSON с мерками в см: подгонка формы вместо постановки")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import refine
    ctx = refine.load_ctx(a.comfy)
    if a.fit:
        return main_fit(a, ctx)
    npz = dict(np.load(a.ref, allow_pickle=True))
    sh = body_shape(json.loads(Path(a.body).read_text(encoding="utf-8")))
    if sh is None:
        raise SystemExit("body_json: телосложение не снято (shape_params/scale_params)")
    out, residual = shaped_ref(npz, mhr_forward(ctx["head"], ctx["torch"], ctx["dev"], npz), *sh)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out).with_suffix(".pending.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(a.out)
    print(f"[body_shape] {Path(a.ref).name}: форма {out['body_shape_hash']}, невязка MHR {residual * 100:.2f} см", flush=True)


if __name__ == "__main__":
    main()
