"""Scene3D plate: metric geometry + camera of a location PHOTO, so the 3D scene matches the photo by construction.

Runs where torch + MoGe are installed (the pod's ComfyUI-next venv):
  /workspace/ComfyUI-next/venv/bin/pip install "git+https://github.com/microsoft/MoGe.git"
  /workspace/ComfyUI-next/venv/bin/python tools/scene3d/plate/plate_geometry.py PHOTO OUT_DIR [--size 1536x864]

Why: H3 builds the frame from a location picture and ignores a depth map rendered from another camera
(pod test 04.11, docs/experiments/3DA_POD_2026-11-04.md). So the 3D camera must BE the photo camera.
MoGe-2 gives a metric point map + intrinsics of the photo; we turn it into
  plate.json        camera for Blender (lens, sensor 36 mm, height above the floor, rotation), floor plane
  plate_mesh.obj    the photo as a metric mesh in the Blender world (Z up, floor z=0), for ray casts/occlusion
  depth_loc.png     16-bit inverse depth of the photo at --size (near = white), the H3 depth control map
  plate_rgb.png     the photo at --size (H3 reference / inpaint source)
  check_floor.jpg   the photo with the floor pixels in green (eyeball the floor fit)
World frame: Z = floor normal (up), floor at z=0, camera at (0, 0, h), Y = camera forward on the floor, X right.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

MODEL = "Ruicheng/moge-2-vitl-normal"


def core_inference(comfy, image):
    """Use installed core weights/classes; no pip, external repo or download."""
    import asyncio
    import torch
    import sys
    sys.path.insert(0, comfy)
    sys.argv = [sys.argv[0], '--lowvram', '--cache-none']
    from comfy_extras import nodes_moge as n
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import comfy.model_management as mm

    async def call(cls, **kwargs):
        result = cls.execute(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return (result.result if hasattr(result, 'result') else result)[0]

    async def run():
        model = await call(n.LoadMoGeModel, model_name=__import__('core_weights').resolve('moge'))
        tensor = torch.tensor(np.asarray(image) / 255., dtype=torch.float32)[None]
        with torch.inference_mode():
            output = await call(n.MoGeInference, moge_model=model, image=tensor, resolution_level=9,
                                fov_x_degrees=0., batch_size=1, force_projection=True, apply_mask=True, **({'refine_steps': 3} if 'refine_steps' in __import__('inspect').signature(n.MoGeInference.execute).parameters else {}))
        return {k: v[0].detach().float().cpu() for k, v in output.items() if k != 'image'}

    try:
        return asyncio.run(run())
    finally:
        mm.unload_all_models()
        mm.soft_empty_cache()


def fit_floor(points, normals, valid, h, rng):
    """RANSAC plane over the lower 45% of the frame where the predicted normal points up (OpenCV: -Y)."""
    rows = np.arange(points.shape[0])[:, None] >= int(points.shape[0] * 0.55)
    # either normal convention (OpenCV y down / OpenGL y up): a floor normal is vertical
    up_like = np.abs(normals[..., 1]) > 0.8 if normals is not None else np.ones(valid.shape, bool)
    cand = points[valid & rows & up_like]
    if len(cand) < 500:
        cand = points[valid & rows]
    if len(cand) < 3:
        return None
    best, best_n = None, -1
    for _ in range(400):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        if n[1] > 0:  # make the normal point up (OpenCV y is down)
            n = -n
        if n[1] > -0.5:  # not a floor-like plane
            continue
        d = -n @ s[0]
        inl = np.abs(cand @ n + d) < 0.02
        if inl.sum() > best_n:
            best, best_n = (n, d, inl), inl.sum()
    if best is None:
        return None  # no floor-like plane in the lower frame (close-up, top-down, no floor)
    n, d, inl = best
    # refine by least squares on the inliers
    p = cand[inl]
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)  # full U would be N x N
    n = vt[-1]
    if n[1] > 0:
        n = -n
    d = -n @ c
    return n, d, int(best_n), len(cand)


# --- 0.2.0: any floor, several levels or no floor in the frame -------------------------------------
LEVEL_TOL = 0.05       # m, one horizontal level
HORIZONTAL_COS = np.cos(np.radians(15))
BELOW_FLOOR_M = 0.10   # m: scene points this far below a level ...
BELOW_FLOOR_FRAC = 0.03  # ... on more than this share of the frame -> that level is not the floor
SIGMA = 0.08          # rad, angular width of a surface orientation in the vote


def point_normals(points, valid, step=None):
    """Normals from the metric point map itself (OpenCV camera), turned towards the camera.

    Differences over `step` pixels (~W/400) average out per-pixel depth noise without crossing objects.
    """
    k = step or max(1, points.shape[1] // 400)
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, :-k] = points[:, k:] - points[:, :-k]
    dy[:-k] = points[k:] - points[:-k]
    ok = valid.copy()
    ok[:, :-k] &= valid[:, k:]
    ok[:-k] &= valid[k:]
    ok[:, -k:] = False
    ok[-k:] = False
    z = np.maximum(points[..., 2], 1e-3)
    ok &= (np.linalg.norm(dx, axis=-1) < 0.05 * k * z) & (np.linalg.norm(dy, axis=-1) < 0.05 * k * z)  # no depth jumps
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=-1)
    ok &= length > 1e-12
    n = n / np.maximum(length, 1e-12)[..., None]
    n[np.einsum("...i,...i->...", n, points) > 0] *= -1
    # 5x5 mean over valid neighbours: smooth per-pixel noise of the predicted depth
    from scipy.ndimage import uniform_filter
    w = uniform_filter(ok.astype(np.float64), 5)
    n = np.stack([uniform_filter(np.where(ok, n[..., i], 0.0), 5) for i in range(3)], -1) / np.maximum(w, 1e-9)[..., None]
    length = np.linalg.norm(n, axis=-1)
    ok &= length > 0.5
    return n / np.maximum(length, 1e-12)[..., None], ok


def _up(pitch_deg, roll_deg):
    """World up in OpenCV camera coords for a camera pitched down by pitch and rolled by roll."""
    t, r = np.radians(pitch_deg), np.radians(roll_deg)
    x, y, z = 0.0 * t, -np.cos(t), -np.sin(t)
    return np.stack([x * np.cos(r) - y * np.sin(r), x * np.sin(r) + y * np.cos(r), z], -1)


def gravity_up(normals, rng, samples=20000):
    """Up direction from the scene itself (horizontal + vertical surfaces), not from a floor.

    Manhattan vote over the camera pitch (-30..89 deg) and roll (+-25 deg); a weak prior for small roll and
    a slightly downward pitch breaks the frontal-wall vs top-down-floor tie.
    """
    if len(normals) > samples:
        normals = normals[rng.choice(len(normals), samples, replace=False)]

    def score(P, R):
        U = _up(P.ravel(), R.ravel())
        D = np.abs(normals @ U.T)
        # ~5 deg width for vertical (D ~ 0, one angle) and horizontal (D ~ 1, two angles: x sqrt2) surfaces
        s = np.exp(-(1 - D) / (2 * SIGMA ** 2)).mean(0) + np.exp(-D ** 2 / (2 * SIGMA ** 2)).mean(0)
        # Weak prior, only to break the one-plane tie (frontal wall vs floor from above): any downward pitch is
        # common (slightly preferring the smaller one), looking up is rare; an upright camera (roll small).
        p = P.ravel()
        pitch_prior = np.where(p < 0, np.exp(-(p / 15) ** 2 / 2), 1 - 0.1 * p / 90)
        return s + 0.2 * pitch_prior + 0.2 * np.exp(-(R.ravel() / 8) ** 2 / 2)

    P, R = np.meshgrid(np.arange(-30, 90, 1.0), np.arange(-24, 25, 2.0))
    k = int(np.argmax(score(P, R)))
    p0, r0 = P.ravel()[k], R.ravel()[k]
    P, R = np.meshgrid(np.arange(p0 - 2, p0 + 2.01, 0.25), np.arange(r0 - 2, r0 + 2.01, 0.25))
    k = int(np.argmax(score(P, R)))
    up = _up(P.ravel()[k], R.ravel()[k])
    # One plane fills the frame: a floor from above and a wall in front give the same normals. Looking up is
    # rare, so a plane tilted towards image-up is a floor (top-down), otherwise a wall (up = image-up on it).
    m = normals.mean(0)
    m /= np.linalg.norm(m)
    if np.mean(normals @ m > np.cos(np.radians(15))) >= 0.8:
        if m[1] < -np.sin(np.radians(3)):
            up = m
        elif abs(up @ m) > 0.9:
            y_up = np.array([0, -1.0, 0])
            up = y_up - (y_up @ m) * m
            up /= np.linalg.norm(up)
    return up, float(np.degrees(np.arcsin(np.clip(-up[2], -1, 1)))), float(np.degrees(np.arctan2(up[0], -up[1])))


def cluster_levels(h, bin_m=0.01, min_frac=0.001):
    """Horizontal levels from heights: 1 cm histogram, levels split at (almost) empty bins."""
    h = np.asarray(h, float)
    if not len(h):
        return []
    k = np.floor((h - h.min()) / bin_m).astype(int)
    counts = np.bincount(k)
    full = counts > max(1, min_frac * len(h))
    levels, start = [], None
    for i, f in enumerate(np.append(full, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            sel = (k >= start) & (k < i)
            levels.append({"h": float(h[sel].mean()), "n": int(sel.sum())})
            start = None
    return levels


def horizontal_levels(points, normals, ok, up):
    """Visible horizontal levels: heights (m, relative to the camera, along up) and pixel counts."""
    horiz = ok & (normals @ up > HORIZONTAL_COS)
    return cluster_levels(points[horiz] @ up), int(horiz.sum())


def scene_frame(points, valid, rng, prior_height=1.5):
    """Floor plane if a floor is visible; otherwise an up direction from the scene and no floor.

    Returns (n, d, report): n·p + d = height above the floor (or above the assumed floor) for p in camera coords.
    """
    normals, ok = point_normals(points, valid)
    if ok.sum() < 100:
        raise SystemExit("plate: too few valid points for the scene geometry")
    up, pitch, roll = gravity_up(normals[ok], rng)
    levels, n_horiz = horizontal_levels(points, normals, ok, up)
    n_valid = int(valid.sum())
    h_all = points[valid] @ up

    def is_floor(lv):
        # A floor has (almost) nothing visible below it: a bed or a table top over a thin floor strip is not
        # the floor (red room 13.11: 0.1.10 took the bed top, camera 0.58 m "above the floor").
        big = lv["n"] >= 0.02 * n_valid and lv["n"] >= 0.1 * max(n_horiz, 1)
        return big and float(np.mean(h_all < lv["h"] - BELOW_FLOOR_M)) <= BELOW_FLOOR_FRAC

    floor = next((lv for lv in levels if is_floor(lv)), None)
    report = {"up_source": "scene_normals", "pitch_deg": pitch, "roll_deg": roll}
    if floor is not None:
        h = points @ up
        sel = ok & (normals @ up > HORIZONTAL_COS) & (np.abs(h - floor["h"]) <= LEVEL_TOL)
        p = points[sel]
        c = p.mean(0)
        _, sv, vt = np.linalg.svd(p - c, full_matrices=False)
        n = vt[-1] if vt[-1] @ up > 0 else -vt[-1]
        if sv[1] < 0.25 * sv[0] or n @ up < np.cos(np.radians(8)):
            n = up                          # thin floor strip: the plane is (almost) a line, keep the scene up
        d = -n @ c
        report.update(visible=True, source="lowest_level", inliers=int(sel.sum()), candidates=n_horiz)
    else:
        n = up
        lowest = float(np.min(points[valid] @ up))
        d = max(prior_height, -lowest + 0.02)
        report.update(visible=False, source="prior_height", inliers=0, candidates=n_horiz)
    report["levels"] = [{"z": round(float(lv["h"] + d), 3), "frac": round(lv["n"] / max(n_horiz, 1), 3)}
                        for lv in levels if lv["n"] >= 0.02 * max(n_horiz, 1)]
    return n, float(d), report


def world_rotation(n):
    """World axes (rows) in camera coords: Z = up n, Y = camera forward on the floor, X right."""
    z_cv = np.array([0, 0, 1.0])
    f = z_cv - (z_cv @ n) * n
    if np.linalg.norm(f) < 0.2:  # looking (almost) straight down: forward = image up, laid on the floor
        y_up = np.array([0, -1.0, 0])
        f = y_up - (y_up @ n) * n
    f /= np.linalg.norm(f)
    return np.stack([np.cross(f, n), f, n])


def choose_floor(points, normals_pred, valid, H, rng):
    """Old lower-frame RANSAC when it agrees with the scene; otherwise the scene frame (0.2.0)."""
    n2, d2, report = scene_frame(points, valid, rng)
    old = fit_floor(points, normals_pred, valid, H, rng)
    if old is not None:
        n, d, n_in, n_cand = old
        same_up = float(n @ n2) > np.cos(np.radians(8))
        same_floor = report["visible"] and abs(float(d) - d2) <= LEVEL_TOL
        if same_up and same_floor:
            report.update(source="ransac_lower", up_source="floor", inliers=n_in, candidates=n_cand)
            return n, float(d), report
        report["rejected_lower_ransac"] = {"height_m": round(float(d), 3), "same_up": same_up}
    return n2, d2, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("out")
    ap.add_argument("--size", default="1536x864")
    ap.add_argument("--stride", type=int, default=4, help="mesh decimation")
    ap.add_argument("--comfy", help="Use existing ComfyUI core MoGe classes")
    a = ap.parse_args()
    W, H = map(int, a.size.split("x"))
    os.makedirs(a.out, exist_ok=True)

    img = Image.open(a.photo).convert("RGB").resize((W, H), Image.LANCZOS)
    img.save(os.path.join(a.out, "plate_rgb.png"))
    if a.comfy:
        o = core_inference(a.comfy, img)
    else:
        import torch
        from moge.model.v2 import MoGeModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = MoGeModel.from_pretrained(MODEL).to(dev).eval()
        x = torch.tensor(np.asarray(img) / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad():
            o = model.infer(x)
    pts = o["points"].float().cpu().numpy()          # H,W,3 metric, OpenCV camera (x right, y down, z fwd)
    dep = o["depth"].float().cpu().numpy()
    valid = o["mask"].cpu().numpy().astype(bool) & np.isfinite(dep) & (dep > 0)
    K = o["intrinsics"].float().cpu().numpy()        # normalised: fx, fy, cx, cy in fractions of W/H
    nrm = o["normal"].float().cpu().numpy() if "normal" in o else None

    rng = np.random.default_rng(0)
    n, d, floor_report = choose_floor(pts, nrm, valid, H, rng)
    R = world_rotation(n)                              # cv -> world rotation (rows)
    cam_h = float(d)                                   # n.p + d = height above floor; camera at p=0
    if cam_h < 0:
        raise SystemExit(f"floor fit failed: camera below the floor ({cam_h:.2f} m)")

    # --- depth control map: inverse depth, near = white, 16 bit
    inv = np.where(valid, 1.0 / np.maximum(dep, 1e-3), 0)
    lo, hi = np.percentile(inv[valid], 1), np.percentile(inv[valid], 99.5)
    dn = np.clip((inv - lo) / (hi - lo), 0, 1)
    Image.fromarray((dn * 65535).astype(np.uint16)).save(os.path.join(a.out, "depth_loc.png"))

    # --- mesh in the Blender world (Z up), decimated grid, no faces across depth jumps
    s = a.stride
    P = pts[::s, ::s]
    V = valid[::s, ::s]
    gh, gw = V.shape
    Pw = P.reshape(-1, 3) @ R.T
    Pw[:, 2] += cam_h
    idx = -np.ones(gh * gw, int)
    keep = V.reshape(-1)
    idx[keep] = np.arange(keep.sum()) + 1
    Z = dep[::s, ::s]
    lines = [f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n" for p in Pw[keep]]
    faces = []
    for i in range(gh - 1):
        for j in range(gw - 1):
            q = [i * gw + j, i * gw + j + 1, (i + 1) * gw + j + 1, (i + 1) * gw + j]
            if min(idx[k] for k in q) <= 0:
                continue
            zz = [Z[i, j], Z[i, j + 1], Z[i + 1, j + 1], Z[i + 1, j]]
            if max(zz) - min(zz) > 0.06 * min(zz) * s:  # depth jump: leave a hole
                continue
            faces.append(f"f {idx[q[0]]} {idx[q[3]]} {idx[q[2]]} {idx[q[1]]}\n")
    with open(os.path.join(a.out, "plate_mesh.obj"), "w") as fh:
        fh.write("o plate\n")
        fh.writelines(lines)
        fh.writelines(faces)

    # --- floor check image
    h_all = pts.reshape(-1, 3) @ n + d
    floor_px = (np.abs(h_all.reshape(H, W)) < 0.03) & valid
    chk = np.asarray(img).copy()
    chk[floor_px] = (chk[floor_px] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    Image.fromarray(chk).save(os.path.join(a.out, "check_floor.jpg"), quality=88)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    info = {
        "photo": os.path.abspath(a.photo), "size": [W, H], "model": MODEL,
        "intrinsics_norm": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
        "lens_mm": float(fx * 36.0), "sensor_width_mm": 36.0,
        "fov_x_deg": float(np.degrees(2 * np.arctan(0.5 / fx))),
        "cam_height_m": cam_h, "R_world_from_cv": R.tolist(),
        "floor": floor_report,
        "inv_norm": {"lo": float(lo), "hi": float(hi)},  # depth_loc = (1/z - lo) / (hi - lo), z in metres
        "depth_m": {"p1": float(np.percentile(dep[valid], 1)), "p99": float(np.percentile(dep[valid], 99))},
        "mesh": {"verts": int(keep.sum()), "faces": len(faces), "stride": s},
    }
    with open(os.path.join(a.out, "plate.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(f"[plate] cam height {cam_h:.2f} m, fov_x {info['fov_x_deg']:.1f} deg, lens {info['lens_mm']:.1f} mm, "
          f"depth {info['depth_m']['p1']:.2f}-{info['depth_m']['p99']:.2f} m, floor {floor_report['source']} {floor_report['inliers']}/{floor_report['candidates']}, "
          f"mesh {info['mesh']['verts']} v / {len(faces)} f")


if __name__ == "__main__":
    main()
