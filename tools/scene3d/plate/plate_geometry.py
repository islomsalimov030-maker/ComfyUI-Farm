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
import torch
from PIL import Image

MODEL = "Ruicheng/moge-2-vitl-normal"


def fit_floor(points, normals, valid, h, rng):
    """RANSAC plane over the lower 45% of the frame where the predicted normal points up (OpenCV: -Y)."""
    rows = np.arange(points.shape[0])[:, None] >= int(points.shape[0] * 0.55)
    # either normal convention (OpenCV y down / OpenGL y up): a floor normal is vertical
    up_like = np.abs(normals[..., 1]) > 0.8 if normals is not None else np.ones(valid.shape, bool)
    cand = points[valid & rows & up_like]
    if len(cand) < 500:
        cand = points[valid & rows]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("out")
    ap.add_argument("--size", default="1536x864")
    ap.add_argument("--stride", type=int, default=4, help="mesh decimation")
    a = ap.parse_args()
    W, H = map(int, a.size.split("x"))
    os.makedirs(a.out, exist_ok=True)

    from moge.model.v2 import MoGeModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(MODEL).to(dev).eval()
    img = Image.open(a.photo).convert("RGB").resize((W, H), Image.LANCZOS)
    img.save(os.path.join(a.out, "plate_rgb.png"))
    x = torch.tensor(np.asarray(img) / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
    with torch.no_grad():
        o = model.infer(x)
    pts = o["points"].float().cpu().numpy()          # H,W,3 metric, OpenCV camera (x right, y down, z fwd)
    dep = o["depth"].float().cpu().numpy()
    valid = o["mask"].cpu().numpy().astype(bool) & np.isfinite(dep) & (dep > 0)
    K = o["intrinsics"].float().cpu().numpy()        # normalised: fx, fy, cx, cy in fractions of W/H
    nrm = o["normal"].float().cpu().numpy() if "normal" in o else None

    rng = np.random.default_rng(0)
    n, d, n_in, n_cand = fit_floor(pts, nrm, valid, H, rng)
    # world axes in camera coords
    z_cv = np.array([0, 0, 1.0])
    f = z_cv - (z_cv @ n) * n
    f /= np.linalg.norm(f)
    r = np.cross(f, n)
    R = np.stack([r, f, n])                            # cv -> world rotation (rows)
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
        "floor": {"inliers": n_in, "candidates": n_cand},
        "inv_norm": {"lo": float(lo), "hi": float(hi)},  # depth_loc = (1/z - lo) / (hi - lo), z in metres
        "depth_m": {"p1": float(np.percentile(dep[valid], 1)), "p99": float(np.percentile(dep[valid], 99))},
        "mesh": {"verts": int(keep.sum()), "faces": len(faces), "stride": s},
    }
    with open(os.path.join(a.out, "plate.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(f"[plate] cam height {cam_h:.2f} m, fov_x {info['fov_x_deg']:.1f} deg, lens {info['lens_mm']:.1f} mm, "
          f"depth {info['depth_m']['p1']:.2f}-{info['depth_m']['p99']:.2f} m, floor inliers {n_in}/{n_cand}, "
          f"mesh {info['mesh']['verts']} v / {len(faces)} f")


if __name__ == "__main__":
    main()
