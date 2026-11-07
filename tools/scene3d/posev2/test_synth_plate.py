"""Синтетическая плита для теста pose_fit: пол z=0, кровать (верх 0.5 м, x -1..1, y 2.5..4.5), стена y=5.
  python test_synth_plate.py <out_dir>"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def make(out, W=768, H=432, h=1.4, pitch_deg=15, fov_x=70.0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t = math.radians(pitch_deg)
    M = np.array([[1, 0, 0], [0, -math.sin(t), math.cos(t)], [0, -math.cos(t), -math.sin(t)]])
    fx = (W / 2) / math.tan(math.radians(fov_x) / 2)
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dcv = np.stack([(uu - W / 2) / fx, (vv - H / 2) / fx, np.ones_like(uu)], -1)
    dw = dcv @ M.T
    C = np.array([0, 0, h])
    best = np.full((H, W), np.inf)

    def cand(tt, ok):
        nonlocal best
        tt = np.where(ok & (tt > 0), tt, np.inf)
        best = np.minimum(best, tt)
    with np.errstate(divide="ignore", invalid="ignore"):
        cand(-C[2] / dw[..., 2], dw[..., 2] < 0)                                    # пол
        tw = (5 - C[1]) / dw[..., 1]
        cand(tw, dw[..., 1] > 0)                                                     # стена
        tb = (0.5 - C[2]) / dw[..., 2]
        pb = C + dw * tb[..., None]
        cand(tb, (dw[..., 2] < 0) & (np.abs(pb[..., 0]) <= 1) & (pb[..., 1] >= 2.5) & (pb[..., 1] <= 4.5))
        tf = (2.5 - C[1]) / dw[..., 1]
        pf = C + dw * tf[..., None]
        cand(tf, (dw[..., 1] > 0) & (np.abs(pf[..., 0]) <= 1) & (pf[..., 2] >= 0) & (pf[..., 2] <= 0.5))
    z = best                                                                         # dcv z = 1 -> t = глубина z
    inv = 1 / z
    lo, hi = float(np.percentile(inv, 1)), float(np.percentile(inv, 99.5))
    dn = np.clip((inv - lo) / (hi - lo), 0, 1)
    Image.fromarray((dn * 65535).astype(np.uint16)).save(out / "depth_loc.png")
    zq = 1 / (lo + dn * (hi - lo))
    P = (dcv * zq[..., None]) @ M.T + C
    s = 4
    Pg = P[::s, ::s]
    gh, gw = Pg.shape[:2]
    with open(out / "plate_mesh.obj", "w") as fh:
        fh.write("o plate\n")
        fh.writelines(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n" for p in Pg.reshape(-1, 3))
        Zg = zq[::s, ::s]
        for i in range(gh - 1):
            for j in range(gw - 1):
                q = [i * gw + j, i * gw + j + 1, (i + 1) * gw + j + 1, (i + 1) * gw + j]
                zz = Zg.reshape(-1)[q]
                if zz.max() - zz.min() > 0.06 * zz.min() * s:
                    continue
                fh.write(f"f {q[0] + 1} {q[3] + 1} {q[2] + 1} {q[1] + 1}\n")
    sh = (np.clip(P[..., 2] / 2, 0, 1) * 120 + 60).astype(np.uint8)
    Image.fromarray(np.stack([sh, sh, sh], -1)).save(out / "plate_rgb.png")
    (out / "plate.json").write_text(json.dumps({"fov_x_deg": fov_x, "cam_height_m": h, "R_world_from_cv": M.tolist(),
                                                "inv_norm": {"lo": lo, "hi": hi}, "size": [W, H]}, indent=1))


if __name__ == "__main__":
    make(sys.argv[1])
