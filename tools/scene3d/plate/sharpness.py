"""Laplacian-variance sharpness of the generated frames vs the location photo (whole frame and the person box).
python tools/scene3d/plate/sharpness.py --maps /workspace/ComfyUI-next/input/plate3d --glob "P4_*/A5*_s101_*"
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import farm_env  # noqa: E402
OUT = str(farm_env.COMFY / "output/plate3d_ab")


def lap(a):
    l = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float(l.var())


ap = argparse.ArgumentParser()
ap.add_argument("--maps", required=True)
ap.add_argument("--glob", default="*/A*_s*_*")
a = ap.parse_args()
loc = os.path.basename(os.path.normpath(a.maps))
photo = np.asarray(Image.open(str(farm_env.COMFY / f"input/{loc}_rgb.png")).convert("L"), np.float32)
print(f"[sharp] photo {lap(photo):.0f}")
for f in sorted(glob.glob(os.path.join(OUT, a.glob + ".png"))):
    sc = os.path.basename(os.path.dirname(f))
    if os.path.basename(f).split("_s")[0].endswith("c"):
        continue
    im = Image.open(f).convert("L")
    g = np.asarray(im.resize((photo.shape[1], photo.shape[0])), np.float32)
    m = Image.open(str(farm_env.COMFY / f"input/plate3d_{sc}_mask_soft.png")).convert("L").resize(im.size)
    bb = m.getbbox()
    box = np.asarray(im.crop(bb), np.float32) if bb else g
    print(f"[sharp] {sc:16s} {os.path.basename(f)[:20]:20s} frame {lap(g):5.0f}  person box {lap(box):5.0f}")
