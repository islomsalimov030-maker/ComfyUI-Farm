"""Scene3D: OpenPose (COCO-18) image from render_scene.py pose.json + 8-bit depth + a check sheet.
python tools/scene3d/draw_pose.py C:/work/scene3d_work/scenes [--only S1_sofa_sit]
Writes <scene>/pose.png (black background, standard OpenPose colours), <scene>/depth8.png and
<root>/_sheet_<scene>.jpg (clay | depth | pose over clay | mask).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def draw(kp, size):
    w, h = size
    img = Image.new("RGB", (w, h), "black")
    over = Image.new("RGB", (w, h), "black")
    d = ImageDraw.Draw(over)
    stick = max(2, round(4 * h / 512))
    for i, (a, b) in enumerate(LIMBS):
        if not (kp[a][2] and kp[b][2]):
            continue
        (x1, y1), (x2, y2) = kp[a][:2], kp[b][:2]
        length = math.hypot(x2 - x1, y2 - y1)
        ang = math.atan2(y2 - y1, x2 - x1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        pts = []
        for t in range(0, 360, 15):
            r = math.radians(t)
            ex, ey = length / 2 * math.cos(r), stick * math.sin(r)
            pts.append((mx + ex * math.cos(ang) - ey * math.sin(ang), my + ex * math.sin(ang) + ey * math.cos(ang)))
        d.polygon(pts, fill=tuple(int(c * 0.6) for c in COLORS[i]))
    img = Image.fromarray(np.maximum(np.asarray(img), np.asarray(over)))
    d = ImageDraw.Draw(img)
    for i, (x, y, v) in enumerate(kp):
        if v:
            d.ellipse((x - stick, y - stick, x + stick, y + stick), fill=COLORS[i])
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    root = Path(a.root)
    for sd in sorted(p for p in root.iterdir() if (p / "pose.json").exists()):
        if a.only and sd.name not in a.only.split(","):
            continue
        info = json.loads((sd / "pose.json").read_text(encoding="utf-8"))
        size = tuple(info["size"])
        pose = draw(info["keypoints_coco18"], size)
        pose.save(sd / "pose.png")
        dep = np.asarray(Image.open(sd / "depth.png")).astype(np.float32)
        dep8 = Image.fromarray(np.clip(dep / dep.max() * 255, 0, 255).astype(np.uint8)).convert("RGB")
        dep8.save(sd / "depth8.png")
        clay = Image.open(sd / "clay.png").convert("RGB")
        mask = Image.open(sd / "mask.png").convert("RGB")
        po = Image.fromarray(np.maximum(np.asarray(clay) // 2, np.asarray(pose)))
        dd = ImageDraw.Draw(po)
        for key, col in (("head_top_px", (255, 255, 255)), ("feet_px", (255, 255, 255))):
            x, y = info[key]
            dd.line((x - 30, y, x + 30, y), fill=col, width=3)
        tw, th = size[0] // 2, size[1] // 2
        sheet = Image.new("RGB", (tw * 2, th * 2))
        for i, im in enumerate((clay, dep8, po, mask)):
            sheet.paste(im.resize((tw, th)), ((i % 2) * tw, (i // 2) * th))
        sheet.save(root / f"_sheet_{sd.name}.jpg", quality=88)
        print(sd.name, "head-feet px", round(info["feet_px"][1] - info["head_top_px"][1]))


main()
