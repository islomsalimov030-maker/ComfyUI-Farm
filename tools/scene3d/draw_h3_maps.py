"""Scene3D 3D-A: draw the H3 control maps from the Blender output (numpy/PIL, run outside Blender).

python tools/scene3d/draw_h3_maps.py C:/work/scene3d_work/gate [--frames 5] [--fps 24]

For every <scene>/ under the root it writes:
  pose25.png       - OpenPose BODY-25 skeleton on black (the H3 "pose" control map)
  depth_loc8.png   - 8-bit RGB depth of the location only (the H3 "depth" control map)
  control/pose_%05d.png, control/depth_%05d.png - the control video: N identical frames
  control.mp4      - the pose sequence as an mp4 (24 fps) for the LoadVideo path
  _sheet_<scene>.jpg - clay-free check sheet (depth | pose | pose over depth)

The control video frames are identical on purpose: the 3D scene is static, so the control
signal must be too. H3 expects a clip of 17n+5 frames.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BODY25_NAMES = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
    "l_thumb1", "l_thumb2", "l_thumb3", "l_index1", "l_index2", "l_index3",
    "l_middle1", "l_middle2", "l_middle3", "l_ring1", "l_ring2", "l_ring3",
    "l_pinky1", "l_pinky2", "l_pinky3",
    "r_thumb1", "r_thumb2", "r_thumb3", "r_index1", "r_index2", "r_index3",
    "r_middle1", "r_middle2", "r_middle3", "r_ring1", "r_ring2", "r_ring3",
    "r_pinky1", "r_pinky2", "r_pinky3",
]
BODY25_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
]
BODY25_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
]
HAND_COLOR = (0, 170, 255)


def draw_pose25(kp, size, out_png):
    w, h = size
    img = Image.new("RGB", (w, h), "black")
    over = Image.new("RGB", (w, h), "black")
    d = ImageDraw.Draw(over)
    stick = max(2, round(4 * h / 512))

    def limb(a, b, color):
        if not (kp[a][2] and kp[b][2]):
            return
        (x1, y1), (x2, y2) = kp[a][:2], kp[b][:2]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1:
            return
        ang = math.atan2(y2 - y1, x2 - x1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        pts = []
        for t in range(0, 360, 15):
            r = math.radians(t)
            ex, ey = length / 2 * math.cos(r), stick * math.sin(r)
            pts.append((mx + ex * math.cos(ang) - ey * math.sin(ang), my + ex * math.sin(ang) + ey * math.cos(ang)))
        d.polygon(pts, fill=tuple(int(c * 0.6) for c in color))

    for i, (a, b) in enumerate(BODY25_LIMBS):
        limb(a, b, BODY25_COLORS[i])
    head_color = (170, 255, 0)
    limb(0, 14, head_color)
    limb(0, 15, head_color)
    limb(14, 16, head_color)
    limb(15, 17, head_color)
    for side, base, wrist in (("l", 18, 7), ("r", 32, 4)):
        for fi in range(5):
            p1 = base + fi * 3
            limb(wrist, p1, HAND_COLOR)
            limb(p1, p1 + 1, HAND_COLOR)
            limb(p1 + 1, p1 + 2, HAND_COLOR)
    img = Image.fromarray(np.maximum(np.asarray(img), np.asarray(over)))
    d = ImageDraw.Draw(img)
    for i, (x, y, v) in enumerate(kp):
        if not v:
            continue
        col = BODY25_COLORS[i] if i < len(BODY25_COLORS) else HAND_COLOR
        r = stick if i < 18 else max(1, stick // 2)
        d.ellipse((x - r, y - r, x + r, y + r), fill=col)
    img.save(out_png)


def depth8(png16, out_png):
    dep = np.asarray(Image.open(png16)).astype(np.float32)
    if dep.ndim == 3:
        dep = dep[..., 0]
    dep8 = Image.fromarray(np.clip(dep / max(dep.max(), 1e-6) * 255, 0, 255).astype(np.uint8)).convert("RGB")
    dep8.save(out_png)


def build_control(pose_png, depth_png, out_dir, frames, fps, mp4):
    """Writes the control video as two numbered image sequences, one folder per map.

    Separate folders per map, because LoadImagesFromFolderKJ loads EVERY image of the folder
    into one batch - pose and depth must not be mixed.
    """
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    pose = Image.open(pose_png).convert("RGB")
    depth = Image.open(depth_png).convert("RGB")
    for key, im in (("pose", pose), ("depth", depth)):
        d = os.path.join(out_dir, key)
        os.makedirs(d, exist_ok=True)
        for i in range(1, frames + 1):
            im.save(os.path.join(d, f"{key}_{i:05d}.png"))
    src = os.path.join(out_dir, "pose", "pose_%05d.png")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", src,
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--frames", type=int, default=5, help="control video length; 5 for a still, 124 for ~5 s")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    root = Path(a.root)
    for sd in sorted(p for p in root.iterdir() if (p / "pose25.json").exists()):
        if a.only and sd.name not in a.only.split(","):
            continue
        info = json.loads((sd / "pose25.json").read_text(encoding="utf-8"))
        size = tuple(info["size"])
        kp = info["keypoints_body25"]
        draw_pose25(kp, size, sd / "pose25.png")
        depth8(sd / "depth_loc.png", sd / "depth_loc8.png")
        ok = build_control(sd / "pose25.png", sd / "depth_loc8.png", sd / "control",
                           a.frames, a.fps, str(sd / "control.mp4"))
        # check sheet: depth | pose | pose over depth | mask
        dep = Image.open(sd / "depth_loc8.png").convert("RGB")
        pose = Image.open(sd / "pose25.png").convert("RGB")
        over = Image.fromarray(np.maximum(np.asarray(dep) // 2, np.asarray(pose)))
        mask = Image.open(sd / "mask.png").convert("RGB")
        tw, th = size[0] // 2, size[1] // 2
        sheet = Image.new("RGB", (tw * 2, th * 2))
        for i, im in enumerate((dep, pose, over, mask)):
            sheet.paste(im.resize((tw, th)), ((i % 2) * tw, (i // 2) * th))
        sheet.save(root / f"_sheet_{sd.name}.jpg", quality=88)
        nhands = sum(1 for i in range(18, len(BODY25_NAMES)) if kp[i][2])
        print(f"{sd.name}: pose25 {sum(v[2] for v in kp)}/{len(kp)} visible (hands {nhands}/{len(kp) - 18}), "
              f"frames {a.frames}, mp4 {'ok' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
