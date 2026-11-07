"""Shot plans without moving the camera (07.11d): crop photo + playblast + shadow + mask by the same rectangle
(photo aspect kept, so the H3 generation size and the body/room proportions stay as in the full frame).
  full   — whole figure fills ~74 % of the frame height (or width for a horizontal figure)
  medium — vertical figure: from the top of the head to ~60 % of the body; horizontal (lying): whole figure, tighter
Zoom is limited to 2.5x (crop >= 40 % of the photo height). Pure PIL; used by pose_h3.py --plan.
  python plan_crop.py <scene_dir> <plate_rgb.png> <out_dir> --plan full|medium
"""
import argparse
import json
from pathlib import Path

from PIL import Image

MIN_FRAC = 0.40


def figure_bbox(mask_path):
    mk = Image.open(mask_path).convert("L")
    if sum(mk.histogram()[128:]) > mk.width * mk.height / 2:      # some renders store the inverse
        mk = mk.point(lambda q: 255 - q)
    return mk.point(lambda q: 255 if q > 127 else 0).getbbox(), mk.size


def plan_rect(bbox, size, plan):
    W, H = size
    A = W / H
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    vertical = bh > bw
    if plan == "full":
        h = max(1.35 * bh, 1.35 * bw / A)
    elif plan == "medium":
        if vertical:
            h, cy = max(0.75 * bh, 1.15 * bw / A), y0 + 0.33 * bh
        else:
            h = max(1.15 * bh, 1.15 * bw / A)
    else:
        raise ValueError(plan)
    h = min(H, max(h, MIN_FRAC * H))
    w = h * A
    left = min(max(cx - w / 2, 0), W - w)
    top = min(max(cy - h / 2, 0), H - h)
    return [int(round(left)), int(round(top)), int(round(left + w)), int(round(top + h))]


def crop_scene(scene_dir, plate_rgb, out_dir, plan):
    scene_dir, out_dir = Path(scene_dir), Path(out_dir)
    bbox, size = figure_bbox(scene_dir / "mask.png")
    photo = Image.open(plate_rgb)
    if photo.size != size:                                         # rect is computed in the render size
        k = photo.width / size[0]
    else:
        k = 1.0
    rect = plan_rect(bbox, size, plan)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("playblast.png", "shadow.png", "mask.png"):
        if (scene_dir / name).is_file():
            Image.open(scene_dir / name).crop(rect).save(out_dir / name)
    pr = [int(round(v * k)) for v in rect]
    photo.crop(pr).save(out_dir / "plate_rgb.png")
    info = {"plan": plan, "rect_render": rect, "rect_photo": pr, "figure_bbox": list(bbox),
            "zoom": round(size[1] / (rect[3] - rect[1]), 2)}
    (out_dir / "plan.json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene_dir")
    ap.add_argument("plate_rgb")
    ap.add_argument("out_dir")
    ap.add_argument("--plan", choices=["full", "medium"], required=True)
    a = ap.parse_args()
    print(json.dumps(crop_scene(a.scene_dir, a.plate_rgb, a.out_dir, a.plan)))
