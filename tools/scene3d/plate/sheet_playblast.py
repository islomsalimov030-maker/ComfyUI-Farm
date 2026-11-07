"""Sheet: photo | playblast _s | playblast _s2, crop 1:1 around the figure (mask bbox + margin), one row per scene.

  python sheet_playblast.py --input $FARM_COMFY/input --out playblast_s_vs_s2.jpg [--suffixes _s,_s2]
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

SCENES = [("plate3d", "P4_table_stand"), ("plate3d", "P8_hand_face"), ("plate3d_m3", "M3_bed_sit")]


def bbox(mask_path, margin=0.35):
    m = Image.open(mask_path).convert("L")
    if sum(m.histogram()[128:]) > m.width * m.height / 2:   # render may paint the room white (see build_ab.prep)
        m = m.point(lambda p: 255 - p)
    x0, y0, x1, y1 = m.point(lambda p: 255 if p > 127 else 0).getbbox()
    pad = int(margin * max(x1 - x0, y1 - y0))
    return max(0, x0 - pad), max(0, y0 - pad), min(m.width, x1 + pad), min(m.height, y1 + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffixes", default="_s,_s2")
    a = ap.parse_args()
    inp = Path(a.input)
    sfx = a.suffixes.split(",")
    rows = []
    for loc, sc in SCENES:
        box = bbox(inp / f"{loc}{sfx[-1]}" / sc / "mask.png")
        photo = Image.open(inp / loc / "_plate" / "plate_rgb.png").convert("RGB")
        tiles = [("photo", photo)] + [(f"{loc}{s}", Image.open(inp / f"{loc}{s}" / sc / "playblast.png").convert("RGB"))
                                     for s in sfx]
        crops = []
        for label, im in tiles:
            if im.size != photo.size:
                im = im.resize(photo.size, Image.LANCZOS)
            c = im.crop(box)
            ImageDraw.Draw(c).text((6, 6), f"{sc} {label}", fill=(255, 255, 0))
            crops.append(c)
        rows.append(crops)
    w = max(sum(c.width for c in r) for r in rows)
    h = sum(r[0].height for r in rows)
    sheet = Image.new("RGB", (w, h), (20, 20, 20))
    y = 0
    for r in rows:
        x = 0
        for c in r:
            sheet.paste(c, (x, y))
            x += c.width
        y += r[0].height
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out, quality=90)
    print(a.out, sheet.size)


if __name__ == "__main__":
    main()
