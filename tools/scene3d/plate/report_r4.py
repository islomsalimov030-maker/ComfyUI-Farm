"""Scene3D round 4b: face-crop sheet for already generated frames (no generation, runs on the PC).

Inputs are copies of pod files (media stay outside git):
  --src     folder with <scene>/<variant>_s<seed>_00001_.png and reference/0003.png
  --boxes   face_boxes.json: MTCNN boxes {"<scene>/<variant>_s<seed>": {"boxes": [[x1,y1,x2,y2,p], ...]}}
  --csv     step1_r4b.csv (frame_lap, bg_lap, bg_ratio, person_lap, face_sim, face_h, faces)
  --sims    optional JSON {"<scene>/<variant>_s<seed>": sim on the person's own face box}
Each row: frame with numbered boxes, then every detected box with context enlarged to 240x240 (including false detections),
then the numbers. Laplacian variance is printed as "lap" - it is a detail/noise measure, not quality.
  .venv/Scripts/python tools/scene3d/plate/report_r4.py --src D/source --boxes D/face_boxes.json \
      --csv D/step1_r4b.csv --out D/step1_faces.jpg
"""
import argparse
import csv
import json
import os

from PIL import Image, ImageDraw, ImageFont

FONTS = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size):
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sims", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="P4_table_stand,P8_hand_face")
    ap.add_argument("--variants", default="A5,A5e,A5n")
    a = ap.parse_args()
    boxes = json.load(open(a.boxes))
    rows = {(r["scene"], r["variant"], r["seed"]): r for r in csv.DictReader(open(a.csv))}
    sims = json.load(open(a.sims)) if a.sims else {}
    tw, th, crop = 640, 360, 240
    f_big, f_small = font(22), font(18)
    keys = [(sc, v, s) for sc in a.scenes.split(",") for v in a.variants.split(",") for s in ("101", "202")]
    row_h = th + 16
    W = 260 + tw + 3 * (crop + 10) + 520
    sheet = Image.new("RGB", (W, row_h * len(keys) + 20), "white")
    d = ImageDraw.Draw(sheet)
    for i, (sc, v, s) in enumerate(keys):
        y = 10 + i * row_h
        d.text((10, y + 10), f"{sc}\n{v} s{s}", fill="black", font=f_big)
        path = os.path.join(a.src, sc, f"{v}_s{s}_00001_.png")
        if not os.path.exists(path):
            d.text((270, y + th // 2), "NO FRAME (not generated)", fill="red", font=f_big)
            continue
        im = Image.open(path).convert("RGB")
        k = tw / im.width
        th_im = im.resize((tw, round(im.height * k)), Image.LANCZOS)
        dd = ImageDraw.Draw(th_im)
        bx = boxes.get(f"{sc}/{v}_s{s}", {}).get("boxes", [])
        for j, b in enumerate(bx):
            dd.rectangle([b[0] * k, b[1] * k, b[2] * k, b[3] * k], outline=(0, 255, 0), width=2)
            dd.text((b[0] * k, b[3] * k + 2), str(j + 1), fill=(0, 255, 0), font=f_small)
        sheet.paste(th_im, (260, y))
        x = 260 + tw + 10
        if sc == "P4_table_stand" and v == "A5e" and s == "101":
            # No detected face on the actual person (the one MTCNN box is a mirror false positive).
            bx = bx + [[1300, 235, 1355, 315, 0.0]]
        for j, b in enumerate(bx[:3]):
            w, h = b[2] - b[0], b[3] - b[1]
            m = max(w, h) * 0.6
            c = im.crop((int(b[0] - m), int(b[1] - m), int(b[2] + m), int(b[3] + m))).resize((crop, crop), Image.LANCZOS)
            sheet.paste(c, (x, y))
            label = f"#{j + 1} h{h}px p{b[4]:.2f}" if b[4] else "manual head crop (no sim)"
            d.text((x + 4, y + crop + 2), label, fill="black", font=f_small)
            x += crop + 10
        r = rows.get((sc, v, s))
        tx = 260 + tw + 3 * (crop + 10) + 10
        if r:
            lines = [f"frame lap {r['frame_lap']}", f"person-box lap {r['person_lap']}",
                     f"outside-person lap {r['bg_lap']} ({r['bg_ratio']} of photo)",
                     f"face sim (largest box) {r['face_sim']}  h{r['face_h']}  n{r['faces']}"]
            if f"{sc}/{v}_s{s}" in sims:
                lines.append(f"face sim (person box) {sims[f'{sc}/{v}_s{s}']}")
            d.text((tx, y + 10), "\n".join(lines), fill="black", font=f_small, spacing=6)
    sheet.save(a.out, quality=90)
    print(a.out, sheet.size)


if __name__ == "__main__":
    main()
