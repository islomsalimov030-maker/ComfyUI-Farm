"""Scene3D plate: comparison sheet of variants - full frame + person crop x2 + numbers (sharpness, face).
Run on the pod with the ComfyUI-next python:
  /workspace/ComfyUI-next/venv/bin/python tools/scene3d/plate/report_sheet.py --maps /workspace/ComfyUI-next/input/plate3d \
      --scenes P4_table_stand,P8_hand_face --variants A5,A5h,A5e,A5n --seeds 101,202 --face face_r4sharp --out x.jpg
Sharpness = Laplacian variance (frame resized to the photo size; person box = bbox of the soft mask) - a rough
blur signal, not a quality score. Face = facenet sim from face_<tag>.json (faces < 40 px unreliable).
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = "/workspace/ComfyUI-next/output/plate3d_ab"
INP = "/workspace/ComfyUI-next/input"


def lap(a):
    l = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float(l.var())


def font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--seeds", default="101,202")
    ap.add_argument("--face", default="", help="face_<tag>.json tag in /workspace/farm/scene3d")
    ap.add_argument("--size-tag", default="", help="file suffix after the variant, e.g. _1344")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    loc = os.path.basename(os.path.normpath(a.maps))
    photo_im = Image.open(f"{INP}/{loc}_rgb.png").convert("RGB")
    photo = np.asarray(photo_im.convert("L"), np.float32)
    face = {}
    if a.face:
        for r in json.load(open(f"/workspace/farm/scene3d/{a.face}.json")):
            face[(r["scene"], r["variant"], r["seed"])] = r
    seeds = [int(s) for s in a.seeds.split(",")]
    vs = a.variants.split(",")
    tw, th, cw = 512, 288, 288           # tile: frame 512x288 + person crop 288x288
    f1, f2 = font(18), font(15)
    rows_ = []
    scenes = a.scenes.split(",")
    nrow = len(scenes) * len(seeds)
    sheet = Image.new("RGB", (170 + len(vs) * (tw + cw + 8), 30 + nrow * (th + 52)), "white")
    dr = ImageDraw.Draw(sheet)
    for j, v in enumerate(vs):
        dr.text((170 + j * (tw + cw + 8) + 4, 6), v, fill="black", font=f1)
    r = 0
    for sc in scenes:
        mpath = f"{INP}/{loc}_{sc}_mask_soft.png" if os.path.exists(f"{INP}/{loc}_{sc}_mask_soft.png") \
            else f"{INP}/plate3d_{sc}_mask_soft.png"
        for s in seeds:
            y = 30 + r * (th + 52)
            dr.text((4, y + 4), f"{sc}\ns{s}\nphoto lap {lap(photo):.0f}", fill="black", font=f2)
            for j, v in enumerate(vs):
                x = 170 + j * (tw + cw + 8)
                fs = sorted(glob.glob(f"{OUT}/{sc}/{v}{a.size_tag}_s{s}_*.png"))
                if not fs:
                    dr.text((x + 4, y + 4), "НЕТ КАДРА", fill="red", font=f1)
                    continue
                im = Image.open(fs[-1]).convert("RGB")
                g = np.asarray(im.convert("L").resize(photo_im.size), np.float32)
                m = Image.open(mpath).convert("L").resize(im.size)
                bb = m.getbbox()
                box = im.crop(bb) if bb else im
                lf, lb = lap(g), lap(np.asarray(box.convert("L"), np.float32))
                # background = frame outside the dilated person box (resized to the photo size)
                bg = g.copy()
                if bb:
                    k = photo_im.width / im.width
                    x0, y0, x1, y1 = [int(round(c * k)) for c in bb]
                    bg[max(0, y0 - 40):y1 + 40, max(0, x0 - 40):x1 + 40] = np.nan
                    pm = photo.copy()
                    pm[np.isnan(bg)] = np.nan
                lbg = float(np.nanvar(_lapn(bg))) if bb else lf
                lph = float(np.nanvar(_lapn(pm))) if bb else lap(photo)
                fr = face.get((sc, v, s), {})
                sim, fh = fr.get("sim"), fr.get("face_h_px", 0)
                sheet.paste(im.resize((tw, th)), (x, y))
                bw, bh = box.size
                kk = min(cw / bw, th / bh)
                sheet.paste(box.resize((max(1, int(bw * kk)), max(1, int(bh * kk))), Image.LANCZOS), (x + tw + 2, y))
                txt = (f"фон {lbg:.0f} ({lbg / lph:.2f} фото)  перс {lb:.0f}  "
                       f"лицо {'—' if sim is None else f'{sim:.2f}'} h{fh}{' мало' if 0 < fh < 40 else ''}"
                       f"{' x' + str(fr.get('faces')) if fr.get('faces', 1) > 1 else ''}")
                dr.text((x + 2, y + th + 4), txt, fill="black", font=f2)
                rows_.append(dict(scene=sc, variant=v, seed=s, frame_lap=round(lf), bg_lap=round(lbg),
                                  bg_ratio=round(lbg / lph, 2), person_lap=round(lb), face_sim=sim, face_h=fh,
                                  faces=fr.get("faces")))
            r += 1
    sheet.save(a.out, quality=88)
    print("[sheet]", a.out)
    for x in rows_:
        print("[row]", json.dumps(x, ensure_ascii=False))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh_:
            w = csv.DictWriter(fh_, fieldnames=list(rows_[0]))
            w.writeheader()
            w.writerows(rows_)


def _lapn(a):
    l = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return l


if __name__ == "__main__":
    main()
