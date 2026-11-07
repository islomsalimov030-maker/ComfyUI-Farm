"""Лист-таблица картинок: строки = записи spec, ячейки = картинки с подписями.

  python tools/rowsheet.py spec.json --out sheet.jpg [--cell-w 520] [--cell-h 390]

spec.json: [{"label": "...", "cells": [[path_or_null, "подпись"], ...]}, ...]
Каждая картинка вписывается в ячейку cell-w × cell-h с сохранением пропорций (letterbox, фон #202020);
нет файла или null — «n/a». Подписи — шрифтом с кириллицей.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG_CELL = (0x20, 0x20, 0x20)
BG_SHEET = (18, 18, 18)
LABEL_H, CAP_H, GAP = 18, 22, 4


def font_ru(size=13):
    for f in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def letterbox(path, cw, ch):
    cell = Image.new("RGB", (cw, ch), BG_CELL)
    if not path or not Path(path).is_file():
        return cell, False
    im = Image.open(path).convert("RGB")
    k = min(cw / im.width, ch / im.height)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
    return cell, True


def build(spec, out, cw=520, ch=390):
    font = font_ru()
    nc = max(len(r["cells"]) for r in spec)
    rh = LABEL_H + ch + CAP_H
    S = Image.new("RGB", (nc * (cw + GAP), len(spec) * rh), BG_SHEET)
    d = ImageDraw.Draw(S)
    for i, r in enumerate(spec):
        y = i * rh
        d.text((4, y + 2), r["label"], fill=(255, 255, 0), font=font)
        for j, (p, cap) in enumerate(r["cells"]):
            x = j * (cw + GAP)
            cell, ok = letterbox(p, cw, ch)
            S.paste(cell, (x, y + LABEL_H))
            if not ok:
                d.text((x + 6, y + LABEL_H + 6), "n/a", fill=(255, 80, 80), font=font)
            d.text((x + 4, y + LABEL_H + ch + 3), cap, fill=(255, 255, 255), font=font)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    S.save(out, quality=88)
    return S.size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-w", type=int, default=520)
    ap.add_argument("--cell-h", type=int, default=390)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    print(a.out, build(spec, a.out, a.cell_w, a.cell_h))


if __name__ == "__main__":
    main()
