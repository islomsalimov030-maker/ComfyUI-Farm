"""Comparison sheet per scene: one column per image, a caption with PCK/face under each (pod-side, PIL only).

python step_sheet.py --out sheet.jpg --title P4 --cell "photo=/path/a.png" --cell "R1 s101=/path/b.png|pck 1.00 face 0.30"
Missing files are drawn as an empty grey cell with the label, so a partial series still gives a sheet.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W = 448


def font(size):
    for name in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='')
    ap.add_argument('--cell', action='append', default=[], help='label=path[|caption]')
    ap.add_argument('--cols', type=int, default=0, help='0 = all cells in one row')
    a = ap.parse_args()
    cells = []
    for c in a.cell:
        label, rest = c.split('=', 1)
        path, _, caption = rest.partition('|')
        cells.append((label, path, caption))
    cols = a.cols or len(cells)
    rows = (len(cells) + cols - 1) // cols
    h = W * 768 // 1344
    pad, top = 44, 30 if a.title else 0
    sheet = Image.new('RGB', (cols * W, top + rows * (h + pad)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    f, fs = font(16), font(13)
    if a.title:
        d.text((6, 6), a.title, fill=(255, 255, 255), font=f)
    for i, (label, path, caption) in enumerate(cells):
        x, y = (i % cols) * W, top + (i // cols) * (h + pad)
        if Path(path).is_file():
            sheet.paste(Image.open(path).convert('RGB').resize((W, h), Image.LANCZOS), (x, y))
        else:
            d.rectangle((x, y, x + W - 2, y + h), fill=(70, 70, 70))
            d.text((x + 8, y + h // 2), 'нет кадра', fill=(220, 220, 220), font=f)
        d.text((x + 6, y + h + 3), label, fill=(255, 230, 120), font=f)
        d.text((x + 6, y + h + 23), caption, fill=(220, 220, 220), font=fs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out, quality=88)
    print('[sheet]', a.out)


if __name__ == '__main__':
    main()
