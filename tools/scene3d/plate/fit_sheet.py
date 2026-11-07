"""3D-A step 2g sheet per scene: photo | playblast _s2 | L_R0 | L_R1 | L_R1 harm | S3 comp | S3 .25/.35 | S3b .35/.5,
each tile = full frame + 1:1 crops of head and feet (from pose25.json), one block per seed; metrics text if given.

  python fit_sheet.py --scene P4_table_stand --loc plate3d --seeds 101,202 --metrics m.json --out fit_P4_table_stand.jpg
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

import farm_env

TW, TH, C = 448, 256, 224   # thumbnail, crop side (crops are 1:1 pixels of the generation, C x C)
COLS = 6


def first(pattern_dir, pattern):
    files = sorted(Path(pattern_dir).glob(pattern))
    return files[-1] if files else None


def tiles(scene, loc, seed):
    inp, out = farm_env.COMFY / 'input', farm_env.COMFY / 'output'
    q, c = out / 'plate3d_quality' / scene, out / 'plate3d_comp' / scene
    return [('photo', inp / f'{loc}_rgb.png'), ('playblast _s2', inp / f'{loc}_s2' / scene / 'playblast.png'),
            ('L_R0', first(q, f'L_R0_s{seed}_*.png')), ('L_R1', first(q, f'L_R1_s{seed}_*.png')),
            ('L_R1 harm .25', first(q, f'L_R1_h25_s{seed}_*.png')), ('L_R1 harm .35', first(q, f'L_R1_h35_s{seed}_*.png')),
            ('S3 comp', first(c, f'comp_s{seed}_*.png')), ('S3 comp2 (no halo)', first(c, f'comp2_s{seed}.png')),('S3 .25', first(c, f'S3_h25_s{seed}_*.png')),
            ('S3 .35', first(c, f'S3_h35_s{seed}_*.png')), ('S3b .35', first(c, f'S3b_h35_s{seed}_*.png')),
            ('S3b .5', first(c, f'S3b_h50_s{seed}_*.png'))]


def anchors(loc, scene):
    pose = json.loads((farm_env.COMFY / 'input' / loc / scene / 'pose25.json').read_text())
    (hx, hy), (fx, fy) = pose['head_top_px'], pose['feet_px']   # photo pixels (render size = plate size)
    size = Image.open(farm_env.COMFY / 'input' / loc / scene / 'mask.png').size
    head_h = abs(fy - hy) / 7.5
    return (hx, hy + 0.6 * head_h), (fx, fy - 0.3 * head_h), size


def crop(im, center, src_size):
    if center is None:
        return Image.new('RGB', (C, C), (40, 40, 40))
    sw, sh = src_size if all(src_size) else (1536, 864)
    x, y = center[0] * im.width / sw, center[1] * im.height / sh
    box = (int(x - C / 2), int(y - C / 2), int(x + C / 2), int(y + C / 2))
    return im.crop(box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--loc', required=True)
    ap.add_argument('--seeds', default='101,202')
    ap.add_argument('--metrics', default='', help='json list of rows with file, pck_strict, face_similarity, people')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    met = {}
    if a.metrics:
        for p in a.metrics.split(','):
            if Path(p).is_file():
                for r in json.loads(Path(p).read_text()):
                    met[(Path(r['file']).parent.name, Path(r['file']).name)] = r
    head, feet, src = anchors(a.loc, a.scene)
    seeds = [int(s) for s in a.seeds.split(',')]
    blocks = []
    for seed in seeds:
        ts = tiles(a.scene, a.loc, seed)
        rows = (len(ts) + COLS - 1) // COLS
        block = Image.new('RGB', (COLS * TW, rows * (TH + C + 20) + 24), (18, 18, 18))
        d = ImageDraw.Draw(block)
        d.text((6, 4), f'{a.scene}  seed {seed}', fill=(255, 255, 0))
        for i, (label, path) in enumerate(ts):
            x, y = (i % COLS) * TW, 24 + (i // COLS) * (TH + C + 20)
            if path is None or not Path(path).is_file():
                d.text((x + 6, y + 6), f'{label}: missing', fill=(255, 80, 80))
                continue
            im = Image.open(path).convert('RGB')
            if label in ('photo', 'playblast _s2'):
                im = im.resize((1344, 768), Image.LANCZOS)
            block.paste(im.resize((TW, TH), Image.LANCZOS), (x, y))
            block.paste(crop(im, head, src), (x, y + TH))
            block.paste(crop(im, feet, src), (x + C, y + TH))
            r = met.get((Path(path).parent.name, Path(path).name))
            txt = label
            if r:
                fs = r.get('face_similarity')
                txt += f"  PCKs {r.get('pck_strict', 0):.2f}  face {'NA' if fs is None else f'{fs:.2f}'}  ppl {r.get('people')}"
            d.rectangle((x, y + TH + C, x + TW - 1, y + TH + C + 19), fill=(0, 0, 0))
            d.text((x + 4, y + TH + C + 4), txt, fill=(255, 255, 255))
        blocks.append(block)
    sheet = Image.new('RGB', (blocks[0].width, sum(b.height for b in blocks)), (0, 0, 0))
    y = 0
    for b in blocks:
        sheet.paste(b, (0, y))
        y += b.height
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out, quality=88)
    print(a.out, sheet.size)


if __name__ == '__main__':
    main()
