"""09.11a fail sheet: five playblast figure crops at exactly 2x, no thumbnails."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IDS = ('Q03', 'Q04', 'Q05', 'T01', 'T03')


def build(root, comfy, out):
    root, comfy = Path(root), Path(comfy)
    prompts = json.loads(Path(__file__).with_name('pose_round_prompts.json').read_text(encoding='utf-8'))['prompts']
    panels = []
    for short in IDS:
        p = next(p for p in prompts if p['id'].startswith(short))
        pid = p['id']
        fit = json.loads((root / 'pose_round/V2' / pid / 'fit.json').read_text(encoding='utf-8'))
        directory = comfy / 'input' / f"pose_V2_m{p['loc']}_s2" / pid
        silhouette = directory / 'silhouette.png'
        if not silhouette.is_file():
            silhouette = directory / 'mask.png'
        mask = Image.open(silhouette).convert('L')
        bb = mask.point(lambda v: 255 if v >= 128 else 0).getbbox()
        if bb is None:
            raise ValueError(f'{pid}: empty figure silhouette, no fabricated crop')
        image = Image.open(directory / 'playblast.png').convert('RGB')
        crop = image.crop(bb)
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.NEAREST)
        text = f"{pid} | {fit['status']} | {', '.join(fit.get('flags', [])) or '-'} | full-resolution figure crop 2x"
        panels.append((crop, text))
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
    width = max(max(p.width for p, _ in panels), 1400)
    sheet = Image.new('RGB', (width, sum(p.height + 44 for p, _ in panels)), '#181818')
    draw, y = ImageDraw.Draw(sheet), 0
    for panel, text in panels:
        draw.text((8, y + 7), text, font=font, fill='white')
        sheet.paste(panel, ((width - panel.width) // 2, y + 44))
        y += panel.height + 44
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=95)
    return sheet.size


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--comfy', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    print(build(args.root, args.comfy, args.out))
