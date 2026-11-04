"""Summarise quality frames with existing DWPose records and person-bound face crops.
Run in ComfyUI-next's Python with PYTHONPATH=/workspace/venvs/face_pkgs,
CUDA_VISIBLE_DEVICES='' so face evaluation does not compete with generation.
Media and per-frame diagnostics are saved outside the repository.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import face_sim as face


def lap(image):
    a = np.asarray(image.convert('L'), dtype=float)
    return a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]


def font(size):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poses', required=True, help='comma-separated pose_metrics JSON files')
    parser.add_argument('--maps', required=True)
    parser.add_argument('--group', default='Q1')
    parser.add_argument('--out', required=True)
    parser.add_argument('--ref', default=face.REF)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    poses = {}
    for path in args.poses.split(','):
        for row in json.loads(Path(path).read_text()):
            if Path(row['file']).name.startswith(args.group + '_'):
                poses[row['file']] = row
    _, _, reference = face.faces(Image.open(args.ref).convert('RGB'))
    if reference is None:
        raise SystemExit('No face in reference')
    photo = Image.open(Path(args.maps) / '_plate/plate_rgb.png').convert('RGB')
    scenes = sorted({r['scene'] for r in poses.values()})
    rows = []
    for scene in scenes:
        records = sorted((r for r in poses.values() if r['scene'] == scene), key=lambda r: r['file'])
        size = Image.open(records[0]['file']).size
        original = photo.resize(size, Image.Resampling.LANCZOS)
        # Common scene-level background ROI: remove every detected person box plus target silhouette.
        background = np.ones((size[1], size[0]), dtype=bool)
        mask = Image.open(Path(args.maps) / scene / 'mask.png').convert('L').resize(size)
        m = np.asarray(mask) > 127
        if m.mean() > .5:
            m = ~m
        background[m] = False
        for record in records:
            if record['person_box']:
                x0, y0, x1, y1 = map(int, record['person_box'])
                background[max(0, y0 - 24):min(size[1], y1 + 24),
                           max(0, x0 - 24):min(size[0], x1 + 24)] = False
        valid = (background[1:-1, 1:-1] & background[:-2, 1:-1] & background[2:, 1:-1]
                 & background[1:-1, :-2] & background[1:-1, 2:])
        original_lap = float(lap(original)[valid].var())
        sheet = Image.new('RGB', (1050, 345 * len(records)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, record in enumerate(records):
            image = Image.open(record['file']).convert('RGB')
            height = 0
            count = 0
            similarity = None
            head = None
            person_lap = None
            if record['person_box']:
                x0, y0, x1, y1 = record['person_box']
                box = (max(0, int(x0)), max(0, int(y0)), min(image.width, int(x1)), min(image.height, int(y1)))
                person = image.crop(box)
                if person.width > 3 and person.height > 3:
                    person_lap = float(lap(person).var())
                # Detection is restricted to the upper part of the actual detected person,
                # rather than selecting the largest MTCNN 'face' anywhere in the room.
                head = image.crop((box[0], box[1], box[2], min(box[3], int(y0 + (y1 - y0) * .4))))
                count, height, embedding = face.faces(head)
                if embedding is not None:
                    similarity = float(embedding @ reference)
            bg = float(lap(image)[valid].var())
            row = dict(file=record['file'], scene=scene, people=record['people'],
                       pck=record['pck'], pck_strict=record['pck_strict'], coverage=record['coverage'],
                       frame_lap=round(float(lap(image).var())), person_lap=person_lap,
                       background_ratio=bg / original_lap, face_similarity=similarity,
                       face_height=round(height), face_candidates=count)
            rows.append(row)
            y = 345 * i
            draw.text((5, y + 4), Path(record['file']).name, fill='black', font=font(17))
            sheet.paste(image.resize((560, 320)), (0, y + 25))
            if head is not None and head.width and head.height:
                head.thumbnail((240, 270), Image.Resampling.LANCZOS)
                sheet.paste(head, (565 + (240 - head.width) // 2, y + 30))
            draw.text((815, y + 35), '\n'.join([
                f"people {record['people']}", f"PCK {record['pck']:.2f}",
                f"strict {record['pck_strict']:.2f}", f"coverage {record['coverage']:.2f}",
                f"bg/photo {bg / original_lap:.2f}", f"face {similarity:.3f}" if similarity is not None else 'face NA',
                f'face h {height:.0f}px', 'SMALL / unreliable' if height < 40 else '',
                'head crop x4 approx', 'visual review required']), fill='black', font=font(16), spacing=6)
            print('[report]', Path(record['file']).name, json.dumps(row), flush=True)
        sheet.save(output / f'{args.group}_{scene}.jpg', quality=90)
    if rows:
        (output / f'{args.group}_metrics.json').write_text(json.dumps(rows, indent=2))
        with (output / f'{args.group}_metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('[report] DONE', len(rows), flush=True)


if __name__ == '__main__':
    main()
