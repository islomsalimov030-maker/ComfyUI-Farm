"""Face similarity to the reference (face_sim: facenet/VGGFace2, diagnostics only) on a head crop placed by the
render's pose25.json (head_top_px / feet_px), so the face is the placed person, not any face in the room.

  python head_sim.py --scene-dir <maps>/<scene> img1.png img2.png ... [--json out.json]
Run with a torch Python + face_pkgs on sys.path (see 3DA_LOCAL_2026-11-07.md: runpy wrapper for python_embeded).
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import face_sim as fs  # noqa: E402

_REF = None


def head_box(scene_dir, size, half=110):
    pose = json.loads((Path(scene_dir) / 'pose25.json').read_text())
    src = Image.open(Path(scene_dir) / 'mask.png').size
    (hx, hy), (fx, fy) = pose['head_top_px'], pose['feet_px']
    hy += 0.6 * abs(fy - hy) / 7.5
    x, y = hx * size[0] / src[0], hy * size[1] / src[1]
    k = size[0] / 1344
    return int(x - half * k), int(y - half * k), int(x + half * k), int(y + half * k)


def sim(image_path, scene_dir):
    global _REF
    if _REF is None:
        _, _, _REF = fs.faces(Image.open(fs.REF).convert('RGB'))
    im = Image.open(image_path).convert('RGB')
    _, h, e = fs.faces(im.crop(head_box(scene_dir, im.size)))
    return (None if e is None else round(float(e @ _REF), 3)), round(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene-dir', required=True)
    ap.add_argument('--json', default='')
    ap.add_argument('images', nargs='+')
    a = ap.parse_args()
    out = {}
    for p in a.images:
        out[p] = sim(p, a.scene_dir)
        print(f'[sim] {Path(p).name}: {out[p][0]} (face {out[p][1]} px)', flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
