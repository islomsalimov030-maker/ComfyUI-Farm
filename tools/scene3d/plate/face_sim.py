"""Scene3D plate: face similarity of the generated frames against the identity reference (0003.png).

facenet-pytorch 2.6.0 (MIT code; InceptionResnetV1 weights 'vggface2' - VGGFace2 is a research dataset, we use the
weights only as a measuring tool, never inside the generation). Installed WITHOUT deps next to ComfyUI-next
(its requirements would downgrade torch):
  /workspace/ComfyUI-next/venv/bin/python -m pip install --no-deps --target /workspace/venvs/face_pkgs facenet-pytorch==2.6.0
Run on the pod:
  PYTHONPATH=/workspace/venvs/face_pkgs /workspace/ComfyUI-next/venv/bin/python tools/scene3d/plate/face_sim.py \
      --glob "P*/A5_s*" [--runs /workspace/farm/scene3d/r4_runs.json] --tag r4
Per frame: number of faces (p > 0.95), height of the largest face box in px, cosine similarity to the reference.
Faces under ~40 px tall give unreliable numbers (flag 'small'). Composites (*c_s*) are skipped: same face.
Writes /workspace/farm/scene3d/face_<tag>.json and prints a per-variant summary.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

OUT = "/workspace/ComfyUI-next/output/plate3d_ab"
REF = "/workspace/ComfyUI-next/input/0003.png"
dev = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, min_face_size=15, device=dev)
net = InceptionResnetV1(pretrained="vggface2").eval().to(dev)


def faces(img):
    """Returns (count, box height, embedding of the largest face) - retries at 2x for small faces."""
    for scale in (1, 2):
        im = img if scale == 1 else img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        boxes, probs = mtcnn.detect(im)
        if boxes is None:
            continue
        keep = [i for i, p in enumerate(probs) if p is not None and p > 0.95]
        if not keep:
            continue
        crops = mtcnn.extract(im, boxes[keep], None)
        hs = [(boxes[i][3] - boxes[i][1]) / scale for i in keep]
        j = max(range(len(keep)), key=lambda k: hs[k])
        with torch.no_grad():
            e = net(crops[j:j + 1].to(dev))[0]
        return len(keep), float(hs[j]), torch.nn.functional.normalize(e, dim=0)
    return 0, 0.0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*/A*_s*", help="relative to output/plate3d_ab, .png added")
    ap.add_argument("--runs", default="", help="ab_runs.json: only these scene/variant/seed")
    ap.add_argument("--tag", default="face")
    ap.add_argument("--ref", default=REF)
    a = ap.parse_args()
    _n, _h, ref = faces(Image.open(a.ref).convert("RGB"))
    if ref is None:
        raise SystemExit(f"no face in the reference {a.ref}")
    files = sorted(glob.glob(os.path.join(OUT, a.glob + "*.png")))
    if a.runs:
        want = {(r["scene"], f"{r['variant']}_s{r['seed']}_") for r in json.load(open(a.runs))}
        files = [f for f in files if any(os.path.basename(os.path.dirname(f)) == sc and
                                         os.path.basename(f).startswith(pre) for sc, pre in want)]
    rows, by = [], defaultdict(list)
    for f in files:
        b = os.path.basename(f)
        m = re.match(r"(.+?)_s(\d+)_", b)
        if not m or m.group(1).endswith("c"):
            continue
        n, h, e = faces(Image.open(f).convert("RGB"))
        sim = float(e @ ref) if e is not None else None
        r = {"scene": os.path.basename(os.path.dirname(f)), "variant": m.group(1), "seed": int(m.group(2)),
             "file": b, "faces": n, "face_h_px": round(h), "sim": None if sim is None else round(sim, 3),
             "small": h < 40}
        rows.append(r)
        by[r["variant"]].append(r)
        print(f"[face] {r['scene']:16s} {r['variant']:8s} s{r['seed']}: faces {n}, h {r['face_h_px']:4d} px, "
              f"sim {r['sim']}{'  (small)' if r['small'] else ''}", flush=True)
    for v, rs in sorted(by.items()):
        ok = [r["sim"] for r in rs if r["sim"] is not None and not r["small"]]
        multi = sum(r["faces"] > 1 for r in rs)
        none = sum(r["faces"] == 0 for r in rs)
        mean = sum(ok) / len(ok) if ok else float("nan")
        print(f"[face] {v:8s}: n {len(rs)}, mean sim {mean:.3f} (faces >= 40 px: {len(ok)}), "
              f"no face {none}, >1 face {multi}")
    path = f"/workspace/farm/scene3d/face_{a.tag}.json"
    json.dump(rows, open(path, "w"), indent=1)
    print("[face] ->", path)


if __name__ == "__main__":
    main()
