"""Scene3D 3D-A: upload the H3 control maps to the studio's input dir.

python tools/scene3d/upload_gate_maps.py C:/work/scene3d_work/gate [--scene all] [--frames 5]

Everything lands under input/scene3d_gate/<scene>/... - a separate folder, nothing of anybody
else is touched. Two kinds of files go up:

  control/pose_%05d.png, control/depth_%05d.png   - the control video as an image sequence
  pose25.png, depth_loc8.png                      - the single-frame maps (stills path)

The Apply node wants an IMAGE batch, so the sequence is uploaded as numbered PNGs and read back
with LoadImagesFromFolderKJ (folder = scene3d_gate/<scene>/control). The single map is uploaded
for the still path and for the visual check.

Only uploads. Nothing is queued, no model is loaded.
"""
import argparse
import os
import sys
from pathlib import Path

import requests

SCENE = "scene3d_gate"


def session():
    s = requests.Session()
    s.auth = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])
    return s, os.environ["COMFY_URL"].rstrip("/")


def upload(s, url, path, name, subfolder):
    with open(path, "rb") as f:
        r = s.post(f"{url}/upload/image", files={"image": (name, f, "image/png")},
                   data={"overwrite": "true", "type": "input", "subfolder": subfolder}, timeout=300)
    r.raise_for_status()
    j = r.json()
    return j.get("subfolder", ""), j.get("name", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--scene", default="all", help="all or a comma separated list of scene names")
    ap.add_argument("--frames", type=int, default=5, help="how many control frames to upload")
    ap.add_argument("--maps", action="store_true", help="also upload the single-frame maps")
    a = ap.parse_args()

    s, url = session()
    root = Path(a.root)
    scenes = [p for p in sorted(root.iterdir()) if (p / "pose25.png").exists()]
    if a.scene != "all":
        want = set(a.scene.split(","))
        scenes = [p for p in scenes if p.name in want]

    total = 0
    for sd in scenes:
        base = f"{SCENE}/{sd.name}"
        n = 0
        for key in ("pose", "depth"):
            src = sd / "control" / key / f"{key}_00001.png"
            if not src.exists():
                continue
            for i in range(1, a.frames + 1):
                p = sd / "control" / key / f"{key}_{i:05d}.png"
                if not p.exists():
                    break
                upload(s, url, p, f"{key}_{i:05d}.png", f"{base}/control/{key}")
                n += 1
        if a.maps:
            for src, name in ((sd / "pose25.png", "pose25.png"), (sd / "depth_loc8.png", "depth_loc8.png")):
                if src.exists():
                    upload(s, url, src, name, base)
                    n += 1
        total += n
        print(f"{sd.name}: {n} files -> input/{base}/control/{{pose,depth}}")
    print(f"total {total} files")


if __name__ == "__main__":
    sys.exit(main())
