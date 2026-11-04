"""Scene3D 3D-A: probe that the uploaded control folders load on the studio.

python tools/scene3d/probe_gate_folders.py [--scene S1_sofa_sit]

Queues a tiny graph (no model, no sampler): LoadImagesFromFolderKJ on the pose and the depth
folder -> SaveImage of the first frame of each. It proves that the folder path convention and the
uploaded files are correct before any expensive generation is started.

Read-only for the studio's state apart from two output PNGs.
"""
import argparse
import json
import os
import sys
import time

import requests

SCENE = "scene3d_gate"


def session():
    s = requests.Session()
    s.auth = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])
    return s, os.environ["COMFY_URL"].rstrip("/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="S1_sofa_sit")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=864)
    a = ap.parse_args()

    s, url = session()
    base = f"input/{SCENE}/{a.scene}/control"
    g = {}
    for key in ("pose", "depth"):
        g[f"load_{key}"] = {"class_type": "LoadImagesFromFolderKJ",
                            "inputs": {"folder": f"{base}/{key}", "width": a.width,
                                       "height": a.height, "keep_aspect_ratio": "stretch"}}
        g[f"first_{key}"] = {"class_type": "ImageFromBatch",
                             "inputs": {"images": [f"load_{key}", 0], "batch_index": 0,
                                        "length": 1}}
        g[f"save_{key}"] = {"class_type": "SaveImage",
                            "inputs": {"images": [f"first_{key}", 0],
                                       "filename_prefix": f"scene3d_gate/probe/{a.scene}/{key}"}}
    # the count output tells us how many frames the folder actually holds
    g["count"] = {"class_type": "PreviewAny", "inputs": {"source": ["load_pose", 2]}}

    r = s.post(f"{url}/prompt", json={"prompt": g}, timeout=120)
    if r.status_code != 200:
        print("queue failed:", r.status_code, r.text[:1500])
        return 1
    pid = r.json()["prompt_id"]
    print(f"queued {pid}")

    for _ in range(120):
        time.sleep(2)
        h = s.get(f"{url}/history/{pid}", timeout=60).json()
        if pid in h:
            out = h[pid]
            status = out.get("status", {})
            if status.get("status_str") == "error":
                print("ERROR:", json.dumps(status.get("messages", []), ensure_ascii=False)[:2000])
                return 1
            for nid, node in out.get("outputs", {}).items():
                for im in node.get("images", []):
                    print(f"{nid}: {im.get('filename')} {im.get('subfolder')}")
            txt = out.get("outputs", {}).get("count", {})
            print("count node:", json.dumps(txt, ensure_ascii=False)[:300])
            return 0
    print("timeout")
    return 1


if __name__ == "__main__":
    sys.exit(main())
