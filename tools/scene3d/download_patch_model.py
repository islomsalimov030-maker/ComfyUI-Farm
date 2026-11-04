"""WS-3D-A: download the H3 Fun ControlNet patch into the studio's models/model_patches/.

python tools/scene3d/download_patch_model.py

The studio has no shell, but it has the "HF Downloader" node (local_path = model_patches), so the
weight is fetched by the studio itself into the right folder. The file is the pruned int8_convrot
variant (2.30 GB) from Comfy-Org/MiniMax-H3 - the bf16 one (4.22 GB) is already on the studio but
is too big to sit next to the 19.5 GB int8 UNet on a 32 GB card.

Only downloads a model file into the models folder; nothing else on the studio is touched.
"""
import json
import os
import sys
import time

import requests

REPO = "Comfy-Org/MiniMax-H3"
FILENAME = "model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"
LOCAL_PATH = "model_patches"


def session():
    s = requests.Session()
    s.auth = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])
    return s, os.environ["COMFY_URL"].rstrip("/")


def main():
    s, url = session()
    before = set(s.get(f"{url}/models/{LOCAL_PATH}", timeout=60).json())
    name = FILENAME.split("/")[-1]
    if name in before:
        print(f"already on the studio: {name}")
        return 0

    g = {"dl": {"class_type": "HF Downloader",
                "inputs": {"repo_id": REPO, "filename": FILENAME,
                           "local_path": LOCAL_PATH, "overwrite": True}}}
    r = s.post(f"{url}/prompt", json={"prompt": g}, timeout=120)
    if r.status_code != 200:
        print("queue failed:", r.status_code, r.text[:600])
        return 1
    pid = r.json()["prompt_id"]
    print(f"queued download {pid} (2.3 GB, may take a few minutes)")

    for _ in range(900):
        time.sleep(4)
        h = s.get(f"{url}/history/{pid}", timeout=60).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                print("ERROR:", json.dumps(st.get("messages", []), ensure_ascii=False)[:1200])
                return 1
            after = set(s.get(f"{url}/models/{LOCAL_PATH}", timeout=60).json())
            new = after - before
            print("new in models/model_patches:", new or "(none visible yet)")
            return 0 if new else 1
    print("timeout")
    return 1


if __name__ == "__main__":
    sys.exit(main())
