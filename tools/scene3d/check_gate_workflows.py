"""Scene3D 3D-A: check that the gate workflows only use nodes the studio actually has.

python tools/scene3d/check_gate_workflows.py workflows/3d_gate

Read-only: it compares every class_type in each API graph against the studio's /object_info and
reports the missing nodes. Nothing is queued, nothing is generated - this is the cheap check that
tells us whether a branch can run on the studio's ComfyUI version at all.
"""
import argparse
import json
import os
import sys

import requests


def session():
    s = requests.Session()
    s.auth = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])
    return s, os.environ["COMFY_URL"].rstrip("/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    s, url = session()
    info = s.get(f"{url}/object_info", timeout=120).json()
    version = s.get(f"{url}/system_stats", timeout=60).json()["system"]["comfyui_version"]
    print(f"studio ComfyUI {version}, {len(info)} nodes\n")
    rc = 0
    for d in a.dirs:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".api.json"):
                continue
            g = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            missing = sorted({v["class_type"] for v in g.values() if v.get("class_type") not in info})
            status = "OK " if not missing else "MISSING"
            print(f"[{status}] {d}/{fn}  nodes={len(g)}" + (f"  -> {missing}" if missing else ""))
            if missing:
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
