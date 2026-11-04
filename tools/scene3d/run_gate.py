"""Scene3D 3D-A: run the H3 Fun ControlNet gate on the studio.

python tools/scene3d/run_gate.py --branch depth_pose --scene S1_sofa_sit
python tools/scene3d/run_gate.py --all --scenes S1_sofa_sit,S2_bench_sit --seeds 101,202

Queues the API graphs from workflows/3d_gate/ one by one, waits for each, and saves the result
next to the graph output. Refuses to start if the studio queue is busy with somebody else's work
(the studio is shared) and never cancels a foreign prompt.

After a batch it calls POST /free {"unload_models":true,"free_memory":true}.

The branch graphs with control (pose, depth, depth_pose, depth_pose_tuned) need
MiniMaxH3FunControlNetApply - ComfyUI >= 0.35. On 0.34.6 they are rejected by the server with
"node not found"; the script reports that and stops instead of burning the queue.
"""
import argparse
import json
import os
import sys
import time

import requests

WF_DIR = "workflows/3d_gate"
BRANCHES = ["base", "pose", "depth", "depth_pose", "depth_pose_tuned"]


def session():
    s = requests.Session()
    s.auth = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])
    return s, os.environ["COMFY_URL"].rstrip("/")


def studio_version(s, url):
    return s.get(f"{url}/system_stats", timeout=60).json()["system"]["comfyui_version"]


def queue_busy(s, url):
    q = s.get(f"{url}/queue", timeout=60).json()
    return len(q.get("queue_running", [])), len(q.get("queue_pending", []))


def has_apply_node(s, url):
    info = s.get(f"{url}/object_info/MiniMaxH3FunControlNetApply", timeout=60)
    try:
        return bool(info.json())
    except Exception:
        return False


def wait(s, url, pid, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = s.get(f"{url}/history/{pid}", timeout=60).json()
        if pid in h:
            out = h[pid]
            st = out.get("status", {})
            if st.get("status_str") == "error":
                return None, json.dumps(st.get("messages", []), ensure_ascii=False)[:1200]
            return out, None
        time.sleep(3)
    return None, "timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="", help="one branch name, or empty with --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scenes", default="", help="comma separated scene names; default = all in the dir")
    ap.add_argument("--seeds", default="101,202")
    ap.add_argument("--dir", default=WF_DIR)
    ap.add_argument("--wait", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true", help="only list what would be queued")
    a = ap.parse_args()

    s, url = session()
    ver = studio_version(s, url)
    print(f"studio ComfyUI {ver}")
    have_apply = has_apply_node(s, url)
    print(f"MiniMaxH3FunControlNetApply: {'yes' if have_apply else 'NO (needs core >= 0.35)'}")

    branches = BRANCHES if a.all else [a.branch]
    if not branches or branches == [""]:
        print("nothing to do: give --branch or --all")
        return 1

    seeds = [x for x in a.seeds.split(",") if x.strip()]
    if a.scenes:
        scenes = a.scenes.split(",")
    else:
        scenes = sorted({f.split(".")[0] for f in os.listdir(a.dir) if f.endswith(".api.json")})

    jobs = []
    for sc in scenes:
        for br in branches:
            for sd in seeds:
                fn = f"{sc}.{br}.s{sd}.api.json"
                path = os.path.join(a.dir, fn)
                if not os.path.exists(path):
                    path = os.path.join(a.dir, f"{sc}.{br}.api.json")
                    if not os.path.exists(path):
                        continue
                jobs.append((path, br, sc, sd))
    print(f"{len(jobs)} jobs")
    for p, br, sc, sd in jobs:
        print(f"  {sc:20s} {br:18s} s{sd}")

    if a.dry_run:
        return 0
    if not have_apply and any(br != "base" for _, br, _, _ in jobs):
        print("\nSTOP: the control branches need MiniMaxH3FunControlNetApply (ComfyUI >= 0.35).")
        print("      Either upgrade the studio core, or run only the 'base' branch for now.")
        return 2

    run, pend = queue_busy(s, url)
    if run or pend:
        print(f"\nSTOP: the studio queue is busy (running={run}, pending={pend}). "
              "Somebody else's work is there - wait, do not preempt.")
        return 3

    done = []
    for path, br, sc, sd in jobs:
        graph = json.load(open(path, encoding="utf-8"))
        run, pend = queue_busy(s, url)
        if run or pend:
            print(f"\nSTOP: queue became busy before {os.path.basename(path)} - pausing.")
            break
        r = s.post(f"{url}/prompt", json={"prompt": graph}, timeout=120)
        if r.status_code != 200:
            print(f"\nERROR queueing {os.path.basename(path)}: {r.status_code} {r.text[:600]}")
            break
        pid = r.json()["prompt_id"]
        print(f"\n>>> {os.path.basename(path)}  pid={pid}")
        t0 = time.time()
        out, err = wait(s, url, pid, timeout=a.wait)
        dt = time.time() - t0
        if err:
            print(f"    FAILED after {dt:.0f}s: {err}")
            break
        imgs = []
        for nid, node in out.get("outputs", {}).items():
            for im in node.get("images", []):
                imgs.append(f"{im.get('subfolder','')}/{im.get('filename')}")
        print(f"    ok in {dt:.0f}s  -> {', '.join(imgs) if imgs else 'no image'}")
        done.append((os.path.basename(path), dt, imgs))

    print("\n--- free memory ---")
    s.post(f"{url}/free", json={"unload_models": True, "free_memory": True}, timeout=120)
    print(f"done {len(done)}/{len(jobs)}")
    for name, dt, _ in done:
        print(f"  {name}  {dt:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
