#!/usr/bin/env python3
"""Подача API-промпта в ComfyUI + опрос + мониторинг памяти (RSS cgroup + VRAM).

    python vsr_run.py <api.json> [--clip vsr_test/clip_x.mp4] [--cap 49]
                      [--prefix name] [--timeout 3600] [--no-monitor]

Мониторинг идёт локально (ComfyUI через туннель 127.0.0.1:8188), память с пода
снимается по SSH. Пики пишутся в <prefix>_mem.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

# Под задаётся переменными окружения (репозиторий публичный — адрес пода
# в коде не хранится):
#   POD_SSH_HOST, POD_SSH_PORT, POD_SSH_KEY, POD_SSH_USER
POD_SSH = [
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    f"{os.environ.get('POD_SSH_USER', 'root')}@{os.environ.get('POD_SSH_HOST', '')}",
    "-p", os.environ.get("POD_SSH_PORT", "22"),
    "-i", os.path.expanduser(os.environ.get("POD_SSH_KEY", "~/.ssh/id_ed25519")),
]


def get(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post(url: str, payload: dict, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def pod_mem() -> tuple[int, int]:
    """(cgroup memory.current, memory.peak) в байтах."""
    try:
        out = subprocess.run(
            POD_SSH + ["cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak"],
            capture_output=True, text=True, timeout=20,
        ).stdout.split()
        return int(out[0]), int(out[1])
    except Exception:
        return -1, -1


def vram_mb() -> int:
    try:
        out = subprocess.run(
            POD_SSH + ["nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        return int(out[0].strip())
    except Exception:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("api")
    ap.add_argument("--clip", help="значение для input'а video (имя файла в ComfyUI/input)")
    ap.add_argument("--cap", type=int, help="frame_load_cap")
    ap.add_argument("--prefix", default="vsr_run")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--out", help="куда сохранить итоговый history json")
    args = ap.parse_args()

    doc = json.load(open(args.api, encoding="utf-8"))
    prompt = doc["prompt"] if "prompt" in doc else doc

    # найти ноду загрузки видео (VHS_LoadVideo) и подменить clip/cap
    if args.clip or args.cap is not None:
        for nid, n in prompt.items():
            if n.get("class_type") == "VHS_LoadVideo":
                if args.clip:
                    n["inputs"]["video"] = args.clip
                if args.cap is not None:
                    n["inputs"]["frame_load_cap"] = args.cap
                print(f"[patch] node {nid} VHS_LoadVideo: video={n['inputs'].get('video')} "
                      f"cap={n['inputs'].get('frame_load_cap')}")

    client_id = "vsr_cli"
    t0 = time.time()
    res = post(f"{COMFY}/prompt", {"prompt": prompt, "client_id": client_id})
    pid = res.get("prompt_id")
    print(f"[submit] prompt_id={pid}  nodes={len(prompt)}")
    if "node_errors" in res and res["node_errors"]:
        print("[node_errors]", json.dumps(res["node_errors"], ensure_ascii=False, indent=2))
        return 2

    csv_path = f"{args.prefix}_mem.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_s,rss_bytes,rss_gb,vram_mb\n")

    last_log = 0
    while True:
        el = time.time() - t0
        if el > args.timeout:
            print(f"[timeout] {args.timeout}s")
            return 3
        try:
            h = get(f"{COMFY}/history/{pid}")
        except Exception:
            time.sleep(2)
            continue
        if pid in h:
            print(f"[done] {el:.1f}s")
            entry = h[pid]
            st = entry.get("status", {})
            print("[status]", json.dumps(st, ensure_ascii=False)[:1500])
            if args.out:
                json.dump(entry, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            outs = entry.get("outputs", {})
            for onid, o in outs.items():
                for kind in ("images", "gifs"):
                    for f in o.get(kind, []):
                        print(f"[out] node {onid} {kind}: {f.get('filename')} "
                              f"({f.get('subfolder','')}) type={f.get('type')}")
            ok = st.get("completed") or st.get("status_str") == "success"
            return 0 if ok else 4

        if not args.no_monitor and time.time() - last_log >= 5:
            last_log = time.time()
            cur, peak = pod_mem()
            v = vram_mb()
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{el:.1f},{cur},{cur/1e9:.2f},{v}\n")
            q = ""
            try:
                q = get(f"{COMFY}/queue")["queue"][0][1]
            except Exception:
                pass
            print(f"  [{el:6.1f}s] rss={cur/1e9:5.1f}GB peak={peak/1e9:5.1f}GB "
                  f"vram={v}MB queue={len(q)}")
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
