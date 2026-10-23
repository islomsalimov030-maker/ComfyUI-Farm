#!/usr/bin/env python3
"""Прогон кусков farm_auto@1 / farm_manual@1 через /prompt (проверка сборки, 23.10).

    python tools/test_wf_api.py auto   --shot SAMH-STD06 [--models 1]
    python tools/test_wf_api.py manual --location loft.png --mods farm/samantha_face farm/samantha_athleisure_body
    python tools/test_wf_api.py facepass --video farm_auto/SAMH-STD06_ref2v_00001.mp4

Оставляет только экземпляры выбранных моделей (как mute в UI), выходы блока, ждёт завершения и печатает итог.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request as U
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8188"


def is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def prune(g: dict, keep_roles) -> dict:
    """Оставить ноды, нужные выходам с ролями keep_roles; ссылки на выкинутые ноды из необязательных входов убрать."""
    outs = [k for k, n in g.items() if n["_meta"].get("role") in keep_roles and
            n["class_type"] in ("SaveImage", "VHS_VideoCombine", "PreviewAny", "NGFarmZip")]
    need, stack = set(), list(outs)
    while stack:
        k = stack.pop()
        if k in need:
            continue
        need.add(k)
        for a, v in g[k]["inputs"].items():
            if is_link(v) and v[0] in g:
                stack.append(v[0])
    res = {k: json.loads(json.dumps(g[k])) for k in need}
    for n in res.values():
        for a in list(n["inputs"]):
            v = n["inputs"][a]
            if is_link(v) and v[0] not in res:
                del n["inputs"][a]
    return res


def drop_models(g: dict, keep: set[int]) -> dict:
    out = {}
    for k, n in g.items():
        r = n["_meta"].get("role", "")
        m = r[5] if r.startswith(("photo", "video")) and len(r) > 5 and r[5].isdigit() else None
        if m and int(m) not in keep:
            continue
        out[k] = n
    for n in out.values():
        for a in list(n["inputs"]):
            v = n["inputs"][a]
            if is_link(v) and v[0] not in out:
                del n["inputs"][a]
    return out


def submit(g: dict) -> dict:
    pid = str(uuid.uuid4())
    body = json.dumps({"prompt": {k: {"class_type": n["class_type"], "inputs": n["inputs"]} for k, n in g.items()},
                       "prompt_id": pid}).encode()
    req = U.Request(URL + "/prompt", body, {"Content-Type": "application/json"})
    try:
        r = json.load(U.urlopen(req))
    except U.HTTPError as e:
        print("ОТКАЗ:", e.read().decode()[:3000]); sys.exit(2)
    print("в очереди", r.get("prompt_id"), "номер", r.get("number"))
    t0 = time.time()
    while True:
        h = json.load(U.urlopen(f"{URL}/history/{pid}"))
        if pid in h:
            st = h[pid]["status"]
            print("статус", st.get("status_str"), f"{time.time() - t0:.0f} с")
            if st.get("status_str") != "success":
                print(json.dumps(st.get("messages", [])[-3:], ensure_ascii=False)[:3000])
            for nid, o in h[pid]["outputs"].items():
                print(nid, {k: (v if k == "text" else [x.get("filename") for x in v] if isinstance(v, list) else v)
                            for k, v in o.items()})
            return h[pid]
        time.sleep(5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["auto", "manual", "facepass", "auto_photo"])
    ap.add_argument("--shot", default="SAMH-STD06")
    ap.add_argument("--models", default="1")
    ap.add_argument("--location", default="loft_living_sheet.png")
    ap.add_argument("--mods", nargs=2, default=["farm/samantha_face", "farm/samantha_athleisure_body"])
    ap.add_argument("--prompt", default="")
    ap.add_argument("--video", default="")
    a = ap.parse_args()
    keep = {int(x) for x in a.models.split(",")}
    name = "farm_auto@1" if a.kind.startswith("auto") else "farm_manual@1"
    g = json.loads((ROOT / "workflows" / f"{name}.api.json").read_text(encoding="utf-8"))
    g = drop_models(g, keep)
    if a.kind.startswith("auto"):
        g["T"]["inputs"]["shot_id"] = a.shot
        roles = {"table", "photo:zip", *(f"photo{i}:out" for i in keep)}
        if a.kind == "auto":
            roles |= {"video:zip", *(f"video{i}:out" for i in keep)}
        g = prune(g, roles)
    elif a.kind == "manual":
        for n in g.values():
            t = n["_meta"]["title"]
            if t.startswith("ФОТО · локация"):
                n["inputs"]["image"] = a.location
            if t.startswith("ФОТО · моды"):
                n["inputs"]["mod_1"], n["inputs"]["mod_2"] = a.mods
            if t.startswith("ФОТО · мод лица"):
                n["inputs"]["mod_1"] = a.mods[0]
            if t.startswith("ФОТО · промпт кадра"):
                n["inputs"]["value"] = a.prompt
            if t.startswith("МАСКА · исходный"):
                n["inputs"]["image"] = a.location
        g = prune(g, {f"photo{i}:out" for i in keep})
    else:
        for n in g.values():
            if n["class_type"] == "StacyVideoInput":
                n["inputs"]["video"] = a.video
        g = prune(g, {"facepass:out"})
    print(name, a.kind, len(g), "нод")
    submit(g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
