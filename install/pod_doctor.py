#!/usr/bin/env python3
"""Проверка пода для фермы (WS-10): хватает ли ComfyUI нод и файлов моделей для движков фермы.

    python install/pod_doctor.py                       # все workflows/*.api.json
    python install/pod_doctor.py workflows/manual_frame@1.api.json --url http://127.0.0.1:8188

Ничего не ставит в очередь: берёт `GET /object_info` и сверяет каждый узел графа — класс есть,
значения выпадающих списков (файлы моделей, LoRA, сэмплеры) есть среди допустимых. LoRA из
Power Lora Loader сверяются со списком `LoraLoaderModelOnly.lora_name`. Только stdlib.
Выход: 0 — всё есть, 1 — чего-то не хватает (список в выводе), 2 — ComfyUI не отвечает.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def object_info(url: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/object_info", timeout=60) as r:
        return json.loads(r.read())


def combo_options(spec) -> list | None:
    """Допустимые значения входа-списка: старый формат [[...], {}] или новый ["COMBO", {"options": [...]}]."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


def check_graph(graph: dict, info: dict) -> tuple[set, list]:
    missing_cls, bad = set(), []
    loras = combo_options(((info.get("LoraLoaderModelOnly") or {}).get("input") or {})
                          .get("required", {}).get("lora_name")) or []
    for nid, node in graph.items():
        cls = node.get("class_type")
        title = (node.get("_meta") or {}).get("title") or nid
        if cls not in info:
            missing_cls.add(cls)
            continue
        spec = info[cls].get("input") or {}
        allspec = {**(spec.get("required") or {}), **(spec.get("optional") or {})}
        for k, v in (node.get("inputs") or {}).items():
            if isinstance(v, dict) and "lora" in v:            # Power Lora Loader (rgthree)
                if v.get("lora") and v.get("lora") not in loras:
                    bad.append((title, k, v["lora"]))
                continue
            if not isinstance(v, str) or (cls.startswith("Load") and k in ("image", "video", "audio")):
                continue                                       # входные картинки подставляет воркер
            opts = combo_options(allspec.get(k))
            if opts and v not in opts and v.replace("\\", "/") not in opts and v.replace("/", "\\") not in opts:
                bad.append((title, k, v))
    return missing_cls, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="*", help="*.api.json (по умолчанию все в workflows/)")
    ap.add_argument("--url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    files = [Path(p) for p in a.graphs] or sorted((REPO / "workflows").glob("*.api.json"))
    try:
        info = object_info(a.url)
    except Exception as e:  # noqa: BLE001
        print(f"ComfyUI {a.url} не отвечает: {e}")
        return 2
    ok = True
    for f in files:
        graph = json.loads(f.read_text(encoding="utf-8"))
        cls, bad = check_graph(graph, info)
        if not cls and not bad:
            print(f"  OK  {f.name}")
            continue
        ok = False
        print(f"  !!  {f.name}")
        for c in sorted(cls):
            print(f"      нет ноды: {c}")
        for title, k, v in bad:
            print(f"      нет значения: {title}.{k} = {v}")
    if not ok:
        print("Значения по умолчанию графа (рука, LoRA) воркер может подменить из профиля — "
              "критично только то, что профиль реально использует.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
