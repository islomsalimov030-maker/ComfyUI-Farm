"""Прогон готового воркфлоу на студии: /prompt -> /history -> /view.

Использование:
    python tools/wf_run_studio.py <api.json> [--model N] [--fizgig 0|1] [--tag NAME]

Печатает итог в stdout и кладёт превью в .tmp_wf/<tag>.png (если есть).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

BASE = os.environ["COMFY_URL"].rstrip("/")
AUTH = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])
OUT = Path(__file__).resolve().parent.parent / ".tmp_wf"

# узлы верхнего уровня: title -> ("class_type", ключ в widgets/inputs)
TOP_INT = "Модель (0 fl2v · 1 ref2v · 2 10Eros · 3 PinkCherry · 4 Singularity)"
TOP_FIZ = "Fizgig (1 кадр, быстрее)"

# слайдеры шагов внутри сабграфа (id узлов в UI-формате)
SG_FIRST_STEPS = "10303"
SG_SECOND_STEPS = "10304"


def find_top(api, title_contains):
    for nid, n in api.items():
        if title_contains.lower() in str(n.get("_meta", {}).get("title", "")).lower():
            return nid, n
    return None, None


def set_value(node, value):
    """Поставить значение в ноду API: пробуем inputs, потом widgets_values."""
    cls = node.get("class_type")
    if cls in ("PrimitiveInt", "PrimitiveBoolean", "PrimitiveFloat"):
        node.setdefault("inputs", {})["value"] = value
        return True
    if "inputs" in node and node["inputs"]:
        for k in list(node["inputs"]):
            if k in ("value", "index", "int", "model_sel", "use_fizgig"):
                node["inputs"][k] = value
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("api_json")
    ap.add_argument("--model", type=int, default=None, help="индекс модели 0..4")
    ap.add_argument("--fizgig", type=int, default=None, help="0/1")
    ap.add_argument("--first-steps", type=int, default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    api = json.load(open(args.api_json, encoding="utf-8"))
    OUT.mkdir(exist_ok=True)

    # --- подменить выбор модели / fizgig -----------------------------------
    changed = []
    for key, val in (("model", args.model), ("fizgig", args.fizgig)):
        if val is None:
            continue
        title = TOP_INT if key == "model" else TOP_FIZ
        nid, node = find_top(api, title)
        if node is None:
            print(f"!! не найден верхний узел: {title}", file=sys.stderr)
            sys.exit(2)
        if not set_value(node, val):
            print(f"!! не смог поставить значение в {nid} {node.get('class_type')}", file=sys.stderr)
            sys.exit(2)
        changed.append(f"{nid}={val}")

    if changed:
        print("подменено:", ", ".join(changed))

    # --- шаги --------------------------------------------------------------
    if args.first_steps is not None:
        for nid, n in api.items():
            if n.get("class_type") == "mxSlider" and nid.endswith(SG_FIRST_STEPS):
                n.setdefault("inputs", {})["Xi"] = args.first_steps
                n["inputs"]["Xf"] = args.first_steps
                print(f"шаги первого прохода: {nid} = {args.first_steps}")

    payload = {"prompt": api, "client_id": f"ws12-{args.tag}"}

    # DNS на этой машине иногда отваливается на пару секунд — ретраим запросы
    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(auth=AUTH, timeout=120, follow_redirects=True, transport=transport) as c:
        t0 = time.time()
        r = c.post(BASE + "/prompt", json=payload)
        print("POST /prompt ->", r.status_code)
        if r.status_code != 200:
            print(r.text[:3000])
            sys.exit(1)
        d = r.json()
        pid = d.get("prompt_id")
        ne = d.get("node_errors") or {}
        print("prompt_id:", pid, "| node_errors:", json.dumps(ne, ensure_ascii=False)[:2500])
        if ne:
            print("!! валидация не прошла")
            sys.exit(1)

        # --- ждать завершения ----------------------------------------------
        last = ""
        while time.time() - t0 < args.timeout:
            time.sleep(10)
            try:
                h = c.get(BASE + f"/history/{pid}", timeout=30)
            except Exception as exc:
                # DNS/сеть иногда отваливается на пару секунд — не теряем промпт
                print(f"[{int(time.time()-t0)}s] опрос не удался ({type(exc).__name__}), ждём дальше")
                continue
            if h.status_code != 200:
                continue
            hist = h.json().get(pid)
            if not hist:
                continue
            status = hist.get("status", {})
            done = status.get("completed")
            msgs = status.get("messages", [])
            tail = ""
            for kind, payload_ in msgs:
                if kind == "execution_error":
                    tail = f"ERROR на узле {payload_.get('node_id')}: {payload_.get('exception_message','')[:300]}"
                elif kind == "execution_success":
                    tail = "SUCCESS"
            if tail != last:
                print(f"[{int(time.time()-t0)}s] {tail or 'выполняется...'} completed={done}")
                last = tail
            if done:
                break
        else:
            print("!! таймаут")
            sys.exit(1)

        outputs = hist.get("outputs", {})
        imgs = []
        for nid, o in outputs.items():
            for k in ("images", "gifs"):
                for it in o.get(k, []) or []:
                    if it.get("filename"):
                        imgs.append(it)
        print("картинок на выходе:", len(imgs))
        for it in imgs:
            print("  ", it.get("filename"), it.get("subfolder", ""), it.get("type", ""))
            v = c.get(BASE + "/view", params={
                "filename": it["filename"],
                "subfolder": it.get("subfolder", ""),
                "type": it.get("type", "output"),
            })
            if v.status_code == 200:
                p = OUT / f"{args.tag}__{it['filename']}"
                p.write_bytes(v.content)
                print("   сохранено:", p)

    print(f"итого {int(time.time()-t0)} с")


if __name__ == "__main__":
    main()
