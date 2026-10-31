#!/usr/bin/env python3
"""Подача воркфлоу на студию с разбором, почему промпт не исполняется.

Диагностика «промпт принят, но не исполняется»:
  * POST /prompt -> 200, node_errors пустые, но запись не появляется в истории
    и очередь пуста — значит сервер отбросил промпт после валидации.

Скрипт печатает по шагам: валидация -> появление в очереди -> появление в
истории -> итоговый статус. Поддерживает `--with-ui` (кладёт UI-граф в
extra_pnginfo.extra_pnginfo.workflow) и `--bisect` (отбрасывает узлы
по одному, начиная с конца, пока промпт не начнёт исполняться).

    python wf_probe_exec.py <api.json> [--with-ui] [--timeout 120]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

BASE = os.environ["COMFY_URL"].rstrip("/")
AUTH = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])
TRANSPORT = httpx.HTTPTransport(retries=6)


def _get(client: httpx.Client, path: str, timeout: int = 45):
    for attempt in range(8):
        try:
            r = client.get(BASE + path, timeout=timeout)
            return r.json()
        except Exception:  # noqa: BLE001 — сеть студии отваливается кратковременно
            if attempt == 7:
                return None
            time.sleep(2)
    return None


def submit(client: httpx.Client, api: dict, ui: dict | None, tag: str):
    payload = {"prompt": api, "client_id": f"probe-{tag}"}
    if ui is not None:
        payload["extra_data"] = {"extra_pnginfo": {"workflow": ui}}
    r = client.post(BASE + "/prompt", json=payload, timeout=90)
    if r.status_code != 200:
        return None, r.status_code, r.text[:600]
    d = r.json()
    return d.get("prompt_id"), r.status_code, d


def wait(client: httpx.Client, pid: str, timeout: int):
    """Ждёт появления записи в истории. Возвращает (история|None, лог)."""
    t0 = time.time()
    log = []
    while time.time() - t0 < timeout:
        time.sleep(4)
        hist = _get(client, f"/history/{pid}") or {}
        if pid in hist:
            return hist[pid], log
        q = _get(client, "/queue") or {}
        log.append(f"[{int(time.time() - t0)}s] running={len(q.get('queue_running', []))} "
                   f"pending={len(q.get('queue_pending', []))}")
    return None, log


def report(hist, log):
    if hist is None:
        print("  !! промпт НЕ появился в истории за отведённое время")
        for line in log[-4:]:
            print("   ", line)
        return False
    st = hist.get("status", {})
    print("  статус:", st.get("status_str"), "completed=", st.get("completed"))
    for kind, m in st.get("messages", []):
        if kind == "execution_error":
            print(f"  ERR node={m.get('node_id')} type={m.get('node_type')} "
                  f"{m.get('exception_type')}: {str(m.get('exception_message'))[:400]}")
        elif kind == "execution_success":
            print("  SUCCESS")
    outs = hist.get("outputs", {})
    if outs:
        for nid, o in outs.items():
            for key in ("images", "gifs", "videos"):
                for it in o.get(key, []) or []:
                    print("   выход:", nid, it.get("filename"), it.get("subfolder", ""))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("api_json")
    ap.add_argument("--ui", default=None, help="UI-граф (по умолчанию ищется <api>.ui.json)")
    ap.add_argument("--with-ui", action="store_true", help="класть UI-граф в extra_pnginfo")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--bisect", action="store_true", help="искать узел-виновник отбрасыванием")
    args = ap.parse_args()

    api = json.load(open(args.api_json, encoding="utf-8"))
    ui = None
    if args.with_ui:
        ui_path = args.ui or args.api_json.replace(".api.json", ".ui.json")
        ui = json.load(open(ui_path, encoding="utf-8"))
        print("UI-граф:", ui_path)

    with httpx.Client(auth=AUTH, timeout=90, follow_redirects=True, transport=TRANSPORT) as c:
        pid, code, payload = submit(c, api, ui, "probe")
        print("POST /prompt ->", code)
        if pid is None:
            print(json.dumps(payload, ensure_ascii=False, indent=1)[:2000])
            sys.exit(1)
        print("prompt_id:", pid, "| node_errors:",
              json.dumps(payload.get("node_errors"), ensure_ascii=False)[:400])
        hist, log = wait(c, pid, args.timeout)
        ok = report(hist, log)

        if not ok and args.bisect:
            print("\n== поиск узла-виновника (отбрасываем с конца) ==")
            ids = list(api.keys())
            for cut in range(1, len(ids) + 1):
                trial = {k: v for k, v in api.items() if k not in ids[-cut:]}
                pid2, code2, payload2 = submit(c, trial, ui, f"bisect{cut}")
                if pid2 is None:
                    print(f"  отброшено {cut}: отказ валидатора "
                          f"({str(payload2)[:120]})")
                    continue
                h2, _ = wait(c, pid2, 25)
                if h2 is not None:
                    print(f"  отброшено {cut} узлов -> промпт ВЫПОЛНИЛСЯ "
                          f"(статус {h2['status'].get('status_str')})")
                    print("  подозрительные (последние отброшенные):", ids[-cut:])
                    break
                print(f"  отброшено {cut}: всё ещё не исполняется")


if __name__ == "__main__":
    main()
