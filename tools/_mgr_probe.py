"""Разведка API ComfyUI-Manager v4.2.1 (legacy-режим) на студии.

Только GET. Печатает список зарегистрированных роутов менеджера (из исходника
пакета на сервере, если доступен) и доступные версии ядра ComfyUI.
"""

import os
import re
import sys

import httpx

BASE = os.environ["COMFY_URL"].rstrip("/")
AUTH = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])


def get(client, path, limit=600):
    try:
        r = client.get(BASE + path)
        body = r.text[:limit] if r.status_code == 200 else ""
        return r.status_code, body
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"[:limit]


def main():
    client = httpx.Client(auth=AUTH, timeout=30, follow_redirects=True)

    print("== routes в исходнике менеджера на сервере ==")
    for path in (
        "/manager/routes",
        "/api/v2/manager/routes",
        "/api/manager/routes",
    ):
        code, body = get(client, path, 4000)
        print(f"{path} -> {code}")
        if code == 200:
            for line in body.splitlines():
                if "/manager" in line or "comfyui" in line.lower():
                    print("   ", line.strip()[:200])

    print()
    print("== исходник менеджера с GitHub ==")
    try:
        r = httpx.get(
            "https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main/__init__.py",
            timeout=30,
            follow_redirects=True,
        )
        text = r.text
        routes = sorted(set(re.findall(r"@(?:routes|app)\.(?:get|post|put|delete)\(\s*[\"']([^\"']+)", text)))
        print("HTTP", r.status_code, "роутов:", len(routes))
        for rt in routes:
            print("  ", rt)
        with open("/tmp/mgr_init.py", "w", encoding="utf-8") as fh:
            fh.write(text)
    except Exception as exc:  # noqa: BLE001
        print("ERR", type(exc).__name__, str(exc)[:200])

    print()
    print("== состояние пакетов ==")
    code, body = get(client, "/api/v2/customnode/installed", 200)
    print("installed HTTP", code)
    if code == 200:
        import json

        data = json.loads(body)
        for key in ("ComfyUI", "ComfyUI-Farm", "ComfyUI-StacyLoop"):
            if key in data:
                print(f"  {key}: ver={data[key].get('ver')} aux={data[key].get('aux_id')}")

    print()
    print("== version ==")
    print(get(client, "/api/v2/manager/version", 100))


if __name__ == "__main__":
    sys.exit(main())
