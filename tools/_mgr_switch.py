"""Разведка состояния ядра ComfyUI на студии через Manager (read-only)."""

import json
import os

import httpx

BASE = os.environ["COMFY_URL"].rstrip("/")
AUTH = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])


def main():
    with httpx.Client(auth=AUTH, timeout=30, follow_redirects=True) as c:
        # доступные ветки/версии + текущая
        r = c.get(BASE + "/api/v2/comfyui_manager/comfyui_versions")
        print("versions:", json.dumps(r.json(), ensure_ascii=False))

        # какие версии уже «известны» менеджеру (fetch кеш)
        for ep in (
            "/api/v2/comfyui_manager/comfyui_versions/branch",
            "/api/v2/comfyui_manager/comfyui_versions/list",
            "/api/v2/manager/startup_alerts",
            "/api/v2/manager/db_mode",
            "/api/v2/manager/preview_method",
            "/api/v2/customnode/getlist",
            "/api/v2/customnode/fetch_updates?mode=default&skip_update=true",
        ):
            try:
                r = c.get(BASE + ep)
                body = r.text[:300] if r.status_code == 200 else ""
                print(f"{ep} -> {r.status_code} {body}")
            except Exception as exc:
                print(f"{ep} -> ERR {type(exc).__name__}")


if __name__ == "__main__":
    main()
