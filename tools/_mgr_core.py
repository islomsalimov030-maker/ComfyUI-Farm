"""Версии ядра ComfyUI, доступные на студии (read-only)."""

import json
import os

import httpx

BASE = os.environ["COMFY_URL"].rstrip("/")
AUTH = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])


def main():
    with httpx.Client(auth=AUTH, timeout=30, follow_redirects=True) as c:
        for ep in (
            "/api/v2/comfyui_manager/comfyui_versions",
            "/api/v2/comfyui_manager/comfyui_versions?skip_update=true",
            "/api/v2/comfyui_manager/comfyui_versions?mode=default",
        ):
            r = c.get(BASE + ep)
            print(f"=== {ep} -> HTTP {r.status_code} ({len(r.content)} b)")
            if r.status_code == 200:
                try:
                    d = r.json()
                    print(json.dumps(d, ensure_ascii=False, indent=1)[:2500])
                except Exception:
                    print(r.text[:1200])
            print()

        r = c.get(BASE + "/api/v2/customnode/installed")
        if r.status_code == 200:
            d = r.json()
            print("ComfyUI installed:", d.get("ComfyUI"))


if __name__ == "__main__":
    main()
