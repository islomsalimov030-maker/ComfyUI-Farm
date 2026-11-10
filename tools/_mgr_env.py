"""Проверка окружения студии перед обновлением ядра ComfyUI (read-only)."""

import json
import os

import httpx

BASE = os.environ["COMFY_URL"].rstrip("/")
AUTH = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])


def main():
    with httpx.Client(auth=AUTH, timeout=40, follow_redirects=True) as c:
        r = c.get(BASE + "/system_stats")
        st = r.json()
        sysinfo = st.get("system", {})
        print("== system ==")
        for key in (
            "os",
            "comfyui_version",
            "required_frontend_version",
            "installed_templates_version",
            "required_templates_version",
            "python_version",
            "pytorch_version",
            "embeded_python",
            "embedded_python",
            "deploy_environment",
            "argv",
        ):
            if key in sysinfo:
                print(f"  {key}: {sysinfo[key]}")
        print("  comfy_package_versions:")
        for p in sysinfo.get("comfy_package_versions", []):
            print(f"    {p['name']}: installed={p['installed']} required={p['required']}")

        print()
        print("== devices ==")
        for d in st.get("devices", []):
            print(f"  {d['name']} vram_total={d.get('vram_total')} free={d.get('vram_free')}")

        print()
        print("== версии ядра ==")
        r = c.get(BASE + "/api/v2/comfyui_manager/comfyui_versions")
        print(" ", json.dumps(r.json(), ensure_ascii=False))

        print()
        print("== ключевые ноды фермы на месте? ==")
        r = c.get(BASE + "/object_info", timeout=90)
        print("  /object_info HTTP", r.status_code, len(r.content), "bytes")
        if r.status_code == 200:
            oi = r.json()
            keys = list(oi.keys())
            print("  всего классов:", len(keys))
            farm = sorted(k for k in keys if k.upper().startswith("NGFARM"))
            print("  NGFarm*:", farm)
            for want in ("ImageGrid", "ImageBatchMultiSkipEmpty", "SaveVideo", "CreateVideo"):
                print(f"  {want}: {'да' if want in oi else 'НЕТ'}")


if __name__ == "__main__":
    main()
