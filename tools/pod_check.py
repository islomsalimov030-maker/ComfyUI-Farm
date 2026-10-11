#!/usr/bin/env python3
"""WS-01 шаг 6 на поде: dry-run и прогон движка через API ComfyUI (127.0.0.1:8188).

    python tools/pod_check.py workflows/video_h3_loop@1.params.yaml values.json           # только валидация
    python tools/pod_check.py workflows/video_h3_loop@1.params.yaml values.json --run     # + прогон, ждать результат

values.json — {имя из params.yaml: значение}; картинки — имена файлов, уже лежащих в ComfyUI/input
(загрузить: curl -F image=@file.png http://127.0.0.1:8188/upload/image).
ComfyUI валидирует весь граф при постановке в очередь, поэтому dry-run = POST /prompt, проверка node_errors
и снятие задачи: из очереди, а если она уже стартовала (очередь была пуста) — /interrupt только своей задачи.
"""
import json, os, sys, time, urllib.request, uuid

sys.path.insert(0, os.path.dirname(__file__))
import wf_params as P

URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


def call(path, data=None):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")


def main():
    params_path, values_path = sys.argv[1], sys.argv[2]
    run = "--run" in sys.argv
    p, api = P.load(params_path)
    errs = P.check(p, api)
    if errs:
        print("params.yaml:", *errs, sep="\n  ")
        return 1
    prompt = P.apply(p, api, json.load(open(values_path, encoding="utf-8")))
    r = call("/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4()), "front": True})
    if r.get("node_errors") or r.get("error"):
        print(json.dumps({"error": r.get("error"), "node_errors": r.get("node_errors")}, ensure_ascii=False, indent=1))
        return 1
    pid = r["prompt_id"]
    print("OK: граф принят, prompt_id", pid)
    if not run:
        call("/queue", {"delete": [pid]})
        running = [x[1] for x in call("/queue").get("queue_running", [])]
        if pid in running:
            call("/interrupt", {"prompt_id": pid})
        print("dry-run: задача снята с очереди")
        return 0
    t0 = time.time()
    while True:
        h = call(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            print("статус:", st.get("status_str"), f"{time.time() - t0:.0f} с")
            for nid, out in h[pid].get("outputs", {}).items():
                title = prompt.get(nid, {}).get("_meta", {}).get("title", nid)
                files = [x.get("filename") for k in ("images", "gifs") for x in out.get(k, [])]
                print(f"  {title}: {files or str(out.get('text', ''))[:300]}")
            return 0 if st.get("status_str") == "success" else 1
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
