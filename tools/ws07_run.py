#!/usr/bin/env python3
"""WS-07: раннер серий на поде (ComfyUI 127.0.0.1:8188). Возобновляемый: готовые задания пропускаются.

    python tools/ws07_run.py plan  s2_scale s1_gen ...      # список заданий и оценка времени (без пода)
    python tools/ws07_run.py check s2_scale s1_gen ...      # /object_info: ноды, модели, dry-run графа (без GPU)
    python tools/ws07_run.py run   s2_scale s1_gen ... [--limit N] [--budget-min M]
    python tools/ws07_run.py refmod                          # извлечь RefMod лица (серия 4)

Задания одной руки идут подряд (модель грузится один раз). Журнал — <out>/<серия>/jobs.jsonl
(время, prompt_id, файлы); картинки — <out>/<серия>/<pose>/<id>.png. Оценка — tools/ws07_eval.py.
Ничего не удаляет; при OOM — POST /free и один повтор.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import uuid
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import ws07_graphs as G  # noqa: E402

CFG = REPO / "tools" / "ws07_series.yaml"
# оценка, с, на один кадр (2 МП, 2 прохода): WS-01 — 130 с на 5090 вместе с проходами лица и загрузкой;
# на RTX PRO 6000 без лица ~35–60 с; первая загрузка руки ~90 с (int8 21 ГБ + Qwen3-VL 27 ГБ с тома).
EST = {"frame": 50, "load": 90, "upscale": {"none": 20, "rtx": 40, "seedvr2_3b": 240, "seedvr2_7b": 420}}


def load_cfg() -> dict:
    c = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    c["kit"] = os.environ.get("WS07_KIT", c["kit"])
    c["out"] = os.environ.get("WS07_OUT", c["out"])
    return c


def jobs(cfg: dict, series: list[str]) -> list[dict]:
    out = []
    for s in series:
        sp = cfg["series"][s]
        if sp["kind"] == "upscale":
            clips = sorted((Path(cfg["kit"]) / "clips").glob("*.mp4"))
            for m, c in itertools.product(sp["method"], clips or [Path("<нет kit/clips>")]):
                out.append({"series": s, "kind": "upscale", "method": m, "clip": c.name,
                            "id": f"{c.stem}__{m}"})
            continue
        for gen, arm, var, pose in itertools.product(sp["gen"], sp["arm"], sp["variant"], cfg["poses"]):
            for seed in cfg["seeds"][pose][: sp.get("seeds", 1)]:
                out.append({"series": s, "kind": "frame", "gen": gen, "arm": arm, "variant": var, "pose": pose,
                            "seed": seed, "id": f"{pose}__{gen}__{arm}__{var}__{seed}"})
    # одна рука/генератор подряд — меньше перезагрузок моделей
    out.sort(key=lambda j: (j.get("arm", ""), j.get("gen", j.get("method", "")), j["series"], j.get("pose", "")))
    return out


def estimate(js: list[dict]) -> tuple[float, int]:
    t, loads, last = 0.0, 0, None
    for j in js:
        if j["kind"] == "upscale":
            t += EST["upscale"][j["method"]]
            continue
        key = (j["arm"], j["gen"].endswith("t1"))
        if key != last:
            t += EST["load"]
            loads += 1
            last = key
        t += EST["frame"]
    return t, loads


def done_ids(cfg: dict, series: str) -> set[str]:
    p = Path(cfg["out"]) / series / "jobs.jsonl"
    if not p.exists():
        return set()
    ok = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("status") == "success":
            ok.add(r["id"])
    return ok


# --------------------------------------------------------------------------------------- граф задания
def _arm(cfg: dict, name: str, model_files: set | None) -> dict:
    a = dict(cfg["arms"][name])
    if model_files is not None and a["model"] not in model_files and a.get("alt_model") in model_files:
        a["model"] = a["alt_model"]
    return a


def frame_job_graph(cfg: dict, j: dict, uploaded: dict, model_files: set | None = None) -> dict:
    kit = Path(cfg["kit"])
    tgt = json.loads((kit / "targets.json").read_text(encoding="utf-8"))[j["pose"]]
    d = f"ws07/{j['pose']}"
    files = tgt["files"]
    var = j["variant"]
    inp = {"plate": uploaded[f"{d}/{files['plate']}"], "sheet": uploaded["ws07/sheet.jpg"]}
    if var == "green":
        inp["plate"] = uploaded[f"{d}/{files['plate_green']}"]
    elif var == "wide":
        inp["plate"] = uploaded[f"{d}/{files['plate_wide']}"]
    elif var == "idcrop":
        inp["sheet"] = uploaded["ws07/idcrop.png"]
    if tgt["has_outfit"]:
        inp["outfit"] = uploaded[f"{d}/{files['outfit']}"]
    if var == "skeleton":
        inp["pose"] = uploaded[f"{d}/{files['skeleton']}"]
    elif tgt["has_pose"]:
        inp["pose"] = uploaded[f"{d}/{files['pose_photo']}"]
    prompt = (kit / "refs" / j["pose"] / f"prompt_{var}.txt").read_text(encoding="utf-8")
    g = G.frame_graph(j["gen"], inp, prompt, j["seed"], _arm(cfg, j["arm"], model_files))
    g[G.nid(g, G.OUT_RAW)]["inputs"]["filename_prefix"] = f"ws07/{j['series']}/{j['id']}"
    return g


def upload_kit(client, cfg: dict) -> dict:
    """Все картинки kit/refs → ComfyUI/input/ws07/… (overwrite). → {относительный путь: имя для LoadImage}."""
    kit = Path(cfg["kit"]) / "refs"
    out = {}
    for f in sorted(kit.rglob("*")):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg") or f.name == "frame_v2.png":
            continue
        rel = f.relative_to(kit).as_posix()
        sub = "ws07/" + rel.rsplit("/", 1)[0] if "/" in rel else "ws07"
        out["ws07/" + rel] = client.upload_image(f, name=f.name, subfolder=sub)
    return out


# --------------------------------------------------------------------------------------- проверка на поде
def object_info(client) -> dict:
    r = client._req("GET", "/object_info", timeout=120)
    r.raise_for_status()
    return r.json()


def combo_values(info: dict, cls: str, inp: str) -> set:
    spec = ((info.get(cls) or {}).get("input") or {})
    for sec in ("required", "optional"):
        s = (spec.get(sec) or {}).get(inp)
        if s:
            if isinstance(s[0], list):
                return set(s[0])
            if len(s) > 1 and isinstance(s[1], dict) and s[1].get("options"):
                return set(s[1]["options"])
    return set()


def check(client, cfg: dict, js: list[dict]) -> int:
    info = object_info(client)
    models = combo_values(info, "DiffusionModelLoaderKJ", "model_name") | combo_values(info, "UNETLoader", "unet_name")
    loras = combo_values(info, "LoraLoaderModelOnly", "lora_name")
    vaes = combo_values(info, "VAELoader", "vae_name")
    bad = 0
    for name, a in cfg["arms"].items():
        m = a["model"] if a["model"] in models else a.get("alt_model")
        miss = ([a["model"]] if m not in models else []) + [l["name"] for l in a["loras"] if l["name"] not in loras]
        print(f"рука {name:20s} {'OK' if not miss else 'НЕТ: ' + ', '.join(miss)}")
    print("t1 image VAE:", "OK" if "minimax_h3_t1_image_vae_step1597.safetensors" in vaes else "нет (A_t1/B_t1/C_t1 упадут)")
    qp = combo_values(info, "H3ReferenceEditPrepare", "quality_profile")
    if any(j.get("gen") == "A_t1" for j in js) and G.SINGLE_PROFILE not in qp:
        print(f"A_t1: в Image Studio нет профиля «{G.SINGLE_PROFILE}» (есть {sorted(qp)}) — нужен пакет ≥ v23 "
              "(требует ComfyUI ≥ 0.37); пропустить A_t1")
        bad += 1
    seen, uploaded = set(), {}
    kit = Path(cfg["kit"]) / "refs"
    for f in kit.rglob("*"):
        if f.is_file():
            rel = f.relative_to(kit).as_posix()
            uploaded["ws07/" + rel] = f"ws07/{rel}"
    for j in js:
        if j["kind"] != "frame":
            need = {"rtx": {"RTXVideoSuperResolution"}, "none": set()}.get(j["method"], {"SeedVR2VideoUpscaler"})
            miss = (need | {"VHS_LoadVideo", "VHS_VideoCombine"}) - set(info)
            if miss and j["method"] not in seen:
                print(f"апскейл {j['method']}: нет нод {sorted(miss)}")
                bad += 1
            seen.add(j["method"])
            continue
        key = (j["gen"], j["variant"], j["pose"])
        if key in seen:
            continue
        seen.add(key)
        g = frame_job_graph(cfg, j, uploaded, models)
        miss = G.classes(g) - set(info)
        if miss:
            if j["gen"] not in seen:
                print(f"генератор {j['gen']}: нет нод {sorted(miss)}")
            seen.add(j["gen"])
            bad += 1
        log = G.fill_defaults(g, info)
        if log and j["gen"] + "_fill" not in seen:
            print(f"генератор {j['gen']}: дозаполнено из object_info: {log}")
            seen.add(j["gen"] + "_fill")
    print("проверка:", "OK" if not bad else f"{bad} проблем")
    return 1 if bad else 0


# --------------------------------------------------------------------------------------- прогон
def run(client, cfg: dict, js: list[dict], limit: int, budget_min: float) -> int:
    from farm.comfy_client import ExecutionFailed, NodeErrors
    info = object_info(client)
    models = combo_values(info, "DiffusionModelLoaderKJ", "model_name")
    uploaded = upload_kit(client, cfg)
    t_start, n = time.time(), 0
    for j in js:
        if j["id"] in done_ids(cfg, j["series"]):
            continue
        if limit and n >= limit:
            break
        if budget_min and (time.time() - t_start) / 60 > budget_min:
            print(f"бюджет {budget_min} мин исчерпан — стоп (повторный запуск продолжит)")
            break
        outdir = Path(cfg["out"]) / j["series"]
        outdir.mkdir(parents=True, exist_ok=True)
        if j["kind"] == "frame":
            g = frame_job_graph(cfg, j, uploaded, models)
        else:
            name = client.upload_image(Path(cfg["kit"]) / "clips" / j["clip"], subfolder="ws07/clips")
            g = G.upscale_graph(name, j["method"], f"ws07/{j['series']}/{j['id']}")
        G.fill_defaults(g, info)
        rec = {**j, "t0": time.strftime("%F %T")}
        for attempt in (1, 2):
            t0 = time.time()
            try:
                pid = client.submit(g, prompt_id=str(uuid.uuid4()))
                entry = client.wait(pid, cfg["timeout_s"])
                rec.update(status="success", sec=round(time.time() - t0, 1), prompt_id=pid,
                           files=_save_outputs(client, entry, outdir, j))
                break
            except NodeErrors as e:
                rec.update(status="node_errors", error=str(e)[:500], node_errors=e.node_errors)
                break
            except ExecutionFailed as e:
                rec.update(status="error", error=str(e)[:800], sec=round(time.time() - t0, 1))
                if e.oom and attempt == 1:
                    client.free()
                    continue
                break
            except Exception as e:  # noqa: BLE001 — таймаут, обрыв: записать и идти дальше
                rec.update(status="error", error=f"{type(e).__name__}: {e}"[:500], sec=round(time.time() - t0, 1))
                try:
                    client.interrupt(rec.get("prompt_id"))
                except Exception:  # noqa: BLE001
                    pass
                break
        with open(outdir / "jobs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        n += 1
        print(f"[{n}] {j['id']}: {rec['status']} {rec.get('sec', '')} с", flush=True)
    print(f"готово заданий: {n}, {(time.time() - t_start) / 60:.1f} мин")
    return 0


def _save_outputs(client, entry: dict, outdir: Path, j: dict) -> list[str]:
    saved = []
    for out in (entry.get("outputs") or {}).values():
        for it in (out.get("images") or []) + (out.get("gifs") or []):
            data = client.view(it["filename"], it.get("subfolder", ""), it.get("type", "output"))
            ext = Path(it["filename"]).suffix
            sub = outdir / (j.get("pose") or "clips")
            sub.mkdir(parents=True, exist_ok=True)
            p = sub / f"{j['id']}{ext}"
            p.write_bytes(data)
            saved.append(str(p))
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "check", "run", "refmod"])
    ap.add_argument("series", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-min", type=float, default=0)
    ap.add_argument("--url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    a = ap.parse_args()
    cfg = load_cfg()
    unknown = [s for s in a.series if s not in cfg["series"]]
    if unknown:
        print("нет серий:", unknown, "есть:", list(cfg["series"]))
        return 2
    js = jobs(cfg, a.series)
    if a.cmd == "plan":
        for s in a.series:
            sj = [j for j in js if j["series"] == s]
            t, loads = estimate(sj)
            print(f"{s:14s} заданий {len(sj):3d}  загрузок модели {loads}  ≈ {t / 60:.0f} мин")
        t, loads = estimate(js)
        print(f"{'ИТОГО':14s} заданий {len(js):3d}  загрузок модели {loads}  ≈ {t / 60:.0f} мин")
        return 0
    from farm.comfy_client import ComfyClient
    client = ComfyClient(a.url, request_timeout_s=120)
    if a.cmd == "check":
        return check(client, cfg, js)
    if a.cmd == "refmod":
        imgs = [client.upload_image(Path(cfg["kit"]) / "refs" / n, subfolder="ws07")
                for n in ("idcrop.png", "face_ref.png")]
        pid = client.submit(G.refmod_extract_graph(imgs), prompt_id=str(uuid.uuid4()))
        t0 = time.time()
        e = client.wait(pid, 1800)
        print(f"RefMod {G.REFMOD}: {time.time() - t0:.0f} с;", json.dumps(e.get("outputs"), ensure_ascii=False)[:400])
        return 0
    return run(client, cfg, js, a.limit, a.budget_min)


if __name__ == "__main__":
    sys.exit(main())
