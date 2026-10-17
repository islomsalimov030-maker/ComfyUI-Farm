#!/usr/bin/env python3
"""SS-340: раунд RefMod на ComfyUI (студия или под): извлечь моды → кадры A/B на движке frame_h3@1 (генератор
WS-07 `A_refmod` / база `A`) → контакт-листы. Конфиг раунда — YAML вне репо (схема — `ss340_round.example.yaml`).

    python tools/ss340_round.py plan    <round.yaml>          # задания и оценка времени, без сервера
    python tools/ss340_round.py check   <round.yaml>          # /object_info: классы, файлы моделей (без очереди)
    python tools/ss340_round.py extract <round.yaml>          # моды → models/refmods/<prefix>/
    python tools/ss340_round.py run     <round.yaml> [--limit N]
    python tools/ss340_round.py sheet   <round.yaml> [--refined]   # контакт-листы по позам/сидам

refine (необязательно): {graph: API-граф доработки готового кадра, seeds: {узел: сид}} — каждый сырой кадр
дополнительно проходит через неё (одинаково для всех вариантов) → <вариант>__<сид>__refined.png.

COMFY_URL / COMFY_USER / COMFY_PASS — из окружения (Basic auth студии, если задан пользователь). Результаты —
`<out>/<round>/`: jobs.jsonl (время, prompt_id), PNG кадров, контакт-листы. Повторный run пропускает готовые.
"""
from __future__ import annotations

import argparse
import base64
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


# --------------------------------------------------------------------------------------- HTTP
class Comfy:
    def __init__(self, url: str):
        import httpx
        auth = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"]) if os.environ.get("COMFY_USER") else None
        self.http = httpx.Client(base_url=url.rstrip("/"), auth=auth, timeout=120)
        self._info = None

    def info(self) -> dict:
        if self._info is None:
            self._info = self.get("/object_info").json()
        return self._info

    def get(self, path: str):
        r = self.http.get(path)
        r.raise_for_status()
        return r

    def upload(self, path: Path, sub: str) -> str:
        with open(path, "rb") as fh:
            r = self.http.post("/upload/image", files={"image": (path.name, fh, "image/png")},
                               data={"subfolder": sub, "overwrite": "true"})
        r.raise_for_status()
        j = r.json()
        return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]

    def submit(self, graph: dict) -> str:
        G.fill_defaults(graph, self.info())   # необязательные в графе входы нод — значения по умолчанию сервера
        r = self.http.post("/prompt", json={"prompt": graph, "client_id": "ss340"})
        if r.status_code != 200:
            raise RuntimeError(f"/prompt {r.status_code}: {r.text[:1500]}")
        j = r.json()
        if j.get("node_errors"):
            raise RuntimeError(f"node_errors: {json.dumps(j['node_errors'], ensure_ascii=False)[:1500]}")
        return j["prompt_id"]

    def wait(self, pid: str, timeout_s: float) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            h = self.get(f"/history/{pid}").json().get(pid)
            if h and (h.get("status", {}).get("completed") or h.get("status", {}).get("status_str") == "error"):
                if h["status"].get("status_str") == "error":
                    msg = [m[1] for m in h["status"].get("messages", []) if m[0] == "execution_error"]
                    raise RuntimeError(f"ошибка выполнения: {json.dumps(msg, ensure_ascii=False)[:1500]}")
                return h
            time.sleep(3)
        raise TimeoutError(f"{pid}: дольше {timeout_s} с")

    def view(self, f: dict) -> bytes:
        return self.http.get("/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                              "type": f.get("type", "output")}).content


# --------------------------------------------------------------------------------------- конфиг
def load(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_dir"] = Path(path).resolve().parent
    return cfg


def p(cfg: dict, rel: str) -> Path:
    q = Path(rel)
    return q if q.is_absolute() else Path(cfg["src"]) / q


def mods(cfg: dict) -> dict:
    """{имя: {images: [Path], masks, concept_type, ref_resolution}} — картинки из папки набора (сортировка
    по имени), take = первые N."""
    out = {}
    for name, m in (cfg.get("mods") or {}).items():
        if m.get("images"):
            imgs = [p(cfg, i) for i in m["images"]]
        else:
            imgs = sorted(f for f in p(cfg, m["dir"]).glob("*.png"))[: m.get("take")]
        if not imgs:
            raise ValueError(f"мод {name}: нет картинок")
        out[name] = dict(m, images=imgs)
    return out


def jobs(cfg: dict) -> list[dict]:
    out = []
    for v in cfg["variants"]:
        for pose in cfg["poses"]:
            for seed in cfg["seeds"][pose]:
                refm = [dict(s, mod=s["mod"].format(pose=pose)) for s in v.get("refmods", [])]
                out.append({"id": f"{v['id']}__{pose}__{seed}", "variant": v["id"], "pose": pose, "seed": seed,
                            "refmods": refm, "retention": v.get("retention", 1.0)})
    return out


def mod_path(cfg: dict, name: str) -> str:
    return f"{cfg['mod_prefix']}/{name}"


# --------------------------------------------------------------------------------------- графы
def extract_graphs(cfg: dict, cli: Comfy | None) -> list[tuple[str, dict]]:
    res = []
    for name, m in mods(cfg).items():
        for pose in (cfg["poses"] if "{pose}" in name else [None]):
            nm = name.format(pose=pose) if pose else name
            imgs = [Path(str(i).format(pose=pose)) for i in m["images"]] if pose else m["images"]
            if m.get("masks"):     # LoadImage без альфы даёт пустую маску → после InvertMask маска молча не работает
                from PIL import Image
                bad = [i.name for i in imgs if "A" not in Image.open(i).getbands()]
                if bad:
                    raise ValueError(f"мод {nm}: masks=true, но без альфы: {bad}")
            up = [cli.upload(i, f"ss340/{nm}") if cli else i.name for i in imgs]
            res.append((nm, G.refmod_extract_graph(
                up, mod=mod_path(cfg, nm), concept_type=m.get("concept_type", "identity"),
                ref_resolution=m.get("ref_resolution", 1024), masks=bool(m.get("masks")),
                max_tokens=m.get("max_tokens", 0), description=m.get("description", nm))))
    return res


def frame_graph(cfg: dict, j: dict, files: dict) -> dict:
    pose = cfg["kit_poses"][j["pose"]]
    inp = {"plate": files[f"{j['pose']}/plate"], "sheet": files["sheet"]}
    for k in ("outfit", "pose"):
        if pose.get(k):
            inp[k] = files[f"{j['pose']}/{k}"]
    prompt = p(cfg, pose["prompt"]).read_text(encoding="utf-8")
    gen = "A_refmod" if j["refmods"] else "A"
    g = G.frame_graph(gen, inp, prompt, j["seed"], cfg["arm"], cfg.get("knobs"),
                      refmods=[dict(s, mod=mod_path(cfg, s["mod"])) for s in j["refmods"]] or None,
                      retention=j["retention"])
    g[G.nid(g, G.OUT_RAW)]["inputs"]["filename_prefix"] = f"ss340/{cfg['round']}/{j['id']}"
    return g


def refine_graph(cfg: dict, image: str, prefix: str) -> dict:
    """Доработка готового кадра — API-граф из конфига (`refine.graph`, например из метаданных одобренного кадра):
    подменяются вход (LoadImage), сиды (`refine.seeds`: {id узла: сид}) и путь сохранения (SaveImage)."""
    r = cfg["refine"]
    g = json.loads(p(cfg, r["graph"]).read_text(encoding="utf-8"))
    loads = [k for k, v in g.items() if v["class_type"] == "LoadImage"]
    saves = [k for k, v in g.items() if v["class_type"] == "SaveImage"]
    if len(loads) != 1 or len(saves) != 1:
        raise ValueError(f"доработка: нужен ровно 1 LoadImage и 1 SaveImage, есть {len(loads)} / {len(saves)}")
    g[loads[0]]["inputs"]["image"] = image
    g[saves[0]]["inputs"]["filename_prefix"] = prefix
    for nid, seed in (r.get("seeds") or {}).items():
        ins = g[str(nid)]["inputs"]
        key = next(k for k in ("seed", "noise_seed") if k in ins)
        ins[key] = int(seed)
    return g


def upload_inputs(cfg: dict, cli: Comfy | None) -> dict:
    files = {"sheet": cli.upload(p(cfg, cfg["sheet"]), "ss340/kit") if cli else "sheet.png"}
    for pose, d in cfg["kit_poses"].items():
        for k in ("plate", "outfit", "pose"):
            if d.get(k):
                files[f"{pose}/{k}"] = cli.upload(p(cfg, d[k]), f"ss340/kit/{pose}") if cli else f"{k}.png"
    return files


# --------------------------------------------------------------------------------------- команды
def cmd_plan(cfg):
    js = jobs(cfg)
    n_mods = len(extract_graphs(cfg, None))
    sec = cfg.get("est_frame_s", 130)
    print(f"модов к извлечению: {n_mods}; кадров: {len(js)} (вариантов {len(cfg['variants'])} × поз "
          f"{len(cfg['poses'])} × сидов); ≈ {len(js) * sec / 60:.0f} мин при {sec} с/кадр")
    for v in cfg["variants"]:
        print(f"  {v['id']:16s} моды: {[s['mod'] for s in v.get('refmods', [])] or '—'}  retention {v.get('retention', 1.0)}")


def cmd_check(cfg, cli):
    info = cli.info()
    sys.path.insert(0, str(REPO / "install"))
    import pod_doctor as D
    errs = []
    graphs = [(f"extract:{n}", g) for n, g in extract_graphs(cfg, None)]
    files = upload_inputs(cfg, None)
    graphs += [(f"frame:{j['id']}", frame_graph(cfg, j, files)) for j in jobs(cfg)[:1]
               + [j for j in jobs(cfg) if j["refmods"]][:1]]
    for name, g in graphs:
        G.fill_defaults(g, info)
        miss, bad = D.check_graph(g, info)
        bad = [b for b in bad if not (b[1].startswith("mod_") or b[1] == "image")]   # моды и входы ещё не загружены
        if miss or bad or G.check_links(g):
            errs.append((name, sorted(miss), bad[:5], G.check_links(g)))
    print("OK" if not errs else json.dumps(errs, ensure_ascii=False, indent=1))
    return 1 if errs else 0


def cmd_extract(cfg, cli):
    for nm, g in extract_graphs(cfg, cli):
        t0 = time.time()
        pid = cli.submit(g)
        h = cli.wait(pid, 900)
        det = [o for o in h.get("outputs", {}).values() if o.get("text")]
        print(f"мод {mod_path(cfg, nm)}: {time.time() - t0:.0f} с", (str(det[0]["text"])[:300] if det else ""),
              flush=True)
    names = cli.get("/object_info/MiniMaxH3RefModsLoader").json()["MiniMaxH3RefModsLoader"]["input"]["required"]["mod_1"][0]
    have = [n for n in names if n.startswith(cfg["mod_prefix"] + "/")]
    print("в загрузчике:", have)


def cmd_run(cfg, cli, limit: int):
    out = Path(cfg["out"]) / cfg["round"]
    out.mkdir(parents=True, exist_ok=True)
    log = out / "jobs.jsonl"
    done = {json.loads(l)["id"] for l in log.read_text(encoding="utf-8").splitlines()} if log.exists() else set()
    files = upload_inputs(cfg, cli)
    todo = [j for j in jobs(cfg) if j["id"] not in done][: limit or None]
    print(f"к запуску {len(todo)} (готово {len(done)})", flush=True)
    for i, j in enumerate(todo, 1):
        t0 = time.time()
        try:
            pid = cli.submit(frame_graph(cfg, j, files))
            h = cli.wait(pid, cfg.get("timeout_s", 900))
        except Exception as e:     # noqa: BLE001 — ошибку пишем и идём дальше
            rec = dict(j, error=str(e)[:800], sec=round(time.time() - t0))
            print(f"[{i}/{len(todo)}] {j['id']}: ОШИБКА {str(e)[:300]}", flush=True)
            if "node_errors" in str(e) or "ошибка выполнения" in str(e):
                with open(out / "errors.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue
        imgs = [f for o in h["outputs"].values() for f in o.get("images", [])]
        saved = []
        for f in imgs:
            dst = out / j["pose"] / f"{j['variant']}__{j['seed']}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(cli.view(f))
            saved.append(str(dst))
        rsec = None
        if cfg.get("refine") and imgs:     # та же доработка для всех вариантов — сравнение на финальной стадии
            t1 = time.time()
            try:
                raw_name = cli.upload(Path(saved[0]), f"ss340/{cfg['round']}/raw")
                g2 = refine_graph(cfg, raw_name, f"ss340/{cfg['round']}/{j['id']}__refined")
                h2 = cli.wait(cli.submit(g2), cfg.get("timeout_s", 900))
                for f in [f for o in h2["outputs"].values() for f in o.get("images", [])][:1]:
                    dst = out / j["pose"] / f"{j['variant']}__{j['seed']}__refined.png"
                    dst.write_bytes(cli.view(f))
                    saved.append(str(dst))
                rsec = round(time.time() - t1)
            except Exception as e:  # noqa: BLE001 — сырой кадр уже сохранён
                print(f"    доработка {j['id']}: ОШИБКА {str(e)[:300]}", flush=True)
        rec = dict(j, prompt_id=pid, sec=round(time.time() - t0), refine_sec=rsec, files=saved)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(todo)}] {j['id']}: {rec['sec']} с", flush=True)


def cmd_sheet(cfg, refined: bool = False):
    from PIL import Image, ImageDraw
    out = Path(cfg["out"]) / cfg["round"]
    suffix = "__refined" if refined else ""
    vids = [v["id"] for v in cfg["variants"]]
    for pose in cfg["poses"]:
        seeds = cfg["seeds"][pose]
        cw, ch = 480, 270
        S = Image.new("RGB", (cw * len(vids), (ch + 22) * len(seeds) + 22), "white")
        D = ImageDraw.Draw(S)
        for x, v in enumerate(vids):
            D.text((x * cw + 6, 4), v, fill="black")
            for y, seed in enumerate(seeds):
                f = out / pose / f"{v}__{seed}{suffix}.png"
                if f.exists():
                    im = Image.open(f).convert("RGB")
                    im.thumbnail((cw, ch))
                    S.paste(im, (x * cw, 22 + y * (ch + 22)))
                else:
                    D.text((x * cw + 6, 22 + y * (ch + 22) + ch // 2), "нет", fill="red")
        dst = out / f"sheet_{pose}{suffix}.jpg"
        S.save(dst, quality=88)
        print(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "check", "extract", "run", "sheet"])
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refined", action="store_true", help="sheet: кадры после доработки")
    ap.add_argument("--url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    a = ap.parse_args()
    cfg = load(a.config)
    if a.cmd == "plan":
        return cmd_plan(cfg) or 0
    if a.cmd == "sheet":
        return cmd_sheet(cfg, a.refined) or 0
    cli = Comfy(a.url)
    if a.cmd == "check":
        return cmd_check(cfg, cli)
    if a.cmd == "extract":
        return cmd_extract(cfg, cli) or 0
    return cmd_run(cfg, cli, a.limit) or 0


if __name__ == "__main__":
    sys.exit(main())
