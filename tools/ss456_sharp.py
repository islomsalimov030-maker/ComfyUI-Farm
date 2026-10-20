#!/usr/bin/env python3
"""SS-456: тест резкости кадров Саманты (16:9) на одной сцене — варианты генерации × доработка без LoRA.

    python tools/ss456_sharp.py gen    <jobs_min.json> <out_dir> [--job 0] [--only B,C]
    python tools/ss456_sharp.py refine <v2_refine_api.json> <out_dir> [--body 0.3] [--face 0.35,0.45] [--only A0,B]
    python tools/ss456_sharp.py sheet  <out_dir>
    python tools/ss456_sharp.py eval   <out_dir> <face_ref_dir>   # FARM_FACE_MODELS=C:/stacy-lora/face_models
    python tools/ss456_sharp.py batch  <v2_refine_api.json> <round.yaml> [--mp 1] [--face 0.85,0.75]
                                    # проход лица по всем кадрам раунда, выражение — из промпта сцены

gen: из графа кадра (генератор T_refmod, turbo 8) — варианты разрешения / модели / шагов (VARIANTS); A0 = готовый кадр.
refine: граф собран из узлов v2 (`_claude/ss340/v2_refine_api.json`) без LoRA Стейси и без yolov8:
  тело — SAM3 «woman» → InpaintCrop → Z-image turbo (denoise --body) → вклейка (лицо исключено);
  лицо — SAM3 «face» → кроп до 4 Мп → H3 fl2va 10 шагов + RefMod `ss456/split_face` (denoise по --face) →
  ColorMatch 0.95 → вклейка, без гейта «лицо > 640»; шарп 0.1 по маске лица SAM3.
COMFY_URL / COMFY_USER / COMFY_PASS — из окружения.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from ss340_round import Comfy  # noqa: E402

FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
# вар.: (модель, turbo, шаги, sampler, w, h)
VARIANTS = {
    "B": (None, True, 8, "euler", 1344, 768),
    "C": (FL2VA, False, 20, "res_multistep", 1888, 1056),
    "D": (FL2VA, False, 20, "res_multistep", 1344, 768),
    "E": (None, False, 20, "res_multistep", 1888, 1056),
}
BODY_PROMPT = "a tall athletic woman, sharp focus, detailed skin."
FACE_PROMPT = ("A tight close-up portrait of a 22-year-old woman with long thick very dark chocolate-brown hair "
               "in a high tight ponytail, large almond-shaped dark brown eyes with a confident gaze, a symmetrical "
               "oval face with high cheekbones and full pink lips. Her face comes from her character reference. "
               "Her face is sharp and natural, with detailed skin. Static camera. "
               "Lighting preserved unchanged. Colors and saturation unchanged.")
# v2 — черты по рефам 07–11 (desc_en мода слишком общий: «oval face»), выражение не задаётся — остаётся из кадра
FACE_PROMPT_V2 = ("A tight close-up portrait of the woman from her character reference: a long face with a strong "
                  "defined square jawline, high sharp cheekbones, thick straight dark eyebrows, large almond-shaped "
                  "dark brown eyes with dark winged makeup and long lashes, a straight defined nose and very full "
                  "glossy nude-pink lips; very dark hair pulled back in a high tight ponytail. Her face, expression "
                  "and head angle stay as in the image. Her face is sharp and natural, with detailed skin. "
                  "Static camera. Lighting preserved unchanged. Colors and saturation unchanged.")
# v3 (19.10, пользователь): на d 0.85 H3 перерисовывает и выражение → выражение из [Shot 1] промпта сцены
FACE_PROMPT_V3 = ("A tight close-up portrait of the woman from her character reference: a long face with a strong "
                  "defined square jawline, high sharp cheekbones, thick straight dark eyebrows, large almond-shaped "
                  "dark brown eyes with dark winged makeup and long lashes, a straight defined nose and very full "
                  "glossy nude-pink lips; very dark hair pulled back in a high tight ponytail. {expr} "
                  "Her head angle stays as in the image. Her face is sharp and natural, with detailed skin. "
                  "Static camera. Lighting preserved unchanged. Colors and saturation unchanged.")
EXPR_DEFAULT = "a calm confident look"


def scene_expr(scene_prompt: str) -> str:
    """Выражение лица из [Shot 1] промпта сцены («with a light smile» → «a light smile»)."""
    import re
    s = re.search(r"\[Shot 1\](.*?)(?:One single photograph|$)", scene_prompt, re.S)
    m = re.search(r"\bwith (an? [a-z ]*?(?:smile|smirk|look|expression))\b", s.group(1) if s else "")
    return m.group(1) if m else EXPR_DEFAULT


def face_prompt_v3(expr: str) -> str:
    return FACE_PROMPT_V3.format(expr=f"She has {expr}, exactly as in the image.")


FACE_MOD = "ss456/split_face"
SEED_Z, SEED_H3 = 612651749928842, 953552952704593


def gen_graph(g: dict, var: str, prefix: str) -> dict:
    g = copy.deepcopy(g)
    g["87"]["inputs"]["filename_prefix"] = prefix
    if var == "A0":                     # граф кадра как есть (ref2va turbo 8, 1888×1056)
        return g
    model, turbo, steps, sampler, w, h = VARIANTS[var]
    g["w7_2"]["inputs"].update(width=w, height=h)
    if model:
        g["62"]["inputs"]["model_name"] = model
    if not turbo:
        g["84:3"]["inputs"]["model"] = ["62", 0]
        del g["63"]
    g["68"]["inputs"]["value"] = steps
    for k in ("84:9", "84:10"):
        g[k]["inputs"]["sampler_name"] = sampler
    g["87"]["inputs"]["filename_prefix"] = prefix
    return g


def refine_graph(v2: dict, image: str, body_d: float, face_ds: list[float], prefix: str,
                 mp: float | None = None, copies: int = 1, steps: int | None = None,
                 blend: int | None = None, face_prompt: str = FACE_PROMPT) -> dict:
    """Узлы v2 → тело Z (без LoRA) + лицо H3 с модом на каждый denoise. Выходы: <prefix>_body, <prefix>_f<d>."""
    src = copy.deepcopy(v2)
    g = {"in": {"class_type": "LoadImage", "inputs": {"image": image}}}

    def take(nid: str, new: str | None = None, **upd) -> dict:
        n = copy.deepcopy(src[nid]); n["inputs"].update(upd); g[new or nid] = n
        return n

    # тело: Z-image turbo без LoRA, VAE ae (Apache-2.0), лицо исключено из вклейки (как в v2: 11195);
    # body_d = 0 — без тела, лицо по сырому кадру
    body = ["in", 0]
    if body_d > 0:
        for nid in [k for k in src if k.startswith("11177:")]:
            take(nid)
        g["11177:11084"]["inputs"]["image"] = ["in", 0]
        g["11177:11090"]["inputs"]["image"] = ["in", 0]
        g["11177:11069"]["inputs"]["model"] = ["11177:11089", 0]
        del g["11177:11091"]
        g["11177:11088"]["inputs"]["vae_name"] = "ae.safetensors"
        g["11177:11077"]["inputs"]["text"] = BODY_PROMPT
        g["11177:11086"]["inputs"].update(seed=SEED_Z, denoise=body_d)
        for nid in ("11193", "11194", "11195"):
            take(nid)
        body = ["11177:11083", 0]
        g["sv_body"] = {"class_type": "SaveImage", "inputs": {"images": body, "filename_prefix": f"{prefix}_body"}}

    # лицо: общие узлы (модель без LoRA и без превью, мод вместо face LoRA, маска SAM3 без гейтов)
    for nid in ("11190:11105", "11190:11106", "11190:11107", "11190:11108", "11190:11128", "11190:11127",
                "11190:11114", "11190:11115", "11190:11116", "11190:11117", "11190:11119", "11190:11120",
                "11190:11121"):
        take(nid)
    g["11190:11108"]["inputs"]["model"] = ["11190:11105", 0]
    g["11190:11128"]["inputs"]["image"] = body
    g["11190:11114"]["inputs"].update(image=body, mask=["11190:11127", 0])
    g["11190:11115"]["inputs"]["megapixels"] = mp or src["11182"]["inputs"]["value"]
    if blend is not None:               # шов вклейки: в v2 40 px — для лица ~80 px это почти всё лицо
        g["11190:11114"]["inputs"]["mask_blend_pixels"] = blend
    g["11190:11117"]["inputs"]["prompt"] = face_prompt
    g["11190:11119"]["inputs"]["noise_seed"] = SEED_H3
    g["ml"] = {"class_type": "MiniMaxH3RefModsLoader", "inputs": {"mod_1": FACE_MOD, "strength_1": 1.0, "copies_1": copies}}
    g["ma"] = {"class_type": "MiniMaxH3RefModApply", "inputs": {
        "conditioning": ["11190:11117", 0], "mods": ["ml", 0], "override": False, "retention": 1.0,
        "curve_direction": "constant", "scramble_seed": -1, "curve_shape": "linear", "curve_value": 1.0}}
    g["11190:11120"]["inputs"].update(model=["11190:11108", 0], conditioning=["ma", 0])
    take("shm", mask=["11190:11128", 1])

    for d in face_ds:
        s = f"f{round(d * 100):02d}"
        take("11190:11118", f"{s}:lat", model=["11190:11108", 0], denoise=d)
        if steps:
            g[f"{s}:lat"]["inputs"]["steps"] = steps
        take("11190:11122", f"{s}:smp", sigmas=[f"{s}:lat", 1], latent_image=[f"{s}:lat", 0])
        take("11190:11123", f"{s}:dec", samples=[f"{s}:smp", 0])
        take("11190:11124", f"{s}:sel", frames=[f"{s}:dec", 0], recommended_index=[f"{s}:dec", 3])
        take("11190:11126", f"{s}:cm", image_target=[f"{s}:sel", 0], strength=src["11185"]["inputs"]["Xf"])
        take("11190:11125", f"{s}:st", inpainted_image=[f"{s}:cm", 0])
        take("shp", f"{s}:shp", image=[f"{s}:st", 0])
        take("shc", f"{s}:shc", destination=[f"{s}:st", 0], source=[f"{s}:shp", 0])
        g[f"{s}:sv"] = {"class_type": "SaveImage", "inputs": {"images": [f"{s}:shc", 0],
                                                              "filename_prefix": f"{prefix}_{s}"}}
    for k, n in g.items():
        n.setdefault("_meta", {"title": k})   # fill_defaults пишет журнал по title
    return g


def fetch(cli: Comfy, h: dict, out: Path, names: dict[str, str]) -> list[Path]:
    """SaveImage-узлы истории → файлы out/<имя>.png (names: id узла → имя)."""
    got = []
    for nid, o in h["outputs"].items():
        if nid in names and o.get("images"):
            p = out / f"{names[nid]}.png"; p.write_bytes(cli.view(o["images"][0])); got.append(p)
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "refine", "sheet", "eval", "batch"])
    ap.add_argument("src", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--url", default=None)
    ap.add_argument("--job", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--body", type=float, default=0.3)
    ap.add_argument("--face", default="0.35,0.45")
    ap.add_argument("--mp", type=float, default=None, help="кроп лица, Мп (по умолч. как в v2 — 4)")
    ap.add_argument("--copies", type=int, default=1, help="копий мода лица (2–3 — сильнее сходство)")
    ap.add_argument("--steps", type=int, default=None, help="шаги H3 лица (в v2 — 10)")
    ap.add_argument("--blend", type=int, default=None, help="шов вклейки лица, px (в v2 — 40)")
    ap.add_argument("--tag", default="", help="суффикс имени серии")
    ap.add_argument("--prompt2", action="store_true", help="промпт лица v2 (черты по рефам 07–11)")
    ap.add_argument("--expr", default=None, help="промпт лица v3 с этим выражением (напр. «a confident smile»)")
    a = ap.parse_args()
    if a.cmd == "batch":                # batch <v2_refine_api.json> <round.yaml>
        return batch(a)
    if a.cmd == "sheet":
        return sheet(Path(a.src))
    if a.cmd == "eval":                 # eval <out_dir> <face_ref_dir>
        return evaluate(Path(a.src), Path(a.out), [x for x in a.only.split(",") if x])
    import os
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cli = Comfy(a.url or os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    only = [x for x in a.only.split(",") if x]
    if a.cmd == "gen":
        g0 = json.loads(Path(a.src).read_text(encoding="utf-8"))[a.job]["g"]
        for var in only or list(VARIANTS):
            name = var + a.tag          # --tag: другая сцена (--job) в той же папке, напр. K → A0K.png
            if (out / f"{name}.png").exists():
                continue
            pid = cli.submit(gen_graph(g0, var, f"ss456/sharp/{name}"))
            h = cli.wait(pid, 1800)
            print(name, fetch(cli, h, out, {"87": name}), flush=True)
    else:
        v2 = json.loads(Path(a.src).read_text(encoding="utf-8"))
        faces = [float(x) for x in a.face.split(",") if x]
        bs = f"z{round(a.body * 100):02d}" + (f"m{a.mp:g}" if a.mp else "") + a.tag
        for var in only or ["A0", *VARIANTS]:
            raw = out / f"{var}.png"
            if not raw.exists() or (out / f"{var}_{bs}_f{round(faces[0] * 100):02d}.png").exists():
                continue
            img = cli.upload(raw, "ss456/sharp")
            g = refine_graph(v2, img, a.body, faces, f"ss456/sharp/{var}_{bs}", a.mp, a.copies, a.steps, a.blend,
                             face_prompt_v3(a.expr) if a.expr else FACE_PROMPT_V2 if a.prompt2 else FACE_PROMPT)
            names = {"sv_body": f"{var}_{bs}"} | {f"f{round(d * 100):02d}:sv": f"{var}_{bs}_f{round(d * 100):02d}"
                                                for d in faces}
            pid = cli.submit(g)
            h = cli.wait(pid, 1800)
            print(var, [p.name for p in fetch(cli, h, out, names)], flush=True)
    return 0


def batch(a) -> int:
    """Проход лица по всем готовым кадрам раунда (jobs.jsonl): промпт v3 с выражением из промпта сцены,
    кроп --mp (1), denoise --face (0.85,0.75) → <вариант>__<сид>__face<d>.png рядом с сырым. Готовые пропускает."""
    import os
    import ss340_round as R
    cfg = R.load(a.out)
    out = Path(cfg["out"]) / cfg["round"]
    v2 = json.loads(Path(a.src).read_text(encoding="utf-8"))
    faces = [float(x) for x in a.face.split(",") if x]
    cli = Comfy(a.url or os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    recs = [json.loads(l) for l in (out / "jobs.jsonl").read_text(encoding="utf-8").splitlines()]
    for j in recs:
        if not j.get("files"):
            continue
        raw = Path(j["files"][0]); stem = raw.with_suffix("")
        names = {f"f{round(d * 100):02d}:sv": f"{stem.name}__face{round(d * 100):02d}" for d in faces}
        if all((raw.parent / f"{n}.png").exists() for n in names.values()):
            continue
        expr = scene_expr(R.p(cfg, cfg["kit_poses"][j["pose"]]["prompt"]).read_text(encoding="utf-8"))
        import httpx
        import time
        for attempt in range(6):            # обрыв связи (19.10: getaddrinfo) — ждём и повторяем
            try:
                img = cli.upload(raw, f"ss340/{cfg['round']}/raw")
                g = refine_graph(v2, img, 0, faces, f"ss340/{cfg['round']}/{j['id']}__face", a.mp or 1.0,
                                 face_prompt=face_prompt_v3(expr))
                h = cli.wait(cli.submit(g), 1800)
                print(j["id"], expr, [p.name for p in fetch(cli, h, raw.parent, names)], flush=True)
                break
            except httpx.TransportError as e:
                print(f"{j['id']}: сеть ({e}), повтор через 60 с", flush=True)
                time.sleep(60)
            except Exception as e:  # noqa: BLE001 — кадр без лица и т. п.: остаётся сырой
                print(f"{j['id']}: ОШИБКА {str(e)[:300]}", flush=True)
                break
    return 0


def sheet(out: Path) -> int:
    """Контакт-лист: строки — варианты генерации, столбцы — сырой / тело / лицо; плюс кропы лица 1:1 по SAM3-боксу."""
    from PIL import Image, ImageDraw
    raws = sorted(p for p in out.glob("*.png") if "_" not in p.stem)
    cols = sorted({p.stem.split("_", 1)[1] for p in out.glob("*_*.png")}, key=lambda s: (len(s), s))
    cols = [""] + cols
    tw, th = 640, 358
    S = Image.new("RGB", (tw * len(cols), (th + 20) * len(raws) + 20), "white"); d = ImageDraw.Draw(S)
    for xi, c in enumerate(cols):
        d.text((xi * tw + 4, 4), c or "raw", fill="black")
    for yi, r in enumerate(raws):
        for xi, c in enumerate(cols):
            p = out / (f"{r.stem}_{c}.png" if c else r.name)
            if p.exists():
                S.paste(Image.open(p).convert("RGB").resize((tw, th), Image.LANCZOS), (xi * tw, 20 + yi * (th + 20) + 20))
            d.text((xi * tw + 4, 20 + yi * (th + 20) + 4), f"{r.stem} {c or 'raw'}", fill="black")
    S.save(out / "sheet.jpg", quality=88)
    print(out / "sheet.jpg")
    return 0


POSE_TOL = 8.0      # градусы: max |Δyaw|, |Δpitch|, |Δroll| между сырым кадром и доработкой


def safe_pick(rows: list[dict], tol: float = POSE_TOL) -> dict[str, str]:
    """Сырой кадр → доработка с лучшим AuraFace среди тех, где поза головы не сдвинулась > tol
    и сходство не ниже сырого; иначе сам сырой кадр (доработка отбрасывается)."""
    by = {r["img"]: r for r in rows}
    res = {}
    for raw in (r for r in rows if "_" not in r["img"] and "face" in r):
        ok = [r for k, r in by.items() if k.startswith(raw["img"] + "_") and "face" in r and r["face"] >= raw["face"]
              and max(abs(a - b) for a, b in zip(r["pose"], raw["pose"])) <= tol]
        res[raw["img"]] = max(ok, key=lambda r: r["face"])["img"] if ok else raw["img"]
    return res


def evaluate(out: Path, ref: Path, only: list[str] | None = None) -> int:
    """AuraFace против рефов (centroid), размер лица, резкость лица и тела (var Лапласиана; кадр приведён к 1888
    по ширине — честно для 1344) → eval.json + faces.jpg (кропы лиц 1:1 после приведения, подпись — метрики)."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    import ws07_lib as L
    from farm.qa import face as F
    cent = F.centroid(str(ref), backend="auraface", det="mediapipe", cache_dir=out / "_cache")
    rows, tiles = [], []
    pngs = [p for p in out.glob("*.png") if not only or p.stem.split("_")[0] in only]   # --only A0,A0K
    for p in sorted(pngs, key=lambda p: (p.stem.split("_")[0], len(p.stem), p.stem)):
        img = L.read_bgr(p)
        if img.shape[1] != 1888:
            img = cv2.resize(img, (1888, round(img.shape[0] * 1888 / img.shape[1])), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = lambda x0, y0, x1, y1: round(float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var()), 1)
        r = {"img": p.stem}
        fs = F.detect(img, "mediapipe")
        fb = None
        if fs:
            fc = max(fs, key=lambda x: x.size_px)
            r["face"] = round(float(F.cos(F.embed(img, fc, "auraface"), cent["emb"])), 3)
            r["face_px"] = round(fc.size_px)
            r["pose"] = [round(fc.yaw, 1), round(float(fc.extra.get("pitch") or 0), 1), round(fc.roll, 1)]
            fb = [int(v) for v in fc.bbox]
            r["lap_face"] = lap(*fb)
        pose = L.pose33(img)
        box = L.body_box(pose, img.shape[1] / img.shape[0]) if pose is not None else None
        if box:
            h, w = gray.shape
            x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
            y0 = max(y0, fb[3]) if fb else y0           # тело ниже лица
            r["lap_body"] = lap(x0, y0, x1, y1)
        rows.append(r); print(r, flush=True)
        if fb:
            cx, cy, s = (fb[0] + fb[2]) // 2, (fb[1] + fb[3]) // 2, 160
            c = img[max(cy - s, 0):cy + s, max(cx - s, 0):cx + s]
            tiles.append((r, Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).resize((2 * s, 2 * s))))
    (out / "eval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for raw, pick in safe_pick(rows).items():
        print(f"safe {raw}: {pick}")
    n = 6; S = Image.new("RGB", (320 * n, 344 * ((len(tiles) + n - 1) // n)), "white"); d = ImageDraw.Draw(S)
    for i, (r, t) in enumerate(tiles):
        x, y = 320 * (i % n), 344 * (i // n)
        S.paste(t, (x, y + 24))
        d.text((x + 3, y + 3), f"{r['img']} af {r.get('face')} lf {r.get('lap_face')} lb {r.get('lap_body')}", fill="black")
    S.save(out / "faces.jpg", quality=90)
    print(out / "faces.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
