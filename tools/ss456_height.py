#!/usr/bin/env python3
"""SS-456: механизм роста. Силуэт нужного роста на плите (A: рост персонажа, B: × --boost) и метрика роста
по длинам костей (не зависит от позы).

    python tools/ss456_height.py kit   <round_d3.yaml> <round_h1.yaml> [--boost 1.15]   # плиты с силуэтом + конфиг
    python tools/ss456_height.py measure <png>... [--card card.json] [--height 230]       # рост по костям, см
    python tools/ss456_height.py fix <png>... --kit <kit_d169> --v2 <v2_refine_api.json> --out <dir>
                                    # C: масштаб персонажа вокруг опоры до роста + склейка шва H3 (denoise 0.3)

Силуэт — капсула ss340_scale_kit (стоя / сидя) + класс «присед»; масштаб на глубине точки опоры:
(y_опоры − горизонт) / камера долей кадра на см. Предметы карточки, стоящие ближе к камере, чем точка опоры
(низ рамки ниже стоп, стопы внутри рамки по x), возвращаются поверх силуэта — персонаж за ними.
Промпт — плитный промпт сцены + определение «силуэт на плите» и правило «занимает место силуэта».
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from farm.prompts.lint import errors, lint  # noqa: E402

GREY = (128, 128, 128)
PIC1 = ("<Picture 1> is the empty room that the image is built on: its layout, camera position, framing and light "
        "are kept exactly.")
SIL_DEF = {"стоя": "where she stands", "сидя": "where she sits", "присед": "where she crouches"}
SIL_RULE = ("She takes the exact place of the flat grey silhouette in <Picture 1>, same position and same height, "
            "facing the camera, her body covering it completely; no grey silhouette remains in the photograph.")
# сцены теста: класс позы, точка опоры (стопы, доли кадра плиты), высота сиденья
SCENES = {"S6_greenhouse": "стоя", "S5_rack": "стоя", "C3_sofa": "сидя", "C4_rack": "присед"}
SEAT_CM = 45.0
# доли роста (антропометрия Drillis–Contini, округлено): кость → длина / рост
BONES = {(11, 13): 0.186, (12, 14): 0.186, (13, 15): 0.146, (14, 16): 0.146,
         (23, 25): 0.245, (24, 26): 0.245, (25, 27): 0.246, (26, 28): 0.246,
         (11, 12): 0.259, (23, 24): 0.191}
TORSO = 0.30          # середина плеч → середина таза


def silhouette(plate: Image.Image, card: dict, feet: tuple[float, float], height_cm: float, cls: str):
    """Капсула класса позы на копии плиты. → (картинка, верх головы y, масштаб px/см)."""
    im = plate.convert("RGB").copy()
    W, H = im.size
    s = (feet[1] - float(card["horizon_y"])) / float(card["camera_height_cm"]) * H
    fx, fy = feet[0] * W, feet[1] * H
    y = lambda cm: fy - cm * s  # noqa: E731
    dx = lambda cm: cm * s / 2  # noqa: E731
    d = ImageDraw.Draw(im)
    h = height_cm
    if cls == "стоя":
        top, sh, hip = h, 0.82 * h, 0.53 * h
        d.polygon([(fx - dx(38), y(sh)), (fx + dx(38), y(sh)), (fx + dx(36), y(hip)), (fx + dx(26), y(0)),
                   (fx - dx(26), y(0)), (fx - dx(36), y(hip))], fill=GREY)
    elif cls == "сидя":
        top, sh = SEAT_CM + 0.52 * h, SEAT_CM + 0.36 * h
        knee = 0.285 * h
        d.polygon([(fx - dx(26), y(0)), (fx + dx(26), y(0)), (fx + dx(30), y(knee)), (fx - dx(30), y(knee))], fill=GREY)
        d.rectangle([fx - dx(36), y(knee), fx + dx(36), y(SEAT_CM)], fill=GREY)
        d.polygon([(fx - dx(40), y(SEAT_CM)), (fx + dx(40), y(SEAT_CM)), (fx + dx(38), y(sh)), (fx - dx(38), y(sh))],
                  fill=GREY)
    elif cls == "присед":       # глубокий присед, пятки на полу: таз ≈ колени ≈ 0.28 роста, макушка ≈ 0.58
        top, sh, hip = 0.58 * h, 0.47 * h, 0.28 * h
        d.polygon([(fx - dx(60), y(0)), (fx + dx(60), y(0)), (fx + dx(75), y(hip)), (fx - dx(75), y(hip))], fill=GREY)
        d.polygon([(fx - dx(40), y(hip)), (fx + dx(40), y(hip)), (fx + dx(42), y(sh)), (fx - dx(42), y(sh))], fill=GREY)
    else:
        raise ValueError(f"класс позы {cls}")
    shoulders = sh
    d.ellipse([fx - dx(10), y(shoulders) - dx(10), fx + dx(10), y(shoulders) + dx(10)], fill=GREY)
    d.ellipse([fx - dx(16), y(top), fx + dx(16), y(top - 23)], fill=GREY)
    d.rectangle([fx - dx(6), y(top - 21), fx + dx(6), y(shoulders) + 2], fill=GREY)
    for a in card.get("anchors", []):             # невысокая мебель перед персонажем — поверх силуэта
        x0, y0, x1, y1 = a["bbox"]                # (стойка/рама, в которой она стоит, — не перед ней)
        if y1 > feet[1] and x0 < feet[0] < x1 and float(a["top_cm"]) < 0.6 * height_cm:
            box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
            im.paste(plate.convert("RGB").crop(box), box[:2])
    return im, round(float(y(top) / H), 4), s


def sil_prompt(base: str, cls: str) -> str:
    if PIC1 not in base or " Her face is sharp and natural." not in base:
        raise ValueError("в плитном промпте нет определения <Picture 1> или «Her face is sharp and natural.»")
    t = base.replace(PIC1, PIC1[:-1] + f"; a flat grey human silhouette drawn on it only marks {SIL_DEF[cls]} "
                                      "and how tall she is there.", 1)
    return t.replace(" Her face is sharp and natural.", f" {SIL_RULE} Her face is sharp and natural.", 1)


def cmd_kit(a) -> int:
    src_cfg = yaml.safe_load(Path(a.src).read_text(encoding="utf-8"))
    root = Path(src_cfg["src"])
    cfg = copy.deepcopy(src_cfg)
    cfg["round"] = Path(a.dst).stem
    cfg["kit_poses"], cfg["poses"], cfg["seeds"], cfg["variants"] = {}, [], {}, []
    seeds = src_cfg["seeds"][f"S6_greenhouse_plate"]
    out = root / "_claude" / "ss456" / "height" / "kit"
    for scene, cls in SCENES.items():
        base_pose = src_cfg["kit_poses"][f"{scene}_plate"]
        kit = root / Path(base_pose["plate"]).parent
        card = json.loads((kit / "card.json").read_text(encoding="utf-8"))
        tgt = json.loads((kit / "target.json").read_text(encoding="utf-8"))
        plate = Image.open(root / base_pose["plate"])
        base = (root / base_pose["prompt"]).read_text(encoding="utf-8")
        pvar = next(v for v in src_cfg["variants"] if v["id"] == f"plate_{scene}")
        for tag, k in (("sil", 1.0), (f"sil{round(a.boost * 100)}", a.boost)):
            im, top_y, _ = silhouette(plate, card, tuple(tgt["feet"]), tgt["height_cm"] * k, cls)
            if top_y < 0.01:                          # голова за кадром — в этой точке такой рост не помещается
                print(f"{scene}_{tag}: пропуск, верх головы {top_y}")
                continue
            d = out / f"{scene}_{tag}"; d.mkdir(parents=True, exist_ok=True)
            im.save(d / "plate.png")
            t = sil_prompt(base, cls)
            errs = [(i.code, i.msg) for i in errors(lint(t, "frame", {"connected": {"<Picture 1>"}}))]
            if errs:
                raise SystemExit(f"{scene} {tag}: линтер {errs}")
            (d / "prompt.txt").write_text(t, encoding="utf-8")
            (d / "target.json").write_text(json.dumps(dict(tgt, sil_k=k, sil_top_y=top_y, cls=cls), ensure_ascii=False,
                                                      indent=1), encoding="utf-8")
            pose = f"{scene}_{tag}"
            rel = lambda p: str(p.relative_to(root)).replace("\\", "/")  # noqa: E731
            cfg["kit_poses"][pose] = {"plate": rel(d / "plate.png"), "prompt": rel(d / "prompt.txt"),
                                      "canvas": base_pose["canvas"], "target_json": rel(d / "target.json"),
                                      "card": base_pose["card"]}
            cfg["poses"].append(pose); cfg["seeds"][pose] = list(seeds)
            v = copy.deepcopy(pvar)
            v.update(id=f"{tag}_{scene}", poses=[pose], seeds={pose: list(seeds)})
            cfg["variants"].append(v)
            print(pose, "верх головы", top_y, "слов", len(t.split()))
    cfg["est_frame_s"] = 110
    Path(a.dst).write_text("# SS-456 механизм роста, серия h1: силуэт роста персонажа (sil) и × boost на плите; "
                           "иначе = round_d3.yaml\n" + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                           encoding="utf-8")
    print("конфиг", a.dst, "кадров", sum(len(s) for s in cfg["seeds"].values()))
    return 0


def bone_height(p: np.ndarray, size: tuple[int, int], card: dict) -> dict | None:
    """Рост по длинам костей на глубине точки опоры (нижняя видимая точка стоп). p — 33×(x, y, vis) в долях.
    Ракурс только укорачивает кости → берём верхний квартиль оценок по костям. → {cm, n, foot_y} или None."""
    W, H = size
    vis = lambda i: p[i, 2] >= 0.5  # noqa: E731
    feet = [p[i, 1] for i in (27, 28, 29, 30, 31, 32) if vis(i)]
    if not feet:
        return None
    fy = max(feet)
    hor, cam = float(card["horizon_y"]), float(card["camera_height_cm"])
    if fy <= hor + 0.02:
        return None
    s = (fy - hor) / cam * H                               # px на см на глубине стоп
    xy = lambda i: np.array([p[i, 0] * W, p[i, 1] * H])  # noqa: E731
    est = [np.linalg.norm(xy(a) - xy(b)) / s / r for (a, b), r in BONES.items() if vis(a) and vis(b)]
    if all(vis(i) for i in (11, 12, 23, 24)):
        est.append(np.linalg.norm((xy(11) + xy(12)) / 2 - (xy(23) + xy(24)) / 2) / s / TORSO)
    if len(est) < 3:
        return None
    return {"cm": round(float(np.percentile(est, 75)), 1), "n": len(est), "foot_y": round(float(fy), 3)}


def cmd_measure(a) -> int:
    import ws07_lib as L
    card = json.loads(Path(a.card).read_text(encoding="utf-8"))
    for f in a.files:
        img = L.read_bgr(f); h, w = img.shape[:2]
        p = L.pose33(img)
        r = bone_height(p, (w, h), card) if p is not None else None
        print(Path(f).name, r, (f"/ {a.height:g} = {r['cm'] / a.height:.2f}" if r else ""))
    return 0


# --------------------------------------------------------------------------- C: поправка роста после генерации
SEAM_PROMPT = ("A photorealistic photograph of a very tall muscular woman in the room, exactly as in the image: her "
               "body, pose, clothes and the room stay unchanged; only the edges where she meets the room are blended "
               "naturally, with matching light and a soft contact shadow on the floor. Static camera. "
               "Lighting preserved unchanged. Colors and saturation unchanged.")


def mask_graph(image: str, prefix: str) -> dict:
    """SAM3 «woman» → маска персонажа PNG (белое = персонаж)."""
    return {"in": {"class_type": "LoadImage", "inputs": {"image": image}},
            "sam": {"class_type": "SAM3Segment", "inputs": {
                "image": ["in", 0], "prompt": "woman", "output_mode": "Merged", "confidence_threshold": 0.5,
                "max_segments": 1, "segment_pick": 0, "mask_blur": 0, "mask_offset": 0, "device": "Auto",
                "invert_output": False, "unload_model": False, "background": "Alpha", "background_color": "#222222"}},
            "mi": {"class_type": "MaskToImage", "inputs": {"mask": ["sam", 1]}},
            "sv": {"class_type": "SaveImage", "inputs": {"images": ["mi", 0], "filename_prefix": prefix}}}


def rescale(frame: Image.Image, mask: Image.Image, plate: Image.Image, card: dict, k: float,
            pivot: tuple[float, float]):
    """Персонаж ×k вокруг точки опоры pivot (доли), фон — плита, невысокая мебель перед опорой — поверх.
    → (картинка, маска шва для склейки: кольцо по краю нового силуэта + старый силуэт вне нового)."""
    from PIL import ImageChops, ImageFilter
    W, H = frame.size
    plate = plate.convert("RGB").resize((W, H), Image.LANCZOS)
    m = mask.convert("L").resize((W, H)).point(lambda v: 255 if v > 127 else 0)
    px, py = pivot[0] * W, pivot[1] * H
    nw, nh = round(W * k), round(H * k)
    ox, oy = round(px - px * k), round(py - py * k)            # точка опоры остаётся на месте
    big, bigm = frame.resize((nw, nh), Image.LANCZOS), m.resize((nw, nh), Image.LANCZOS)
    out = plate.copy()
    layer = Image.new("RGB", (W, H)); lm = Image.new("L", (W, H))
    layer.paste(big, (ox, oy)); lm.paste(bigm, (ox, oy))
    out.paste(layer, (0, 0), lm)
    for a in card.get("anchors", []):                           # невысокая мебель перед опорой — поверх
        x0, y0, x1, y1 = a["bbox"]
        if y1 > pivot[1] and x0 < pivot[0] < x1 and float(a["top_cm"]) < 140:
            box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
            out.paste(plate.crop(box), box[:2])
    ring = ImageChops.subtract(lm.filter(ImageFilter.MaxFilter(25)), lm.filter(ImageFilter.MinFilter(25)))
    seam = ImageChops.lighter(ring, ImageChops.subtract(m, lm)).filter(ImageFilter.MaxFilter(9))
    return out, seam


def seam_graph(v2: dict, image: str, seam: str, prefix: str, denoise: float) -> dict:
    """Склейка шва: H3 fl2va по маске шва (InpaintCrop/Stitch из v2, кроп до 2 Мп), промпт SEAM_PROMPT, без модов."""
    src = copy.deepcopy(v2)
    g = {"in": {"class_type": "LoadImage", "inputs": {"image": image}},
         "seam": {"class_type": "LoadImage", "inputs": {"image": seam}},
         "sm": {"class_type": "ImageToMask", "inputs": {"image": ["seam", 0], "channel": "red"}}}
    for nid in ("11190:11105", "11190:11106", "11190:11107", "11190:11108", "11190:11114", "11190:11115",
                "11190:11116", "11190:11117", "11190:11118", "11190:11119", "11190:11120", "11190:11121",
                "11190:11122", "11190:11123", "11190:11124", "11190:11126", "11190:11125"):
        g[nid] = copy.deepcopy(src[nid])
    g["11190:11108"]["inputs"]["model"] = ["11190:11105", 0]
    g["11190:11114"]["inputs"].update(image=["in", 0], mask=["sm", 0], context_from_mask_extend_factor=1.2,
                                      mask_blend_pixels=16)
    g["11190:11115"]["inputs"]["megapixels"] = 2.0
    g["11190:11117"]["inputs"]["prompt"] = SEAM_PROMPT
    g["11190:11118"]["inputs"].update(model=["11190:11108", 0], denoise=denoise)
    g["11190:11119"]["inputs"]["noise_seed"] = 4242
    g["11190:11120"]["inputs"].update(model=["11190:11108", 0])
    g["11190:11126"]["inputs"]["strength"] = 0.95
    g["sv"] = {"class_type": "SaveImage", "inputs": {"images": ["11190:11125", 0], "filename_prefix": prefix}}
    for k, n in g.items():
        n.setdefault("_meta", {"title": k})
    return g


def cmd_fix(a) -> int:
    """Поправка роста кадров раунда: рост по костям → k = цель / рост (если промах > --tol) → масштаб вокруг опоры →
    склейка шва H3. Кадры: <out>/<сцена>__<вар>__s<сид>.png (+ __mask, __scaled, __seam, __fixed)."""
    import os
    import ws07_lib as L
    from ss340_round import Comfy
    cli = Comfy(a.url or os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    v2 = json.loads(Path(a.v2).read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in map(Path, a.files):
        scene = f.name.split("__")[0]
        kit = Path(a.kit) / scene
        card = json.loads((kit / "card.json").read_text(encoding="utf-8"))
        img = L.read_bgr(f); h, w = img.shape[:2]
        p = L.pose33(img)
        r = bone_height(p, (w, h), card) if p is not None else None
        if not r:
            print(f.name, "рост не измерен — пропуск"); continue
        k = a.height / r["cm"]
        rec = {"frame": f.name, "cm": r["cm"], "k": round(k, 3)}
        if abs(k - 1) <= a.tol:
            print(rec, "в допуске"); continue
        stem = out / f.stem
        if not Path(f"{stem}__mask.png").exists():
            name = cli.upload(f, "ss456/height/raw")
            h1 = cli.wait(cli.submit(mask_graph(name, f"ss456/height/{f.stem}__mask")), 900)
            mf = next(o["images"][0] for o in h1["outputs"].values() if o.get("images"))
            Path(f"{stem}__mask.png").write_bytes(cli.view(mf))
        xs = [p[i, 0] for i in (27, 28, 29, 30, 31, 32) if p[i, 2] >= 0.5]
        pivot = (float(np.mean(xs)) if xs else 0.5, r["foot_y"])
        scaled, seam = rescale(Image.open(f).convert("RGB"), Image.open(f"{stem}__mask.png"),
                               Image.open(kit / "plate.png"), card, k, pivot)
        scaled.save(f"{stem}__scaled.png"); seam.save(f"{stem}__seam.png")
        n_img, n_seam = cli.upload(Path(f"{stem}__scaled.png"), "ss456/height/fix"), \
            cli.upload(Path(f"{stem}__seam.png"), "ss456/height/fix")
        h2 = cli.wait(cli.submit(seam_graph(v2, n_img, n_seam, f"ss456/height/{f.stem}__fixed", a.denoise)), 900)
        ff = next(o["images"][0] for o in h2["outputs"].values() if o.get("images"))
        Path(f"{stem}__fixed.png").write_bytes(cli.view(ff))
        img2 = L.read_bgr(f"{stem}__fixed.png"); p2 = L.pose33(img2)
        r2 = bone_height(p2, (img2.shape[1], img2.shape[0]), card) if p2 is not None else None
        rec["cm_fixed"] = r2 and r2["cm"]
        print(rec, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kit"); k.add_argument("src"); k.add_argument("dst"); k.add_argument("--boost", type=float, default=1.15)
    m = sub.add_parser("measure"); m.add_argument("files", nargs="+"); m.add_argument("--card", required=True)
    m.add_argument("--height", type=float, default=230)
    fx = sub.add_parser("fix"); fx.add_argument("files", nargs="+"); fx.add_argument("--kit", required=True)
    fx.add_argument("--v2", required=True); fx.add_argument("--out", required=True)
    fx.add_argument("--height", type=float, default=230); fx.add_argument("--tol", type=float, default=0.08)
    fx.add_argument("--denoise", type=float, default=0.3); fx.add_argument("--url", default=None)
    a = ap.parse_args()
    return {"kit": cmd_kit, "measure": cmd_measure, "fix": cmd_fix}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
