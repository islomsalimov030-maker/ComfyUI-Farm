#!/usr/bin/env python3
"""SS-340 / WS-07 серия s2_scale: масштаб позы образом, а не текстом. Из карточки плиты (якоря, горизонт,
высота камеры, слот) и позы одобренного кадра v2 → рамка тела нужного роста в нужном месте пола →
зелёная рамка на копии плиты, скелет того же размера (Picture 4), промпты, card.json и target.json.

    python tools/ss340_scale_kit.py --plates C:/stacy-lora/_claude/ws03/plates_apartment.yaml \\
        --plate apartment/window --pose Standing-2 --height 162 --slot A --name s2A
    python tools/ss340_scale_kit.py ... --feet 0.511,0.987 --name s2v2     # стопы там же, где у v2, рост 162

Рост в кадре: (y_стоп − горизонт) · рост / камера (WS-03, `farm.qa.image.scale.expected_height`); камера —
медиана top·(y1−гор)/(y1−y0) по якорям на полу. Поза v2 масштабируется вокруг середины стоп (k = нужная высота
рамки тела / высота рамки на v2) и переносится в точку стоп. Медиа — в <src>/_claude/ss340/kit_s2/<name>/ (не в репо).
Промпт — Standing-2d кита раунда C (рост 162, зеркало как якорь) + варианты green / skeleton / green_skeleton /
guide (чистая плита + плита с рамкой отдельным гидом <Picture 4>) / silhouette (плоский серый силуэт нужного роста
на плите, персонаж его заменяет).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import ws07_lib as L  # noqa: E402
from farm.prompts.lint import errors, lint  # noqa: E402
from farm.qa.image.scale import expected_height  # noqa: E402


def card_from_draft(p: dict) -> dict:
    """Черновик WS-03 (plates_*.yaml) → карточка в формате фермы (CONTRACTS §10) + camera_height_cm."""
    hor, cams, anchors = float(p["hor"]), [], []
    for a in p["anchors"]:
        name, cm, x0, y0, x1, y1 = a[:6]
        anchors.append({"name": name, "top_cm": float(cm), "bbox": [x0, y0, x1, y1]})
        if len(a) <= 7 and y1 - hor >= 0.03:          # nofloor/nocam — не для высоты камеры
            cams.append(cm * (y1 - hor) / (y1 - y0))
    return {"id": p["id"], "horizon_y": hor, "camera_height_cm": round(st.median(cams), 1) if cams else None,
            "cams": [round(c, 1) for c in cams], "anchors": anchors,
            "slots": [{"id": s[0], "pose": s[1], "x": s[2], "y": s[3]} for s in p.get("slots", [])]}


GUIDE_DEF = ("<Picture 4> is a layout guide only: the same room as <Picture 1> with a thin bright green rectangle that "
             "marks where she stands and how tall she is there.")
GUIDE_RET = "<Picture 4>: attribute_transfer - only her position and height in the frame are taken from it."
GUIDE_RULE = ("She stands exactly where the green rectangle of <Picture 4> is, her body filling it from top to bottom; "
              "the room, the light and every object come from <Picture 1>, and the green rectangle never appears "
              "in the photograph.")


SIL_DEF = ("<Picture 1> is the empty room that the image is built on: its layout, camera position, framing and light are "
           "kept exactly; a flat grey human silhouette drawn on it only marks where she stands and how tall she is.")
SIL_RULE = ("She takes the exact place of the flat grey silhouette in <Picture 1>, same position and same height, her body "
            "covering it completely; no grey silhouette remains in the photograph.")


def silhouette_variant(base: str) -> str:
    """Плита с плоским серым силуэтом нужного роста: персонаж заменяет силуэт (правка «заменить», без рамки)."""
    return L._insert_rule(base.replace(L.PIC1_DEF, SIL_DEF, 1), SIL_RULE)


def draw_silhouette(plate, p: np.ndarray, color=(128, 128, 128)):
    """Плоский серый силуэт по 33 точкам позы. Толщины — от ширины плеч и роста (голова фронтально по точкам лица
    выходит слишком мелкой): голова ≈ 1/7.5 роста, бёдра 0.4 ширины плеч, руки 0.22."""
    from PIL import ImageDraw
    im = plate.convert("RGB").copy()
    w, h = im.size
    d = ImageDraw.Draw(im)
    vis = lambda i: p[i, 2] >= L.MIN_VIS  # noqa: E731
    xy = lambda i: (float(p[i, 0] * w), float(p[i, 1] * h))  # noqa: E731
    box = L.body_box(p, w / h)
    body_h = (box[3] - box[1]) * h
    sw = abs(xy(11)[0] - xy(12)[0]) if vis(11) and vis(12) else body_h / 4.5
    head_r = body_h / 15
    def limb(a, b, t):
        if vis(a) and vis(b):
            t = max(2, int(t))
            d.line([xy(a), xy(b)], fill=color, width=t)
            for c in (a, b):
                d.ellipse([xy(c)[0] - t / 2, xy(c)[1] - t / 2, xy(c)[0] + t / 2, xy(c)[1] + t / 2], fill=color)
    if all(vis(i) for i in (11, 12, 23, 24)):
        (lx, ly), (rx, ry) = xy(23), xy(24)
        k = 0.15 * sw * (1 if lx > rx else -1)
        d.polygon([xy(11), xy(12), (rx - k, ry), (lx + k, ly)], fill=color)
    for a, b in ((23, 25), (24, 26)):
        limb(a, b, 0.42 * sw)
    for a, b in ((25, 27), (26, 28)):
        limb(a, b, 0.30 * sw)
    for a, b in ((27, 31), (28, 32)):
        limb(a, b, 0.18 * sw)
    for a, b in ((11, 13), (12, 14), (13, 15), (14, 16)):
        limb(a, b, 0.22 * sw)
    if vis(0):
        cx = xy(0)[0]
        top = box[1] * h
        d.ellipse([cx - 0.8 * head_r, top, cx + 0.8 * head_r, top + 2 * head_r], fill=color)
        if vis(11) and vis(12):                   # шея
            nx, ny = (xy(11)[0] + xy(12)[0]) / 2, (xy(11)[1] + xy(12)[1]) / 2
            d.line([(nx, ny), (cx, top + 1.6 * head_r)], fill=color, width=max(2, int(0.55 * head_r)))
    return im


def capsule(plate, card: dict, feet: tuple[float, float], height_cm: float, cls: str = "стоя",
            seat_cm: float = 45.0, color=(128, 128, 128)):
    """Обобщённый силуэт без конечностей — только место и рост, поза остаётся за текстом. Размеры — антропометрия
    в см (голова 23×16, плечи 38, таз 36, ноги вместе 26), масштаб на глубине стоп: (y_стоп − горизонт) / камера
    долей кадра на см. Стоя — «столбик» до пола; сидя — голени до колен (50 см) + корпус от сиденья, верх головы
    на сиденье + 0.52 роста. → (картинка, рамка [x0, y0, x1, y1], верх головы y)."""
    from PIL import ImageDraw
    im = plate.convert("RGB").copy()
    W, H = im.size
    hor, cam = float(card["horizon_y"]), float(card["camera_height_cm"])
    s = (feet[1] - hor) / cam * H                      # px на см на глубине стоп
    fx, fy = feet[0] * W, feet[1] * H
    y = lambda cm: fy - cm * s  # noqa: E731
    dx = lambda cm: cm * s / 2  # noqa: E731
    d = ImageDraw.Draw(im)
    if cls == "стоя":
        top = height_cm
        sh, hip = 0.82 * height_cm, 0.53 * height_cm
        body = [(fx - dx(38), y(sh)), (fx + dx(38), y(sh)), (fx + dx(36), y(hip)), (fx + dx(26), y(0)),
                (fx - dx(26), y(0)), (fx - dx(36), y(hip))]
        d.polygon(body, fill=color)
        d.ellipse([fx - dx(10), y(sh) - dx(10), fx + dx(10), y(sh) + dx(10)], fill=color)
    elif cls == "сидя":
        top = seat_cm + 0.52 * height_cm
        sh = seat_cm + 0.36 * height_cm
        d.polygon([(fx - dx(26), y(0)), (fx + dx(26), y(0)), (fx + dx(30), y(50)), (fx - dx(30), y(50))], fill=color)
        d.polygon([(fx - dx(40), y(seat_cm)), (fx + dx(40), y(seat_cm)), (fx + dx(38), y(sh)), (fx - dx(38), y(sh))],
                  fill=color)
        d.rectangle([fx - dx(36), y(50), fx + dx(36), y(seat_cm)], fill=color)
    else:
        raise ValueError(f"класс позы {cls}: только стоя | сидя")
    d.ellipse([fx - dx(16), y(top), fx + dx(16), y(top - 23)], fill=color)          # голова
    d.rectangle([fx - dx(6), y(top - 21), fx + dx(6), y(sh) + 2], fill=color)        # шея
    half = 20 * s                                   # рамка-цель: 40 см по ширине
    box = [(fx - half) / W, y(top) / H, (fx + half) / W, fy / H]
    return im, [round(float(v), 4) for v in box], round(float(y(top) / H), 4)


def guide_variant(base: str) -> str:
    """Чистая плита <Picture 1> + плита с зелёной рамкой как отдельный гид <Picture 4> (рамка не на исходнике)."""
    t = base.replace(L.PIC1_DEF, L.PIC1_DEF + "\n" + GUIDE_DEF, 1)
    t = t.replace("<Picture 1>: fully_preserved", GUIDE_RET + "\n<Picture 1>: fully_preserved", 1)
    return L._insert_rule(t, GUIDE_RULE)


def transform(p: np.ndarray, k: float, src: tuple[float, float], dst: tuple[float, float]) -> np.ndarray:
    q = p.copy()
    q[:, 0] = dst[0] + k * (p[:, 0] - src[0])
    q[:, 1] = dst[1] + k * (p[:, 1] - src[1])
    return q


def green_box(plate, box, pad_x: float = 0.15):
    """Зелёная рамка: по высоте ровно рост (промпт: «заполняет её сверху донизу»), по ширине с запасом на руки."""
    from PIL import ImageDraw
    im = plate.convert("RGB").copy()
    w, h = im.size
    bw = box[2] - box[0]
    xy = [max(0, (box[0] - pad_x * bw) * w), max(0, box[1] * h), min(w - 1, (box[2] + pad_x * bw) * w),
          min(h - 1, box[3] * h)]
    ImageDraw.Draw(im).rectangle(xy, outline=(0, 255, 0), width=max(3, int(0.004 * w)))
    return im


def main() -> int:
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="C:/stacy-lora")
    ap.add_argument("--plates", required=True)
    ap.add_argument("--plate", required=True, help="ID плиты в черновике WS-03")
    ap.add_argument("--pose", default="Standing-2", help="поза кита WS-07 (плита, кадр v2)")
    ap.add_argument("--prompt", default="_claude/ss340/kit_c/Standing-2d/prompt_fixe.txt")
    ap.add_argument("--height", type=float, default=162)
    ap.add_argument("--slot", default="A")
    ap.add_argument("--feet", help="x,y стоп в долях кадра вместо слота")
    ap.add_argument("--drop", action="append", default=[], help="фраза, которую убрать из промпта")
    ap.add_argument("--name", required=True)
    ap.add_argument("--shape", default="pose", choices=["pose", "capsule"],
                    help="pose — силуэт/рамка/скелет из позы кадра v2; capsule — обобщённый силуэт без позы")
    ap.add_argument("--cls", default="стоя", choices=["стоя", "сидя"], help="capsule: класс позы")
    ap.add_argument("--plate-file", help="capsule: плита (от --src), по умолчанию — плита кита позы")
    ap.add_argument("--sub", action="append", default=[], help="замена в промпте «было=>стало»")
    a = ap.parse_args()
    src = Path(a.src)
    kit = src / "_claude/ws07/kit/refs" / a.pose
    out = src / "_claude/ss340/kit_s2" / a.name
    out.mkdir(parents=True, exist_ok=True)
    draft = next(p for p in yaml.safe_load(Path(a.plates).read_text(encoding="utf-8")) if p["id"] == a.plate)
    card = card_from_draft(draft)
    if a.feet:
        fx, fy = (float(v) for v in a.feet.split(","))
    else:
        s = next(s for s in card["slots"] if s["id"] == a.slot)
        fx, fy = s["x"], s["y"]
    base = (src / a.prompt).read_text(encoding="utf-8").split("\n# inputs:")[0].rstrip() + "\n"
    for d in a.drop:
        if d not in base:
            print(f"!! в промпте нет «{d}»", file=sys.stderr)
            return 1
        base = base.replace(d, "")
    for pair in a.sub:
        old, new = pair.split("=>", 1)
        if old not in base:
            print(f"!! в промпте нет «{old}»", file=sys.stderr)
            return 1
        base = base.replace(old, new)
    if a.shape == "capsule":
        seat = next((x["top_cm"] for x in card["anchors"] if "seat" in x["name"]), 45.0)
        plate = Image.open(src / a.plate_file if a.plate_file else kit / "plate.png").convert("RGB")
        plate.save(out / "plate.png")
        im, tbox, top_y = capsule(plate, card, (fx, fy), a.height, a.cls, seat)
        im.save(out / "plate_silhouette.png")
        prompts = {"fixe": base, "silhouette": silhouette_variant(base)}
        rep = {}
        for v, t in prompts.items():
            if "<Picture 4>" in t:
                print(f"!! {v}: <Picture 4> в промпте, а входа нет", file=sys.stderr)
                return 1
            errs = [(i.code, i.msg) for i in errors(lint(t, "frame", {"connected": {"<Picture 1>", "<Picture 2>"}}))]
            rep[v] = {"words": len(t.split()), "errors": errs}
            (out / f"prompt_{v}.txt").write_text(t, encoding="utf-8")
        (out / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        exp_h = expected_height(card, fy, a.height) if a.cls == "стоя" else None
        tgt = {"plate": a.plate, "cls": a.cls, "height_cm": a.height, "feet": [fx, fy], "shape": "capsule",
               "expected_h": exp_h and round(exp_h, 4), "expected_top_y": top_y, "target_box": tbox, "lint": rep}
        (out / "target.json").write_text(json.dumps(tgt, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k_: v for k_, v in tgt.items() if k_ != "lint"}, ensure_ascii=False))
        bad = {v: r["errors"] for v, r in rep.items() if r["errors"]}
        print("линтер:", bad or "0 ошибок", "→", out)
        return 1 if bad else 0
    exp_h = expected_height(card, fy, a.height)
    if not exp_h:
        print("!! в карточке нет горизонта/камеры или стопы выше горизонта", file=sys.stderr)
        return 1
    v2 = L.read_bgr(kit / "frame_v2.png")
    vh, vw = v2.shape[:2]
    p = L.pose33(v2)
    box = L.body_box(p, vw / vh)
    feet_v2 = (float(np.mean(p[[27, 28], 0])), box[3])
    k = exp_h / (box[3] - box[1])
    q = transform(p, k, feet_v2, (fx, fy))
    plate = Image.open(kit / "plate.png").convert("RGB")
    W, H = plate.size
    tbox = L.body_box(q, W / H)
    tbox = [round(float(x), 4) for x in tbox]
    green_box(plate, tbox).save(out / "plate_green.png")
    L.draw_skeleton(q, (W, H)).save(out / "skeleton.png")
    draw_silhouette(plate, q).save(out / "plate_silhouette.png")
    # контроль: скелет поверх плиты (для глаз, на студию не идёт)
    Image.blend(plate, L.draw_skeleton(q, (W, H)), 0.45).save(out / "_check_overlay.jpg", quality=88)

    prompts = {"fixe": base, "green": L.prompt_variant(base, "green"), "skeleton": L.prompt_variant(base, "skeleton")}
    prompts["green_skeleton"] = L.prompt_variant(prompts["skeleton"], "green")
    prompts["guide"] = guide_variant(base)
    prompts["silhouette"] = silhouette_variant(base)
    rep = {}
    for v, t in prompts.items():
        labels = {"<Picture 1>", "<Picture 2>"} | ({"<Picture 4>"} if ("skeleton" in v or v == "guide") else set())
        if ("skeleton" in v or v == "guide") and "<Picture 4>" not in t or "green" in v and "green rectangle" not in t \
                or v == "guide" and GUIDE_RULE not in t or v == "silhouette" and SIL_RULE not in t:
            print(f"!! вариант {v}: правило не вставилось", file=sys.stderr)
            return 1
        errs = [(i.code, i.msg) for i in errors(lint(t, "frame", {"connected": labels}))]
        rep[v] = {"words": len(t.split()), "errors": errs}
        (out / f"prompt_{v}.txt").write_text(t, encoding="utf-8")
    (out / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    tgt = {"pose": a.pose, "plate": a.plate, "expected_top_y": round(float(tbox[1]), 4), "height_cm": a.height, "feet": [fx, fy], "expected_h": round(exp_h, 4),
           "k_vs_v2": round(k, 3), "target_box": tbox, "v2_box": [round(float(x), 4) for x in box],
           "v2_expected_h": round(expected_height(card, box[3], a.height) or 0, 4), "lint": rep}
    (out / "target.json").write_text(json.dumps(tgt, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k_: v for k_, v in tgt.items() if k_ != "lint"}, ensure_ascii=False))
    bad = {v: r["errors"] for v, r in rep.items() if r["errors"]}
    print("линтер:", bad or "0 ошибок", "→", out)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
