#!/usr/bin/env python3
"""SS-340: сборка папок RefMod на ПК — по папке на будущий мод, картинки одного аспекта + альфа-маска + refmod.yaml.

    .venv/Scripts/python tools/refmod_build.py [--src C:/stacy-lora] [--out C:/stacy-lora/refmod_inputs/next] [--only face,zones]

Раскладка (одна подпапка = один мод; H3RefModCreateFromFolder use_subfolders / ws07_graphs.refmod_extract_graph):
  characters/<id>/face/          1:1 768, лицо по центру, маска лицо+волосы (MediaPipe face → эллипс с запасом)
  characters/<id>/body_scale/    3:4 768x1024, ПОЛНЫЙ РОСТ с полями (рост виден целиком, не крупный план),
                                 нейтральный фон листа; вписывание без растяжения (маски нет)
  characters/<id>/parts/<часть>/ 1:1 512, плотный кроп части; SFW-аналоги считаются автоматически (hands, feet),
                                 уникальные приметы — вручную: картинки + mark.json (см. --marks), слоты без картинок
                                 остаются пустыми с README (заполняет человек)
  locations/<loc>/<zone>/        16:9 1344x768, плита зоны целиком без персонажа, без маски
Картинки кропаются и масштабируются здесь, чтобы холст мода (задаёт первая картинка) не обрезал остальные.
Сами медиа в репо не кладём — только этот скрипт. Маски: альфа PNG (255 = субъект); в графе InvertMask.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageOps

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

FACE, BODY, PART, ZONE = (768, 768), (768, 1024), (512, 512), (1344, 768)
# токены мода (Full Reference): T * (H/32) * (W/32), T = число картинок (стопка по времени)
TOK = {k: (v[0] // 32) * (v[1] // 32) for k, v in {"face": FACE, "body": BODY, "part": PART, "zone": ZONE}.items()}


# --------------------------------------------------------------------------------------- геометрия
def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter or 1)


def square_box(cx, cy, side, w, h):
    side = min(side, w, h)
    x0 = int(min(max(cx - side / 2, 0), w - side))
    y0 = int(min(max(cy - side / 2, 0), h - side))
    return x0, y0, x0 + int(side), y0 + int(side)


def rgba(img: Image.Image, mask: Image.Image | None, box, size) -> Image.Image:
    out = img.crop(box).resize(size, Image.LANCZOS).convert("RGBA")
    if mask is not None:
        out.putalpha(mask.crop(box).resize(size, Image.LANCZOS))
    return out


# --------------------------------------------------------------------------------------- детекторы (MediaPipe, локально)
def _mp():
    from farm.qa import mp as M
    return M


def face_box(img: Image.Image):
    M = _mp()
    a = np.asarray(img.convert("RGB"))
    r = M._task("face_landmarker").detect(M._mp_image(a))
    if not r.face_landmarks:
        return None
    p = np.array([[q.x * a.shape[1], q.y * a.shape[0]] for q in r.face_landmarks[0]])
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def pose_points(img: Image.Image):
    M = _mp()
    a = np.asarray(img.convert("RGB"))
    r = M._task("pose_landmarker").detect(M._mp_image(a))
    if not r.pose_landmarks:
        return None
    return np.array([[q.x * a.shape[1], q.y * a.shape[0], q.visibility] for q in r.pose_landmarks[0]])


def ellipse_mask(size, box, grow=0.25, blur=12):
    m = Image.new("L", size, 0)
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * grow, (y1 - y0) * grow
    ImageDraw.Draw(m).ellipse([x0 - dx, y0 - dy * 1.6, x1 + dx, y1 + dy * 0.6], fill=255)  # волосы сверху
    return m.filter(ImageFilter.GaussianBlur(blur))


# --------------------------------------------------------------------------------------- сборка
def manifest(d: Path, **kw):
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("*.png")))
    kw.setdefault("tokens_per_image", None)
    if kw["tokens_per_image"]:
        kw["tokens_total"] = kw["tokens_per_image"] * n
    kw["images"] = n
    (d / "refmod.yaml").write_text(yaml.safe_dump(kw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return n


def characters(src: Path) -> list[dict]:
    rows = list(csv.DictReader(open(src / "_claude/ws03/characters_stacy.tsv", encoding="utf-8"), delimiter="\t"))
    return [{"id": r["ID"], "height_cm": int(r["Рост, см"] or 0) or None, "trigger": r["Триггер LoRA"]} for r in rows]


def build_face(src, out, ch, limit=10):
    d = out / "characters" / ch["id"] / "face"
    files = [f for f in sorted((src / "dataset_face_v1_ready").glob("*.png")) if "_r" not in f.stem]  # _r2 — дубли
    picks, skipped = [], 0
    for f in files:
        if len(picks) >= limit:
            break
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        b = face_box(img)
        if b is None:
            skipped += 1
            continue
        w, h = img.size
        side = max(b[2] - b[0], b[3] - b[1]) * 1.9
        box = square_box((b[0] + b[2]) / 2, (b[1] + b[3]) / 2 - (b[3] - b[1]) * 0.1, side, w, h)
        m = ellipse_mask(img.size, b)
        d.mkdir(parents=True, exist_ok=True)
        rgba(img, m, box, FACE).save(d / f"{len(picks):02d}_{f.stem}.png")
        picks.append(f.name)
    return manifest(d, kind="face", character=ch["id"], preset="identity_encode", concept_type="identity",
                    ref_resolution=FACE[0], aspect="1:1", mask="alpha: лицо+волосы", tokens_per_image=TOK["face"],
                    sources=picks, skipped_no_face=skipped,
                    use="1 / 5 / 10 первых — серия r1 (кривая числа кадров)")


def fit_pad(img: Image.Image, size, fill=None) -> Image.Image:
    """Вписать без искажения (пропорции тела важнее всего) и добить полями цвета фона (угол картинки)."""
    fill = fill or img.getpixel((2, 2))
    im = img.copy()
    im.thumbnail(size, Image.LANCZOS)
    out = Image.new("RGB", size, fill)
    out.paste(im, ((size[0] - im.size[0]) // 2, (size[1] - im.size[1]) // 2))
    return out


def build_body(src, out, ch):
    """Полный рост, панель листа целиком (голова и стопы с полями), вписана без растяжения в 3:4, без маски
    (фон листа нейтральный и одинаковый). Кроп по позе не делаем — он режет волосы/плечи и искажает пропорции."""
    d = out / "characters" / ch["id"] / "body_scale"
    sheet = ImageOps.exif_transpose(Image.open(src / "character_sheets/sheet_body_sfw.png")).convert("RGB")
    W, H = sheet.size
    made, src_note = 0, []
    # Проверенные границы исходного sheet_body_sfw 2232×1152: портретная панель шире остальных.
    # Делить на пять равных частей нельзя — соседний персонаж попадёт в референс.
    edges = [round(x * W / 2232) for x in (0, 576, 990, 1404, 1818, 2232)]
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, len(edges) - 1):   # панель 0 — портреты (лицо уже в face/)
        panel = sheet.crop((edges[i] + 3, 0, edges[i + 1] - 3, H))
        pts = pose_points(panel)
        if pts is None:                 # не человек в полный рост — пропуск
            continue
        fit_pad(panel, BODY).save(d / f"{made:02d}_sheet_panel{i}.png")
        src_note.append(f"sheet_body_sfw panel {i}")
        made += 1
    return manifest(d, kind="body_scale", character=ch["id"], height_cm=ch["height_cm"], preset="identity_encode",
                    concept_type="identity", ref_resolution=BODY[0], aspect="3:4", tokens_per_image=TOK["body"],
                    mask="нет (нейтральный фон листа)", sources=src_note,
                    note=("Масштаб задаёт не мод, а плита + якоря карточки + промпт роста; мод держит пропорции "
                          "тела. В промпте: рост {h} см и якоря «X reaches her Y».").format(h=ch["height_cm"]))


# точки позы MediaPipe: кисть = запястье + мизинец + указательный + большой; стопа = щиколотка + пятка + носок
PART_PTS = {"hands": [(15, 17, 19, 21), (16, 18, 20, 22)], "feet": [(27, 29, 31), (28, 30, 32)]}
MIN_SIDE = 140          # меньше — после апскейла до 512 мыло, а не референс
EXCLUDE: set[str] = set()   # «часть/кадр_L|R» — отбраковка глазами (--exclude); детектор кистей на кропах
                            # отбрасывает и нормальные руки, лежащие на теле, поэтому финальный отбор ручной


def build_parts(src, out, ch, part: str, limit=20):
    """Кроп части по точкам позы (детектор кистей не видит мелкие руки в полном кадре). Источники: одобренные
    кадры v2 + листы персонажа. Сторона кропа ~ 3.2 × размах точек, но ≥ 0.9 длины предплечья/голени."""
    d = out / "characters" / ch["id"] / "parts" / part
    srcs = sorted((src / "frames/sfw_final_frames_v2").glob("*.png")) + \
        sorted((src / "character_sheets").glob("sheet_body*.png"))
    made, used, small, dup, nohand = 0, [], 0, 0, 0
    d.mkdir(parents=True, exist_ok=True)
    for f in srcs:
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        pts = pose_points(img)
        if pts is None:
            continue
        boxes_in_img = []
        for side_idx, ids in enumerate(PART_PTS[part]):
            p = pts[list(ids)]
            if made >= limit or (p[:, 2] < 0.5).any():      # часть не видна / закрыта
                continue
            limb = (13 + side_idx, 15 + side_idx) if part == "hands" else (25 + side_idx, 27 + side_idx)
            seg = float(np.linalg.norm(pts[limb[0], :2] - pts[limb[1], :2]))
            span = float(np.ptp(p[:, :2], axis=0).max())
            side = max(span * 3.2, seg * 0.9)
            if side < MIN_SIDE:
                small += 1
                continue
            c = p[:, :2].mean(axis=0)
            box = square_box(c[0], c[1], side, *img.size)
            if any(iou(box, b) > 0.4 for b in boxes_in_img):   # L и R в одном кропе — одна картинка
                dup += 1
                continue
            if f"{part}/{f.stem}_{'LR'[side_idx]}" in EXCLUDE:  # отбраковано глазами (файл исключений)
                nohand += 1
                continue
            crop = rgba(img, None, box, PART).convert("RGB")
            boxes_in_img.append(box)
            crop.save(d / f"{made:02d}_{f.stem}_{'LR'[side_idx]}.png")
            used.append(f.name)
            made += 1
    return manifest(d, kind="part", part=part, character=ch["id"], preset="identity_encode", concept_type="generic",
                    ref_resolution=PART[0], aspect="1:1", tokens_per_image=TOK["part"], mask="нет (плотный кроп)",
                    sources=sorted(set(used)), skipped_small=small, skipped_dup=dup, skipped_excluded=nohand,
                    use=f"SFW-аналог проблемной части ({part}): 5 / 10 / 20 — серия r3; отбраковать глазами")


def build_marks(out, ch, marks: Path | None):
    """Уникальные приметы: <marks>/<id>/<примета>/*.png (+ mark.json {desc_en, side, crop: [x0,y0,x1,y1]})."""
    base = out / "characters" / ch["id"] / "parts"
    res = {}
    if marks and (marks / ch["id"]).is_dir():
        for md in sorted(p for p in (marks / ch["id"]).iterdir() if p.is_dir()):
            meta = json.loads((md / "mark.json").read_text(encoding="utf-8")) if (md / "mark.json").exists() else {}
            d = base / f"mark_{md.name}"
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(sorted(md.glob("*.[pj][np]g"))):
                img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                c = meta.get("crop")
                box = tuple(c) if c else square_box(img.size[0] / 2, img.size[1] / 2, min(img.size), *img.size)
                rgba(img, None, box, PART).save(d / f"{i:02d}_{f.stem}.png")
            res[md.name] = manifest(d, kind="unique_mark", mark=md.name, character=ch["id"], desc_en=meta.get("desc_en"),
                                    side=meta.get("side"), preset="identity_encode", concept_type="identity",
                                    ref_resolution=PART[0], aspect="1:1", tokens_per_image=TOK["part"],
                                    use="мод приметы + её описание в промпте (desc_en); проверка — VLM «есть/нет/где»")
    readme = base / "README.txt"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "Части тела: одна подпапка = один мод.\n"
        "hands/ — собрано автоматически (SFW-аналог).\n"
        "mark_<имя>/ — уникальные приметы персонажа (родинка, тату, шрам...): 5–10 плотных кропов 1:1 с разных\n"
        "  ракурсов, одинаковый свет; положить в <marks>/<персонаж>/<имя>/ + mark.json и пересобрать.\n"
        "Остальные категории SS-470 заполняются вручную по тем же правилам (кроп 1:1 ≥ 512, 5/10/20 кадров).\n",
        encoding="utf-8")
    return res


def build_zones(src, out):
    """Зоны = карточки плит WS-03 (`_claude/ws03/plates_*.yaml`, файлы в `Locations/`): id `локация/зона`,
    описание и якоря известной высоты — масштаб персонажа задают они (в манифест), мод держит вид зоны."""
    made = {}
    for card in sorted((src / "_claude/ws03").glob("plates_*.yaml")):
        for c in yaml.safe_load(card.read_text(encoding="utf-8")) or []:
            f = src / "Locations" / c["file"]
            if not f.exists():
                made[c["id"]] = f"нет файла {c['file']}"
                continue
            d = out / "locations" / c["id"]
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            w, h = img.size
            im = ImageOps.contain(img, ZONE, Image.LANCZOS)
            ox, oy = (ZONE[0] - im.width) // 2, (ZONE[1] - im.height) // 2
            plate = Image.new("RGB", ZONE, img.getpixel((2, 2)))
            plate.paste(im, (ox, oy))
            # Не растягиваем мебель и не режем якоря: вписываем плиту, координаты карточки переводим в поля.
            tx = lambda x: (ox + x * im.width) / ZONE[0]
            ty = lambda y: (oy + y * im.height) / ZONE[1]
            anchors = [a[:2] + [tx(a[2]), ty(a[3]), tx(a[4]), ty(a[5])] + a[6:] for a in c.get("anchors", [])]
            slots = [a[:2] + [tx(a[2]), ty(a[3])] + a[4:] for a in c.get("slots", [])]
            d.mkdir(parents=True, exist_ok=True)
            plate.save(d / "00_plate.png")
            made[c["id"]] = manifest(d, kind="zone", plate_id=c["id"], zone=c.get("zone"), desc_en=c.get("desc"),
                                     anchors=anchors, slots=slots, horizon=ty(c["hor"]) if c.get("hor") is not None else None,
                                     preset="identity_encode", concept_type="background", ref_resolution=ZONE[1],
                                     aspect="16:9", tokens_per_image=TOK["zone"], sources=[c["file"]],
                                     use="серия r4: плита <Picture 1> vs + мод зоны; ракурсы 2–5 — дополнить")
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="C:/stacy-lora")
    ap.add_argument("--out", default="C:/stacy-lora/refmod_inputs/next")
    ap.add_argument("--marks", default="C:/stacy-lora/refmods_marks", help="исходники уникальных примет")
    ap.add_argument("--only", default="face,body,hands,feet,marks,zones")
    ap.add_argument("--exclude", default="C:/stacy-lora/refmod_inputs/exclude.txt",
                    help="строки «часть/кадр_L|R» — кропы, отбракованные глазами")
    a = ap.parse_args()
    src, out, only = Path(a.src), Path(a.out), set(a.only.split(","))
    if out.exists():
        ap.error("Выходная папка уже существует: выберите новую (--out); исходники и прошлые наборы не трогаем.")
    ex = Path(a.exclude)
    if ex.exists():
        EXCLUDE.update(l.split("#")[0].strip() for l in ex.read_text(encoding="utf-8").splitlines() if l.strip())
    report = {}
    for ch in characters(src):
        if ch["id"] != "stacy":
            raise ValueError("Для другого персонажа нужно явно задать его исходники; Стейси подставлять нельзя.")
        r = report.setdefault(ch["id"], {})
        if "face" in only:
            r["face"] = build_face(src, out, ch)
        if "body" in only:
            r["body_scale"] = build_body(src, out, ch)
        for part in ("hands", "feet"):
            if part in only:
                r[part] = build_parts(src, out, ch, part)
        if "marks" in only:
            r["marks"] = build_marks(out, ch, Path(a.marks))
    if "zones" in only:
        report["zones"] = build_zones(src, out)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
