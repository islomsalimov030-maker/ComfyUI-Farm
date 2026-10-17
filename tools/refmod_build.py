#!/usr/bin/env python3
"""SS-340: сборка папок RefMod на ПК — по папке на будущий мод: картинки одного аспекта (+ альфа-маска) + refmod.yaml.

    .venv/Scripts/python tools/refmod_build.py --sources <вне репо>/sources.yaml --out <новая папка>

Код не знает ни персонажей, ни перечня частей тела: всё задаёт sources.yaml (лежит вне репо, конкретику ведёт
человек). Схема — `tools/refmod_sources.example.yaml`. Пути в нём — относительно `src` (или абсолютные).

Раскладка (одна подпапка = один мод; H3RefModCreateFromFolder use_subfolders / ws07_graphs.refmod_extract_graph):
  characters/<id>/face/          1:1 768, лицо по центру, маска лицо+волосы (MediaPipe face → эллипс с запасом)
  characters/<id>/body_scale/    3:4 768x1024, ПОЛНЫЙ РОСТ вписан без растяжения и без кропа по телу
  characters/<id>/<категория>/   любые категории персонажа (части тела, приметы…):
                                   auto: pose_hands | pose_feet — кропы по точкам позы из указанных кадров;
                                   иначе — готовые картинки человека из `dir` (альфа сохраняется как маска)
  locations/<loc>/<zone>/        16:9 1344x768, плита вписана целиком; якоря и слоты карточки пересчитаны
Картинки готовятся здесь, чтобы холст мода (задаёт первая картинка) не обрезал остальные.
Медиа в репо не кладём. Маски: альфа PNG (255 = субъект); в графе извлечения — InvertMask.
Отбраковка глазами: файл exclude (строки «<персонаж>/<категория>/<имя исходника>[_L|_R]»).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageOps

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

FACE, BODY, PART, ZONE = (768, 768), (768, 1024), (512, 512), (1344, 768)
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
MIN_SIDE = 256              # кроп части меньше — апскейл до 512 больше чем в 2 раза: мыло, а не референс
# точки позы MediaPipe: кисть = запястье + мизинец + указательный + большой; стопа = щиколотка + пятка + носок;
# limb — сегмент конечности (предплечье / голень) для нижней границы размера кропа
AUTO = {"pose_hands": {"pts": [(15, 17, 19, 21), (16, 18, 20, 22)], "limb": [(13, 15), (14, 16)]},
        "pose_feet": {"pts": [(27, 29, 31), (28, 30, 32)], "limb": [(25, 27), (26, 28)]}}


def tokens(size) -> int:
    """Токены одной картинки мода (Full Reference): (H/32) * (W/32); мод = сумма по стопке."""
    return (size[0] // 32) * (size[1] // 32)


def check_size(size, where: str):
    """Размер картинки мода: стороны кратны 32 (иначе H3Mod округлит и токены разойдутся с манифестом),
    короткая ≥ 320 (минимум H3Mod) и ≤ 1024 (пресет identity_encode фиксирует 1024, крупнее — уменьшит сам)."""
    w, h = size
    if w % 32 or h % 32 or min(w, h) < 320 or min(w, h) > 1024:
        raise ValueError(f"{where}: size {list(size)} — стороны кратны 32, короткая 320…1024")


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


def fit_pad(img: Image.Image, size, fill=None) -> Image.Image:
    """Вписать без искажения и добить полями цвета фона (угол картинки); альфа (если есть) — тоже вписывается."""
    has_a = img.mode == "RGBA"
    im = ImageOps.contain(img, size, Image.LANCZOS)
    fill = fill or img.convert("RGB").getpixel((2, 2))
    out = Image.new("RGBA" if has_a else "RGB", size, (*fill, 0) if has_a else fill)
    out.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return out


def ellipse_mask(size, box, grow=0.25, blur=12):
    m = Image.new("L", size, 0)
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * grow, (y1 - y0) * grow
    ImageDraw.Draw(m).ellipse([x0 - dx, y0 - dy * 1.6, x1 + dx, y1 + dy * 0.6], fill=255)  # волосы сверху
    return m.filter(ImageFilter.GaussianBlur(blur))


# --------------------------------------------------------------------------------------- детекторы (MediaPipe, локально)
def _mp():
    from farm.qa import mp as M
    return M


def _bgr(img: Image.Image) -> np.ndarray:
    """farm.qa.mp._mp_image ждёт BGR (OpenCV) и сам переводит в RGB — подаём BGR, иначе R и B переставлены."""
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])


def face_box(img: Image.Image):
    M = _mp()
    a = _bgr(img)
    r = M._task("face_landmarker").detect(M._mp_image(a))
    if not r.face_landmarks:
        return None
    p = np.array([[q.x * a.shape[1], q.y * a.shape[0]] for q in r.face_landmarks[0]])
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def pose_points(img: Image.Image):
    """Самая крупная поза кадра (детектор ищет до 2: отражение в зеркале, второй человек) — [33, (x, y, vis)]."""
    M = _mp()
    a = _bgr(img)
    r = M._task("pose_landmarker").detect(M._mp_image(a))
    if not r.pose_landmarks:
        return None
    poses = [np.array([[q.x * a.shape[1], q.y * a.shape[0], q.visibility] for q in lm]) for lm in r.pose_landmarks]
    area = lambda p: float(np.prod(np.ptp(p[p[:, 2] > 0.3][:, :2], axis=0))) if (p[:, 2] > 0.3).sum() > 2 else 0.0  # noqa: E731
    return max(poses, key=area)


def full_body_ok(pts, size, margin=0.015, vis=0.5) -> bool:
    """Полный рост в кадре: нос, пятки и носки (0, 29–32) видимы и не ближе margin к краям."""
    if pts is None:
        return False
    w, h = size
    p = pts[[0, 29, 30, 31, 32]]
    return bool((p[:, 2] >= vis).all() and (p[:, 0] > margin * w).all() and (p[:, 0] < (1 - margin) * w).all()
                and (p[:, 1] > margin * h).all() and (p[:, 1] < (1 - margin) * h).all())


# --------------------------------------------------------------------------------------- конфиг и файлы
class Ctx:
    def __init__(self, src: Path, out: Path, exclude: set[str]):
        self.src, self.out, self.exclude = src, out, exclude

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else self.src / q

    def images(self, spec: dict) -> list[Path]:
        """dir | dirs + glob (по умолчанию все картинки) + skip_regex — по имени файла без расширения."""
        dirs = spec.get("dirs") or [spec["dir"]]
        skip = re.compile(spec["skip_regex"]) if spec.get("skip_regex") else None
        out = []
        for d in dirs:
            for f in sorted(self.path(d).glob(spec.get("glob", "*"))):
                if f.suffix.lower() in IMG_EXT and not (skip and skip.search(f.stem)):
                    out.append(f)
        return out

    def excluded(self, ch: str, cat: str, key: str) -> bool:
        return f"{ch}/{cat}/{key}" in self.exclude


def load_sources(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for cid, ch in (cfg.get("characters") or {}).items():
        if not re.fullmatch(r"[a-z0-9_-]+", cid):
            raise ValueError(f"id персонажа «{cid}»: только [a-z0-9_-] (как в справочнике WS-03)")
        for cat, spec in (ch.get("categories") or {}).items():
            if cat in ("face", "body_scale") or not re.fullmatch(r"[a-z0-9_-]+", cat):
                raise ValueError(f"{cid}: категория «{cat}» — [a-z0-9_-], не face/body_scale")
            if spec.get("auto") and spec["auto"] not in AUTO:
                raise ValueError(f"{cid}/{cat}: auto={spec['auto']}, есть {sorted(AUTO)}")
            if not spec.get("auto") and not spec.get("dir"):
                raise ValueError(f"{cid}/{cat}: нужен dir (готовые картинки) или auto")
    return cfg


def manifest(d: Path, size, **kw) -> int:
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("*.png")))
    kw.update(images=n, size=list(size), tokens_per_image=tokens(size), tokens_total=tokens(size) * n)
    (d / "refmod.yaml").write_text(yaml.safe_dump(kw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return n


# --------------------------------------------------------------------------------------- категории
def build_face(ctx: Ctx, cid: str, spec: dict) -> int:
    d = ctx.out / "characters" / cid / "face"
    picks, skipped = [], 0
    for f in ctx.images(spec):
        if len(picks) >= spec.get("limit", 10):
            break
        if ctx.excluded(cid, "face", f.stem):
            continue
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        b = face_box(img)
        if b is None:
            skipped += 1
            continue
        side = max(b[2] - b[0], b[3] - b[1]) * 1.9
        box = square_box((b[0] + b[2]) / 2, (b[1] + b[3]) / 2 - (b[3] - b[1]) * 0.1, side, *img.size)
        d.mkdir(parents=True, exist_ok=True)
        rgba(img, ellipse_mask(img.size, b), box, FACE).save(d / f"{len(picks):02d}_{f.stem}.png")
        picks.append(f.name)
    return manifest(d, FACE, kind="face", character=cid, preset="identity_encode", concept_type="identity",
                    mask="alpha: лицо+волосы", sources=picks, skipped_no_face=skipped,
                    use="1 / 5 / 10 первых — серия r1 (кривая числа кадров)")


def build_body(ctx: Ctx, cid: str, spec: dict, height_cm) -> int:
    """Полный рост: либо готовые кадры (`dir`), либо панели листа (`sheet` + `panels` — границы в пикселях
    исходника). Вписывается без растяжения, без маски и без кропа по позе (режет волосы, искажает пропорции)."""
    d = ctx.out / "characters" / cid / "body_scale"
    d.mkdir(parents=True, exist_ok=True)
    made, note = 0, []
    items: list[tuple[str, Image.Image]] = []
    if spec.get("sheet"):
        sheet = ImageOps.exif_transpose(Image.open(ctx.path(spec["sheet"]))).convert("RGB")
        k = sheet.width / spec.get("sheet_width", sheet.width)
        for i, (x0, x1) in enumerate(spec["panels"]):
            items.append((f"{Path(spec['sheet']).stem}_p{i}",
                          sheet.crop((round(x0 * k) + 3, 0, round(x1 * k) - 3, sheet.height))))
    for f in ctx.images(spec) if spec.get("dir") or spec.get("dirs") else []:
        items.append((f.stem, ImageOps.exif_transpose(Image.open(f)).convert("RGB")))
    rejected = []
    for key, im in items:
        if ctx.excluded(cid, "body_scale", key):
            continue
        if not full_body_ok(pose_points(im), im.size, margin=spec.get("margin", 0.015)):   # обрезан / не человек
            rejected.append(key)
            continue
        fit_pad(im, BODY).save(d / f"{made:02d}_{key}.png")
        note.append(key)
        made += 1
    return manifest(d, BODY, kind="body_scale", character=cid, height_cm=height_cm, preset="identity_encode",
                    concept_type="identity", mask="нет", sources=note, rejected_not_full_body=rejected,
                    note="Масштаб задают плита + якоря карточки + рост в промпте; мод держит пропорции тела.")


def build_auto(ctx: Ctx, cid: str, cat: str, spec: dict) -> int:
    """Кропы части по точкам позы (детектор кистей не видит мелкие руки в полном кадре).
    Сторона ~ 3.2 × размах точек, но ≥ 0.9 сегмента конечности; L и R в одном кропе — одна картинка."""
    a = AUTO[spec["auto"]]
    d = ctx.out / "characters" / cid / cat
    d.mkdir(parents=True, exist_ok=True)
    made, used, small, dup, excl = 0, [], 0, 0, 0
    for f in ctx.images(spec):
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        pts = pose_points(img)
        if pts is None:
            continue
        boxes = []
        for side_idx, ids in enumerate(a["pts"]):
            p = pts[list(ids)]
            if made >= spec.get("limit", 20) or (p[:, 2] < 0.5).any():     # часть не видна / закрыта
                continue
            l0, l1 = a["limb"][side_idx]
            side = max(float(np.ptp(p[:, :2], axis=0).max()) * 3.2,
                       float(np.linalg.norm(pts[l0, :2] - pts[l1, :2])) * 0.9)
            if side < MIN_SIDE:
                small += 1
                continue
            c = p[:, :2].mean(axis=0)
            box = square_box(c[0], c[1], side, *img.size)
            if any(iou(box, b) > 0.4 for b in boxes):
                dup += 1
                continue
            key = f"{f.stem}_{'LR'[side_idx]}"
            if ctx.excluded(cid, cat, key):
                excl += 1
                continue
            boxes.append(box)
            # маска части: эллипс по точкам части + конец сегмента конечности (без лица/одежды/фона кропа)
            q = np.vstack([p[:, :2], pts[[l1], :2]])
            pad = max(float(np.ptp(q, axis=0).max()) * 0.35, side * 0.08)
            m = Image.new("L", img.size, 0)
            ImageDraw.Draw(m).ellipse([q[:, 0].min() - pad, q[:, 1].min() - pad, q[:, 0].max() + pad,
                                       q[:, 1].max() + pad], fill=255)
            m = m.filter(ImageFilter.GaussianBlur(max(2, side * 0.03)))
            rgba(img, m, box, PART).save(d / f"{made:02d}_{key}.png")
            used.append(f.name)
            made += 1
    return manifest(d, PART, kind="part", category=cat, character=cid, auto=spec["auto"], preset="identity_encode",
                    concept_type="generic", desc_en=spec.get("desc_en"), mask="alpha: эллипс по точкам части",
                    sources=sorted(set(used)), skipped_small=small, skipped_dup=dup, skipped_excluded=excl,
                    use=spec.get("use", "5 / 10 / 20 кадров — кривая числа кадров; отбраковать глазами"))


def build_manual(ctx: Ctx, cid: str, cat: str, spec: dict) -> int:
    """Готовые картинки человека (части тела, приметы, что угодно): вписываются без кропа (fit) или
    центр-кропом (fit: crop); альфа-канал картинки сохраняется как маска мода."""
    size = tuple(spec.get("size", PART))
    check_size(size, f"{cid}/{cat}")
    d = ctx.out / "characters" / cid / cat
    d.mkdir(parents=True, exist_ok=True)
    used, alpha = [], []
    for f in ctx.images(spec):
        if ctx.excluded(cid, cat, f.stem):
            continue
        img = ImageOps.exif_transpose(Image.open(f))
        has_a = "A" in img.getbands() or (img.mode == "P" and "transparency" in img.info)   # палитра + tRNS
        img = img.convert("RGBA") if has_a else img.convert("RGB")
        if spec.get("fit", "pad") == "crop":
            img = ImageOps.fit(img, size, Image.LANCZOS)
        else:
            img = fit_pad(img, size)
        img.save(d / f"{len(used):02d}_{f.stem}.png")
        used.append(f.name)
        alpha.append(has_a)
    if any(alpha) and not all(alpha):
        raise ValueError(f"{cid}/{cat}: маска (альфа) есть не у всех картинок — "
                         f"без альфы: {[u for u, a_ in zip(used, alpha) if not a_]}")
    return manifest(d, size, kind="manual", category=cat, character=cid, preset=spec.get("preset", "identity_encode"),
                    concept_type=spec.get("concept_type", "identity"), desc_en=spec.get("desc_en"),
                    placement=spec.get("placement"), mask="alpha" if alpha and all(alpha) else "нет", sources=used,
                    use=spec.get("use", "мод категории + desc_en в промпте; проверка — есть / нет / где"))


def build_zones(ctx: Ctx, spec: dict) -> dict:
    """Зоны = карточки плит WS-03: id `локация/зона`, описание, якоря известной высоты и слоты — масштаб персонажа
    задают они (переносятся в манифест под вписанную плиту), мод держит вид зоны."""
    made = {}
    for pattern in spec.get("cards", []):
        for card in sorted(ctx.src.glob(pattern)) if not Path(pattern).is_absolute() else [Path(pattern)]:
            for c in yaml.safe_load(card.read_text(encoding="utf-8")) or []:
                f = ctx.path(spec["images_dir"]) / c["file"]
                if not f.exists():
                    made[c["id"]] = f"нет файла {c['file']}"
                    continue
                img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                # Мод зоны — одна картинка: общий холст не нужен. Только уменьшение до ZONE (мелкие плиты не
                # растягиваем) и центр-подрезка до кратного 32 (сетка латента); без полей — поля стали бы
                # содержимым мода.
                im = img.copy()
                im.thumbnail(ZONE, Image.LANCZOS)
                cw, ch = im.width // 32 * 32, im.height // 32 * 32
                if min(cw, ch) < 320:
                    made[c["id"]] = f"плита {img.size} слишком мала (меньше 320 px после подрезки)"
                    continue
                ox, oy = (im.width - cw) // 2, (im.height - ch) // 2
                plate = im.crop((ox, oy, ox + cw, oy + ch))
                tx = lambda x: (x * im.width - ox) / cw       # noqa: E731
                ty = lambda y: (y * im.height - oy) / ch      # noqa: E731
                anchors = [a[:2] + [tx(a[2]), ty(a[3]), tx(a[4]), ty(a[5])] + a[6:] for a in c.get("anchors", [])]
                slots = [s[:2] + [tx(s[2]), ty(s[3])] + s[4:] for s in c.get("slots", [])]
                d = ctx.out / "locations" / c["id"]
                d.mkdir(parents=True, exist_ok=True)
                plate.save(d / "00_plate.png")
                made[c["id"]] = manifest(d, (cw, ch), kind="zone", plate_id=c["id"], zone=c.get("zone"),
                                         source_size=list(img.size), small_source=img.width < ZONE[0],
                                         desc_en=c.get("desc"), anchors=anchors, slots=slots,
                                         horizon=ty(c["hor"]) if c.get("hor") is not None else None,
                                         preset="identity_encode", concept_type="background", sources=[c["file"]],
                                         use="серия r4: плита <Picture 1> vs + мод зоны; ракурсы 2–5 — дополнить")
    return made


def build(cfg: dict, ctx: Ctx, only: set[str] | None = None) -> dict:
    want = lambda k: not only or k in only      # noqa: E731
    report: dict = {}
    for cid, ch in (cfg.get("characters") or {}).items():
        r = report.setdefault(cid, {})
        if ch.get("face") and want("face"):
            r["face"] = build_face(ctx, cid, ch["face"])
        if ch.get("body") and want("body"):
            r["body_scale"] = build_body(ctx, cid, ch["body"], ch.get("height_cm"))
        for cat, spec in (ch.get("categories") or {}).items():
            if want(cat) or want("categories"):
                r[cat] = build_auto(ctx, cid, cat, spec) if spec.get("auto") else build_manual(ctx, cid, cat, spec)
    if cfg.get("locations") and want("zones"):
        report["zones"] = build_zones(ctx, cfg["locations"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help="sources.yaml вне репо (схема — refmod_sources.example.yaml)")
    ap.add_argument("--out", required=True, help="новая папка (существующую не трогаем)")
    ap.add_argument("--exclude", default=None, help="по умолчанию — exclude.txt рядом с sources.yaml")
    ap.add_argument("--only", default="", help="face,body,zones,categories или имена категорий")
    a = ap.parse_args()
    src_cfg = Path(a.sources)
    cfg = load_sources(src_cfg)
    out = Path(a.out)
    if out.exists():
        ap.error("Выходная папка уже существует: выберите новую (--out); прошлые наборы не трогаем.")
    ex = Path(a.exclude) if a.exclude else src_cfg.with_name("exclude.txt")
    exclude = set()
    if ex.exists():
        exclude = {l.split("#")[0].strip() for l in ex.read_text(encoding="utf-8").splitlines()} - {""}
    ctx = Ctx(Path(cfg.get("src", src_cfg.parent)), out, exclude)
    report = build(cfg, ctx, set(filter(None, a.only.split(","))) or None)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
