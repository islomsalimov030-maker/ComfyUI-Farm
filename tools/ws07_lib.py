"""WS-07: общее для серий — поза MediaPipe (33 точки), рамка тела, скелет, зелёная рамка, широкая плита,
варианты промптов кадра (серия «масштаб») и перенумерация меток <Picture N> для узлов с Autogrow-слотами.

Рамка тела считается одинаково для эталона (одобренный кадр v2) и для результата — масштаб сравнивается
как отношение диагоналей рамок в долях кадра (плита и кадры v2 — 16:9, координаты нормированные).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------------------- поза и рамка тела
# пары точек MediaPipe Pose (33) для скелета; цвета — как у OpenPose (по конечностям)
LIMBS = [((11, 12), (255, 0, 0)), ((11, 13), (255, 170, 0)), ((13, 15), (255, 255, 0)),
         ((12, 14), (170, 255, 0)), ((14, 16), (85, 255, 0)), ((11, 23), (0, 255, 85)),
         ((12, 24), (0, 255, 170)), ((23, 24), (0, 255, 255)), ((23, 25), (0, 170, 255)),
         ((25, 27), (0, 85, 255)), ((24, 26), (0, 0, 255)), ((26, 28), (85, 0, 255)),
         ((27, 31), (170, 0, 255)), ((28, 32), (255, 0, 255))]
HEAD = (0, 2, 5, 7, 8)          # нос, глаза, уши
MIN_VIS = 0.3


def pose33(img_bgr: np.ndarray, long_side: int = 1280) -> Optional[np.ndarray]:
    """Самая крупная поза: массив (33, 3) — x, y в долях кадра, visibility. None — человека нет."""
    from farm.qa import mp as M
    small, _s = M._resize(img_bgr, long_side)
    res = M._task("pose_landmarker").detect(M._mp_image(small))
    best, area = None, -1.0
    for lms in res.pose_landmarks or []:
        a = np.array([(p.x, p.y, float(p.visibility or 0)) for p in lms], dtype=np.float32)
        ar = float(np.ptp(a[:, 0]) * np.ptp(a[:, 1]))
        if ar > area:
            best, area = a, ar
    return best


def head_radius(p: np.ndarray, aspect: float) -> float:
    """Радиус головы в долях высоты кадра: 1.6 × расстояние нос–середина ушей (минимум 0.02)."""
    ears = [p[i] for i in (7, 8) if p[i, 2] >= MIN_VIS]
    if not ears or p[0, 2] < MIN_VIS:
        return 0.03
    mid = np.mean([e[:2] for e in ears], 0)
    d = math.hypot((p[0, 0] - mid[0]) * aspect, p[0, 1] - mid[1])
    return max(0.02, 1.6 * d)


def body_box(p: np.ndarray, aspect: float = 16 / 9) -> Optional[list[float]]:
    """Рамка тела [x0, y0, x1, y1] в долях кадра: видимые точки + круг головы вокруг носа."""
    vis = p[p[:, 2] >= MIN_VIS]
    if len(vis) < 6:
        return None
    x0, y0 = float(vis[:, 0].min()), float(vis[:, 1].min())
    x1, y1 = float(vis[:, 0].max()), float(vis[:, 1].max())
    if p[0, 2] >= MIN_VIS:
        r = head_radius(p, aspect)
        x0, x1 = min(x0, p[0, 0] - r / aspect), max(x1, p[0, 0] + r / aspect)
        y0, y1 = min(y0, p[0, 1] - r), max(y1, p[0, 1] + r)
    return [max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)]


def box_diag(b: list[float], aspect: float = 16 / 9) -> float:
    """Диагональ рамки в долях высоты кадра."""
    return math.hypot((b[2] - b[0]) * aspect, b[3] - b[1])


def scale_metrics(box: Optional[list[float]], target: Optional[list[float]], aspect: float = 16 / 9) -> dict:
    """ratio = диагональ результата / диагональ эталона (>1 — персонаж крупнее), shift — сдвиг центра
    в долях диагонали эталона, err = |ln ratio|."""
    if not box or not target:
        return {"scale_ratio": None, "scale_err": None, "pos_shift": None, "clipped": None}
    r = box_diag(box, aspect) / max(box_diag(target, aspect), 1e-6)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    tx, ty = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
    sh = math.hypot((cx - tx) * aspect, cy - ty) / max(box_diag(target, aspect), 1e-6)
    # фигура упирается в край кадра, а у эталона — нет: рамка обрезана, ratio занижен
    clipped = any(abs(v - e) < 0.01 and abs(t - e) >= 0.01 for v, t, e in zip(box, target, (0, 0, 1, 1)))
    return {"scale_ratio": round(r, 3), "scale_err": round(abs(math.log(r)), 3), "pos_shift": round(sh, 3),
            "clipped": clipped}


# --------------------------------------------------------------------------------------- картинки входов
def draw_skeleton(p: np.ndarray, size: tuple[int, int]):
    """Скелет нужного размера на чёрном холсте размера плиты (PIL.Image)."""
    from PIL import Image, ImageDraw
    w, h = size
    im = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(im)
    t = max(3, int(0.008 * h))
    for (a, b), col in LIMBS:
        if p[a, 2] >= MIN_VIS and p[b, 2] >= MIN_VIS:
            d.line([(p[a, 0] * w, p[a, 1] * h), (p[b, 0] * w, p[b, 1] * h)], fill=col, width=t)
    if all(p[i, 2] >= MIN_VIS for i in (0, 11, 12)):
        neck = ((p[11, 0] + p[12, 0]) / 2 * w, (p[11, 1] + p[12, 1]) / 2 * h)
        d.line([neck, (p[0, 0] * w, p[0, 1] * h)], fill=(0, 0, 255), width=t)
    if p[0, 2] >= MIN_VIS:
        r = head_radius(p, w / h) * h
        d.ellipse([p[0, 0] * w - r, p[0, 1] * h - r, p[0, 0] * w + r, p[0, 1] * h + r], outline=(255, 0, 85), width=t)
    for i in range(33):
        if p[i, 2] >= MIN_VIS:
            d.ellipse([p[i, 0] * w - t, p[i, 1] * h - t, p[i, 0] * w + t, p[i, 1] * h + t], fill=(255, 255, 255))
    return im


def draw_green_box(plate, box: list[float], pad: float = 0.06):
    """Копия плиты с тонкой зелёной рамкой вокруг места персонажа (рамка тела + pad с каждой стороны)."""
    from PIL import ImageDraw
    im = plate.convert("RGB").copy()
    w, h = im.size
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0, y0 = max(0, (box[0] - pad * bw) * w), max(0, (box[1] - pad * bh) * h)
    x1, y1 = min(w - 1, (box[2] + pad * bw) * w), min(h - 1, (box[3] + pad * bh) * h)
    ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=max(3, int(0.004 * w)))
    return im


def widen_plate(plate, pad: float = 0.2):
    """Плита в центре холста (1 + 2·pad)×: поля — растянутые и размытые края плиты. → (картинка, кроп в долях)."""
    from PIL import Image, ImageFilter
    im = plate.convert("RGB")
    w, h = im.size
    W, H = int(round(w * (1 + 2 * pad))), int(round(h * (1 + 2 * pad)))
    bg = im.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=max(8, w // 60)))
    ox, oy = (W - w) // 2, (H - h) // 2
    bg.paste(im, (ox, oy))
    return bg, [ox / W, oy / H, (ox + w) / W, (oy + h) / H]


# --------------------------------------------------------------------------------------- промпты
POSE_DEF_RE = re.compile(r"^<Subject 3> is the body pose of the person in <Picture 4>.*$", re.M)
PIC1_DEF = "<Picture 1> is the empty room that the image is built on: its layout, camera position, framing and light are kept exactly."
SKELETON_DEF = ("<Subject 3> is the body pose, the size and the position given by the stick-figure skeleton in "
                "<Picture 4>; it shows exactly where she is in the frame and how tall she is there, and the skeleton "
                "itself is never drawn in the photograph.")
SKELETON_RULE = "Her body matches the skeleton in <Picture 4> in pose, size and position."
GREEN_RULE = ("She is placed inside the thin bright green rectangle marked on <Picture 1>, her body filling it from "
              "top to bottom; the green rectangle is only a guide and never appears in the photograph.")
WIDE_RULE = ("The blurred border around <Picture 1> is only padding: the room simply continues there with the "
             "same walls, floor and light.")
ID_SUBJECT = ("<Subject 1> is {trig}, the young woman whose face is shown in the head-and-shoulders portrait <Picture 2>, "
              "with exactly that face, hair and skin tone, an adult woman of average height and slim build")


def _insert_rule(text: str, rule: str) -> str:
    return text.replace(" Her face is sharp,", f" {rule} Her face is sharp,", 1)


def prompt_variant(base: str, variant: str) -> str:
    """Промпт кадра из эталонного (golden WS-04, шаблон frame_ref_v1) для варианта серии «масштаб»."""
    t = base.split("\n# inputs:")[0].rstrip() + "\n"
    if variant in ("fixe", "base"):
        return t
    if variant == "skeleton":
        if POSE_DEF_RE.search(t):
            t = POSE_DEF_RE.sub(SKELETON_DEF, t)
            t = t.replace("<Subject 3>: attribute_transfer - only the body pose is transferred to <Subject 1>.",
                          "<Subject 3>: attribute_transfer - only the body pose, size and position are transferred "
                          "to <Subject 1>.")
        else:
            t = t.replace(PIC1_DEF, SKELETON_DEF + "\n" + PIC1_DEF, 1)
            t = re.sub(r"(in <Subject 2>)(, placed into <Picture 1>)", r"\1, in the body pose of <Subject 3>\2", t, 1)
            t = t.replace("<Picture 1>: fully_preserved", "<Subject 3>: attribute_transfer - only the body pose, size "
                          "and position are transferred to <Subject 1>.\n<Picture 1>: fully_preserved", 1)
        return _insert_rule(t, SKELETON_RULE)
    if variant == "green":
        t = t.replace(PIC1_DEF, PIC1_DEF[:-1] + "; a thin bright green rectangle drawn on it only marks where she "
                      "is and how tall she is.", 1)
        return _insert_rule(t, GREEN_RULE)
    if variant == "wide":
        return _insert_rule(t, WIDE_RULE)
    if variant == "idcrop":
        m = re.search(r"^<Subject 1> is (\S+), the young woman in <Picture 2>, with exactly the face, hair, skin tone "
                      r"and body proportions shown there", t, re.M)
        if m:
            t = t.replace(m.group(0), ID_SUBJECT.format(trig=m.group(1)), 1)
        return t.replace("her face, hair, skin tone and body proportions", "her face, hair and skin tone", 1) \
                .replace("her face, hair, skin tone, body proportions and outfit", "her face, hair, skin tone and outfit", 1)
    raise ValueError(f"неизвестный вариант промпта: {variant}")


LABEL_RE = re.compile(r"<Picture (\d)>")


def renumber_pictures(text: str, present: list[int]) -> str:
    """Для Autogrow-слотов (ref_image_0..): метки идут подряд. present — номера картинок по слотам
    исходного движка (1 плита, 2 лист, 3 одежда, 4 поза), в порядке подачи."""
    m = {old: new for new, old in enumerate(sorted(present), 1)}
    missing = sorted({int(x) for x in LABEL_RE.findall(text)} - set(m))
    if missing:
        raise ValueError(f"в промпте <Picture {missing}>, а входа нет")
    return LABEL_RE.sub(lambda mo: f"<Picture {m[int(mo.group(1))]}>", text)


def gen_size(megapixels: float = 2.0, aspect: float = 16 / 9, multiple: int = 32) -> tuple[int, int]:
    w = math.sqrt(megapixels * 1e6 * aspect)
    return int(round(w / multiple) * multiple), int(round(w / aspect / multiple) * multiple)


def read_bgr(path: str | Path) -> np.ndarray:
    import cv2
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img
