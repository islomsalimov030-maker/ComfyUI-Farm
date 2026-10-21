#!/usr/bin/env python3
"""SS-456, двухпроходная схема роста (идея пользователя 20.10): быстрый черновик → поза из черновика →
силуэт той же позы нужного роста → финальная генерация по силуэту. Всё автоматически; выключается —
просто не запускать финальный конфиг (в ферме — флаг профиля).

    python tools/ss456_twopass.py draft <round_456.yaml> <round_456d.yaml> [--only s1,s2]  # черновики 1344×768
    (python tools/ss340_round.py run <round_456d.yaml>)
    python tools/ss456_twopass.py build <round_456.yaml> <round_456d.yaml[,round_h3d.yaml]> <round_456f.yaml>
    (python tools/ss340_round.py run <round_456f.yaml>)
    python tools/ss456_twopass.py eval <round_456d.yaml[,more] > <round_456f.yaml>

Посадка (ревизия пользователя 21.10): место персонажа — из ЧЕРНОВИКА, куда его ставит генератор по
смыслу сцены; посадка по карточным слотам («на глаз») была ошибкой. Гейт совпадения сцены (ORB+RANSAC
черновик↔плита) гарантирует кадрирование черновика = плите, поэтому якорь (низ маски / таз сидя)
переносится ТОЖДЕСТВЕННО — та же доля кадра. Слот карточки — только отчёт-отклонение.
Масштаб ×k — на глубине ЕЁ опоры (не слота), классы поз (target.json cls):
- стоя/присед — expected_height(карточка, стопы черновика, ЦЕЛЬ top_cm) / рамка черновика:
  у стоя цель = рост 230, у приседа — высота приседа (~133);
- сидя — пол под её тазом (проекция через горизонт), торс до цели top_cm, без кость-метрики;
- лежа — поверхность под её нижней точкой (низ маски); k = длина цели / длина нос→щиколотка.
Мебель карточки перед ПЕРСОНАЖЕМ (по его позиции, не по слоту; низ ниже опоры, < 140 см)
возвращается поверх силуэта.

Автоприёмка (решение пользователя 21.10): допуск ±15 % от цели (target.json top_cm) — achieved()
пересчитывает кадр в см по карточке: стоя/присед высота рамки на глубине стоп, сидя макушка над полом
через таз на сиденье, лежа длина тела. Ретрай: build --poses <не прошедшие> --boost 1.05–1.15.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import ss340_round as R  # noqa: E402
import ws07_lib as L  # noqa: E402
from ss456_height import sil_prompt  # noqa: E402
from ss340_scale_kit import draw_silhouette, transform  # noqa: E402
from farm.qa.image.scale import expected_height  # noqa: E402

H_CM = 230.0
DRAFT_CANVAS = [1344, 768]
TOL_FRAC = 0.15                     # допуск автоприёмки: ±15 % от цели (решение пользователя 21.10)
# гейт совпадения сцены: совпавшие черновики дают ~900 inliers из ~1100 при dx, dy < 0.003;
# перестроенная H3 комната — 3–4 inliers из ~20 (замер 21.10, рама C4)
SHIFT_MIN = dict(inliers=30, ratio=0.4, dxy=0.03, scale=0.04)


def scenes_of(cfg: dict) -> list[str]:
    """Сцены базового конфига: ключи <сцена>_plate, для которых есть вариант plate_<сцена>."""
    vs = {v["id"] for v in cfg.get("variants", [])}
    return sorted(k[:-6] for k in cfg.get("kit_poses", {}) if k.endswith("_plate") and k[:-6] in
                  {v[6:] for v in vs if v.startswith("plate_")})


def kind_of(cls: str) -> str:
    if "сидя" in cls:
        return "сидя"
    if "лежа" in cls:
        return "лежа"
    if "корточк" in cls:
        return "присед"
    return "стоя"


def load_drafts(parts: str) -> tuple[Path, dict]:
    """Конфиги черновиков через запятую → (root, job_id → job). Ретраи добавляют новые сиды в свой раунд."""
    root, dj = None, {}
    for part in parts.split(","):
        dcfg = yaml.safe_load(Path(part.strip()).read_text(encoding="utf-8"))
        root = Path(dcfg["src"])
        drun = Path(dcfg["out"]) / dcfg["round"]
        for line in (drun / "jobs.jsonl").read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            dj[j["id"]] = j
    if root is None or not dj:
        raise RuntimeError("нет заданий черновиков")
    return root, dj


def plate_variant(cfg: dict, scene: str) -> dict:
    return next(v for v in cfg["variants"] if v["id"] == f"plate_{scene}")


def cmd_draft(a) -> int:
    cfg = yaml.safe_load(Path(a.src).read_text(encoding="utf-8"))
    out = copy.deepcopy(cfg)
    out.update(round=Path(a.dst).stem, kit_poses={}, poses=[], seeds={}, variants=[], est_frame_s=40)
    only = set(a.only.split(",")) if a.only else None
    for scene in scenes_of(cfg):
        if only and scene not in only:
            continue
        old = cfg["kit_poses"][f"{scene}_plate"]
        pose = f"{scene}_draft"
        out["kit_poses"][pose] = dict(old, canvas=DRAFT_CANVAS)
        seeds = list(cfg["seeds"][f"{scene}_plate"])
        out["poses"].append(pose)
        out["seeds"][pose] = seeds
        v = copy.deepcopy(plate_variant(cfg, scene))
        v.update(id=f"draft_{scene}", poses=[pose], seeds={pose: seeds})
        out["variants"].append(v)
    Path(a.dst).write_text("# SS-456: быстрые черновики (поза для силуэта), 1344×768, turbo 8\n"
                           + yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(a.dst, "кадров", sum(len(s) for s in out["seeds"].values()))
    return 0


def scene_shift(draft_png: Path, plate: Image.Image) -> dict:
    """Совпадение сцены черновика с чистой плитой: ORB + RANSAC (partial affine), плита приводится
    к размеру черновика. ok = достаточно inliers и трансляция/масштаб в допуске — только такой
    черновик можно сажать по карточке плиты. → {matches, inliers, dx, dy, scale, ok}."""
    import cv2
    d = cv2.imdecode(np.fromfile(str(draft_png), np.uint8), cv2.IMREAD_GRAYSCALE)
    p = cv2.cvtColor(np.asarray(plate.convert("RGB").resize((d.shape[1], d.shape[0]))), cv2.COLOR_RGB2GRAY)
    kp_d, des_d = cv2.ORB_create(3000).detectAndCompute(d, None)
    kp_p, des_p = cv2.ORB_create(3000).detectAndCompute(p, None)
    good = []
    if des_d is not None and des_p is not None:
        for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_p, des_d, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
    r = {"matches": len(good), "inliers": 0, "dx": None, "dy": None, "scale": None, "ok": False}
    if len(good) >= 8:
        src = np.float32([kp_p[g.queryIdx].pt for g in good])
        dst = np.float32([kp_d[g.trainIdx].pt for g in good])
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4, maxIters=5000)
        if A is not None:
            r.update(inliers=int(inl.sum()), dx=round(float(A[0, 2] / d.shape[1]), 4),
                     dy=round(float(A[1, 2] / d.shape[0]), 4), scale=round(float(np.hypot(A[0, 0], A[1, 0])), 4))
            r["ok"] = (r["inliers"] >= SHIFT_MIN["inliers"]
                       and r["inliers"] / r["matches"] >= SHIFT_MIN["ratio"]
                       and abs(r["dx"]) <= SHIFT_MIN["dxy"] and abs(r["dy"]) <= SHIFT_MIN["dxy"]
                       and abs(r["scale"] - 1) <= SHIFT_MIN["scale"])
    return r


def mask_bottom(mb: np.ndarray) -> tuple[float, float]:
    """Низ тела по МАСКЕ, доли кадра: последний ряд шире четверти самой широкой строки (нити волос/тени
    отбрасываются), x — центр нижней полосы 2 % высоты (обе стопы, не одна)."""
    h, w = mb.shape
    wide = np.where(mb.sum(1) >= max(mb.sum(1).max() * 0.25, 4))[0]
    if not wide.size:
        raise RuntimeError("маска пуста")
    y = int(wide[-1])
    xs = np.where(mb[max(0, y - round(0.02 * h)):y + 1].any(0))[0]
    return float((xs[0] + xs[-1]) / 2) / w, float(y) / h


def seat_point(card: dict, slot: tuple[float, float]) -> tuple[tuple[float, float], float]:
    """Поверхность сиденья над слотом: (x слота, y пола слота − высота сиденья). Высота — якорь 'seat'
    карточки, иначе 45 см (типичный диван). → (точка, см)."""
    cm = next((float(a["top_cm"]) for a in card.get("anchors", []) if "seat" in str(a.get("name", ""))), 45.0)
    s = (slot[1] - float(card["horizon_y"])) / float(card["camera_height_cm"])
    return (slot[0], slot[1] - cm * s), cm


def body_length(p: np.ndarray, aspect: float) -> float | None:
    """Длина тела по позе (доли высоты кадра): нос → самая дальняя щиколотка. Поднятые руки исключены —
    максимум по всем точкам их считал и ломал метрику (рука с батончиком над головой)."""
    vis = lambda i: p[i, 2] >= L.MIN_VIS  # noqa: E731
    if not vis(0):
        return None
    ankl = [i for i in (27, 28, 31, 32) if vis(i)]
    if not ankl:
        return None
    nx, ny = float(p[0, 0]) * aspect, float(p[0, 1])
    return max(math.hypot(float(p[i, 0]) * aspect - nx, float(p[i, 1]) - ny) for i in ankl)


def head_top(p: np.ndarray, aspect: float) -> float | None:
    """Верх головы (доля высоты кадра): нос − радиус головы (уши→нос). Поднятые руки не участвуют."""
    if p[0, 2] < L.MIN_VIS:
        return None
    return float(p[0, 1]) - L.head_radius(p, aspect)


def mask_span(mb: np.ndarray) -> float:
    """Протяжённость тела по МАСКЕ вдоль главной оси (доли высоты кадра): фолбэк длины лежа,
    когда поза не даёт щиколоток (L2 «скрестив ноги» — колени/лодыжки перекрыты)."""
    ys, xs = np.where(mb)
    pts = np.stack([xs / mb.shape[0], ys / mb.shape[0]], 1)     # сразу в долях ВЫСОТЫ (x/w·aspect)
    c = pts - pts.mean(0)
    _, _, v = np.linalg.svd(c, full_matrices=False)
    proj = c @ v[0]                                        # проекции на главную ось тела
    return float(proj.max() - proj.min())


def plan_anchor(kind: str, mb: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Якорь посадки в ДОЛЯХ кадра черновика: сидя — таз (неподвижен при ×k), иначе низ МАСКИ
    (контакт с опорой). Гейт сцены гарантирует кадрирование черновика = плите, поэтому якорь
    переносится в плиту тождественно; слот карточки местом больше не является."""
    if kind == "сидя":
        return (float((p[23, 0] + p[24, 0]) / 2), float((p[23, 1] + p[24, 1]) / 2))
    return mask_bottom(mb)


def plan_k(p: np.ndarray, box, mb: np.ndarray, card: dict, kind: str, tgt: dict,
           anchor: tuple[float, float], seat_cm: float, aspect: float) -> float:
    """Масштаб ×k на глубине ЕЁ опоры, не слота. стоя/присед — expected_height по стопам ЧЕРНОВИКА
    с ЦЕЛЬЮ top_cm (рост стоя / высота приседа) / рамка черновика; сидя — пол под её тазом
    (проекция через горизонт), торс до цели top_cm, кость-метрика не используется; лежа —
    поверхность под низом её маски, k = длина цели (top_cm) / длина тела нос→щиколотка
    (фолбэк — протяжённость маски)."""
    hor, cam = float(card["horizon_y"]), float(card["camera_height_cm"])
    if kind == "сидя":
        top = head_top(p, aspect)
        if top is None and box is None:
            raise RuntimeError("сидя: голова не найдена для масштаба")
        r = seat_cm / cam
        if r >= 1 or anchor[1] <= hor:                     # пол под тазом через горизонт
            raise RuntimeError("сидя: пол под тазом не определяется")
        ground = (anchor[1] - hor * r) / (1 - r)
        if ground <= hor + 0.02:
            raise RuntimeError("сидя: пол под тазом за горизонтом")
        s = (ground - hor) / cam                           # px/см на глубине ПОЛА под её тазом
        k = float(tgt["top_cm"] - seat_cm) * s / (anchor[1] - (top if top is not None else box[1]))
    elif kind == "лежа":
        length = body_length(p, aspect) or mask_span(mb)   # щиколотки перекрыты — длина по маске
        if not length:
            raise RuntimeError("лежа: длина тела не измерена")
        s = (anchor[1] - hor) / cam                        # поверхность под её нижней точкой
        k = float(tgt["top_cm"]) * s / length
    else:
        if box is None:
            raise RuntimeError("стоя: нет рамки тела")
        feet = [p[i, 1] for i in (27, 28, 31, 32) if p[i, 2] >= L.MIN_VIS]
        # цель ПОЗЫ (top_cm): у стоя это рост 230, у приседа — высота приседа (~133); делить полный
        # рост на рамку приседа нельзя — перекос ×230/133 (C4/C5 «267 из 133» в наборе 21.10)
        eh = expected_height(card, float(np.mean(feet)), float(tgt["top_cm"])) if feet else None
        if not eh:
            raise RuntimeError("стоя: карточка не даёт рост на глубине стоп черновика")
        k = eh / (box[3] - box[1])
    return float(k)


def clip_k(k: float, anchor_y: float, dst_y: float, box_top: float) -> tuple[float, float, bool]:
    """Верх головы в кадре плиты: dst_y − k·(anchor_y − box_top). Если < 0.02 — урезать k ровно до
    верха 0.02 (голова не выходит за кадр). → (k, top_y, clipped)."""
    top = dst_y - k * (anchor_y - box_top)
    if top >= 0.02:
        return k, top, False
    k2 = (dst_y - 0.02) / (anchor_y - box_top)
    return k2, dst_y - k2 * (anchor_y - box_top), True


def person_mask(cli, png: Path, cache: Path) -> Image.Image:
    """Маска человека SAM3 «woman» (белое = человек). Кэш по имени файла."""
    import ss456_height as HH
    cache.mkdir(parents=True, exist_ok=True)
    c = cache / f"{png.stem}__mask.png"
    if not c.exists():
        name = cli.upload(png, "ss456/twopass/raw")
        h = cli.wait(cli.submit(HH.mask_graph(name, f"ss456/twopass/{png.stem}__mask")), 900)
        f = next(o["images"][0] for o in h["outputs"].values() if o.get("images"))
        c.write_bytes(cli.view(f))
    return Image.open(c).convert("L")


def cut_silhouette(draft_png: Path, mb: np.ndarray, plate: Image.Image, k: float,
                   anchor: tuple[float, float], card: dict):
    """Силуэт = вырезанное тело черновика (маска SAM3) серым ×k на чистую плиту ТОЖДЕСТВЕННЫМ
    переносом долей кадра: якорь (низ маски / таз сидя) → та же доля плиты — гейт сцены уже
    гарантирует совпадение кадрирований. Мебель карточки ПЕРЕД персонажем (по его позиции)
    возвращается поверх. → (плита с силуэтом, отчёт)."""
    W0, H0 = Image.open(draft_png).size
    ys, xs = np.where(mb)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    plate = plate.convert("RGB")
    W, H = plate.size
    cut = Image.open(draft_png).convert("RGB").crop((x0, y0, x1, y1))
    cmask = Image.fromarray((mb[y0:y1, x0:x1] * 255).astype(np.uint8)).convert("L")
    # px черновика → px плиты: ×k (рост) и ×W/W0 (черновик 1344 → финал 2336); якорь — в ту же долю
    sx, sy = k * W / W0, k * H / H0
    nw, nh = round(cut.width * sx), round(cut.height * sy)
    cut = cut.resize((nw, nh), Image.LANCZOS)
    cmask = cmask.resize((nw, nh), Image.LANCZOS)
    sil = Image.composite(Image.new("RGB", cut.size, (128, 128, 128)),
                          Image.new("RGB", cut.size, (0, 0, 0)), cmask)
    ax, ay = anchor[0] * W0, anchor[1] * H0               # якорь в px черновика → та же доля плиты
    dx, dy = anchor[0] * W - (ax - x0) * sx, anchor[1] * H - (ay - y0) * sy
    out = plate.copy()
    out.paste(sil, (round(dx), round(dy)), cmask)
    for a in card.get("anchors", []):         # мебель перед ПЕРСОНАЖЕМ — поверх силуэта (он за ней)
        a_x0, a_y0, a_x1, a_y1 = a["bbox"]
        if a_y1 > anchor[1] and a_x0 < anchor[0] < a_x1 and float(a["top_cm"]) < 140:
            box = (int(a_x0 * W), int(a_y0 * H), int(a_x1 * W), int(a_y1 * H))
            out.paste(plate.crop(box), box[:2])
    rep = {"anchor": [round(anchor[0], 3), round(anchor[1], 3)],
           "top_y": round(dy / H, 3), "mask_top": round(y0 / H0, 4), "w_frac": round(nw / W, 3),
           "out_x": round(min(dx, W - (dx + nw)) / W, 3)}                 # запас до края кадра (− = вылез)
    return out, rep


def clip_k_mask(k: float, anchor_y: float, dst_y: float, mask_top: float) -> float:
    """Урезание k по фактическому верху МАСКИ (волосы выше круга головы позы): верх = dst − k·(anchor−top)
    ≥ 0.01. Вызывается повторно после первой вырезки, если маска вылезла за кадр."""
    top = dst_y - k * (anchor_y - mask_top)
    if top >= 0.01:
        return k
    return (dst_y - 0.01) / (anchor_y - mask_top)


def clip_k_width(k: float, w_frac: float) -> float:
    """Урезание k, если вырезка шире кадра: w_frac — доля ширины кадра при текущем k; цель ≤ 94 %."""
    if w_frac <= 0.94:
        return k
    return k * 0.94 / w_frac


def cmd_build(a) -> int:
    import os
    cli = R.Comfy(os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    root, dj = load_drafts(a.draft)
    base = yaml.safe_load(Path(a.src).read_text(encoding="utf-8"))
    fcfg = copy.deepcopy(base)
    fcfg.update(round=Path(a.dst).stem, kit_poses={}, poses=[], seeds={}, variants=[], est_frame_s=150)
    only = set(a.poses.split(",")) if a.poses else None
    meta, rejected = {}, []
    for scene in scenes_of(base):
        kp = base["kit_poses"].get(f"{scene}_plate")
        if not kp:
            continue
        kit = root / Path(kp["plate"]).parent
        card = json.loads((kit / "card.json").read_text(encoding="utf-8"))
        tgt = json.loads((kit / "target.json").read_text(encoding="utf-8"))
        kind = kind_of(str(tgt.get("cls", "")))
        slot = tuple(tgt["feet"])
        seat, seat_cm = seat_point(card, slot) if kind == "сидя" else (None, None)
        final_canvas = kp["canvas"]
        # финал — от ЧИСТОЙ плиты: персонаж в source («черновик с человеком») заставляет H3 копировать
        # его обычный размер и игнорировать силуэт (проверено 20.10, h_ratio 0.99–1.02 при k 1.2–1.45)
        plate = Image.open(root / kp["plate"]).convert("RGB") \
            .resize((final_canvas[0], final_canvas[1]), Image.Resampling.LANCZOS)
        built = False
        for jid in sorted(dj):                                  # сиды — из черновиков раунда (ретраи — там же)
            if not jid.startswith(f"draft_{scene}"):
                continue
            seed = dj[jid]["seed"]
            pose = f"{scene}_tp_s{seed % 1000}"
            if only and pose not in only:
                continue
            draft_png = Path(dj[jid]["files"][0])
            shift = scene_shift(draft_png, plate)
            if not shift["ok"]:
                # H3 перестроил сцену — доли кадра черновика не переносимы; другой сид/черновик
                rejected.append(pose)
                meta[pose] = {"rejected": shift}
                print(f"!! {pose}: сцена черновика не совпала с плитой — пропуск", shift, flush=True)
                continue
            img = L.read_bgr(draft_png)
            h0, w0 = img.shape[:2]
            p = L.pose33(img)
            if p is None:
                raise RuntimeError(f"{draft_png.name}: поза не найдена")
            box = L.body_box(p, w0 / h0)
            m = person_mask(cli, draft_png, draft_png.parent / "_masks")
            mb = np.asarray(m.resize((w0, h0), Image.NEAREST)) > 127
            if not mb.any():
                raise RuntimeError(f"{draft_png.name}: маска пуста")
            anchor = plan_anchor(kind, mb, p)
            k = plan_k(p, box, mb, card, kind, tgt, anchor, seat_cm, w0 / h0) * a.boost
            if kind == "сидя":                 # голени ×k не должны уйти за низ кадра (таз фиксирован)
                feet_y = max((p[i, 1] for i in (27, 28, 31, 32) if p[i, 2] >= L.MIN_VIS), default=None)
                if feet_y and feet_y > anchor[1]:
                    k = min(k, (0.99 - anchor[1]) / (feet_y - anchor[1]))
            if kind == "лежа":                 # верх головы не над опорой — клип по краям не считаем
                top_y, clipped = None, False
            else:
                k, top_y, clipped = clip_k(k, anchor[1], anchor[1], box[1])
            if a.shape == "cut":           # 21.10: рисованный силуэт ломал телосложение (раздутые бёдра) —
                sil, rep = cut_silhouette(draft_png, mb, plate, k, anchor, card)  # тело из черновика
                if kind != "лежа" and rep["top_y"] < 0.01:   # маска (волосы) выше рамки позы — переделать
                    k = clip_k_mask(k, anchor[1], anchor[1], rep["mask_top"])
                    sil, rep = cut_silhouette(draft_png, mb, plate, k, anchor, card)
                    clipped = True
                if rep["out_x"] < -0.04:                       # тело шире кадра — урезать до 94 % ширины
                    k = clip_k_width(k, rep["w_frac"])
                    sil, rep = cut_silhouette(draft_png, mb, plate, k, anchor, card)
                    clipped = True
                rep.update(kind=kind, k=round(k, 3), clipped=clipped, shift=shift,
                           dev=[round(anchor[0] - slot[0], 3),        # отклонение от слота — только отчёт
                                round(anchor[1] - (seat or slot)[1], 3)])
            else:
                q = transform(p, k, anchor, anchor)
                sil = draw_silhouette(plate, q)
                rep = {"kind": kind, "k": round(k, 3), "top_y": top_y and round(float(top_y), 3),
                       "clipped": clipped, "shift": shift, "slot": [round(slot[0], 4), round(slot[1], 4)]}
            if seat:
                rep["seat_cm"] = seat_cm
            d = root / "_claude" / "ss456" / "twopass" / f"{scene}_s{seed % 1000}"
            d.mkdir(parents=True, exist_ok=True)
            sil.save(d / "plate.png")
            plate.save(d / "clean.png")
            prompt = (root / kp["prompt"]).read_text(encoding="utf-8")
            (d / "prompt.txt").write_text(sil_prompt(prompt, kind), encoding="utf-8")
            (d / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
            (d / "target.json").write_text(json.dumps(tgt, ensure_ascii=False, indent=1), encoding="utf-8")
            rel = lambda x: x.relative_to(root).as_posix()  # noqa: E731
            fcfg["kit_poses"][pose] = {"plate": rel(d / "plate.png"), "prompt": rel(d / "prompt.txt"),
                                       "canvas": final_canvas}
            fcfg["poses"].append(pose)
            fcfg["seeds"][pose] = [seed]
            v = copy.deepcopy(plate_variant(base, scene))
            v.update(id=f"tp_{scene}_s{seed % 1000}", poses=[pose], seeds={pose: [seed]})
            fcfg["variants"].append(v)
            meta[pose] = rep
            built = True
            print(pose, rep, flush=True)
        if built or a.fallback != "capsule":
            continue
        # все черновики сцены перестроены H3 — капсула по карточке (место + рост/длина, поза текстом;
        # H1: стоя 9/10). Итог проверяется глазами и метрикой, сцена в финале может тоже перестроиться.
        import ss456_height as HH
        seed = base["seeds"][f"{scene}_plate"][0]
        pose = f"{scene}_tp_s{seed % 1000}"
        if only and pose not in only:
            continue
        sil, top_c, _ = HH.silhouette(plate, card, slot, float(tgt["top_cm"]) * a.boost, kind)
        d = root / "_claude" / "ss456" / "twopass" / f"{scene}_s{seed % 1000}"
        d.mkdir(parents=True, exist_ok=True)
        sil.save(d / "plate.png")
        plate.save(d / "clean.png")
        prompt = (root / kp["prompt"]).read_text(encoding="utf-8")
        (d / "prompt.txt").write_text(sil_prompt(prompt, kind), encoding="utf-8")
        (d / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "target.json").write_text(json.dumps(tgt, ensure_ascii=False, indent=1), encoding="utf-8")
        rel = lambda x: x.relative_to(root).as_posix()  # noqa: E731
        fcfg["kit_poses"][pose] = {"plate": rel(d / "plate.png"), "prompt": rel(d / "prompt.txt"),
                                   "canvas": final_canvas}
        fcfg["poses"].append(pose)
        fcfg["seeds"][pose] = [seed]
        v = copy.deepcopy(plate_variant(base, scene))
        v.update(id=f"tp_{scene}_s{seed % 1000}", poses=[pose], seeds={pose: [seed]})
        fcfg["variants"].append(v)
        meta[pose] = {"kind": kind, "fallback": "capsule", "top_y": top_c,
                      "slot": [round(slot[0], 4), round(slot[1], 4)],
                      "rejected_drafts": [p for p in rejected if p.startswith(f"{scene}_")]}
        print(pose, meta[pose], flush=True)
    (root / "_claude" / "ss456" / "twopass" / "build.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    Path(a.dst).write_text("# SS-456: финал по силуэту из черновика (двухпроходная схема роста)\n"
                           + yaml.safe_dump(fcfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(a.dst, "кадров", len(fcfg["poses"]), end=" ")
    if rejected:
        print("| пропущено, сцена не совпала:", ", ".join(rejected))
    else:
        print()
    return 1 if not fcfg["poses"] else 0


def achieved(p: np.ndarray, box, card: dict, kind: str, slot: tuple[float, float],
             top_cm: float, seat_cm: float, aspect: float) -> dict | None:
    """Автоприёмка: кадр → «рост» в см по карточке, вердикт ±15 % от цели top_cm.
    стоя/присед — макушка..стопы на глубине стоп; сидя — сиденье + (таз − макушка)/s;
    лежа — длина тела (нос→щиколотка) на глубине таза. Верх — по ГОЛОВЕ (нос − радиус), не по рамке:
    рамка тянет поднятые руки (батончик над головой) и завышала рост на десятки см. Сдвиг по глубине
    учтён (s по фактической опоре)."""
    hor, cam = float(card["horizon_y"]), float(card["camera_height_cm"])
    vis = lambda i: p[i, 2] >= L.MIN_VIS  # noqa: E731
    top = head_top(p, aspect)
    if kind in ("стоя", "присед"):
        feet = [p[i, 1] for i in (27, 28, 31, 32) if vis(i)]
        if not feet or (top is None and box is None):
            return None
        fy = float(np.mean(feet))
        s = (fy - hor) / cam                        # вертикальный px/см на глубине СТОП (глубина опоры)
        cm = (fy - (top if top is not None else box[1])) / s
    elif kind == "сидя":
        hp = [p[i, 1] for i in (23, 24) if vis(i)]
        if not hp or (top is None and box is None):
            return None
        hy = float(np.mean(hp))
        r = seat_cm / cam                           # пол под тазом: проекция вниз через горизонт
        ground = (hy - hor * r) / (1 - r) if r < 1 else slot[1]
        s = (ground - hor) / cam                    # масштаб на глубине ПОЛА под сиденьем
        cm = seat_cm + (hy - (top if top is not None else box[1])) / s
    else:
        length = body_length(p, aspect)
        if length is None:
            return None
        pts = [float(p[i, 1]) for i in (0, 27, 28, 31, 32) if vis(i)]   # точки длины: нос и щиколотки
        s = (max(pts) - hor) / cam                  # глубина ЕЁ опоры — низ тела на поверхности
        cm = length / s
    tol = TOL_FRAC * top_cm
    return {"cm": round(cm, 1), "target": top_cm, "tol": round(tol, 1),
            "ok": bool(abs(cm - top_cm) <= tol),
            "soft": kind in ("лежа", "присед")}     # позозависимая цель (глубина приседа/сгиба) — глаза решают


def cmd_eval(a) -> int:
    from ss340_eval import sil_leak
    root, dj_jobs = load_drafts(a.draft)
    dj = [j for j in dj_jobs.values()]
    fj = []
    for part in a.src.split(","):
        fcfg = yaml.safe_load(Path(part.strip()).read_text(encoding="utf-8"))
        frun = Path(fcfg["out"]) / fcfg["round"]
        fj += [json.loads(l) for l in (frun / "jobs.jsonl").read_text(encoding="utf-8").splitlines()]
    build = json.loads((root / "_claude" / "ss456" / "twopass" / "build.json").read_text(encoding="utf-8"))
    rows, failed = [], []
    for j in fj:
        scene = j["pose"].rsplit("_tp_", 1)[0]
        seed_dir = root / "_claude" / "ss456" / "twopass" / f"{scene}_s{j['seed'] % 1000}"
        draft = next(Path(x["files"][0]) for x in dj if x["pose"] == f"{scene}_draft" and x["seed"] == j["seed"])
        fin = Path(j["files"][0])
        img = L.read_bgr(fin)
        h, w = img.shape[:2]
        p = L.pose33(img)
        box = L.body_box(p, w / h) if p is not None else None
        r = {"id": j["id"], "round": Path(fin).parent.parent.name, "scene": scene,
             "seed": j["seed"] % 1000, "build": build.get(j["pose"])}
        bld = build.get(j["pose"]) or {}
        card = json.loads((seed_dir / "card.json").read_text(encoding="utf-8"))
        tgt = json.loads((seed_dir / "target.json").read_text(encoding="utf-8"))
        kind = kind_of(str(tgt.get("cls", "")))
        slot = tuple(tgt["feet"])
        seat, seat_cm = seat_point(card, slot) if kind == "сидя" else (None, 45.0)
        pd = L.pose33(L.read_bgr(draft))
        bd = L.body_box(pd, w / h) if pd is not None else None
        if box is not None and bd is not None:
            r["h_ratio_final"] = round(float((box[3] - box[1]) / (bd[3] - bd[1])), 3)
        if p is not None:                                  # посадка финала + автоприёмка
            feet = [p[i, 1] for i in (27, 28, 31, 32) if p[i, 2] >= L.MIN_VIS]
            low = float(np.mean(feet)) if feet else (box[3] if box else None)
            if low is not None:
                r["contact_dy"] = round(low - slot[1], 3)              # vs слот карточки (отчёт-отклонение)
                if bld.get("anchor"):
                    r["anchor_dy"] = round(low - bld["anchor"][1], 3)  # vs посадка силуэта (авторитет)
            if kind == "сидя":
                hp = [p[i, 1] for i in (23, 24) if p[i, 2] >= L.MIN_VIS]
                if hp:
                    r["hip_seat_dy"] = round(float(np.mean(hp)) - seat[1], 3)
            ach = achieved(p, box, card, kind, slot, float(tgt["top_cm"]), seat_cm, w / h)
            if ach:
                r["achieved"] = ach
                if not ach["ok"] and not ach.get("soft"):
                    failed.append(j["pose"])
        r["sil_leak"] = sil_leak(fin, seed_dir / "plate.png", seed_dir / "clean.png")
        rows.append(r)
        print(r, flush=True)
    out = root / "_claude" / "ss456" / "twopass" / "eval.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("ok", len(rows), "->", out)
    if failed:
        print("НЕ ПРОШЛИ допуск ±15 %:", ", ".join(failed))
        print("ретрай: python tools/ss456_twopass.py build <base> <drafts> <cfg> --poses",
              ",".join(failed), "--boost 1.05")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft"); d.add_argument("src"); d.add_argument("dst")
    d.add_argument("--only", help="только эти сцены (через запятую)")
    b = sub.add_parser("build"); b.add_argument("src"); b.add_argument("draft"); b.add_argument("dst")
    b.add_argument("--shape", default="cut", choices=["cut", "draw"],
                   help="cut — тело из черновика по маске SAM3 (анатомия сохранена); draw — рисованный по точкам")
    b.add_argument("--poses", help="только эти позы <сцена>_tp_s<сид> — ретрай бустом")
    b.add_argument("--boost", type=float, default=1.0, help="умножить k (недобор роста)")
    b.add_argument("--fallback", default="capsule", choices=["capsule", "none"],
                   help="все черновики сцены перестроены H3 → капсула по карточке (место + рост, поза текстом)")
    e = sub.add_parser("eval"); e.add_argument("draft"); e.add_argument("src")
    a = ap.parse_args()
    return {"draft": cmd_draft, "build": cmd_build, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
