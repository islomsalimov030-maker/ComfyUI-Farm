#!/usr/bin/env python3
"""SS-456, двухпроходная схема роста (идея пользователя 20.10): быстрый черновик → поза из черновика →
силуэт той же позы нужного роста → финальная генерация по силуэту. Всё автоматически; выключается —
просто не запускать финальный конфиг (в ферме — флаг профиля).

    python tools/ss456_twopass.py draft <round_d3.yaml> <round_h3d.yaml>   # конфиг черновиков 1344×768
    (python tools/ss340_round.py run <round_h3d.yaml>)
    python tools/ss456_twopass.py build <round_d3.yaml> <round_h3d.yaml> <round_h3f.yaml>
    (python tools/ss340_round.py run <round_h3f.yaml>)
    python tools/ss456_twopass.py eval <round_h3d.yaml> <round_h3f.yaml>

Посадка (фикс 21.10 дефекта «силуэт в случайном месте»): вырезанное тело ставится на СЛОТ чистой плиты
из карточки кита (target.json: feet), а не в доли кадра черновика — кадрирование черновика ≠ плите.
Вертикаль: стоя/присед низ МАСКИ (не bbox — тот тянет мебель и тень) → пол слота; сидя таз (неподвижен
при ×k вокруг себя) → сиденье над слотом (якорь 'seat' карточки, иначе 45 см). Черновики, где H3
перестроил сцену (сравнение с плитой ORB+RANSAC), отбраковываются — нужен другой сид, а не перенос точки.
Мебель карточки перед слотом (низ ниже пола слота, < 140 см) возвращается поверх силуэта — персонаж за ней.
Рост цели: стоя/присед — expected_height по карточке НА СЛОТЕ (размер и место из одной геометрии);
сидя — bone_height(черновик), метрика экспериментальная — финал судим отношением до/после, кропами и
sil_leak, не абсолютными см.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import ss340_round as R  # noqa: E402
import ws07_lib as L  # noqa: E402
from ss456_height import bone_height, sil_prompt  # noqa: E402
from ss340_scale_kit import draw_silhouette, transform  # noqa: E402
from farm.qa.image.scale import expected_height  # noqa: E402

SCENES = ["S6_greenhouse", "C3_sofa", "C4_rack"]
H_CM = 230.0
DRAFT_CANVAS = [1344, 768]
SIT = {"C3_sofa"}
# гейт совпадения сцены: совпавшие черновики дают ~900 inliers из ~1100 при dx, dy < 0.003;
# перестроенная H3 комната — 3–4 inliers из ~20 (замер 21.10, рама C4)
SHIFT_MIN = dict(inliers=30, ratio=0.4, dxy=0.03, scale=0.04)


def plate_variant(cfg: dict, scene: str) -> dict:
    return next(v for v in cfg["variants"] if v["id"] == f"plate_{scene}")


def cmd_draft(a) -> int:
    cfg = yaml.safe_load(Path(a.src).read_text(encoding="utf-8"))
    out = copy.deepcopy(cfg)
    out.update(round=Path(a.dst).stem, kit_poses={}, poses=[], seeds={}, variants=[], est_frame_s=40)
    for scene in SCENES:
        old = cfg["kit_poses"][f"{scene}_plate"]
        pose = f"{scene}_draft"
        out["kit_poses"][pose] = dict(old, canvas=DRAFT_CANVAS)
        seeds = list(cfg["seeds"][f"{scene}_plate"])
        out["poses"].append(pose)
        out["seeds"][pose] = seeds
        v = copy.deepcopy(plate_variant(cfg, scene))
        v.update(id=f"draft_{scene}", poses=[pose], seeds={pose: seeds})
        out["variants"].append(v)
    Path(a.dst).write_text("# SS-456 H3d: быстрые черновики (поза для силуэта), 1344×768, turbo 8\n"
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


def plan_scaling(draft_png: Path, card: dict, scene: str, slot: tuple[float, float]):
    """Поза черновика → (p, k, pivot, hip, box). Стоя/присед k = ожидаемый рост НА СЛОТЕ / рамка тела
    черновика: размер и место берутся из одной геометрии плиты, кадрирование черновика не участвует.
    Сидя k = 230 / кость-рост (экспериментально). pivot — центр масштабирования (стопы / таз)."""
    img = L.read_bgr(draft_png)
    h, w = img.shape[:2]
    p = L.pose33(img)
    if p is None:
        raise RuntimeError(f"{draft_png.name}: поза не найдена")
    box = L.body_box(p, w / h)
    feet = [p[i] for i in (27, 28, 31, 32) if p[i, 2] >= L.MIN_VIS]
    if not feet:
        raise RuntimeError(f"{draft_png.name}: стопы не видны")
    fx, fy = float(np.mean([q[0] for q in feet])), float(np.mean([q[1] for q in feet]))
    hip_xy = (float((p[23, 0] + p[24, 0]) / 2), float((p[23, 1] + p[24, 1]) / 2))
    if scene in SIT:
        cur = bone_height(p, (w, h), card)
        if not cur:
            raise RuntimeError(f"{draft_png.name}: рост сидя не измерен")
        k = H_CM / cur["cm"]
        pivot = hip_xy
    else:
        eh = expected_height(card, slot[1], H_CM)
        if not eh:
            raise RuntimeError(f"{draft_png.name}: карточка не даёт рост на глубине слота")
        k = eh / (box[3] - box[1])
        pivot = (fx, fy)
    return p, float(k), pivot, hip_xy, box


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


def cut_silhouette(cli, draft_png: Path, plate: Image.Image, k: float, hip_xy, card: dict,
                   slot: tuple[float, float], seat: tuple[float, float] | None):
    """Силуэт = вырезанное тело черновика (маска SAM3) серым ×k на чистую плиту ПО СЛОТУ: стоя/присед
    низ маски → пол слота, сидя таз → сиденье. Мебель карточки перед слотом возвращается поверх.
    → (плита с силуэтом, отчёт: посадка, контакты, выход за кадр)."""
    m = person_mask(cli, draft_png, draft_png.parent / "_masks")
    W0, H0 = Image.open(draft_png).size
    mb = np.asarray(m.resize((W0, H0), Image.NEAREST)) > 127
    if not mb.any():
        raise RuntimeError(f"{draft_png.name}: маска пуста")
    ys, xs = np.where(mb)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    plate = plate.convert("RGB")
    W, H = plate.size
    cut = Image.open(draft_png).convert("RGB").crop((x0, y0, x1, y1))
    cmask = Image.fromarray((mb[y0:y1, x0:x1] * 255).astype(np.uint8)).convert("L")
    # px черновика → px плиты: ×k (рост) и ×W/W0 (черновик 1344 → финал 2336); без пересчёта в плиту
    # кроп вставлялся в ~1.7 раза меньше и «висел в воздухе» (дефект 21.10)
    sx, sy = k * W / W0, k * H / H0
    nw, nh = round(cut.width * sx), round(cut.height * sy)
    cut = cut.resize((nw, nh), Image.LANCZOS)
    cmask = cmask.resize((nw, nh), Image.LANCZOS)
    sil = Image.composite(Image.new("RGB", cut.size, (128, 128, 128)),
                          Image.new("RGB", cut.size, (0, 0, 0)), cmask)
    bx, by = mask_bottom(mb)
    bmy = by * H0                             # низ маски в px черновика
    if seat:                                  # сидя: таз (неподвижен при ×k вокруг себя) → сиденье
        ax, ay, tx, ty = hip_xy[0] * W0, hip_xy[1] * H0, seat[0] * W, seat[1] * H
    else:                                     # стоя/присед: низ маски → пол слота
        ax, ay, tx, ty = bx * W0, bmy, slot[0] * W, slot[1] * H
    dx, dy = tx - (ax - x0) * sx, ty - (ay - y0) * sy
    out = plate.copy()
    out.paste(sil, (round(dx), round(dy)), cmask)
    for a in card.get("anchors", []):         # мебель перед слотом — поверх силуэта (персонаж за ней)
        a_x0, a_y0, a_x1, a_y1 = a["bbox"]
        if a_y1 > slot[1] and a_x0 < slot[0] < a_x1 and float(a["top_cm"]) < 140:
            box = (int(a_x0 * W), int(a_y0 * H), int(a_x1 * W), int(a_y1 * H))
            out.paste(plate.crop(box), box[:2])
    rep = {"slot": [round(slot[0], 4), round(slot[1], 4)], "low": [round(bx, 3), round(by, 3)],
           "top_y": round(dy / H, 3),
           "bottom_dy": round((dy + (bmy - y0) * sy) / H - slot[1], 3),   # контакт низа с полом слота
           "out_x": round(min(dx, W - (dx + nw)) / W, 3)}                # запас до края кадра (− = вылез)
    return out, rep


def cmd_build(a) -> int:
    import os
    cli = R.Comfy(os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    dcfg = yaml.safe_load(Path(a.draft).read_text(encoding="utf-8"))
    root = Path(dcfg["src"])
    fout = root / "_claude" / "ss456" / "twopass"
    drun = Path(dcfg["out"]) / dcfg["round"]
    dj = {json.loads(l)["id"]: json.loads(l) for l in (drun / "jobs.jsonl").read_text(encoding="utf-8").splitlines()}
    base = yaml.safe_load(Path(a.src).read_text(encoding="utf-8"))
    fcfg = copy.deepcopy(base)
    fcfg.update(round=Path(a.dst).stem, kit_poses={}, poses=[], seeds={}, variants=[], est_frame_s=120)
    cards, meta, rejected = {}, {}, []
    for scene in SCENES:
        kit = root / Path(base["kit_poses"][f"{scene}_plate"]["plate"]).parent
        cards[scene] = json.loads((kit / "card.json").read_text(encoding="utf-8"))
        tgt = json.loads((kit / "target.json").read_text(encoding="utf-8"))
        slot = tuple(tgt["feet"])
        seat, seat_cm = seat_point(cards[scene], slot) if scene in SIT else (None, None)
        final_canvas = base["kit_poses"][f"{scene}_plate"]["canvas"]
        # финал — от ЧИСТОЙ плиты: персонаж в source («черновик с человеком») заставляет H3 копировать
        # его обычный размер и игнорировать силуэт (проверено 20.10, h_ratio 0.99–1.02 при k 1.2–1.45)
        plate = Image.open(root / base["kit_poses"][f"{scene}_plate"]["plate"]).convert("RGB") \
            .resize((final_canvas[0], final_canvas[1]), Image.Resampling.LANCZOS)
        for jid in sorted(dj):                                  # сиды — из черновиков раунда (ретраи — там же)
            if not jid.startswith(f"draft_{scene}"):
                continue
            seed = dj[jid]["seed"]
            pose = f"{scene}_tp_s{seed % 1000}"
            draft_png = Path(dj[jid]["files"][0])
            shift = scene_shift(draft_png, plate)
            if not shift["ok"]:
                # H3 перестроил сцену — посадка по карточке невозможна; другой сид/черновик, не перенос точки
                rejected.append(pose)
                meta[pose] = {"rejected": shift}
                print(f"!! {pose}: сцена черновика не совпала с плитой — пропуск", shift, flush=True)
                continue
            p, k, pivot, hip_xy, box = plan_scaling(draft_png, cards[scene], scene, slot)
            k, top_y, clipped = clip_k(k, pivot[1], (seat or slot)[1], box[1])
            if a.shape == "cut":           # 21.10: рисованный силуэт ломал телосложение (раздутые бёдра) —
                sil, rep = cut_silhouette(cli, draft_png, plate, k, hip_xy,   # тело берём из черновика
                                          cards[scene], slot, seat)
                rep.update(k=round(k, 3), clipped=clipped, shift=shift)
            else:
                q = transform(p, k, pivot, seat[:2] if seat else slot)
                if scene in SIT:           # стопы — на пол слота (голени к полу плиты)
                    floor = max(q[i][1] for i in (27, 28, 31, 32))
                    for i in (27, 28, 31, 32):
                        q[i][1] += slot[1] - floor
                sil = draw_silhouette(plate, q)
                rep = {"k": round(k, 3), "top_y": round(float(top_y), 3), "clipped": clipped, "shift": shift,
                       "slot": [round(slot[0], 4), round(slot[1], 4)]}
            if seat:
                rep["seat_cm"] = seat_cm
            d = fout / f"{scene}_s{seed % 1000}"
            d.mkdir(parents=True, exist_ok=True)
            sil.save(d / "plate.png")
            plate.save(d / "clean.png")
            prompt = (root / base["kit_poses"][f"{scene}_plate"]["prompt"]).read_text(encoding="utf-8")
            sil_cls = "сидя" if scene in SIT else ("присед" if "корточк" in str(tgt.get("cls", "")) else "стоя")
            (d / "prompt.txt").write_text(sil_prompt(prompt, sil_cls), encoding="utf-8")
            rel = lambda x: x.relative_to(root).as_posix()  # noqa: E731
            fcfg["kit_poses"][pose] = {"plate": rel(d / "plate.png"), "prompt": rel(d / "prompt.txt"),
                                       "canvas": final_canvas}
            fcfg["poses"].append(pose)
            fcfg["seeds"][pose] = [seed]
            v = copy.deepcopy(plate_variant(base, scene))
            v.update(id=f"tp_{scene}_s{seed % 1000}", poses=[pose], seeds={pose: [seed]})
            fcfg["variants"].append(v)
            meta[pose] = rep
            print(pose, rep, flush=True)
    (fout / "build.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    Path(a.dst).write_text("# SS-456 H3f: финал по силуэту из черновика (двухпроходная схема роста)\n"
                           + yaml.safe_dump(fcfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(a.dst, "кадров", len(fcfg["poses"]), end=" ")
    if rejected:
        print("| пропущено, сцена не совпала:", ", ".join(rejected))
    else:
        print()
    return 1 if not fcfg["poses"] else 0


def cmd_eval(a) -> int:
    from ss340_eval import sil_leak
    dcfg = yaml.safe_load(Path(a.draft).read_text(encoding="utf-8"))
    root = Path(dcfg["src"])
    drun = Path(dcfg["out"]) / dcfg["round"]
    frun = Path(yaml.safe_load(Path(a.src).read_text(encoding="utf-8"))["out"]) / Path(a.src).stem
    dj = [json.loads(l) for l in (drun / "jobs.jsonl").read_text(encoding="utf-8").splitlines()]
    fj = [json.loads(l) for l in (frun / "jobs.jsonl").read_text(encoding="utf-8").splitlines()]
    build = json.loads((root / "_claude" / "ss456" / "twopass" / "build.json").read_text(encoding="utf-8"))
    rows = []
    for j in fj:
        scene = next(s for s in SCENES if s in j["pose"])
        seed_dir = root / "_claude" / "ss456" / "twopass" / f"{scene}_s{j['seed'] % 1000}"
        draft = next(Path(x["files"][0]) for x in dj if x["pose"] == f"{scene}_draft" and x["seed"] == j["seed"])
        fin = Path(j["files"][0])
        img = L.read_bgr(fin)
        h, w = img.shape[:2]
        p = L.pose33(img)
        box = L.body_box(p, w / h) if p is not None else None
        r = {"id": j["id"], "scene": scene, "seed": j["seed"] % 1000, "build": build.get(j["pose"])}
        pd = L.pose33(L.read_bgr(draft))
        bd = L.body_box(pd, w / h) if pd is not None else None
        if box and bd:
            r["h_ratio_final"] = round((box[3] - box[1]) / (bd[3] - bd[1]), 3)      # финал/черновик по рамке
        card = json.loads((root / base_card(root, scene) / "card.json").read_text(encoding="utf-8"))
        slot = tuple(json.loads((root / base_card(root, scene) / "target.json").read_text(encoding="utf-8"))["feet"])
        if p is not None:                                  # контакт финала с опорой слота
            feet = [p[i][1] for i in (27, 28, 31, 32) if p[i, 2] >= L.MIN_VIS]
            if feet:
                r["contact_dy"] = round(float(np.mean(feet)) - slot[1], 3)          # стопы vs пол слота
            if scene in SIT:
                hp = [p[i][1] for i in (23, 24) if p[i, 2] >= L.MIN_VIS]
                if hp:
                    r["hip_seat_dy"] = round(float(np.mean(hp)) - seat_point(card, slot)[0][1], 3)
        bh = bone_height(p, (w, h), card) if p is not None else None
        bhd = bone_height(pd, (w, h), card) if pd is not None else None
        if bh and bhd:
            r["bone_final"] = bh["cm"]
            r["bone_draft"] = bhd["cm"]
        r["sil_leak"] = sil_leak(fin, seed_dir / "plate.png", seed_dir / "clean.png")
        rows.append(r)
        print(r, flush=True)
    out = root / "_claude" / "ss456" / "twopass" / "eval.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("ok", len(rows), "->", out)
    return 0


def base_card(root: Path, scene: str) -> Path:
    f = yaml.safe_load((root / "refmod_inputs" / "round_d3.yaml").read_text(encoding="utf-8"))
    return root / Path(f["kit_poses"][f"{scene}_plate"]["plate"]).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft"); d.add_argument("src"); d.add_argument("dst")
    b = sub.add_parser("build"); b.add_argument("src"); b.add_argument("draft"); b.add_argument("dst")
    b.add_argument("--shape", default="cut", choices=["cut", "draw"],
                   help="cut — тело из черновика по маске SAM3 (анатомия сохранена); draw — рисованный по точкам")
    e = sub.add_parser("eval"); e.add_argument("draft"); e.add_argument("src")
    a = ap.parse_args()
    return {"draft": cmd_draft, "build": cmd_build, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
