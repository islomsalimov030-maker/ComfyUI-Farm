#!/usr/bin/env python3
"""SS-340: оценка раунда RefMod — метрики WS-07/WS-05 (масштаб против кадра v2, лицо AuraFace, возраст, резкость)
по каждому кадру и медианы по варианту → <out>/<round>/report.md + eval.json.

    python tools/ss340_eval.py <round.yaml> [--age none|mivolo]

Масштаб: scale_ratio = рамка тела (MediaPipe) / рамка на одобренном кадре v2 той же позы; > 1 — крупнее эталона,
scale_err = |ln ratio| (0 — как в v2). Эталоны рамок — targets.json кита WS-07 (`kit` в конфиге раунда).
kit_poses.<поза>.target_json — эталон рамки из `ss340_scale_kit.py` (рост по карточке, а не v2);
верх головы Δy — рамка тела против expected_top_y цели (работает и для «сидя»).
kit_poses.<поза>.card — карточка плиты (card.json): scale@1 по якорям и scale_abs = высота рамки тела / ожидаемый
рост на глубине стоп ((y_стоп − горизонт) · рост / камера; 1 — ровно height_cm, > 1 — персонаж велик).
kit_poses.<поза>.leak: {sil, clean} — остаток силуэта на кадре (доля области силуэта; в отчёте — максимум по сидам).
checks: [..] в конфиге — запускать только эти проверки QA (без моделей лица — [scale]).
Метрики — подсказка; решение по контакт-листам принимает человек.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import yaml  # noqa: E402

import ss340_round as R  # noqa: E402
import ws07_eval as E  # noqa: E402
from farm.qa.image.scale import expected_height  # noqa: E402


def sil_leak(img_path: Path, sil: Path, clean: Path, grey=(128, 128, 128), tol: int = 14) -> float | None:
    """Остаток силуэта: доля пикселей силуэта (плита с силуэтом ≠ чистая плита), которые на кадре остались серыми
    цвета силуэта и лежат вне маски человека (MediaPipe, расширена на 1.5 % кадра — серая одежда не считается)."""
    import cv2
    import numpy as np
    from PIL import Image
    import refmod_build as B
    out = Image.open(img_path).convert("RGB")
    W, H = out.size
    a = np.asarray(Image.open(sil).convert("RGB").resize((W, H)), dtype=np.int16)
    b = np.asarray(Image.open(clean).convert("RGB").resize((W, H)), dtype=np.int16)
    region = np.abs(a - b).max(2) > 10
    if region.sum() < 100:
        return None
    o = np.asarray(out, dtype=np.int16)
    is_grey = (np.abs(o - np.array(grey)).max(2) <= tol)
    _, seg = B.pose_seg(out)
    person = np.zeros((H, W), bool) if seg is None else seg > 0.5
    k = max(3, int(0.015 * H)) | 1
    person = cv2.dilate(person.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return round(float((is_grey & region & ~person).sum() / region.sum()), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--age", default="none", choices=["none", "mivolo"])
    ap.add_argument("--refined", action="store_true", help="оценивать кадры после доработки (__refined.png)")
    a = ap.parse_args()
    cfg = R.load(a.config)
    sdir = Path(cfg["out"]) / cfg["round"]
    kit = R.p(cfg, cfg["kit"])
    targets = json.loads((kit / "targets.json").read_text(encoding="utf-8"))
    th = yaml.safe_load((REPO / "farm/qa/thresholds_sfw_base.yaml").read_text(encoding="utf-8"))
    th["frame"]["age"]["model"] = a.age          # как в ws07_eval: none — без MiVOLO
    height = float(cfg.get("height_cm", 162))
    th["frame"]["scale"]["default_height_cm"] = height   # справочника персонажей в оценке нет — рост из конфига
    if cfg.get("checks"):
        th["frame"]["enabled"] = list(cfg["checks"])
    cache = sdir / "_cache"
    cache.mkdir(exist_ok=True)
    rows = []
    suffix = "__refined" if a.refined else ""
    for j in E.load_jobs(sdir):
        f = sdir / j["pose"] / f"{j['variant']}__{j['seed']}{suffix}.png"
        if not f.is_file():
            continue
        kp = cfg["kit_poses"].get(j["pose"], {})
        if kp.get("target_json"):
            tgt = json.loads(R.p(cfg, kp["target_json"]).read_text(encoding="utf-8"))
        else:
            tgt = targets[kp.get("target", j["pose"])]
        card = json.loads(R.p(cfg, kp["card"]).read_text(encoding="utf-8")) if kp.get("card") else None
        m = E.evaluate(f, dict(j, variant="fixe"), tgt, th, str(R.p(cfg, cfg["face_dir"])), cache,
                       extra={"plate_card": card} if card else None)
        b = [float(x) for x in m["box"]] if m.get("box") else None
        exp = expected_height(card, b[3], height) if (card and b) else None
        m["scale_abs"] = round((b[3] - b[1]) / exp, 3) if exp else None
        m["feet_y"] = round(b[3], 3) if b else None
        # верх головы против цели (доля высоты кадра; < 0 — голова выше цели, персонаж велик) — и для «сидя»
        m["top_dy"] = round(b[1] - tgt["expected_top_y"], 3) if (b and tgt.get("expected_top_y") is not None) else None
        if kp.get("leak"):        # {sil: плита с силуэтом, clean: чистая плита}
            m["sil_leak"] = sil_leak(f, R.p(cfg, kp["leak"]["sil"]), R.p(cfg, kp["leak"]["clean"]))
        rows.append({**{k: j.get(k) for k in ("id", "variant", "pose", "seed", "sec")}, **m})
        print(f"{j['id']}: scale {m.get('scale_ratio')} abs {m['scale_abs']} feet {m['feet_y']} "
              f"face {m.get('face_id')} hard {m.get('hard_fail')}", flush=True)
    (sdir / f"eval{suffix}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    by = defaultdict(list)
    for r in rows:
        by[(r["variant"], r["pose"]) if cfg.get("report_by_pose") else r["variant"]].append(r)
    lines = [f"# {cfg['round']}{' (после доработки)' if a.refined else ''} — метрики (подсказка; решение по контакт-листам)", "",
             f"| вариант | кадров | масштаб (медиана, 1 = эталон) | ошибка масштаба | рост к карточке (1 = {height:g} см) "
             "| стопы y | верх головы Δy | остаток силуэта | лицо AuraFace | без hard-отказа | время, с |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = list(dict.fromkeys((r["variant"], r["pose"]) for r in rows)) if cfg.get("report_by_pose") \
        else [x["id"] for x in cfg["variants"]]
    for v in keys:
        rs = by.get(v, [])
        if not rs:
            continue
        ok = sum(1 for r in rs if not r["hard_fail"])
        name = " / ".join(v) if isinstance(v, tuple) else v
        lines.append(f"| {name} | {len(rs)} | {E._med([r.get('scale_ratio') for r in rs])} | "
                     f"{E._med([r.get('scale_err') for r in rs])} | {E._med([r.get('scale_abs') for r in rs])} | "
                     f"{E._med([r.get('feet_y') for r in rs])} | {E._med([r.get('top_dy') for r in rs])} | "
                     f"{max([r['sil_leak'] for r in rs if r.get('sil_leak') is not None], default=None)} | "
                     f"{E._med([r.get('face_id') for r in rs])} | "
                     f"{ok}/{len(rs)} | {E._med([r.get('sec') for r in rs])} |")
    (sdir / f"report{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
