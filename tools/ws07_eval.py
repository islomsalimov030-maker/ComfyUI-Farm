#!/usr/bin/env python3
"""WS-07: оценка серии кадров — метрики WS-05 + масштаб против эталона (рамка тела кадра v2) → report.md и контакт-листы.

    python tools/ws07_eval.py <out>/<серия> [--kit <kit>] [--age mivolo|none] [--vlm]

Работает и на поде, и на ПК (CPU; MediaPipe + AuraFace на onnxruntime, MiVOLO — если есть venv).
Масштаб: рамка тела MediaPipe (ws07_lib.body_box) против той же рамки на одобренном кадре v2 той же позы:
scale_ratio > 1 — персонаж крупнее, scale_err = |ln ratio|; pos_shift — сдвиг центра. Плюс scale из WS-05
(якорь карточки плиты), если карточка есть. Для варианта wide результат сначала кадрируется по wide_crop.
Итог по группе (gen × arm × variant): медианы, доля кадров без hard-отказа, время генерации из jobs.jsonl.
Решение «берём / не берём» — пользователь по контакт-листам, метрики — подсказка (бриф WS-07).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import yaml  # noqa: E402

import ws07_lib as L  # noqa: E402
from farm.contracts import Asset, QAContext  # noqa: E402
from farm.qa import registry  # noqa: E402

POSE_RU = {"Lying": "лежит", "Sitting": "сидит", "Standing": "стоит"}
SHOT_PLATE = {"Sitting-8": "stretch_studio/stool", "Standing-2": "apartment/window",
              "Sitting-2": "apartment/sofa", "Lying-4": "apartment/bed"}


def load_jobs(sdir: Path) -> list[dict]:
    p = sdir / "jobs.jsonl"
    last = {}
    for line in p.read_text(encoding="utf-8").splitlines() if p.exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        last[r["id"]] = r
    return list(last.values())


def local_file(sdir: Path, j: dict) -> Path | None:
    """Файлы ищутся рядом с серией (работает и после scp на ПК — абсолютные пути пода не нужны)."""
    for f in j.get("files") or []:
        p = sdir / j["pose"] / Path(f).name
        if p.is_file():
            return p
    return None


def evaluate(img_path: Path, j: dict, tgt: dict, th: dict, face_dir: str, cache: Path) -> dict:
    import cv2
    img = L.read_bgr(img_path)
    if j["variant"] == "wide":
        h, w = img.shape[:2]
        x0, y0, x1, y1 = tgt["wide_crop"]
        img = cv2.resize(img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], (w, h), interpolation=cv2.INTER_AREA)
        img_path = cache / f"{img_path.stem}__crop.png"
        cv2.imwrite(str(img_path), img)
    h, w = img.shape[:2]
    p = L.pose33(img)
    box = L.body_box(p, w / h) if p is not None else None
    m = L.scale_metrics(box, tgt["target_box"], w / h)
    shot = {"shot_id": j["pose"], "character": "stacy", "pose_ru": POSE_RU[j["pose"].split("-")[0]],
            "plate": SHOT_PLATE.get(j["pose"]), "extra_json": {}}
    ctx = QAContext(shot=shot, stage="frame", profile={"name": "sfw_base"}, attempt={}, thresholds=th,
                    refs={"face_dir": face_dir})
    res = registry.run_stage("frame", Asset(j["id"], "-", "frame", str(img_path), j["id"], w, h), ctx)
    chk = {r.name: {"value": round(float(r.value), 4), "passed": r.passed, "severity": r.severity,
                    "fail_code": r.fail_code} for r in res}
    hard = [r.fail_code for r in res if not r.passed and r.severity == "hard"]
    return {**m, "box": box and [round(x, 3) for x in box], "checks": chk, "hard_fail": hard,
            "face_id": chk.get("face_id", {}).get("value"), "age": chk.get("age", {}).get("value"),
            "sharp": chk.get("sharp", {}).get("value")}


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.median(xs), 3) if xs else None


def contact_sheet(rows: list[dict], sdir: Path, kit: Path, name: str, cols_key: str) -> Path:
    """Строки — позы, столбцы — значения cols_key (gen/arm/variant), первый столбец — эталон v2. Подписи — метрики."""
    from PIL import Image, ImageDraw
    poses = sorted({r["pose"] for r in rows})
    cols = sorted({r[cols_key] for r in rows})
    W, H = 384, 216
    sheet = Image.new("RGB", (W * (len(cols) + 1), (H + 34) * len(poses) + 24), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), f"{name}: столбцы = {cols_key}; подпись: scale×, id, age, sharp", fill=(255, 255, 255))
    for i, pose in enumerate(poses):
        y = 24 + i * (H + 34)
        ref = kit / "refs" / pose / "frame_v2.png"
        if ref.is_file():
            sheet.paste(Image.open(ref).convert("RGB").resize((W, H)), (0, y))
        d.text((4, y + H + 2), f"{pose} — эталон v2", fill=(200, 200, 200))
        for c, col in enumerate(cols, 1):
            rs = [r for r in rows if r["pose"] == pose and r[cols_key] == col and r.get("path")]
            if not rs:
                continue
            r = rs[0]                                         # первый сид; остальные — в report.md
            sheet.paste(Image.open(r["path"]).convert("RGB").resize((W, H)), (c * W, y))
            col_txt = (255, 120, 120) if r.get("hard_fail") else (200, 255, 200)
            clip = "✂" if r.get("clipped") else ""
            d.text((c * W + 4, y + H + 2), f"{col} | s×{r.get('scale_ratio')}{clip} id {r.get('face_id')} "
                   f"age {r.get('age')} sh {r.get('sharp')}", fill=col_txt)
    p = sdir / f"contact_{cols_key}.jpg"
    sheet.save(p, quality=85)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("series_dir")
    ap.add_argument("--kit", default=None)
    ap.add_argument("--age", default="mivolo")
    ap.add_argument("--vlm", action="store_true", help="включить checklist (Ollama qwen3-vl:8b-instruct)")
    a = ap.parse_args()
    sdir = Path(a.series_dir)
    kit = Path(a.kit) if a.kit else sdir.parent / "kit"
    targets = json.loads((kit / "targets.json").read_text(encoding="utf-8"))
    th = yaml.safe_load((REPO / "farm/qa/thresholds_sfw_base.yaml").read_text(encoding="utf-8"))
    th["frame"]["stop_cost"] = None
    th["frame"]["age"]["model"] = a.age
    th["frame"]["checklist"]["enabled"] = bool(a.vlm)
    cache = sdir / "_eval"
    cache.mkdir(exist_ok=True)
    rows = []
    for j in sorted(load_jobs(sdir), key=lambda r: r["id"]):
        if j.get("kind") != "frame":
            continue
        path = local_file(sdir, j) if j.get("status") == "success" else None
        r = {k: j.get(k) for k in ("id", "pose", "gen", "arm", "variant", "seed", "status", "sec")}
        if path:
            t0 = time.perf_counter()
            r.update(evaluate(path, j, targets[j["pose"]], th, str(kit / "face_ref_dir"), cache), path=str(path),
                     eval_sec=round(time.perf_counter() - t0, 1))
        rows.append(r)
        print(f"{r['id']:70s} {r['status']:8s} s×{r.get('scale_ratio')} id {r.get('face_id')} "
              f"hard {r.get('hard_fail')}", flush=True)
    (sdir / "eval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["gen"], r["arm"], r["variant"])].append(r)
    lines = [f"# {sdir.name} — итоги ({time.strftime('%F %H:%M')})", "",
             "Медианы по кадрам группы. scale× — диагональ рамки тела / эталон v2 (1.00 = как v2, >1 крупнее); "
             "|ln| — ошибка масштаба; ok — доля кадров без hard-отказа WS-05; сек — время генерации.", "",
             "✂ — фигура обрезана краем кадра (scale× занижен).", "",
             "| генератор | рука | вариант | кадров | ok | scale× | \\|ln\\| | сдвиг | ✂ | face_id | возраст Δ | резкость | сек |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (gen, arm, var), rs in sorted(groups.items()):
        ok = [r for r in rs if r.get("path")]
        good = sum(1 for r in ok if not r.get("hard_fail"))
        lines.append(f"| {gen} | {arm} | {var} | {len(ok)}/{len(rs)} | {good}/{len(ok) or 1} | "
                     f"{_med([r.get('scale_ratio') for r in ok])} | {_med([r.get('scale_err') for r in ok])} | "
                     f"{_med([r.get('pos_shift') for r in ok])} | {sum(1 for r in ok if r.get('clipped'))} | "
                     f"{_med([r.get('face_id') for r in ok])} | "
                     f"{_med([r.get('age') for r in ok])} | {_med([r.get('sharp') for r in ok])} | "
                     f"{_med([r.get('sec') for r in rs])} |")
    lines += ["", "## По позам (scale× / face_id по сидам)", ""]
    for (gen, arm, var), rs in sorted(groups.items()):
        per = defaultdict(list)
        for r in rs:
            per[r["pose"]].append(f"{r.get('scale_ratio')}/{r.get('face_id')}" if r.get("path") else r["status"])
        lines.append(f"- {gen} · {arm} · {var}: " + "; ".join(f"{p} {', '.join(v)}" for p, v in sorted(per.items())))
    fails = [r for r in rows if r["status"] != "success"]
    if fails:
        lines += ["", "## Ошибки", ""] + [f"- {r['id']}: {r['status']}" for r in fails]
    sheets = []
    for key in ("gen", "arm", "variant"):
        if len({r[key] for r in rows if r.get("path")}) > 1:
            sheets.append(contact_sheet([r for r in rows if r.get("path")], sdir, kit, sdir.name, key))
    lines += ["", "Контакт-листы: " + ", ".join(p.name for p in sheets), "",
              "Решение по серии (заполняет пользователь): берём / не берём / ещё тест — ___"]
    (sdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{sdir / 'report.md'}; контакт-листы: {[p.name for p in sheets]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
