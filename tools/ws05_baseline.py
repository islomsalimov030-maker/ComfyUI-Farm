"""WS-05: прогон реестра проверок кадра (farm.qa.registry.run_stage) по 26 кадрам v2 Стейси.

    .venv/Scripts/python tools/ws05_baseline.py [--set new|old|src] [--age mivolo|genderage|none] [--out docs/qa/ws05_baseline_<set>.json]

Эталон лица — dataset_face_v1_ready (77 кадров, как в STACY_CONTEXT §6). Решение «глаз» — по STACY_CONTEXT §6:
Лежа-2/5 — не дотянуто (отказ), Стоя-11 — ненадёжно (к человеку), остальные — принять, Лежа-6 — без лица.
Для наборов old/src решения пользователя нет — эталон CSV (buffalo_l на полном кадре): id < 0.55 → «отказ»,
0.55–0.65 → «пограничное» (засчитывается любой вердикт), иначе «принять»; профиль/мелкое лицо (Стоя-11) → «человек».
Вердикт реестра как у воркера + routing.yaml: FACE_UNRELIABLE → «человек», другой hard-отказ → «отказ».
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

from farm.contracts import Asset, QAContext  # noqa: E402
from farm.qa import registry  # noqa: E402
from ws05_face_eval import ROOT, SETS, SKIP, read_csv, src_path  # noqa: E402

EYE = {"Lying-2": "отказ", "Lying-5": "отказ", "Standing-11": "человек"}
CSV_COL = {"old": "id_old_c83", "src": "id_src"}
POSE_RU = {"Lying": "лежит", "Sitting": "сидит", "Standing": "стоит"}


def verdict(results) -> str:
    """Как воркер + routing.yaml: FACE_ID → отказ (ретрай), FACE_UNRELIABLE → человек."""
    codes = [r.fail_code for r in results if not r.passed and r.severity == "hard"]
    if any(c != "FACE_UNRELIABLE" for c in codes):
        return "отказ"
    return "человек" if codes else "принять"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="new", choices=list(SETS))
    ap.add_argument("--age", default="mivolo", help="mivolo | genderage (НЕкоммерч., только тест) | none")
    ap.add_argument("--backend", default="auraface")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"docs/qa/ws05_baseline_{a.set}.json"
    th = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "farm/qa/thresholds_sfw_base.yaml", encoding="utf-8"))
    th["frame"]["age"]["model"] = a.age
    th["frame"]["face_id"]["backend"] = a.backend
    th["frame"]["stop_cost"] = None  # для отчёта — все проверки
    gt = read_csv()
    rows, agree, n = {}, 0, 0
    for p in sorted(SETS["new"].glob("*.png")):
        name = p.stem
        if name in SKIP:
            continue
        path = src_path(name) if a.set == "src" else SETS[a.set] / f"{name}.png"
        if not path or not path.is_file():
            continue
        shot = {"shot_id": name, "character": "stacy", "pose_ru": POSE_RU[name.split("-")[0]],
                "extra_json": {"no_face": name == "Lying-6"}}
        ctx = QAContext(shot=shot, stage="frame", profile={"name": "sfw_base"}, attempt={}, thresholds=th,
                        refs={"face_dir": str(ROOT / "dataset_face_v1_ready")})
        t0 = time.perf_counter()
        res = registry.run_stage("frame", Asset(name, "-", "frame", str(path), name), ctx)
        dt = time.perf_counter() - t0
        v = verdict(res)
        eye = "без лица" if name == "Lying-6" else EYE.get(name, "принять")
        if a.set in CSV_COL and name not in ("Lying-6", "Standing-11"):
            g = float(gt[name][CSV_COL[a.set]])
            eye = "отказ" if g < 0.55 else ("пограничное" if g < 0.65 else "принять")
        hit = v == eye or (eye == "без лица" and v == "принять") or eye == "пограничное"             or (v == "человек" and eye == "отказ")  # к человеку вместо отказа — не пропуск плохого
        n += 1
        agree += hit
        rows[name] = {"verdict": v, "eye": eye, "hit": hit, "sec": round(dt, 2),
                      "checks": [{"name": r.name, "value": round(float(r.value), 4), "passed": r.passed,
                                  "severity": r.severity, "fail_code": r.fail_code, "evidence": r.evidence}
                                 for r in res]}
        short = " ".join(f"{r.name}={float(r.value):.3g}{'' if r.passed else '!'}" for r in res)
        print(f"{name:12s} {v:8s} eye={eye:8s} {'OK ' if hit else 'MISS'} {dt:5.1f}s {short}", flush=True)
    print(f"\nсовпадение с глазом: {agree}/{n} = {agree / n:.0%}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps({"agree": agree, "n": n, "frames": rows}, ensure_ascii=False, indent=1, default=float)
    txt = txt.replace(json.dumps(str(ROOT))[1:-1] + "\\\\", "")   # в отчёт без локальных путей ПК
    Path(out).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
