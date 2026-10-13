"""WS-05: face_id/возраст/резкость по кадрам Стейси — сверка с face_metrics_final.csv и AuraFace vs buffalo_l.

    .venv/Scripts/python tools/ws05_face_eval.py --det mediapipe   # и --det buffalo_l для сверки

Данные только читаются (C:\\stacy-lora). Пишет JSON с метриками по каждому кадру/набору.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from farm.qa import face as F  # noqa: E402
from farm.qa.image.sharp import face_sharpness  # noqa: E402

ROOT = Path(r"C:\stacy-lora")
SETS = {"new": ROOT / "frames/sfw_final_frames", "old": ROOT / "frames/sfw_final_frames/_old_c83",
        "src": ROOT / "frames/sfw"}
RU = {"lying": "лежа", "sitting": "сидя", "standing": "стоя"}
SKIP = {"Lying-3"}  # не открывать (бриф)


def src_path(name: str) -> Path | None:
    pose, n = name.split("-")
    for cand in (f"{RU[pose.lower()]}-{n}.png", f"{RU[pose.lower()]} -{n}.png",
                 f"{RU[pose.lower()].capitalize()}-{n}.png"):
        p = SETS["src"] / cand
        if p.is_file():
            return p
    return None


def read_csv() -> dict:
    p = ROOT / "frames/face_metrics_final.csv"
    with open(p, encoding="utf-8-sig") as f:
        return {r["frame"]: r for r in csv.DictReader(f, delimiter=";")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="по умолчанию docs/qa/ws05_face_eval_<det>.json")
    ap.add_argument("--backends", default="buffalo_l,auraface")
    ap.add_argument("--det", default="mediapipe", help="mediapipe | buffalo_l (SCRFD, только тест)")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "farm_qa"))
    a = ap.parse_args()
    a.out = a.out or f"docs/qa/ws05_face_eval_{a.det}.json"
    gt = read_csv()
    names = sorted(p.stem for p in SETS["new"].glob("*.png") if p.stem not in SKIP)
    ds = ROOT / "dataset_face_v1_ready"
    res: dict = {"frames": {}, "dataset": {}, "centroid": {}}
    cents = {}
    for be in a.backends.split(","):
        c = F.centroid(ds, backend=be, det=a.det, cache_dir=a.cache)
        cents[be] = c
        sims = c["embs"] @ c["emb"]
        res["centroid"][be] = {"n": c["n"], "age_median": c["age_median"],
                               "self_sim_p10": float(np.percentile(sims, 10)),
                               "self_sim_mean": float(sims.mean())}
        res["dataset"][be] = dict(zip(c["names"], map(float, sims)))
    for name in names:
        row = {"gt": {k: gt.get(name, {}).get(k) for k in ("id_src", "id_old_c83", "id_new", "age_new", "sharp_new")}}
        for set_name in ("new", "old", "src"):
            p = src_path(name) if set_name == "src" else SETS[set_name] / f"{name}.png"
            if not p or not p.is_file():
                continue
            img = F.load_bgr(p)
            faces = F.detect(img, det=a.det)
            if not faces:
                row[set_name] = {"face": None}
                continue
            f0 = faces[0]
            r = {"face_px": round(f0.size_px), "yaw": round(f0.yaw, 1), "det_score": round(f0.score, 3),
                 "n_faces": len(faces)}
            for be in a.backends.split(","):
                r[f"id_{be}"] = round(F.cos(F.embed(img, f0, be), cents[be]["emb"]), 4)
            r["src"] = f0.extra.get("source")
            if set_name == "new":
                r["age"] = round(F.age(img, f0), 1)
                r["sharp"] = round(face_sharpness(img, f0), 1)
            row[set_name] = r
        res["frames"][name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k != "gt"}, ensure_ascii=False)[:300], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
