"""WS-05: сводка по выходам ws05_face_eval.py — сверка с CSV, AuraFace vs buffalo_l, перебор порога.

    .venv/Scripts/python tools/ws05_face_stats.py docs/qa/ws05_face_eval_buffalo_l.json docs/qa/ws05_face_eval_mediapipe.json
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

CSV = r"C:/stacy-lora/frames/face_metrics_final.csv"
GT_COL = {"new": "id_new", "old": "id_old_c83", "src": "id_src"}
# решение пользователя по кадрам v2 (STACY_CONTEXT §6): не Стейси/не дотянуто — Лежа-2, Лежа-5;
# ненадёжно — Стоя-11 (профиль, мелкое лицо); остальные — принять (Сидя-1/2 «на глаз нормально»)
EYE_FAIL = {"Lying-2", "Lying-5"}
EYE_UNRELIABLE = {"Standing-11"}


def load_gt() -> dict:
    with open(CSV, encoding="utf-8-sig") as f:
        return {r["frame"]: r for r in csv.DictReader(f, delimiter=";")}


def summary(path: str, gt: dict, min_px: int = 80, max_yaw: float = 45) -> None:
    r = json.load(open(path, encoding="utf-8"))
    print(f"\n=== {path}")
    for be, c in r["centroid"].items():
        print(f"  centroid {be}: n={c['n']} self p10={c['self_sim_p10']:.3f} mean={c['self_sim_mean']:.3f}")
    pooled = []
    for s, col in GT_COL.items():
        a = [(float(gt[n][col]), v[s]["id_buffalo_l"], v[s]["id_auraface"], n)
             for n, v in r["frames"].items() if "id_buffalo_l" in v.get(s, {}) and gt.get(n, {}).get(col)]
        g, b, au = (np.array(x) for x in list(zip(*a))[:3])
        pooled += [(x[1], x[2]) for x in a]
        print(f"  {s}: n={len(a)} buf~csv spearman {spearmanr(g, b)[0]:.3f} mae {np.abs(g - b).mean():.3f}"
              f" | aura~buf spearman {spearmanr(b, au)[0]:.3f} | aura~csv spearman {spearmanr(g, au)[0]:.3f}")
    b, au = np.array(pooled).T
    d = r["dataset"]
    db = np.array(list(d["buffalo_l"].values()))
    da = np.array([d["auraface"].get(k, np.nan) for k in d["buffalo_l"]])
    ok = ~np.isnan(da)
    print(f"  pooled 3 sets n={len(b)} aura~buf spearman {spearmanr(b, au)[0]:.3f};"
          f" dataset n={ok.sum()} spearman {spearmanr(db[ok], da[ok])[0]:.3f}")
    # вердикт по кадрам v2 vs глаз
    for be in ("buffalo_l", "auraface"):
        best = []
        for t in np.arange(0.50, 0.72, 0.01):
            agree, flags = 0, []
            for n, v in r["frames"].items():
                x = v["new"]
                if f"id_{be}" not in x:
                    continue
                unrel = x["face_px"] < min_px or abs(x["yaw"]) > max_yaw
                if unrel:
                    verdict = "unreliable"
                else:
                    verdict = "pass" if x[f"id_{be}"] >= t else "fail"
                eye = "fail" if n in EYE_FAIL else ("unreliable" if n in EYE_UNRELIABLE else "pass")
                # «ненадёжно» = к человеку; совпадение, если глаз тоже не «pass» или это Стоя-11
                hit = verdict == eye or (verdict == "unreliable" and eye != "pass")
                agree += hit
                if not hit:
                    flags.append(f"{n}:{verdict}")
            best.append((agree, round(float(t), 2), flags))
        top = max(a for a, _, _ in best)
        rng = [t for a, t, _ in best if a == top]
        ex = next(f for a, t, f in best if a == top)
        print(f"  {be}: лучшее совпадение с глазом {top}/26 при пороге {rng[0]}..{rng[-1]}; промахи {ex}")


if __name__ == "__main__":
    gt = load_gt()
    for p in sys.argv[1:]:
        summary(p, gt)
