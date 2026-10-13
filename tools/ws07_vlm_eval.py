#!/usr/bin/env python3
"""WS-07 серия 6: VLM-судья (Ollama, P(Yes) из logprobs — farm/qa/vlm.py) на чек-листах WS-05.

    python tools/ws07_vlm_eval.py --models qwen3-vl:8b-instruct,qwen3-vl:32b-instruct,qwen3.6:27b \
        --frames C:/stacy-lora/frames/sfw_final_frames [--out /workspace/farm/experiments/s6_vlm]

Набор: 26 кадров v2 + решение глаза (STACY_CONTEXT §6 / tools/ws05_baseline.EYE): Лежа-2/5 — плохо,
Стоя-11 — к человеку, остальные — хорошо. Вопросы — постоянные чек-листа (одна женщина, без текста,
анатомия) + «соответствует позе строки» (поза из имени файла; негатив — вопрос с чужой позой).
Меряет: AUC/разделение хорошо-плохо по позе (свой vs чужой вопрос), долю ответов из logprobs, с/вопрос,
пик VRAM (nvidia-smi) рядом с загруженным ComfyUI. Модель после прогона выгружается (keep_alive 0).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from farm.qa import vlm  # noqa: E402
from ws07_lib import read_bgr  # noqa: E402

POSE_Q = {"Lying": "Is the woman lying down?", "Sitting": "Is the woman sitting?",
          "Standing": "Is the woman standing upright on her feet?"}
ALWAYS = ["Is there exactly one woman in this image?",
          "Is the image free of any text, captions, logos or watermarks?",
          "Is the woman's whole body proportioned naturally, without extra or missing limbs?"]
SKIP = {"Lying-3"}


def vram_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:  # noqa: BLE001
        return None


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(s / (len(pos) * len(neg)), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-vl:8b-instruct")
    ap.add_argument("--frames", default="/workspace/farm/experiments/kit/frames_v2")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--out", default="/workspace/farm/experiments/s6_vlm")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(p for p in Path(a.frames).glob("*.png") if p.stem not in SKIP)
    summary = {}
    for model in a.models.split(","):
        cfg = {"model": model, "url": a.url, "keep_alive": "10m"}
        rows, t_q, n_q, src_lp, v_peak = [], 0.0, 0, 0, vram_mb() or 0
        for f in frames:
            img = read_bgr(f)
            kind = f.stem.split("-")[0]
            other = [q for k, q in POSE_Q.items() if k != kind]
            qs = [("own_pose", POSE_Q[kind])] + [("other_pose", q) for q in other] + [("always", q) for q in ALWAYS]
            ans = []
            for tag, q in qs:
                t0 = time.perf_counter()
                try:
                    r = vlm.p_yes(img, q, cfg, keep_alive="10m")
                except vlm.VLMUnavailable as e:
                    print("VLM недоступна:", e)
                    return 1
                t_q += time.perf_counter() - t0
                n_q += 1
                src_lp += r["source"] == "logprobs"
                ans.append({"tag": tag, **r})
                v_peak = max(v_peak, vram_mb() or 0)
            rows.append({"frame": f.stem, "answers": ans})
            print(model, f.stem, " ".join(f"{x['tag'][:5]}={x['p_yes']:.2f}" for x in ans), flush=True)
        vlm.unload(cfg)
        own = [x["p_yes"] for r in rows for x in r["answers"] if x["tag"] == "own_pose"]
        oth = [x["p_yes"] for r in rows for x in r["answers"] if x["tag"] == "other_pose"]
        alw = [x["p_yes"] for r in rows for x in r["answers"] if x["tag"] == "always"]
        summary[model] = {"pose_auc": auc(own, oth), "always_mean": round(sum(alw) / len(alw), 3),
                          "always_below_0.5": sum(p < 0.5 for p in alw), "logprobs_share": round(src_lp / n_q, 3),
                          "sec_per_q": round(t_q / n_q, 2), "vram_peak_mb": v_peak, "n_frames": len(rows)}
        (out / f"{model.replace(':', '_')}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")
        print(model, summary[model])
    lines = ["# s6_vlm — VLM-судья на чек-листах WS-05", "",
             "pose_auc — P(Yes) на «своей» позе против «чужой» (1.0 = идеально); always — постоянные вопросы "
             "на одобренных кадрах (должны быть ≈1); VRAM — пик всей карты (вместе с ComfyUI).", "",
             "| модель | pose_auc | always ср. | always <0.5 | logprobs | с/вопрос | VRAM пик, МБ |", "|---|---|---|---|---|---|---|"]
    for m, s in summary.items():
        lines.append(f"| {m} | {s['pose_auc']} | {s['always_mean']} | {s['always_below_0.5']} | {s['logprobs_share']} | "
                     f"{s['sec_per_q']} | {s['vram_peak_mb']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
