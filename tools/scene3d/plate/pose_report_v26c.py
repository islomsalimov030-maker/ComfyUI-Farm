"""Step 9c: offline Markdown tables from saved fits and full-frame H3 metrics.

No generation, network, model imports or implicit re-evaluation.
"""
import argparse
import json
import math
from pathlib import Path

VARIANTS = ("D1", "D1H", "D2", "D2H", "D3", "D3H")
METRICS = ("pck_lr", "bg_ssim", "obj_shift_px", "fig_same", "proxy_same", "noise_ratio")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:.4g}"


def tables(root):
    root = Path(root)
    lines = ["## Постановка 17 поз", "",
             "| Шот | Класс | Статус | Флаги | p95 soft/hard, см | Gap опоры, см | yaw / pitch / roll, ° |",
             "|---|---|---|---|---|---|---|"]
    records = {}
    gold = read(Path(__file__).resolve().parents[3] / "eval/shot_spec_gold.json")
    for item in gold["specs"]:
        # Gold specs are keyed by id.
        if isinstance(item, str):
            pid = item
        else:
            pid = item["id"]
        path = root / "pose_round/V2" / pid / "fit.json"
        if not path.is_file():
            lines.append(f"| {pid} | — | отсутствует | — | — | — | — |")
            continue
        r = read(path)
        records[pid] = r
        spec, pen, orient = (r.get(k) or {} for k in ("shot_spec", "penetration", "orient"))
        yaw = (orient.get("yaw") or {}).get("yaw_deg")
        angles = " / ".join(number(v) for v in (yaw, orient.get("tilt_final_deg"), orient.get("roll_final_deg")))
        lines.append(f"| {pid} | {spec.get('pose_class', '—')} | {r['status']} | "
                     f"{', '.join(r.get('flags', [])) or '—'} | "
                     f"{number(pen.get('soft_p95_cm'))} / {number(pen.get('hard_p95_cm'))} | "
                     f"{number(r.get('support_gap_cm'))} | {angles} |")
    counts = {}
    for r in records.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines += ["", "Статусы: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
              "## Диагностика Q03", ""]
    q03 = next((v for k, v in records.items() if k.startswith("Q03")), {})
    for key in ("zone_diagnostics", "diagnostics", "support_info", "target_info", "validation", "height_adjust"):
        if key in q03:
            lines.append(f"- {key}: " + json.dumps(q03[key], ensure_ascii=False, default=str))
    lines += ["", "## H3: средние по веткам", "",
              "Средние считаются отдельно по доступным числам; отсутствующая метрика не заменяется нулём.", "",
              "| Ветка | N | PCK_lr | bg_ssim | obj_shift_px | fig_same | proxy_same | noise_ratio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    metrics_file = root / "pose_round/h3_v26/metrics.json"
    metrics = read(metrics_file) if metrics_file.is_file() else {}
    means = {}
    for variant in VARIANTS:
        rows = [values[variant] for values in metrics.values() if variant in values]
        means[variant] = {}
        for key in METRICS:
            values = [r[key] for r in rows if r.get(key) is not None and math.isfinite(r[key])]
            means[variant][key] = sum(values) / len(values) if values else None
        lines.append(f"| {variant} | {len(rows)} | " + " | ".join(number(means[variant][k]) for k in METRICS) + " |")
    lines += ["", "## H3: по шотам", "",
              "| Шот | Ветка | PCK_lr | bg_ssim | obj_shift_px | fig_same | proxy_same | noise_ratio | Ориентиры |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for pid, row in metrics.items():
        for variant in VARIANTS:
            if variant not in row:
                continue
            r = row[variant]
            marks = []
            for key, good in (("bg_ssim", lambda x: x >= .85), ("obj_shift_px", lambda x: x <= 5),
                              ("pck_lr", lambda x: x >= .8), ("noise_ratio", lambda x: .8 <= x <= 1.25)):
                marks.append(key + (" ?" if r.get(key) is None else " ✓" if good(r[key]) else " ✗"))
            lines.append(f"| {pid} | {variant} | " + " | ".join(number(r.get(k)) for k in METRICS)
                         + " | " + ", ".join(marks) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).write_text(tables(args.root), encoding="utf-8")
