#!/usr/bin/env python3
"""SS-340: оценка раунда RefMod — метрики WS-07/WS-05 (масштаб против кадра v2, лицо AuraFace, возраст, резкость)
по каждому кадру и медианы по варианту → <out>/<round>/report.md + eval.json.

    python tools/ss340_eval.py <round.yaml> [--age none|mivolo]

Масштаб: scale_ratio = рамка тела (MediaPipe) / рамка на одобренном кадре v2 той же позы; > 1 — крупнее эталона,
scale_err = |ln ratio| (0 — как в v2). Эталоны рамок — targets.json кита WS-07 (`kit` в конфиге раунда).
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
    cache = sdir / "_cache"
    cache.mkdir(exist_ok=True)
    rows = []
    suffix = "__refined" if a.refined else ""
    for j in E.load_jobs(sdir):
        f = sdir / j["pose"] / f"{j['variant']}__{j['seed']}{suffix}.png"
        if not f.is_file():
            continue
        m = E.evaluate(f, dict(j, variant="fixe"), targets[j["pose"]], th, str(R.p(cfg, cfg["face_dir"])), cache)
        rows.append({**{k: j[k] for k in ("id", "variant", "pose", "seed", "sec")}, **m})
        print(f"{j['id']}: scale {m.get('scale_ratio')} face {m.get('face_id')} hard {m.get('hard_fail')}", flush=True)
    (sdir / f"eval{suffix}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    by = defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    lines = [f"# {cfg['round']}{' (после доработки)' if a.refined else ''} — метрики (подсказка; решение по контакт-листам)", "",
             "| вариант | кадров | масштаб (медиана, 1 = как v2) | ошибка масштаба | лицо AuraFace | без hard-отказа | время, с |",
             "|---|---|---|---|---|---|---|"]
    for v in [x["id"] for x in cfg["variants"]]:
        rs = by.get(v, [])
        if not rs:
            continue
        ok = sum(1 for r in rs if not r["hard_fail"])
        lines.append(f"| {v} | {len(rs)} | {E._med([r.get('scale_ratio') for r in rs])} | "
                     f"{E._med([r.get('scale_err') for r in rs])} | {E._med([r.get('face_id') for r in rs])} | "
                     f"{ok}/{len(rs)} | {E._med([r.get('sec') for r in rs])} |")
    (sdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
