"""WS-06: прогон проверок видео и post.loop по папке готовых клипов (офлайн, CPU).

    python tools/ws06_video_eval.py --clips <папка mp4> [--kf <папка ключевых кадров>] \
        [--out docs/qa/ws06_eval.json] [--post-dir <куда класть исправленные>] [--no-identity] [--labels labels.json]

Клипы, имя которых есть в --transitions (через запятую), идут как loop=0 (переходы, шов не проверяется).
--labels: {"<клип>": {"verdict": "good|bad", "codes": [...], "note": "..."}} — сверка с человеком.
Выход: JSON по клипам (метрики, коды, маршрут, результат post.loop) + таблица в stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from farm.config import load_routing, load_config  # noqa: E402
from farm.contracts import Asset, QAContext  # noqa: E402
from farm.qa import registry  # noqa: E402


def route_for(codes: list[str], routing: dict) -> tuple[str, dict]:
    routes = routing.get("routes") or {}
    for code in routes:
        if code in codes:
            return code, routes[code]
    return (codes[0] if codes else "—"), (routing.get("default") or {})


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # консоль Windows (cp1251) не знает «→»
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--kf")
    ap.add_argument("--face-ref")
    ap.add_argument("--out", default="docs/qa/ws06_eval.json")
    ap.add_argument("--post-dir")
    ap.add_argument("--transitions", default="Standing-4,Standing-9")
    ap.add_argument("--no-identity", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--labels")
    ap.add_argument("--thresholds", default="farm/qa/thresholds_sfw_base.yaml")
    a = ap.parse_args()

    th = yaml.safe_load(open(a.thresholds, encoding="utf-8"))
    if a.no_identity:
        th.setdefault("video", {})["disabled"] = ["identity_dip"]
    routing = load_routing(load_config())
    trans = {x.strip() for x in a.transitions.split(",") if x.strip()}
    labels = json.load(open(a.labels, encoding="utf-8")) if a.labels else {}
    clips = sorted(Path(a.clips).glob("*.mp4"))
    if a.only:
        keep = set(a.only.split(","))
        clips = [c for c in clips if c.stem in keep]
    rows = []
    for p in clips:
        name = p.stem
        base = name[2:].split("_")[0] if name.startswith("x_") else name
        refs = {"loop": base not in trans}
        if a.kf:
            for ext in (".png", ".jpg"):
                k = Path(a.kf) / f"{base}{ext}"
                if k.exists():
                    refs["keyframe"] = str(k)
                    break
        if a.face_ref:
            refs["face_ref"] = a.face_ref
        ctx = QAContext(shot={"shot_id": name, "loop": int(refs["loop"])}, stage="video", profile={},
                        attempt={"params_json": {}}, thresholds=th, refs=refs, extra={})
        asset = Asset(asset_id=name, attempt_id=name, kind="video", path=str(p), sha256="")
        t0 = time.time()
        res = registry.run_stage("video", asset, ctx)
        sec = time.time() - t0
        codes = [r.fail_code for r in res if not r.passed and r.severity == "hard" and r.fail_code]
        soft = [r.fail_code for r in res if not r.passed and r.severity == "soft" and r.fail_code]
        code, route = route_for(codes, routing)
        row = {"clip": name, "loop": refs["loop"], "sec": round(sec, 1), "hard": codes, "soft": soft,
               "route": f"{code}→{route.get('action', '—')}" if codes else "pass",
               "checks": {r.name: {"value": r.value, "passed": r.passed, "sev": r.severity,
                                   "code": r.fail_code, "ev": r.evidence} for r in res}}
        if a.post_dir and refs["loop"] and any(c in ("SEAM",) for c in codes + soft):
            from farm.post import loop as PL
            from farm.qa.video import _frames as F
            fr, fps = F.read_full(p)
            out, rep = PL.repair(fr, (th.get("video") or {}).get("post_loop"))
            dst = F.write(Path(a.post_dir) / f"{name}_loop.mp4", out, fps)
            sub = Asset(asset_id=name + "_fix", attempt_id=name, kind="video", path=str(dst), sha256="")
            res2 = registry.run_stage("video", sub, ctx)
            rep["qa_after"] = {r.name: r.passed for r in res2}
            rep["hard_after"] = [r.fail_code for r in res2 if not r.passed and r.severity == "hard" and r.fail_code]
            row["post_loop"] = rep
        if name in labels:
            row["human"] = labels[name]
        rows.append(row)
        pl = row.get("post_loop")
        print(f"{name:24} {row['route']:22} hard={','.join(codes) or '-':24} soft={','.join(soft) or '-':22} "
              f"{'post: ' + str(pl['seam_before']) + '→' + str(pl['seam_after']) + ' ' + ','.join(pl['hard_after'] or ['ok']) if pl else ''}"
              f" {sec:.0f}s", flush=True)
    lab = [r for r in rows if r.get("human", {}).get("verdict")]
    if lab:
        # «брак» для фермы = hard-отказ (→ seed/постобработка) или мягкий флаг лица (→ к человеку)
        for r in lab:
            r["qa_verdict"] = "bad" if (r["hard"] or "IDENTITY_DIP" in r["soft"]) else "good"
        hit = sum(r["qa_verdict"] == r["human"]["verdict"] for r in lab)
        bad = [r for r in lab if r["human"]["verdict"] == "bad"]
        good = [r for r in lab if r["human"]["verdict"] == "good"]
        print(f"\nсовпадение с разметкой: {hit}/{len(lab)} = {hit / len(lab):.0%}; брак пойман "
              f"{sum(r['qa_verdict'] == 'bad' for r in bad)}/{len(bad)}; ложных отказов "
              f"{sum(r['qa_verdict'] == 'bad' for r in good)}/{len(good)}")
        for r in lab:
            if r["qa_verdict"] != r["human"]["verdict"]:
                print(f"  расхождение {r['clip']}: человек={r['human']['verdict']} "
                      f"({','.join(r['human'].get('codes') or [])}) qa={r['qa_verdict']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
