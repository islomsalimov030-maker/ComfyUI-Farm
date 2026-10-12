"""WS-04: прогон LLM-сборщика на поде — 20 SFW-строк × (кадр, видео) × повторы через Ollama.

Меряет: валидный JSON с первой попытки (критерий ≥ 95 %), итог без ошибок линтера, раунды правок, время.
Запуск на поде (Ollama на 127.0.0.1:11434, модель уже скачана):

    cd /workspace/farm/repo && python tools/ws04_llm_eval.py --model qwen3-vl:8b-instruct --reps 3

Отчёт: $FARM_HOME/logs/ws04_llm_eval_<время>.json и .md (+ примеры промптов). Модель выгружается в конце
(keep_alive 0). Только SFW-строки из tests/fixtures/prompts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tests")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stages", default="frame,video")
    ap.add_argument("--rows", default="", help="ID через запятую; пусто — все 20")
    ap.add_argument("--format", default="schema", choices=["schema", "json"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="ws04_"))
    real_home = Path(os.environ.get("FARM_HOME", "/workspace/farm"))
    os.environ["FARM_HOME"] = str(work)               # справочник и кэш — во временной папке
    from prompts_helpers import PROFILE, make_env     # noqa: E402
    from farm.prompts import builder as B             # noqa: E402
    from farm.prompts.llm import OllamaClient         # noqa: E402

    lib, shots = make_env(work)
    rows = [r for r in (a.rows.split(",") if a.rows else shots) if r]
    prof = {**PROFILE, "llm": {"prompt_model": a.model, "url": a.url, "keep_alive": 0, "format_mode": a.format}}
    client = OllamaClient.from_profile(prof)
    if not client.ping():
        print(f"Ollama не отвечает на {a.url}", file=sys.stderr)
        return 2
    B._clients[(a.model, a.url)] = client

    res, examples = [], {}
    t0 = time.time()
    with client.batch():
        for rep in range(a.reps):
            for sid in rows:
                for st in a.stages.split(","):
                    r = B.build(st, shots[sid], lib, prof, ctx={"try_no": rep, "no_cache": True})
                    m = r.fields_json.get("meta", {})
                    errs = [f"{i.code}: {i.msg}" for i in r.lint if i.severity == "error"]
                    warns = [i.code for i in r.lint if i.severity != "error"]
                    res.append({"id": sid, "stage": st, "rep": rep, "first_json_ok": m.get("llm_first_json_ok"),
                                "attempts": (m.get("llm") or {}).get("attempts"), "lint_rounds": m.get("lint_rounds"),
                                "ms": (m.get("llm") or {}).get("ms"), "errors": errs, "warns": warns})
                    if rep == 0 and not errs:
                        examples[f"{sid}.{st}"] = r.text
                    print(f"{sid:7} {st:5} rep{rep} json1={m.get('llm_first_json_ok')} "
                          f"rounds={m.get('lint_rounds')} err={len(errs)} {(m.get('llm') or {}).get('ms')} ms",
                          flush=True)
    n = len(res)
    first = sum(1 for x in res if x["first_json_ok"])
    clean = sum(1 for x in res if not x["errors"])
    clean0 = sum(1 for x in res if not x["errors"] and x["lint_rounds"] == 0)
    ms = sorted(x["ms"] or 0 for x in res)
    summary = {"model": a.model, "format": a.format, "n": n, "first_try_json_ok": first,
               "first_try_json_ok_pct": round(100 * first / n, 1) if n else 0,
               "lint_clean_final": clean, "lint_clean_without_feedback": clean0,
               "median_ms": ms[len(ms) // 2] if ms else None, "p90_ms": ms[int(len(ms) * 0.9)] if ms else None,
               "wall_s": round(time.time() - t0, 1),
               "error_codes": Counter(e.split(":")[0] for x in res for e in x["errors"]).most_common(),
               "warn_codes": Counter(w for x in res for w in x["warns"]).most_common(),
               "pass": bool(n) and first / n >= 0.95}
    out = Path(a.out) if a.out else real_home / "logs" / f"ws04_llm_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "results": res}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    md = [f"# WS-04 · прогон LLM ({a.model})", "", "```", json.dumps(summary, ensure_ascii=False, indent=1), "```", ""]
    for k in list(examples)[:6]:
        md += [f"## {k}", "", "```text", examples[k].strip(), "```", ""]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"отчёт: {out.with_suffix('.md')}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
