"""Реестр проверок качества (CONTRACTS §8). Общий для кадров (WS-05) и видео (WS-06).

    @register(stage="frame", name="face_id", cost=1)
    def check(asset: Asset, ctx: QAContext) -> CheckResult: ...

`run_stage(stage, asset, ctx)` запускает проверки стадии по `cost` (дешёвые первыми).
Кадр: после первого hard-отказа дорогие проверки (cost ≥ `stop_cost`) не гоняются — ретрай всё равно
будет. Видео: прогоняются все (маршрут выбирается по приоритету кодов — SEAM не должен прятать CUT).
Проверка упала с исключением → `CheckResult(TECH, severity=info)` и лог — не роняем воркер.
Модули `farm/qa/image/*.py`, `farm/qa/video/*.py` импортируются автоматически (кроме `_*.py`).
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..contracts import Asset, CheckResult, QAContext

log = logging.getLogger("farm.qa")


@dataclass
class Check:
    stage: str
    name: str
    cost: float
    fn: Callable[[Asset, QAContext], Optional[CheckResult]]


_CHECKS: dict[tuple[str, str], Check] = {}
_loaded = False


def register(stage: str, name: str, cost: float = 1):
    def deco(fn):
        _CHECKS[(stage, name)] = Check(stage, name, cost, fn)
        return fn
    return deco


def load_all() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for sub in ("image", "video"):
        try:
            pkg = importlib.import_module(f"{__package__}.{sub}")
        except ModuleNotFoundError:
            continue
        for m in pkgutil.iter_modules(pkg.__path__):
            if not m.name.startswith("_"):
                importlib.import_module(f"{pkg.__name__}.{m.name}")


def checks(stage: str) -> list[Check]:
    load_all()
    return sorted((c for c in _CHECKS.values() if c.stage == stage), key=lambda c: (c.cost, c.name))


def _stage_key(stage: str) -> str:
    return {"image": "frame"}.get(stage, stage)


def run_stage(stage: str, asset: Asset, ctx: QAContext) -> list[CheckResult]:
    stage = _stage_key(stage)
    th = (ctx.thresholds or {}).get(stage) or {}
    enabled = th.get("enabled")             # список имён — запускать только их
    disabled = set(th.get("disabled") or [])
    stop_cost = th.get("stop_cost", 3 if stage == "frame" else None)
    out: list[CheckResult] = []
    hard_fail = False
    for c in checks(stage):
        if (enabled is not None and c.name not in enabled) or c.name in disabled:
            continue
        if hard_fail and stop_cost is not None and c.cost >= stop_cost:
            out.append(CheckResult(c.name, 0.0, True, "info", None, {"skipped": "после hard-отказа"}))
            continue
        t0 = time.perf_counter()
        try:
            r = c.fn(asset, ctx)
        except Exception as e:  # noqa: BLE001 — проверка не должна ронять воркер
            log.exception("проверка %s/%s упала на %s", stage, c.name, asset.path)
            r = CheckResult(c.name, 0.0, False, "info", "TECH", {"error": f"{type(e).__name__}: {e}"})
        if r is None:
            continue
        r.evidence = dict(r.evidence or {})
        r.evidence.setdefault("ms", round((time.perf_counter() - t0) * 1000))
        out.append(r)
        if not r.passed and r.severity == "hard":
            hard_fail = True
    return out
