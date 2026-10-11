"""Общие типы данных между потоками (CONTRACTS §7, §8).

Соседние модули могут импортировать эти классы или возвращать свои объекты
с теми же полями — оркестратор читает их через getattr (утиная типизация).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class Asset:
    asset_id: str
    attempt_id: str
    kind: str                 # frame | video | last_frame | crop
    path: str
    sha256: str
    w: Optional[int] = None
    h: Optional[int] = None
    frames: Optional[int] = None


@dataclass
class QAContext:
    shot: dict
    stage: str                # frame | video
    profile: dict
    attempt: dict
    thresholds: dict = field(default_factory=dict)
    refs: dict = field(default_factory=dict)       # пути к референсам (лицо, плита…)
    extra: dict = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    value: float
    passed: bool
    severity: str = "hard"    # hard | soft | info
    fail_code: Optional[str] = None
    evidence: dict = field(default_factory=dict)
    checker_ver: str = "0"


@dataclass
class LintIssue:
    code: str
    msg: str
    severity: str = "warn"    # error | warn | info


@dataclass
class PromptResult:
    text: str
    fields_json: dict = field(default_factory=dict)
    lint: list = field(default_factory=list)       # list[LintIssue]
    template_ver: str = "stub"


# --- интерфейсы плагинов (реализуют соседние потоки) -------------------------

@runtime_checkable
class QARunner(Protocol):
    def run_stage(self, stage: str, asset: Asset, ctx: QAContext) -> list: ...


@runtime_checkable
class PromptBuilder(Protocol):
    def build(self, stage: str, shot: dict, lib: Any, profile: dict, ctx: Optional[dict] = None) -> PromptResult: ...


@runtime_checkable
class Library(Protocol):
    def resolve_inputs(self, shot: dict, stage: str) -> dict: ...


@runtime_checkable
class Uploader(Protocol):
    def upload_final(self, asset_path: str, shot: dict) -> Optional[str]: ...


@runtime_checkable
class TableSink(Protocol):
    def on_shot_update(self, shot: dict, links: list, scores: dict) -> None: ...


@runtime_checkable
class PostFixer(Protocol):
    def fix(self, asset: Asset, ctx: QAContext) -> str: ...
