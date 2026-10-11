"""Подключение модулей соседних потоков через интерфейсы (contracts.py) с заглушками.

Модуль отсутствует → заглушка + предупреждение в лог. Модуль есть, но падает при импорте →
ошибка (не прячем баги соседей за заглушкой).
"""
from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Optional

from .contracts import Asset, CheckResult, PromptResult, QAContext
from .storage import NullUploader

log = logging.getLogger("farm.plugins")


def _try_import(rel: str) -> Optional[Any]:
    pkg = __package__  # "farm" в тестах, "<ComfyUI-Farm>.farm" внутри ComfyUI
    full = f"{pkg}.{rel}"
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as e:
        # отсутствует сам модуль (или его пакет) — норм для заглушки; иначе — баг соседа
        if e.name and (full == e.name or full.startswith(e.name + ".")):
            return None
        raise


# --- заглушки -------------------------------------------------------------------------

class StubQA:
    """Нет QA-реестра (WS-05/06): проверок нет, всё «проходит», решает человек на ревью."""
    is_stub = True

    def run_stage(self, stage: str, asset: Asset, ctx: QAContext) -> list:
        return [CheckResult(name="stub", value=1.0, passed=True, severity="info",
                            evidence={"note": "QA-реестр не подключён"})]


class StubPromptBuilder:
    """Нет сборщика (WS-04): служебный текст-заглушка. Не для прода."""
    is_stub = True

    def build(self, stage: str, shot: dict, lib: Any, profile: dict, ctx: Optional[dict] = None):
        parts = [f"[stub {stage}]", shot.get("character") or "", shot.get("location") or "",
                 shot.get("pose_ru") or "", shot.get("action_ru") or ""]
        if ctx and ctx.get("comment"):
            parts.append(f"note: {ctx['comment']}")
        return PromptResult(text=" | ".join(p for p in parts if p), template_ver="stub")


class StubLibrary:
    is_stub = True

    def resolve_inputs(self, shot: dict, stage: str) -> dict:
        return {}


class NullTableSink:
    is_stub = True

    def on_shot_update(self, shot: dict, links: list, scores: dict) -> None:
        return None


# --- адаптеры -----------------------------------------------------------------------

class _ModuleQA:
    def __init__(self, mod):
        self.mod = mod

    def run_stage(self, stage, asset, ctx):
        return list(self.mod.run_stage(stage, asset, ctx))


class _ModulePromptBuilder:
    def __init__(self, mod):
        self.fn = mod.build
        self._ctx_ok = "ctx" in inspect.signature(self.fn).parameters

    def build(self, stage, shot, lib, profile, ctx=None):
        if self._ctx_ok:
            return self.fn(stage, shot, lib, profile, ctx=ctx)
        return self.fn(stage, shot, lib, profile)


class _FnUploader:
    def __init__(self, fn):
        self.fn = fn

    def upload_final(self, asset_path, shot):
        return self.fn(asset_path, shot)


class _FnSink:
    def __init__(self, fn):
        self.fn = fn

    def on_shot_update(self, shot, links, scores):
        return self.fn(shot, links, scores)


class _FnFixer:
    def __init__(self, fn):
        self.fn = fn

    def fix(self, asset, ctx):
        return self.fn(asset, ctx)


class Plugins:
    """Набор зависимостей воркера. Любое поле можно передать явно (тесты, ручная сборка)."""

    def __init__(self, qa=None, prompts=None, library=None, uploader=None, sink=None,
                 fixers: Optional[dict] = None, autoload: bool = True):
        self.qa = qa or (self._load_qa() if autoload else StubQA())
        self.prompts = prompts or (self._load_prompts() if autoload else StubPromptBuilder())
        self.library = library or (self._load_library() if autoload else StubLibrary())
        self.uploader = uploader or (self._load_uploader() if autoload else NullUploader())
        self.sink = sink or (self._load_sink() if autoload else NullTableSink())
        self._fixers = dict(fixers or {})
        self._autoload = autoload
        for name, obj in self.describe().items():
            if obj == "stub":
                log.warning("плагин %s не найден — работает заглушка", name)

    def describe(self) -> dict:
        return {n: ("stub" if getattr(getattr(self, n), "is_stub", False) else "real")
                for n in ("qa", "prompts", "library", "uploader", "sink")}

    @staticmethod
    def _load_qa():
        mod = _try_import("qa.registry")
        return _ModuleQA(mod) if mod is not None and hasattr(mod, "run_stage") else StubQA()

    @staticmethod
    def _load_prompts():
        mod = _try_import("prompts.builder")
        return _ModulePromptBuilder(mod) if mod is not None and hasattr(mod, "build") else StubPromptBuilder()

    @staticmethod
    def _load_library():
        mod = _try_import("library")
        if mod is not None:
            if hasattr(mod, "get_library"):
                return mod.get_library()
            if hasattr(mod, "resolve_inputs"):
                return mod
        return StubLibrary()

    @staticmethod
    def _load_uploader():
        mod = _try_import("sheets")
        if mod is not None and hasattr(mod, "upload_final"):
            return _FnUploader(mod.upload_final)
        u = NullUploader()
        u.is_stub = True
        return u

    @staticmethod
    def _load_sink():
        mod = _try_import("sheets")
        if mod is not None and hasattr(mod, "on_shot_update"):
            return _FnSink(mod.on_shot_update)
        return NullTableSink()

    def fixer(self, name: str):
        """name из routing.yaml, например 'post.loop' → farm/post/loop.py: fix(asset, ctx) -> path."""
        if name in self._fixers:
            return self._fixers[name]
        if not self._autoload:
            return None
        mod = _try_import(name)
        f = _FnFixer(mod.fix) if mod is not None and hasattr(mod, "fix") else None
        self._fixers[name] = f
        return f
