"""Состояние пульта (старт/пауза, активные профили) — файл на томе, переживает рестарт пода.

По умолчанию ферма на паузе: после первой установки ничего не запускается само.
Нажали «start» — после Terminate/нового пода продолжит работу без участия человека.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path


class Control:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._state = {"running": False, "profiles": ["*"]}
        if self.path.exists():
            try:
                self._state.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    @property
    def running(self) -> bool:
        return bool(self._state["running"])

    @property
    def profiles(self) -> list[str]:
        return list(self._state.get("profiles") or ["*"])

    def set_running(self, value: bool) -> None:
        with self._lock:
            self._state["running"] = bool(value)
            self._save()

    def set_profiles(self, profiles: list[str]) -> None:
        with self._lock:
            self._state["profiles"] = list(profiles) or ["*"]
            self._save()

    def allows(self, profile: str) -> bool:
        p = self.profiles
        return "*" in p or profile in p

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)
