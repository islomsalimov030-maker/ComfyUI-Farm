"""Состояние пульта (старт/пауза, активные профили) — файл на томе, переживает рестарт пода.

По умолчанию ферма на паузе: после первой установки ничего не запускается само.
Нажали «start» — после Terminate/нового пода продолжит работу без участия человека.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional


class Control:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._state = {"running": False, "profiles": ["*"], "filter": None}
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

    # --- v0.5: фильтр пакета (бот ▶️): какие НОВЫЕ шоты запускать; начатые доделываются ---
    @property
    def filter(self) -> Optional[dict]:
        return self._state.get("filter") or None

    def set_filter(self, flt: Optional[dict]) -> None:
        """flt = {min_priority: int, shot_ids: [..]}; None/{} — снять фильтр."""
        clean = {}
        if flt:
            if flt.get("min_priority") is not None:
                clean["min_priority"] = int(flt["min_priority"])
            if flt.get("shot_ids"):
                clean["shot_ids"] = [str(x) for x in flt["shot_ids"]]
        with self._lock:
            self._state["filter"] = clean or None
            self._save()

    def allows_shot(self, shot: dict) -> bool:
        if not self.allows(shot.get("profile") or ""):
            return False
        f = self.filter
        if not f:
            return True
        if f.get("shot_ids") and shot.get("shot_id") not in f["shot_ids"]:
            return False
        if f.get("min_priority") is not None and int(shot.get("priority") or 0) < f["min_priority"]:
            return False
        return True

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)
