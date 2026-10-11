"""Мок-данные для тестов бота (WS-08): БД фермы заполняется напрямую, без воркера."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mock_comfy import tiny_png

from bot import config as Cfg
from bot.notifier import Notifier
from bot.service import Service
from bot.state import BotState
from bot.store import FarmStore
from farm import states as S
from farm.db import DB, now

REV, OP, STRANGER = 111, 222, 999


class FakeCtl:
    def __init__(self, supports_filter=True, down=False):
        self.calls = []
        self.state = {"running": False, "profiles": ["*"], "filter": None, "thread_alive": True,
                      "attempts_in_flight": 2, "last_error": None}
        self.supports_filter = supports_filter
        self.down = down

    def _check(self):
        if self.down:
            from bot.farmctl import FarmUnavailable
            raise FarmUnavailable("ConnectError: refused")

    def status(self):
        self._check()
        self.calls.append(("status",))
        return self._out()

    def start(self, profiles=None, flt=None, set_filter=False):
        self._check()
        self.calls.append(("start", profiles, flt, set_filter))
        self.state["running"] = True
        if profiles:
            self.state["profiles"] = profiles
        if set_filter:
            self.state["filter"] = flt or None
        return self._out()

    def pause(self):
        self._check()
        self.calls.append(("pause",))
        self.state["running"] = False
        return self._out()

    def _out(self):
        d = dict(self.state)
        if not self.supports_filter:
            d.pop("filter")
        return d


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


def make_cfg(home: Path, **over) -> Cfg.BotConfig:
    base = {"farm_home": str(home), "access_file": None,
            "access": {"reviewers": [REV], "operators": [OP], "notify_chats": [REV, OP]},
            "notify": {"digest_min": 15, "max_cards_per_digest": 2, "urgent_debounce_s": 60}}
    base.update(over)
    return Cfg.load(None, base)


class Env:
    def __init__(self, home: Path, **cfg_over):
        self.home = home
        self.cfg = make_cfg(home, **cfg_over)
        self.db = DB(self.cfg.db_path)
        self.store = FarmStore(self.db)
        self.state = BotState(self.cfg.state_path)
        self.ctl = FakeCtl()
        self.clock = Clock()
        self.svc = Service(self.cfg, self.store, self.state, self.ctl)
        self.notifier = Notifier(self.cfg, self.store, self.state, clock=self.clock)
        self.files = home / "results"
        self.files.mkdir(parents=True, exist_ok=True)

    def reopen_state(self):
        """Имитация рестарта бота: новое соединение к тому же файлу состояния."""
        self.state.close()
        self.state = BotState(self.cfg.state_path)
        self.svc = Service(self.cfg, self.store, self.state, self.ctl)
        self.notifier = Notifier(self.cfg, self.store, self.state, clock=self.clock)

    # ------------------------------------------------------------ мок-данные
    def shot(self, shot_id, state, kind="video", profile="test_sfw", priority=0, **kw):
        self.db.add_shot({"shot_id": shot_id, "profile": profile, "kind": kind, "state": state,
                          "priority": priority, "location": "loft", "character": "stacy", **kw})
        if state == S.NEEDS_HUMAN:
            self.db.insert("events", {"ts": now(), "shot_id": shot_id, "from_state": "video_qa",
                                      "to_state": S.NEEDS_HUMAN, "reason": kw.get("last_error") or "SLOWMO: QA"})

    def asset(self, shot_id, stage="frame", candidate=1, try_no=0, qa_score=0.9, approved=None,
              superseded=0, checks=None, ext=None, data=None):
        n = self.db.one("SELECT COUNT(*) c FROM attempts")["c"]
        att = f"at_{shot_id}_{stage}_{n}"
        self.db.insert("attempts", {"attempt_id": att, "shot_id": shot_id, "stage": stage,
                                    "arm_id": f"test_sfw/{stage}/arm_a", "workflow": "frame_test",
                                    "workflow_ver": "1", "seed": 4242 + n, "status": "done",
                                    "idem_key": hashlib.sha256(att.encode()).hexdigest(), "try_no": try_no,
                                    "superseded": superseded, "started_at": now(),
                                    "params_json": {"model": "/m/model_a.safetensors",
                                                    "loras": [{"name": "stacy_v3.safetensors", "w": 0.8}]}})
        aid = f"as_{shot_id}_{stage}_{n}"
        path = self.files / f"{aid}.{ext or ('png' if stage == 'frame' else 'mp4')}"
        path.write_bytes(data if data is not None else (tiny_png(16, 16, n) if stage == "frame" else b"\0" * 2048))
        self.db.insert("assets", {"asset_id": aid, "attempt_id": att, "kind": stage, "path": str(path),
                                  "sha256": "x" * 64, "candidate": candidate, "approved": approved,
                                  "qa_passed": 1, "qa_score": qa_score, "created_at": now()})
        for name, value, passed, sev, code in checks or [("face_id", 0.82, 1, "hard", None),
                                                         ("seam", 0.31, 0, "soft", "SEAM")]:
            self.db.insert("checks", {"attempt_id": att, "asset_id": aid, "name": name, "value": value,
                                      "passed": passed, "severity": sev, "fail_code": code, "created_at": now()})
        col = "frame_asset_id" if stage == "frame" else "video_asset_id"
        if candidate:
            self.db.update("shots", "shot_id", shot_id, {col: aid})
        return aid

    def reviews(self):
        return self.db.all("SELECT * FROM reviews ORDER BY created_at, review_id")
