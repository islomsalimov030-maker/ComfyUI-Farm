"""Выбор следующих шотов: приоритет + группировка по (профиль, стадия, воркфлоу).

Расписание «по моделям, а не по шотам»: держимся текущей группы, пока в ней есть работа,
чтобы не перегружать модели H3/LoRA в VRAM. Переключение группы → POST /free (в воркере).
"""
from __future__ import annotations

from typing import Callable, Optional

from . import states as S
from .db import DB

# состояния, которые воркер двигает сам (pending продвигает promote, review — человек)
RUNNABLE = [S.READY, S.FRAME_GEN, S.FRAME_QA, S.FRAME_RETRY, S.FRAME_REVIEW,
            S.VIDEO_GEN, S.VIDEO_QA, S.VIDEO_RETRY, S.POST_FIX]

# незавершённое — первым (возобновление после рестарта)
_STATE_RANK = {S.FRAME_GEN: 0, S.VIDEO_GEN: 0, S.FRAME_QA: 1, S.VIDEO_QA: 1, S.POST_FIX: 1,
               S.FRAME_RETRY: 2, S.VIDEO_RETRY: 2, S.FRAME_REVIEW: 3, S.READY: 4}


def next_stage(shot: dict) -> str:
    st = shot["state"]
    if st in S.VIDEO_STATES or (st == S.FRAME_REVIEW and shot.get("kind") == "video"):
        return "video"
    if st == S.READY and shot.get("end_from"):
        return "video"
    return "frame"


def group_key(shot: dict, workflow_of: Callable[[dict, str], Optional[str]]) -> tuple:
    stage = next_stage(shot)
    return (shot.get("profile") or "", stage, workflow_of(shot, stage) or "")


def select(db: DB, allows: Callable[[str], bool], workflow_of: Callable[[dict, str], Optional[str]],
           limit: int = 8, sticky: Optional[tuple] = None) -> list[dict]:
    qs = ",".join("?" for _ in RUNNABLE)
    rows = db.all(f"SELECT * FROM shots WHERE state IN ({qs})", RUNNABLE)
    rows = [r for r in rows if allows(r.get("profile") or "")]
    # кадр-шоты в frame_review ждут человека — не работа для воркера
    rows = [r for r in rows if not (r["state"] == S.FRAME_REVIEW and r.get("kind") != "video")]
    if not rows:
        return []

    def key(r):
        g = group_key(r, workflow_of)
        return (_STATE_RANK.get(r["state"], 9) > 1,       # сначала доделываем начатое
                g != sticky if sticky else False,          # затем держимся текущей группы
                -(r.get("priority") or 0),
                g,
                r.get("created_at") or "",
                r["shot_id"])

    rows.sort(key=key)
    return rows[:limit]
