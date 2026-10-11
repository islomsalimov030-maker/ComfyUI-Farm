"""Машина состояний шота (CONTRACTS §4).

pending → ready → frame_gen → frame_qa → [frame_retry ≤N] → frame_review* → video_gen → video_qa
        → [video_retry ≤N | post_fix] → video_review* → approved | needs_human | rejected

*review не блокирует: для видео-шотов frame_review сразу переходит в video_gen на лучшем кандидате;
вердикт человека (reviews) возвращает шот на нужную стадию в любой момент (рёбра REDO).
"""
from __future__ import annotations

from typing import Optional

from .db import DB, now

PENDING, READY = "pending", "ready"
FRAME_GEN, FRAME_QA, FRAME_RETRY, FRAME_REVIEW = "frame_gen", "frame_qa", "frame_retry", "frame_review"
VIDEO_GEN, VIDEO_QA, VIDEO_RETRY, POST_FIX, VIDEO_REVIEW = (
    "video_gen", "video_qa", "video_retry", "post_fix", "video_review")
APPROVED, NEEDS_HUMAN, REJECTED = "approved", "needs_human", "rejected"

ALL_STATES = [PENDING, READY, FRAME_GEN, FRAME_QA, FRAME_RETRY, FRAME_REVIEW,
              VIDEO_GEN, VIDEO_QA, VIDEO_RETRY, POST_FIX, VIDEO_REVIEW,
              APPROVED, NEEDS_HUMAN, REJECTED]
TERMINAL = {APPROVED, REJECTED}
FRAME_STATES = {FRAME_GEN, FRAME_QA, FRAME_RETRY, FRAME_REVIEW}
VIDEO_STATES = {VIDEO_GEN, VIDEO_QA, VIDEO_RETRY, POST_FIX, VIDEO_REVIEW}

# Прямой ход конвейера
FORWARD: dict[str, set[str]] = {
    PENDING: {READY},
    READY: {FRAME_GEN, VIDEO_GEN},             # VIDEO_GEN — если кадр задан через end_from
    FRAME_GEN: {FRAME_QA, FRAME_RETRY, NEEDS_HUMAN},
    FRAME_QA: {FRAME_REVIEW, FRAME_RETRY, NEEDS_HUMAN},
    FRAME_RETRY: {FRAME_GEN, NEEDS_HUMAN},
    FRAME_REVIEW: {VIDEO_GEN, APPROVED},       # APPROVED — для kind=frame после «Принять»
    VIDEO_GEN: {VIDEO_QA, VIDEO_RETRY, NEEDS_HUMAN},
    VIDEO_QA: {VIDEO_REVIEW, VIDEO_RETRY, POST_FIX, NEEDS_HUMAN},
    VIDEO_RETRY: {VIDEO_GEN, NEEDS_HUMAN},
    POST_FIX: {VIDEO_QA, VIDEO_RETRY, NEEDS_HUMAN},
    VIDEO_REVIEW: {APPROVED},
    NEEDS_HUMAN: set(),
    APPROVED: set(),
    REJECTED: set(),
}

# Вердикты человека: redo кадра / redo видео / accept после needs_human / reject.
_AFTER_FRAME = {FRAME_REVIEW, *VIDEO_STATES, APPROVED, NEEDS_HUMAN}
REDO: dict[str, set[str]] = {s: set() for s in ALL_STATES}
for s in _AFTER_FRAME | {FRAME_QA, FRAME_RETRY}:
    REDO[s].add(FRAME_GEN)
for s in {VIDEO_REVIEW, VIDEO_QA, VIDEO_RETRY, POST_FIX, APPROVED, NEEDS_HUMAN}:
    REDO[s].add(VIDEO_GEN)
for s in {FRAME_REVIEW, VIDEO_REVIEW, NEEDS_HUMAN}:
    REDO[s] |= {APPROVED, REJECTED}
REDO[NEEDS_HUMAN].add(VIDEO_GEN)
REDO[PENDING].add(REJECTED)
REDO[READY].add(REJECTED)

TRANSITIONS: dict[str, set[str]] = {s: FORWARD[s] | REDO[s] for s in ALL_STATES}


class IllegalTransition(ValueError):
    pass


class StaleState(RuntimeError):
    """Состояние шота изменилось параллельно (например, пришёл redo от бота)."""


def can(src: str, dst: str) -> bool:
    return dst in TRANSITIONS.get(src, set())


def transition(db: DB, shot_id: str, dst: str, reason: str = "",
               expect: Optional[str] = None, **fields) -> str:
    """Атомарный переход с проверкой таблицы и оптимистической блокировкой по текущему состоянию.

    expect — состояние, в котором вызывающий видел шот; если оно уже другое → StaleState.
    fields — доп. колонки shots, обновляемые в той же транзакции.
    """
    with db.tx():
        row = db.one("SELECT state FROM shots WHERE shot_id=?", (shot_id,))
        if row is None:
            raise KeyError(shot_id)
        src = row["state"]
        if expect is not None and src != expect:
            raise StaleState(f"{shot_id}: ожидалось {expect}, сейчас {src}")
        if not can(src, dst):
            raise IllegalTransition(f"{shot_id}: {src} → {dst} запрещён")
        upd = {"state": dst, "updated_at": now(), **fields}
        sets = ", ".join(f"{k}=?" for k in upd)
        db.execute(f"UPDATE shots SET {sets} WHERE shot_id=? AND state=?",
                   [*upd.values(), shot_id, src])
        db.execute("INSERT INTO events(ts, shot_id, from_state, to_state, reason) VALUES (?,?,?,?,?)",
                   (now(), shot_id, src, dst, reason))
    return src


def stage_of(state: str) -> Optional[str]:
    if state in FRAME_STATES:
        return "frame"
    if state in VIDEO_STATES:
        return "video"
    return None
