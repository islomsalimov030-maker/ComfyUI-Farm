"""Собственное состояние бота (отдельный SQLite, схема фермы не трогается).

- `outbox` — очередь уведомлений. Строка удаляется из очереди (sent_at) только после успешной
  отправки, поэтому рестарт бота ничего не теряет (доставка «хотя бы один раз»).
- `cards` — где лежат карточки ассета (чтобы после вердикта снять кнопки во всех чатах).
- `awaiting` — ждём от пользователя текст (комментарий к «✍️» или фильтр пакета).
- `meta` — время последнего дайджеста и т.п.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

DDL = [
    "CREATE TABLE IF NOT EXISTS outbox (key TEXT PRIMARY KEY, kind TEXT, asset_id TEXT, shot_id TEXT, "
    "urgent INTEGER DEFAULT 0, note TEXT, created_at TEXT, sent_at TEXT, tries INTEGER DEFAULT 0, "
    "last_error TEXT)",
    "CREATE INDEX IF NOT EXISTS ix_outbox_open ON outbox(sent_at, urgent, created_at)",
    "CREATE TABLE IF NOT EXISTS cards (asset_id TEXT, chat_id INTEGER, message_id INTEGER, media INTEGER, "
    "PRIMARY KEY(asset_id, chat_id, message_id))",
    "CREATE TABLE IF NOT EXISTS awaiting (user_id INTEGER PRIMARY KEY, kind TEXT, asset_id TEXT, created_at TEXT)",
    "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)",
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


class BotState:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        for d in DDL:
            self.conn.execute(d)

    def _all(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, tuple(args)).fetchall()]

    def _exec(self, sql, args=()):
        with self._lock:
            return self.conn.execute(sql, tuple(args))

    # ------------------------------------------------------------ outbox
    def enqueue(self, key: str, kind: str, asset_id: Optional[str], shot_id: Optional[str],
                urgent: bool = False, note: str = "", at: Optional[datetime] = None) -> bool:
        cur = self._exec("INSERT OR IGNORE INTO outbox(key, kind, asset_id, shot_id, urgent, note, created_at) "
                         "VALUES (?,?,?,?,?,?,?)", (key, kind, asset_id, shot_id, int(urgent), note,
                                                    iso(at or utcnow())))
        return cur.rowcount == 1

    def unsent(self) -> list[dict]:
        return self._all("SELECT * FROM outbox WHERE sent_at IS NULL ORDER BY urgent DESC, created_at, key")

    def mark_sent(self, key: str, note: Optional[str] = None) -> None:
        self._exec("UPDATE outbox SET sent_at=?, last_error=COALESCE(?, last_error) WHERE key=?",
                   (iso(utcnow()), note, key))

    def mark_failed(self, key: str, error: str) -> int:
        self._exec("UPDATE outbox SET tries=tries+1, last_error=? WHERE key=?", (error[:500], key))
        r = self._all("SELECT tries FROM outbox WHERE key=?", (key,))
        return r[0]["tries"] if r else 0

    # ------------------------------------------------------------ карточки
    def add_card(self, asset_id: str, chat_id: int, message_id: int, media: bool) -> None:
        self._exec("INSERT OR IGNORE INTO cards VALUES (?,?,?,?)", (asset_id, chat_id, message_id, int(media)))

    def cards(self, asset_id: str) -> list[dict]:
        return self._all("SELECT * FROM cards WHERE asset_id=?", (asset_id,))

    def drop_cards(self, asset_id: str) -> None:
        self._exec("DELETE FROM cards WHERE asset_id=?", (asset_id,))

    # ------------------------------------------------------------ ожидание текста
    def set_awaiting(self, user_id: int, kind: str, asset_id: Optional[str] = None) -> None:
        self._exec("INSERT OR REPLACE INTO awaiting VALUES (?,?,?,?)", (user_id, kind, asset_id, iso(utcnow())))

    def pop_awaiting(self, user_id: int, ttl_min: float) -> Optional[dict]:
        rows = self._all("SELECT * FROM awaiting WHERE user_id=?", (user_id,))
        self._exec("DELETE FROM awaiting WHERE user_id=?", (user_id,))
        if not rows:
            return None
        r = rows[0]
        if datetime.fromisoformat(r["created_at"]) < utcnow() - timedelta(minutes=ttl_min):
            return None
        return r

    def peek_awaiting(self, user_id: int) -> Optional[dict]:
        rows = self._all("SELECT * FROM awaiting WHERE user_id=?", (user_id,))
        return rows[0] if rows else None

    # ------------------------------------------------------------ meta
    def get(self, k: str) -> Optional[str]:
        r = self._all("SELECT v FROM meta WHERE k=?", (k,))
        return r[0]["v"] if r else None

    def put(self, k: str, v: str) -> None:
        self._exec("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))

    def close(self) -> None:
        with self._lock:
            self.conn.close()
