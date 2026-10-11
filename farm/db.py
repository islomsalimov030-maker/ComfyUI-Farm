"""SQLite-хранилище фермы (CONTRACTS §3). Миграции — только добавлением колонок/индексов.

Одно соединение на процесс под RLock + WAL: бот (отдельный процесс) читает/пишет параллельно.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 2

# (колонка, объявление). PRIMARY KEY допустим только у первой колонки при создании таблицы.
SCHEMA: dict[str, list[tuple[str, str]]] = {
    "shots": [
        ("shot_id", "TEXT PRIMARY KEY"),
        ("sheet_row", "INTEGER"),
        ("profile", "TEXT"),
        ("character", "TEXT"),
        ("location", "TEXT"),
        ("plate", "TEXT"),
        ("outfit", "TEXT"),
        ("objects_json", "TEXT"),
        ("pose_ru", "TEXT"),
        ("action_ru", "TEXT"),
        ("loop", "INTEGER"),
        ("duration_s", "REAL"),
        ("end_from", "TEXT"),
        ("pose_ref", "TEXT"),
        ("motion_ref", "TEXT"),
        ("priority", "INTEGER DEFAULT 0"),
        ("depends_on_json", "TEXT"),
        ("state", "TEXT DEFAULT 'pending'"),
        ("kind", "TEXT DEFAULT 'video'"),
        ("updated_at", "TEXT"),
        # --- v0.2 (WS-02) ---
        ("created_at", "TEXT"),
        ("stage_try", "INTEGER DEFAULT 0"),      # номер попытки на текущей стадии (0 = первая)
        ("frame_asset_id", "TEXT"),              # выбранный кандидат-кадр (ключ для видео)
        ("video_asset_id", "TEXT"),              # выбранный кандидат-видео
        ("review_comment", "TEXT"),              # комментарий человека для следующей попытки
        ("last_error", "TEXT"),
    ],
    "attempts": [
        ("attempt_id", "TEXT PRIMARY KEY"),
        ("shot_id", "TEXT REFERENCES shots(shot_id)"),
        ("stage", "TEXT"),
        ("arm_id", "TEXT"),
        ("workflow", "TEXT"),
        ("workflow_ver", "TEXT"),
        ("params_json", "TEXT"),
        ("prompt_text", "TEXT"),
        ("prompt_ver", "TEXT"),
        ("seed", "INTEGER"),
        ("idem_key", "TEXT"),
        ("comfy_prompt_id", "TEXT"),
        ("status", "TEXT"),                      # queued | running | done | failed
        ("gpu_sec", "REAL"),
        ("started_at", "TEXT"),
        ("finished_at", "TEXT"),
        ("error", "TEXT"),
        # --- v0.2 ---
        ("try_no", "INTEGER DEFAULT 0"),         # номер попытки стадии (0 — первая, дальше ретраи)
        ("superseded", "INTEGER DEFAULT 0"),     # шот ушёл на redo — результат не используется
    ],
    "assets": [
        ("asset_id", "TEXT PRIMARY KEY"),
        ("attempt_id", "TEXT REFERENCES attempts(attempt_id)"),
        ("kind", "TEXT"),
        ("path", "TEXT"),
        ("sha256", "TEXT"),
        ("w", "INTEGER"),
        ("h", "INTEGER"),
        ("frames", "INTEGER"),
        ("approved", "INTEGER"),                 # NULL — нет вердикта, 1 — принят, 0 — отклонён
        # --- v0.2 ---
        ("candidate", "INTEGER DEFAULT 0"),      # 1 — выбран лучшим и отправлен на ревью (бот)
        ("parent_asset_id", "TEXT"),             # результат post_fix ссылается на исходник
        ("qa_passed", "INTEGER"),                # NULL — QA ещё не было; 1/0 — итог hard-проверок
        ("qa_score", "REAL"),                    # 0..1, для выбора лучшего кандидата
        ("fail_codes_json", "TEXT"),
        ("created_at", "TEXT"),
    ],
    "checks": [
        ("attempt_id", "TEXT REFERENCES attempts(attempt_id)"),
        ("name", "TEXT"),
        ("value", "REAL"),
        ("passed", "INTEGER"),
        ("severity", "TEXT"),
        ("fail_code", "TEXT"),
        ("evidence_json", "TEXT"),
        ("checker_ver", "TEXT"),
        # --- v0.2 ---
        ("asset_id", "TEXT"),
        ("created_at", "TEXT"),
    ],
    "reviews": [
        ("review_id", "TEXT PRIMARY KEY"),
        ("asset_id", "TEXT REFERENCES assets(asset_id)"),
        ("reviewer", "TEXT"),
        ("verdict", "TEXT"),                     # accept | redo | redo_comment
        ("comment", "TEXT"),
        ("created_at", "TEXT"),
        ("consumed", "INTEGER DEFAULT 0"),
    ],
    "arms": [
        ("arm_id", "TEXT PRIMARY KEY"),
        ("profile", "TEXT"),
        ("model", "TEXT"),
        ("loras_json", "TEXT"),
        ("sampler_json", "TEXT"),
        ("enabled", "INTEGER DEFAULT 1"),
        # --- v0.2 ---
        ("stage", "TEXT"),
    ],
    "arm_stats": [
        ("arm_id", "TEXT"),
        ("category", "TEXT"),
        ("n", "INTEGER DEFAULT 0"),
        ("successes", "INTEGER DEFAULT 0"),
        ("score_sum", "REAL DEFAULT 0"),
    ],
    # --- v0.2: журнал переходов (аудит, статус для бота) ---
    "events": [
        ("event_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("ts", "TEXT"),
        ("shot_id", "TEXT"),
        ("from_state", "TEXT"),
        ("to_state", "TEXT"),
        ("reason", "TEXT"),
    ],
}

INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_attempts_idem ON attempts(idem_key)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_arm_stats ON arm_stats(arm_id, category)",
    "CREATE INDEX IF NOT EXISTS ix_shots_state ON shots(state)",
    "CREATE INDEX IF NOT EXISTS ix_attempts_shot ON attempts(shot_id, stage, try_no)",
    "CREATE INDEX IF NOT EXISTS ix_assets_attempt ON assets(attempt_id)",
    "CREATE INDEX IF NOT EXISTS ix_checks_attempt ON checks(attempt_id)",
    "CREATE INDEX IF NOT EXISTS ix_reviews_open ON reviews(consumed)",
]

JSON_COLS = {"objects_json", "depends_on_json", "params_json", "fail_codes_json",
             "evidence_json", "loras_json", "sampler_json"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def _strip_for_alter(decl: str) -> str:
    # ALTER TABLE ADD COLUMN не умеет PRIMARY KEY/UNIQUE; REFERENCES с DEFAULT NULL допустим.
    for kw in ("PRIMARY KEY AUTOINCREMENT", "PRIMARY KEY", "UNIQUE"):
        decl = decl.replace(kw, "")
    return decl.strip()


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30,
                                    isolation_level=None)  # autocommit; транзакции явно
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    # --- миграции -------------------------------------------------------------
    def migrate(self) -> None:
        with self.tx():
            for table, cols in SCHEMA.items():
                exists = self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
                if not exists:
                    body = ", ".join(f"{c} {d}" for c, d in cols)
                    self.conn.execute(f"CREATE TABLE {table} ({body})")
                    continue
                have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
                for c, d in cols:
                    if c not in have:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {c} {_strip_for_alter(d)}")
            for ddl in INDEXES:
                self.conn.execute(ddl)
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # --- примитивы ------------------------------------------------------------
    @contextmanager
    def tx(self):
        with self._lock:
            if self.conn.in_transaction:      # вложенная — просто продолжаем внешнюю
                yield self.conn
                return
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def execute(self, sql: str, args: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, tuple(args))

    def all(self, sql: str, args: Iterable = ()) -> list[dict]:
        with self._lock:
            return [self._row(r) for r in self.conn.execute(sql, tuple(args)).fetchall()]

    def one(self, sql: str, args: Iterable = ()) -> Optional[dict]:
        with self._lock:
            r = self.conn.execute(sql, tuple(args)).fetchone()
            return self._row(r) if r else None

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in JSON_COLS & d.keys():
            if isinstance(d[k], str) and d[k]:
                try:
                    d[k] = json.loads(d[k])
                except ValueError:
                    pass
        return d

    def insert(self, table: str, row: dict, or_ignore: bool = False) -> None:
        row = {k: (dumps(v) if k in JSON_COLS and not isinstance(v, (str, type(None))) else v)
               for k, v in row.items()}
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        verb = "INSERT OR IGNORE" if or_ignore else "INSERT"
        self.execute(f"{verb} INTO {table} ({cols}) VALUES ({qs})", row.values())

    def update(self, table: str, key: str, key_val: Any, fields: dict) -> int:
        fields = {k: (dumps(v) if k in JSON_COLS and not isinstance(v, (str, type(None))) else v)
                  for k, v in fields.items()}
        sets = ", ".join(f"{k}=?" for k in fields)
        cur = self.execute(f"UPDATE {table} SET {sets} WHERE {key}=?", [*fields.values(), key_val])
        return cur.rowcount

    # --- удобства -------------------------------------------------------------
    def add_shot(self, shot: dict) -> None:
        """Добавить шот (используется синхронизацией таблицы WS-03 и тестами). Повтор — игнор."""
        row = dict(shot)
        row.setdefault("state", "pending")
        row.setdefault("kind", "video")
        row.setdefault("priority", 0)
        row.setdefault("created_at", now())
        row["updated_at"] = now()
        if isinstance(row.get("loop"), bool):
            row["loop"] = int(row["loop"])
        self.insert("shots", row, or_ignore=True)

    def shot(self, shot_id: str) -> Optional[dict]:
        return self.one("SELECT * FROM shots WHERE shot_id=?", (shot_id,))

    def close(self) -> None:
        with self._lock:
            self.conn.close()
