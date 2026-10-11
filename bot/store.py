"""Доступ бота к БД фермы (CONTRACTS §3, §9). Бот только читает и пишет в `reviews`.

Кандидат на ревью = `assets.candidate=1 AND approved IS NULL` у неотменённой попытки
плюс лучший ассет шота в `needs_human`. Решённым ассет считается, если `approved` уже задан
или по нему есть непрочитанный оркестратором (`consumed=0`) вердикт.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

from farm.db import DB, now

VERDICTS = ("accept", "redo", "redo_comment")
MAIN_KIND = {"frame": "frame", "video": "video"}
Allows = Callable[[Optional[str]], bool]


def priority_limits() -> tuple[int, int]:
    """Шкала приоритета таблицы (farm/sheet_schema.yaml, WS-03): в таблице 1 — срочно, в БД больше = раньше."""
    try:
        import yaml

        from farm.config import PKG_DIR
        lo, hi = yaml.safe_load((PKG_DIR / "sheet_schema.yaml").read_text(encoding="utf-8"))["limits"]["priority"]
        return int(lo), int(hi)
    except Exception:
        return 1, 3


def priority_sheet_to_db(p: int) -> int:
    lo, hi = priority_limits()
    return lo + hi - int(p)


def priority_db_to_sheet(p: int) -> int:
    lo, hi = priority_limits()
    return lo + hi - int(p)


def review_id_for(bot_name: str, key: str) -> str:
    """Детерминированный id вердикта: повтор того же нажатия/сообщения Telegram не создаёт дубль."""
    return "rv_" + hashlib.sha256(f"tg|{bot_name}|{key}".encode()).hexdigest()[:20]


class FarmStore:
    def __init__(self, db: DB | str | Path):
        self.db = db if isinstance(db, DB) else DB(db)

    # ------------------------------------------------------------ что показывать
    def candidates(self, allows: Allows) -> list[dict]:
        rows = self.db.all(
            "SELECT a.asset_id, a.created_at, s.shot_id, s.profile, s.state, t.stage "
            "FROM assets a JOIN attempts t ON t.attempt_id=a.attempt_id JOIN shots s ON s.shot_id=t.shot_id "
            "WHERE a.candidate=1 AND a.approved IS NULL AND t.superseded=0 AND s.state<>'rejected' "
            "AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id=a.asset_id AND r.consumed=0) "
            "ORDER BY a.created_at, a.asset_id")
        return [r for r in rows if allows(r["profile"])]

    def needs_human(self, allows: Allows) -> list[dict]:
        """Шоты в needs_human: последнее событие (ключ уведомления), причина и лучший ассет стадии."""
        out = []
        for s in self.db.all("SELECT * FROM shots WHERE state='needs_human' ORDER BY updated_at, shot_id"):
            if not allows(s.get("profile")):
                continue
            ev = self.db.one("SELECT event_id, reason FROM events WHERE shot_id=? AND to_state='needs_human' "
                             "ORDER BY event_id DESC LIMIT 1", (s["shot_id"],))
            asset = self.best_asset(s["shot_id"])
            if asset and self.decided(asset["asset_id"]):
                asset = None
            out.append({"shot_id": s["shot_id"], "profile": s.get("profile"),
                        "event_id": ev["event_id"] if ev else 0,
                        "reason": (ev or {}).get("reason") or s.get("last_error") or "",
                        "asset_id": asset["asset_id"] if asset else None})
        return out

    def best_asset(self, shot_id: str) -> Optional[dict]:
        """Лучший основной ассет последней стадии шота (по qa_score), среди неотменённых попыток."""
        last = self.db.one("SELECT stage FROM attempts WHERE shot_id=? AND superseded=0 "
                           "ORDER BY try_no DESC, started_at DESC, attempt_id DESC LIMIT 1", (shot_id,))
        if not last:
            return None
        stage = last["stage"]
        return self.db.one(
            "SELECT a.* FROM assets a JOIN attempts t ON t.attempt_id=a.attempt_id "
            "WHERE t.shot_id=? AND t.stage=? AND t.superseded=0 AND a.kind=? "
            "ORDER BY t.try_no DESC, COALESCE(a.qa_score, -1) DESC, a.created_at DESC LIMIT 1",
            (shot_id, stage, MAIN_KIND.get(stage, stage)))

    # ------------------------------------------------------------ карточка
    def card(self, asset_id: str) -> Optional[dict]:
        a = self.db.one("SELECT * FROM assets WHERE asset_id=?", (asset_id,))
        if a is None:
            return None
        t = self.db.one("SELECT * FROM attempts WHERE attempt_id=?", (a["attempt_id"],)) or {}
        s = self.db.shot(t.get("shot_id")) or {}
        checks = self.db.all("SELECT name, value, passed, severity, fail_code FROM checks WHERE asset_id=? "
                             "ORDER BY passed, CASE severity WHEN 'hard' THEN 0 WHEN 'soft' THEN 1 ELSE 2 END, "
                             "name", (asset_id,))
        if not checks:   # проверки старого формата без asset_id
            checks = self.db.all("SELECT name, value, passed, severity, fail_code FROM checks "
                                 "WHERE attempt_id=? AND asset_id IS NULL ORDER BY passed, name", (a["attempt_id"],))
        params = t.get("params_json") if isinstance(t.get("params_json"), dict) else {}
        arm = {"arm_id": t.get("arm_id"), "model": params.get("model"), "loras": params.get("loras")}
        if not arm["model"] and t.get("arm_id"):
            row = self.db.one("SELECT model, loras_json FROM arms WHERE arm_id=?", (t["arm_id"],))
            if row:
                arm["model"], arm["loras"] = row["model"], row["loras_json"]
        return {"asset": a, "attempt": t, "shot": s, "checks": checks, "arm": arm}

    # ------------------------------------------------------------ вердикты
    def decided(self, asset_id: str) -> Optional[str]:
        """None — можно ревьюить; иначе причина, почему нельзя."""
        a = self.db.one("SELECT a.approved, a.candidate, t.superseded, s.state FROM assets a "
                        "JOIN attempts t ON t.attempt_id=a.attempt_id JOIN shots s ON s.shot_id=t.shot_id "
                        "WHERE a.asset_id=?", (asset_id,))
        if a is None:
            return "ассет не найден"
        if a["approved"] is not None:
            return "уже принят" if a["approved"] else "уже отправлен на переделку"
        r = self.db.one("SELECT verdict, reviewer FROM reviews WHERE asset_id=? AND consumed=0 "
                        "ORDER BY created_at LIMIT 1", (asset_id,))
        if r:
            return f"уже решено ({r['reviewer']}: {r['verdict']}), ждёт оркестратора"
        if a["superseded"]:
            return "кандидат устарел (шот уже переделывается)"
        if a["state"] == "rejected":
            return "шот отклонён"
        if not a["candidate"] and a["state"] != "needs_human":
            return "не кандидат на ревью"
        return None

    def record(self, review_id: str, asset_id: str, reviewer: str, verdict: str,
               comment: Optional[str] = None) -> tuple[bool, str]:
        """Записать вердикт атомарно. (True, '') — записан; (False, причина) — нет."""
        if verdict not in VERDICTS:
            return False, f"неизвестный вердикт {verdict}"
        if verdict == "redo_comment" and not (comment or "").strip():
            return False, "пустой комментарий"
        with self.db.tx():
            if self.db.one("SELECT 1 FROM reviews WHERE review_id=?", (review_id,)):
                return True, "повтор"            # тот же апдейт Telegram пришёл второй раз
            why = self.decided(asset_id)
            if why:
                return False, why
            self.db.insert("reviews", {"review_id": review_id, "asset_id": asset_id, "reviewer": reviewer,
                                       "verdict": verdict, "comment": (comment or "").strip() or None,
                                       "created_at": now(), "consumed": 0})
        return True, ""

    # ------------------------------------------------------------ статус и пакеты
    def state_counts(self, allows: Allows) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.db.all("SELECT profile, state, COUNT(*) c FROM shots GROUP BY profile, state"):
            if allows(r["profile"]):
                out[r["state"]] = out.get(r["state"], 0) + r["c"]
        return out

    def profiles(self, allows: Allows) -> list[str]:
        return [r["profile"] for r in self.db.all("SELECT DISTINCT profile FROM shots WHERE profile IS NOT NULL "
                                                  "ORDER BY profile") if allows(r["profile"])]

    def open_reviews(self) -> int:
        return self.db.one("SELECT COUNT(*) c FROM reviews WHERE consumed=0")["c"]

    def match_batch(self, allows: Allows, flt: dict) -> list[str]:
        """Шоты, которые запустит пакет: ещё не начатые (pending/ready) и подходящие под фильтр."""
        rows = self.db.all("SELECT shot_id, profile, priority FROM shots WHERE state IN ('pending','ready') "
                           "ORDER BY priority DESC, created_at, shot_id")
        profs = flt.get("profiles") or ["*"]
        ids = set(flt.get("shot_ids") or [])
        pmin = flt.get("min_priority")
        out = []
        for r in rows:
            if not allows(r["profile"]):
                continue
            if "*" not in profs and r["profile"] not in profs:
                continue
            if ids and r["shot_id"] not in ids:
                continue
            if pmin is not None and (r["priority"] or 0) < pmin:
                continue
            out.append(r["shot_id"])
        return out


def fmt_loras(loras) -> str:
    if isinstance(loras, str):
        try:
            loras = json.loads(loras)
        except ValueError:
            return loras
    parts = []
    for lo in loras or []:
        if isinstance(lo, dict):
            n = Path(str(lo.get("name") or "?")).stem
            w = lo.get("w", lo.get("weight"))
            parts.append(f"{n}@{w:g}" if isinstance(w, (int, float)) else n)
        else:
            parts.append(Path(str(lo)).stem)
    return " + ".join(parts)
