"""Очередь уведомлений: что и когда отправить (без Telegram — тестируется на мок-БД).

scan() кладёт новых кандидатов и «нужен человек» в outbox (идемпотентно по ключу).
plan() решает, что отправить сейчас: срочные — сразу (с антидребезгом), обычные — дайджестом
раз в digest_min. Неотправленное остаётся в outbox и переживает рестарт.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from .config import BotConfig
from .state import BotState, iso, utcnow
from .store import FarmStore


@dataclass
class Plan:
    items: list[dict] = field(default_factory=list)
    header: Optional[str] = None          # текст шапки дайджеста
    skipped: int = 0                      # устаревшие, снятые с очереди без отправки


class Notifier:
    def __init__(self, cfg: BotConfig, store: FarmStore, state: BotState,
                 clock: Callable[[], datetime] = utcnow):
        self.cfg, self.store, self.state, self.clock = cfg, store, state, clock
        n = cfg.notify
        self.digest = timedelta(minutes=float(n.get("digest_min", 15)))
        self.debounce = timedelta(seconds=float(n.get("urgent_debounce_s", 60)))
        self.max_cards = int(n.get("max_cards_per_digest", 10))
        self.urgent = bool(n.get("urgent", True))

    def scan(self) -> int:
        now, added = self.clock(), 0
        for c in self.store.candidates(self.cfg.allows):
            added += self.state.enqueue(f"cand:{c['asset_id']}", "candidate", c["asset_id"], c["shot_id"],
                                        urgent=False, at=now)
        for h in self.store.needs_human(self.cfg.allows):
            added += self.state.enqueue(f"nh:{h['shot_id']}:{h['event_id']}", "needs_human", h["asset_id"],
                                        h["shot_id"], urgent=self.urgent, note=h["reason"][:500], at=now)
        return added

    def _stale(self, item: dict) -> Optional[str]:
        if item["kind"] == "candidate":
            return self.store.decided(item["asset_id"])
        shot = self.store.db.shot(item["shot_id"])
        if shot is None or shot["state"] != "needs_human":
            return "шот уже не ждёт человека"
        ev = self.store.db.one("SELECT MAX(event_id) e FROM events WHERE shot_id=? AND to_state='needs_human'",
                               (item["shot_id"],))
        if ev and ev["e"] is not None and item["key"] != f"nh:{item['shot_id']}:{ev['e']}":
            return "есть более свежее уведомление"
        return None

    def plan(self) -> Plan:
        now, plan = self.clock(), Plan()
        fresh = []
        for it in self.state.unsent():
            why = self._stale(it)
            if why:
                self.state.mark_sent(it["key"], note=f"skip: {why}")
                plan.skipped += 1
            else:
                fresh.append(it)
        urgent = [i for i in fresh if i["urgent"]]
        normal = [i for i in fresh if not i["urgent"]]

        last_u = self.state.get("last_urgent_at")
        if urgent and (last_u is None or now - datetime.fromisoformat(last_u) >= self.debounce):
            plan.items += urgent

        last_d = self.state.get("last_digest_at")
        if normal and (last_d is None or now - datetime.fromisoformat(last_d) >= self.digest):
            batch = normal[: self.max_cards]
            rest = len(normal) - len(batch)
            plan.items += batch
            plan.header = f"📬 Новых кандидатов на ревью: {len(normal)}" + (
                f" (покажу {len(batch)}, ещё {rest} — в следующем дайджесте или /pending)" if rest else "")
        return plan

    def committed(self, plan: Plan) -> None:
        """Отметить, что рассылка по плану состоялась (часы дайджеста/антидребезга)."""
        now = iso(self.clock())
        if any(i["urgent"] for i in plan.items):
            self.state.put("last_urgent_at", now)
        if plan.header:
            self.state.put("last_digest_at", now)
