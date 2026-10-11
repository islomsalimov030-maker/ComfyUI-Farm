"""Логика кнопок бота без Telegram: проверка доступа, запись вердиктов, пульт, статус.

main.py лишь переводит апдейты aiogram в вызовы этих методов и рисует ответы. Тесты гоняют
этот слой на мок-БД (CONTRACTS §9).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from . import cards as C
from .config import BotConfig
from .farmctl import FarmControl, FarmUnavailable
from .state import BotState
from .store import FarmStore, priority_sheet_to_db, review_id_for

log = logging.getLogger("farm.bot")

NO_ACCESS = "⛔ Нет доступа. Ваш id: {uid} — попросите добавить его в whitelist бота."


@dataclass
class Reply:
    text: str
    buttons: Optional[list[list[tuple[str, str]]]] = None
    alert: bool = False                   # для ответа на кнопку: всплывающее окно
    done: Optional[dict] = None           # вердикт записан → снять кнопки с карточек ассета
    menu: bool = False                    # показать постоянное меню
    items: list[dict] = field(default_factory=list)   # карточки к отправке (/pending)


class Service:
    def __init__(self, cfg: BotConfig, store: FarmStore, state: BotState, ctl: Optional[FarmControl]):
        self.cfg, self.store, self.state, self.ctl = cfg, store, state, ctl
        self.ttl = float(cfg.notify.get("comment_ttl_min", 30))

    # ------------------------------------------------------------ доступ
    def _who(self, uid: int, name: Optional[str]) -> str:
        return f"{name or 'tg'}#{uid}"

    def start(self, uid: int) -> Reply:
        a = self.cfg.access
        if not (a.can_review(uid) or a.can_operate(uid)):
            return Reply(NO_ACCESS.format(uid=uid))
        return Reply(f"Бот ревью фермы «{self.cfg.name}». Карточки приходят сами: срочные сразу, "
                     f"остальные дайджестом. Ваш id: {uid}.", menu=True)

    # ------------------------------------------------------------ вердикты
    def verdict(self, uid: int, name: Optional[str], asset_id: str, verdict: str, key: str,
                comment: Optional[str] = None) -> Reply:
        if not self.cfg.access.can_review(uid):
            return Reply(NO_ACCESS.format(uid=uid), alert=True)
        if not self._visible(asset_id):
            return Reply("Этот шот не в профилях этого бота", alert=True)
        who = self._who(uid, name)
        ok, why = self.store.record(review_id_for(self.cfg.name, key), asset_id, who, verdict, comment)
        if not ok:
            return Reply(f"Не записано: {why}", alert=True)
        log.info("вердикт %s %s от %s", verdict, asset_id, who)
        text = {"accept": "✅ Принято", "redo": "🔁 Отправлено на переделку",
                "redo_comment": "✍️ Отправлено на переделку с комментарием"}[verdict]
        return Reply(text, done={"asset_id": asset_id, "verdict": verdict, "who": who, "comment": comment})

    def ask_comment(self, uid: int, asset_id: str) -> Reply:
        if not self.cfg.access.can_review(uid):
            return Reply(NO_ACCESS.format(uid=uid), alert=True)
        why = self.store.decided(asset_id)
        if why:
            return Reply(f"Нельзя: {why}", alert=True)
        self.state.set_awaiting(uid, "comment", asset_id)
        return Reply("✍️ Напишите комментарий одним сообщением — что исправить. /cancel — отмена.")

    def _visible(self, asset_id: str) -> bool:
        r = self.store.db.one("SELECT s.profile FROM assets a JOIN attempts t ON t.attempt_id=a.attempt_id "
                              "JOIN shots s ON s.shot_id=t.shot_id WHERE a.asset_id=?", (asset_id,))
        return r is None or self.cfg.allows(r["profile"])   # нет ассета — ответит store.record

    # ------------------------------------------------------------ свободный текст
    def text(self, uid: int, name: Optional[str], text: str, key: str) -> Optional[Reply]:
        """Ответ на «следующее сообщение» (комментарий или фильтр пакета). None — текст не ожидался."""
        t = (text or "").strip()
        if t.lower() in ("/cancel", "отмена"):
            had = self.state.pop_awaiting(uid, self.ttl)
            return Reply("Отменено." if had else "Нечего отменять.", menu=True)
        if t in (C.BTN_BATCH, C.BTN_STATUS, C.BTN_PAUSE, C.BTN_PENDING) or t.startswith("/"):
            return None                       # кнопки меню и команды — не комментарий
        aw = self.state.peek_awaiting(uid)
        if not aw:
            return None
        aw = self.state.pop_awaiting(uid, self.ttl)
        if not aw:
            return Reply("Время на комментарий вышло — нажмите ✍️ ещё раз.")
        if aw["kind"] == "comment":
            return self.verdict(uid, name, aw["asset_id"], "redo_comment", key, comment=t)
        if aw["kind"] == "batch":
            return self.batch_text(uid, t)
        return None

    # ------------------------------------------------------------ пакеты и пульт
    def batch_menu(self, uid: int) -> Reply:
        if not self.cfg.access.can_operate(uid):
            return Reply(NO_ACCESS.format(uid=uid))
        profs = [p for p in self.store.profiles(self.cfg.allows) if len(("bt:p:" + p).encode()) <= 64]
        rows = [[("Все профили этого бота", "bt:all")]]
        rows += [[(f"Профиль {p}", f"bt:p:{p}")] for p in profs[:20]]
        rows.append([("🔎 Фильтр текстом…", "bt:txt")])
        return Reply("▶️ Что запустить? (стартуют шоты в «ждут»/«готовы»; начатые доделываются)", buttons=rows)

    def ask_batch_filter(self, uid: int) -> Reply:
        if not self.cfg.access.can_operate(uid):
            return Reply(NO_ACCESS.format(uid=uid), alert=True)
        self.state.set_awaiting(uid, "batch")
        return Reply("Напишите фильтр, например:\n<code>профиль=test_sfw приоритет=1</code>"
                     " (1 — срочные, как в таблице)\n"
                     "<code>id=S-01,S-02</code>\n«все» — без фильтра. /cancel — отмена.")

    def batch_text(self, uid: int, text: str) -> Reply:
        try:
            flt = C.parse_filter(text)
        except ValueError as e:
            self.state.set_awaiting(uid, "batch")
            return Reply(f"Фильтр не понят: {e}. Попробуйте ещё раз или /cancel.")
        return self.batch(uid, flt)

    def batch(self, uid: int, flt: dict) -> Reply:
        if not self.cfg.access.can_operate(uid):
            return Reply(NO_ACCESS.format(uid=uid))
        flt = {k: v for k, v in (flt or {}).items() if v not in (None, [], "")}
        if "priority_max" in flt:                 # таблица (1 — срочно) → БД/воркер (больше = раньше)
            flt["min_priority"] = priority_sheet_to_db(flt.pop("priority_max"))
        bot_profs = self.cfg.profiles
        profs = flt.get("profiles") or bot_profs
        if "*" not in bot_profs:
            bad = [p for p in profs if p not in bot_profs]
            if bad or "*" in profs:
                return Reply(f"Профили вне этого бота: {', '.join(bad or ['*'])}")
        matched = self.store.match_batch(self.cfg.allows, {**flt, "profiles": profs})
        worker_flt = {k: flt[k] for k in ("min_priority", "shot_ids") if k in flt} or None
        if self.ctl is None:
            return Reply("Пульт фермы не настроен (farm_url)")
        try:
            st = self.ctl.start(profiles=profs, flt=worker_flt, set_filter=True)
        except FarmUnavailable as e:
            return Reply(f"⚠ Не запущено: нет связи с фермой — {e}")
        note = ""
        if worker_flt and "filter" not in st:
            note = ("\n⚠ Воркер не знает фильтр по приоритету/ID (нужен ng-farm с CONTRACTS v0.5) — "
                    "запущены все шоты выбранных профилей.")
        lst = ", ".join(matched[:15]) + (f" +{len(matched) - 15}" if len(matched) > 15 else "")
        return Reply(f"▶️ Ферма запущена. Фильтр: {C.fmt_filter({**flt, 'profiles': profs})}.\n"
                     f"Новых шотов к старту: {len(matched)}" + (f" ({lst})" if matched else "") + note, menu=True)

    def pause(self, uid: int) -> Reply:
        if not self.cfg.access.can_operate(uid):
            return Reply(NO_ACCESS.format(uid=uid))
        if self.ctl is None:
            return Reply("Пульт фермы не настроен (farm_url)")
        try:
            st = self.ctl.pause()
        except FarmUnavailable as e:
            return Reply(f"⚠ Пауза не поставлена: нет связи с фермой — {e}")
        return Reply(f"⏸ Воркер на паузе. Новые задания не отправляются; в ComfyUI ещё "
                     f"{st.get('attempts_in_flight', 0)} — доделаются.",
                     buttons=[[("▶️ Продолжить как было", "ctl:resume")]])

    def resume(self, uid: int) -> Reply:
        if not self.cfg.access.can_operate(uid):
            return Reply(NO_ACCESS.format(uid=uid), alert=True)
        try:
            self.ctl.start()                  # без profiles/filter — прежние настройки пульта
        except (FarmUnavailable, AttributeError) as e:
            return Reply(f"⚠ Не запущено: {e}", alert=True)
        return Reply("▶️ Продолжаем.")

    def status(self, uid: int) -> Reply:
        a = self.cfg.access
        if not (a.can_review(uid) or a.can_operate(uid)):
            return Reply(NO_ACCESS.format(uid=uid))
        ctl, err = None, None
        if self.ctl is not None:
            try:
                ctl = self.ctl.status()
            except FarmUnavailable as e:
                err = str(e)
        else:
            err = "пульт не настроен"
        counts = self.store.state_counts(self.cfg.allows)
        return Reply(C.status_text(ctl, counts, self.store.open_reviews(), len(self.state.unsent()), err),
                     buttons=[[("⏸ Пауза", "ctl:pause"), ("▶️ Продолжить", "ctl:resume")]]
                     if a.can_operate(uid) else None)

    def pending(self, uid: int) -> Reply:
        """Все открытые кандидаты и «нужен человек» прямо сейчас (мимо дайджеста)."""
        a = self.cfg.access
        if not a.can_review(uid):
            return Reply(NO_ACCESS.format(uid=uid))
        items = [{"kind": "needs_human", "asset_id": h["asset_id"], "shot_id": h["shot_id"],
                  "note": h["reason"]} for h in self.store.needs_human(self.cfg.allows)]
        items += [{"kind": "candidate", "asset_id": c["asset_id"], "shot_id": c["shot_id"], "note": ""}
                  for c in self.store.candidates(self.cfg.allows)]
        if not items:
            return Reply("📥 Ничего не ждёт ревью.")
        return Reply(f"📥 Ждут ревью: {len(items)}", items=items)
