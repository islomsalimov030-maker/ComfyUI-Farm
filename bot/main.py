"""Telegram-бот асинхронного ревью (WS-08): aiogram 3 (MIT), long polling — входящих портов нет.

Запуск: python -m bot --config /workspace/farm/secrets/bot_main.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (BotCommand, CallbackQuery, ForceReply, FSInputFile, InlineKeyboardButton,
                           InlineKeyboardMarkup, KeyboardButton, Message, ReplyKeyboardMarkup)

from . import cards as C
from . import config as Cfg
from . import media
from .farmctl import FarmControl
from .notifier import Notifier, Plan
from .service import Reply, Service
from .state import BotState
from .store import FarmStore

log = logging.getLogger("farm.bot")
CAPTION_MAX = 1024


def inline(rows) -> Optional[InlineKeyboardMarkup]:
    if not rows:
        return None
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=d) for t, d in r]
                                                 for r in rows])


MENU_KB = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text=t) for t in row] for row in C.MENU],
                              resize_keyboard=True, is_persistent=True)


class App:
    def __init__(self, cfg: Cfg.BotConfig, bot: Bot, ctl: Optional[FarmControl] = None,
                 store: Optional[FarmStore] = None, state: Optional[BotState] = None):
        self.cfg, self.bot = cfg, bot
        self.store = store or FarmStore(cfg.db_path)
        self.state = state or BotState(cfg.state_path)
        self.ctl = ctl if ctl is not None else FarmControl(cfg.farm_url)
        self.svc = Service(cfg, self.store, self.state, self.ctl)
        self.notifier = Notifier(cfg, self.store, self.state)
        self.dp = Dispatcher()
        self.dp.include_router(self._router())
        self.max_tries = int(cfg.notify.get("max_send_tries", 5))

    # ================================================================ отправка карточек
    async def send_item(self, chat_id: int, item: dict, text_only: bool = False) -> None:
        aid = item.get("asset_id")
        if not aid:
            shot = await asyncio.to_thread(self.store.db.shot, item["shot_id"]) or {}
            await self.bot.send_message(chat_id, C.nh_text(item["shot_id"], shot.get("profile"),
                                                           item.get("note") or "", shot.get("last_error")))
            return
        card = await asyncio.to_thread(self.store.card, aid)
        if card is None:
            return
        text = C.card_text(card, item.get("kind", "candidate"), item.get("note") or "")
        kb = inline(C.review_buttons(aid))
        pv = media.Preview("none")
        if not text_only:
            pv = await asyncio.to_thread(media.prepare, card["asset"]["path"], card["asset"]["kind"],
                                         self.cfg.preview, self.cfg.cache_dir, aid)
        if pv.note and pv.method != "none":
            text += f"\n<i>превью: {pv.note}</i>"
        elif pv.method == "none" and pv.note:
            text += f"\n<i>без превью: {pv.note}</i>"
        senders = {"photo": self.bot.send_photo, "animation": self.bot.send_animation,
                   "video": self.bot.send_video, "document": self.bot.send_document}
        if pv.method in senders:
            arg = {"photo": "photo", "animation": "animation", "video": "video", "document": "document"}[pv.method]
            if len(text) <= CAPTION_MAX:
                msg = await senders[pv.method](chat_id, **{arg: FSInputFile(pv.path)}, caption=text, reply_markup=kb)
                self.state.add_card(aid, chat_id, msg.message_id, True)
                return
            await senders[pv.method](chat_id, **{arg: FSInputFile(pv.path)})
        msg = await self.bot.send_message(chat_id, text, reply_markup=kb)
        self.state.add_card(aid, chat_id, msg.message_id, False)

    async def deliver(self, item: dict) -> bool:
        """Отправить пункт outbox во все чаты уведомлений. True — доставлено (или безнадёжно, но отмечено)."""
        done_chats = {c["chat_id"] for c in self.state.cards(item["asset_id"])} if item.get("asset_id") else set()
        text_only = (item.get("tries") or 0) >= self.max_tries - 1
        try:
            for chat in self.cfg.access.notify_chats:
                if chat in done_chats:
                    continue                      # повтор после частичного сбоя — без дублей
                await self.send_item(chat, item, text_only=text_only)
        except TelegramRetryAfter as e:
            log.warning("Telegram просит подождать %s с", e.retry_after)
            await asyncio.sleep(e.retry_after)
            return False
        except Exception as e:
            tries = self.state.mark_failed(item["key"], f"{type(e).__name__}: {e}")
            log.warning("не отправлено %s (попытка %s): %s", item["key"], tries, e)
            if tries >= self.max_tries:
                self.state.mark_sent(item["key"], note=f"gave up: {e}")
                return True
            return False
        self.state.mark_sent(item["key"])
        return True

    async def notify_once(self) -> Plan:
        await asyncio.to_thread(self.notifier.scan)
        plan = await asyncio.to_thread(self.notifier.plan)
        if not plan.items:
            return plan
        if plan.header:
            for chat in self.cfg.access.notify_chats:
                try:
                    await self.bot.send_message(chat, plan.header)
                except Exception as e:
                    log.warning("шапка дайджеста не ушла в %s: %s", chat, e)
        for it in plan.items:
            await self.deliver(it)
        self.notifier.committed(plan)
        return plan

    async def notifier_loop(self) -> None:
        period = float(self.cfg.notify.get("scan_s", 30))
        while True:
            try:
                await self.notify_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("цикл уведомлений")
            await asyncio.sleep(period)

    async def close_cards(self, done: dict) -> None:
        """После вердикта: в каждой копии карточки — строка вердикта, кнопки убрать."""
        aid = done["asset_id"]
        card = await asyncio.to_thread(self.store.card, aid)
        suffix = C.verdict_suffix(done["verdict"], done["who"], done.get("comment"))
        base = C.card_text(card) if card else ""
        for c in self.state.cards(aid):
            try:
                if c["media"] and len(base + suffix) <= CAPTION_MAX:
                    await self.bot.edit_message_caption(chat_id=c["chat_id"], message_id=c["message_id"],
                                                        caption=base + suffix, reply_markup=None)
                elif c["media"]:
                    await self.bot.edit_message_reply_markup(chat_id=c["chat_id"], message_id=c["message_id"],
                                                             reply_markup=None)
                else:
                    await self.bot.edit_message_text(text=base + suffix, chat_id=c["chat_id"],
                                                     message_id=c["message_id"], reply_markup=None)
            except TelegramBadRequest as e:
                log.info("карточку %s/%s не отредактировать: %s", c["chat_id"], c["message_id"], e)
        self.state.drop_cards(aid)

    # ================================================================ ответы
    async def reply(self, m: Message, r: Reply) -> None:
        kb = inline(r.buttons) if r.buttons else (MENU_KB if r.menu else None)
        await m.answer(r.text, reply_markup=kb)
        for it in r.items:
            await self.send_item(m.chat.id, it)
        if r.done:
            await self.close_cards(r.done)

    async def answer_cb(self, cb: CallbackQuery, r: Reply, as_message: bool = False) -> None:
        try:
            await cb.answer(r.text[:200] if not as_message else None, show_alert=r.alert)
        except TelegramBadRequest:
            pass                                  # старый callback после рестарта — уже не ответить
        if as_message and cb.message:
            kb = inline(r.buttons) if r.buttons else ForceReply(selective=True)
            await self.bot.send_message(cb.message.chat.id, r.text, reply_markup=kb)
        if r.done:
            await self.close_cards(r.done)

    def _router(self) -> Router:
        rt = Router()
        svc = self.svc

        def uname(u) -> str:
            return u.username or u.first_name or "tg"

        @rt.message(CommandStart())
        @rt.message(Command("help"))
        async def _start(m: Message):
            await self.reply(m, svc.start(m.from_user.id))

        @rt.message(Command("whoami"))
        async def _whoami(m: Message):
            await m.answer(f"user_id: <code>{m.from_user.id}</code>, chat_id: <code>{m.chat.id}</code>")

        @rt.message(Command("status"))
        @rt.message(F.text == C.BTN_STATUS)
        async def _status(m: Message):
            await self.reply(m, await asyncio.to_thread(svc.status, m.from_user.id))

        @rt.message(Command("pending"))
        @rt.message(F.text == C.BTN_PENDING)
        async def _pending(m: Message):
            await self.reply(m, await asyncio.to_thread(svc.pending, m.from_user.id))

        @rt.message(Command("pause"))
        @rt.message(F.text == C.BTN_PAUSE)
        async def _pause(m: Message):
            await self.reply(m, await asyncio.to_thread(svc.pause, m.from_user.id))

        @rt.message(Command("batch"))
        async def _batch_cmd(m: Message, command: CommandObject):
            if command.args:
                r = await asyncio.to_thread(svc.batch_text, m.from_user.id, command.args)
            else:
                r = svc.batch_menu(m.from_user.id)
            await self.reply(m, r)

        @rt.message(F.text == C.BTN_BATCH)
        async def _batch_btn(m: Message):
            await self.reply(m, await asyncio.to_thread(svc.batch_menu, m.from_user.id))

        @rt.callback_query(F.data.startswith("rv:"))
        async def _review(cb: CallbackQuery):
            _, code, aid = cb.data.split(":", 2)
            uid, who = cb.from_user.id, uname(cb.from_user)
            if code == "c":
                r = await asyncio.to_thread(svc.ask_comment, uid, aid)
                await self.answer_cb(cb, r, as_message=not r.alert)
                return
            verdict = {"a": "accept", "r": "redo"}.get(code)
            if verdict is None:
                await cb.answer("?")
                return
            r = await asyncio.to_thread(svc.verdict, uid, who, aid, verdict, f"cb:{cb.id}")
            await self.answer_cb(cb, r)

        @rt.callback_query(F.data.startswith("bt:"))
        async def _batch_cb(cb: CallbackQuery):
            uid = cb.from_user.id
            if cb.data == "bt:txt":
                r = svc.ask_batch_filter(uid)
                await self.answer_cb(cb, r, as_message=not r.alert)
                return
            flt = {"profiles": [cb.data[5:]]} if cb.data.startswith("bt:p:") else {}
            r = await asyncio.to_thread(svc.batch, uid, flt)
            await self.answer_cb(cb, r, as_message=True)

        @rt.callback_query(F.data.startswith("ctl:"))
        async def _ctl_cb(cb: CallbackQuery):
            uid = cb.from_user.id
            r = await asyncio.to_thread(svc.pause if cb.data == "ctl:pause" else svc.resume, uid)
            await self.answer_cb(cb, r, as_message=not r.alert)

        @rt.message(Command("cancel"))
        @rt.message(F.text)
        async def _text(m: Message):
            r = await asyncio.to_thread(svc.text, m.from_user.id, uname(m.from_user), m.text,
                                        f"msg:{m.chat.id}:{m.message_id}")
            if r is None:
                if m.chat.type == "private":
                    await m.answer("Не понял. Кнопки меню ниже, /help.", reply_markup=MENU_KB)
                return
            await self.reply(m, r)

        return rt

    # ================================================================ запуск
    async def run(self) -> None:
        await self.bot.set_my_commands([
            BotCommand(command="status", description="📊 Статус очереди"),
            BotCommand(command="pending", description="📥 Что ждёт ревью"),
            BotCommand(command="batch", description="▶️ Запустить пакет [фильтр]"),
            BotCommand(command="pause", description="⏸ Пауза воркера"),
            BotCommand(command="cancel", description="Отменить ввод"),
            BotCommand(command="whoami", description="Мой id"),
        ])
        loop = asyncio.create_task(self.notifier_loop(), name="notifier")
        try:
            await self.dp.start_polling(self.bot, allowed_updates=self.dp.resolve_used_update_types(),
                                        handle_signals=True)
        finally:
            loop.cancel()


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Бот ревью фермы Next Genium")
    ap.add_argument("--config", help="YAML бота (по умолчанию $FARM_HOME/secrets/bot_main.yaml, если есть)")
    ap.add_argument("--check", action="store_true", help="проверить конфиг, БД и пульт без запуска бота")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = args.config
    if not path:
        import os
        d = Path(os.environ.get("FARM_HOME", "/workspace/farm")) / "secrets" / "bot_main.yaml"
        path = str(d) if d.exists() else None
    cfg = Cfg.load(path)
    if not cfg.access.reviewers:
        log.warning("whitelist пуст — бот всем ответит «нет доступа». Узнать id: /whoami")
    if args.check:
        store = FarmStore(cfg.db_path)
        print("БД:", cfg.db_path, store.state_counts(cfg.allows))
        print("Состояние бота:", cfg.state_path)
        print("Доступ: reviewers", sorted(cfg.access.reviewers), "operators", sorted(cfg.access.operators),
              "чаты", cfg.access.notify_chats)
        try:
            print("Пульт:", FarmControl(cfg.farm_url).status().get("running"))
        except Exception as e:
            print("Пульт недоступен:", e)
        try:
            cfg.token()
            print("Токен: найден")
        except RuntimeError as e:
            print("Токен:", e)
        return
    bot = Bot(cfg.token(), default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    app = App(cfg, bot)
    log.info("бот «%s»: БД %s, профили %s", cfg.name, cfg.db_path, cfg.profiles)
    asyncio.run(app.run())
