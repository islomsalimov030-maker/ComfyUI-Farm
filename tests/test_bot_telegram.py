"""WS-08: обвязка aiogram на фейковой сессии Telegram + связка «бот → воркер» на мок-ComfyUI."""
import asyncio
from datetime import datetime

import pytest

aiogram = pytest.importorskip("aiogram")

from aiogram import Bot  # noqa: E402
from aiogram.client.session.base import BaseSession  # noqa: E402
from aiogram.types import Chat, Message, Update  # noqa: E402
from bot_helpers import OP, REV, Env  # noqa: E402
from conftest import ScriptedQA, add_shot  # noqa: E402

from bot.main import App  # noqa: E402
from bot.service import Service  # noqa: E402
from bot.state import BotState  # noqa: E402
from bot.store import FarmStore  # noqa: E402
from farm import states as S  # noqa: E402


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.fail = None
        self.n = 100

    async def make_request(self, bot, method, timeout=None):
        if self.fail:
            raise self.fail
        self.calls.append(method)
        if method.__api_method__.startswith("send") and method.__api_method__ != "sendChatAction":
            self.n += 1
            return Message(message_id=self.n, date=datetime.now(), chat=Chat(id=method.chat_id, type="private"))
        return True

    async def stream_content(self, *a, **kw):  # pragma: no cover
        yield b""

    async def close(self):
        pass

    def names(self):
        return [m.__api_method__ for m in self.calls]


@pytest.fixture
def tg(tmp_path):
    env = Env(tmp_path / "farm")
    sess = FakeSession()
    bot = Bot("42:TEST", session=sess)
    app = App(env.cfg, bot, ctl=env.ctl, store=env.store, state=env.state)
    app.notifier.clock = env.clock
    yield env, app, sess
    env.state.close()
    env.db.close()


def cb_update(data, uid=REV, cb_id="cb1"):
    return {"update_id": abs(hash(cb_id)) % 10**6, "callback_query": {
        "id": cb_id, "from": {"id": uid, "is_bot": False, "first_name": "Islam", "username": "islam"},
        "chat_instance": "ci", "data": data,
        "message": {"message_id": 101, "date": 0, "chat": {"id": uid, "type": "private"}, "text": "card"}}}


def msg_update(text, uid=REV, mid=500):
    return {"update_id": mid, "message": {"message_id": mid, "date": 0, "text": text,
                                          "chat": {"id": uid, "type": "private"},
                                          "from": {"id": uid, "is_bot": False, "first_name": "Islam"}}}


async def feed(app, raw):
    await app.dp.feed_update(app.bot, Update.model_validate(raw, context={"bot": app.bot}))


def test_digest_sends_cards_with_buttons(tg):
    env, app, sess = tg
    env.shot("F1", S.FRAME_REVIEW, kind="frame")
    fa = env.asset("F1", "frame")
    env.shot("V1", S.VIDEO_REVIEW)
    va = env.asset("V1", "video")
    plan = asyncio.run(app.notify_once())
    assert len(plan.items) == 2
    names = sess.names()
    assert names.count("sendMessage") == 2                       # шапка дайджеста в 2 чата
    assert names.count("sendPhoto") == 2 and names.count("sendAnimation") == 2
    photo = next(m for m in sess.calls if m.__api_method__ == "sendPhoto")
    assert "F1" in photo.caption and photo.reply_markup.inline_keyboard[0][0].callback_data == f"rv:a:{fa}"
    assert len(env.state.cards(fa)) == 2 and len(env.state.cards(va)) == 2
    assert env.state.unsent() == []


def test_accept_button_updates_db_and_closes_cards(tg):
    env, app, sess = tg
    env.shot("V1", S.VIDEO_REVIEW)
    va = env.asset("V1", "video")
    asyncio.run(app.notify_once())
    sess.calls.clear()
    asyncio.run(feed(app, cb_update(f"rv:a:{va}")))
    (r,) = env.reviews()
    assert r["verdict"] == "accept" and r["asset_id"] == va
    names = sess.names()
    assert names.count("answerCallbackQuery") == 1 and names.count("editMessageCaption") == 2
    edit = next(m for m in sess.calls if m.__api_method__ == "editMessageCaption")
    assert "Принято" in edit.caption and edit.reply_markup is None
    assert env.state.cards(va) == []


def test_comment_button_flow(tg):
    env, app, sess = tg
    env.shot("V1", S.VIDEO_REVIEW)
    va = env.asset("V1", "video")
    asyncio.run(feed(app, cb_update(f"rv:c:{va}", cb_id="c1")))
    ask = next(m for m in sess.calls if m.__api_method__ == "sendMessage")
    assert "комментарий" in ask.text and ask.reply_markup.force_reply
    asyncio.run(feed(app, msg_update("поза не та, сядь глубже")))
    (r,) = env.reviews()
    assert r["verdict"] == "redo_comment" and r["comment"] == "поза не та, сядь глубже"


def test_redo_and_stranger(tg):
    env, app, sess = tg
    env.shot("V1", S.VIDEO_REVIEW)
    va = env.asset("V1", "video")
    asyncio.run(feed(app, cb_update(f"rv:r:{va}", uid=777, cb_id="x")))
    assert env.reviews() == []
    ans = next(m for m in sess.calls if m.__api_method__ == "answerCallbackQuery")
    assert ans.show_alert and "Нет доступа" in ans.text
    asyncio.run(feed(app, cb_update(f"rv:r:{va}", cb_id="y")))
    assert env.reviews()[0]["verdict"] == "redo"


def test_menu_status_batch_pause(tg):
    env, app, sess = tg
    env.shot("P1", S.PENDING, priority=3)
    asyncio.run(feed(app, msg_update("📊 Статус очереди", uid=OP, mid=1)))
    assert "Статус фермы" in sess.calls[-1].text
    asyncio.run(feed(app, msg_update("/batch приоритет=1", uid=OP, mid=2)))
    assert env.ctl.calls[-1] == ("start", ["*"], {"min_priority": 3}, True)
    assert "P1" in sess.calls[-1].text
    asyncio.run(feed(app, cb_update("bt:p:test_sfw", uid=OP, cb_id="b1")))
    assert env.ctl.calls[-1] == ("start", ["test_sfw"], None, True)
    asyncio.run(feed(app, msg_update("⏸ Пауза", uid=OP, mid=3)))
    assert env.ctl.calls[-1] == ("pause",)
    asyncio.run(feed(app, cb_update("ctl:resume", uid=OP, cb_id="r1")))
    assert env.ctl.calls[-1] == ("start", None, None, False)


def test_pending_command_sends_cards_here(tg):
    env, app, sess = tg
    env.shot("V1", S.VIDEO_REVIEW)
    env.asset("V1", "video")
    env.shot("H1", S.NEEDS_HUMAN, last_error="node_errors: граф не валиден")   # без ассета
    asyncio.run(feed(app, msg_update("/pending", uid=REV, mid=9)))
    texts = [getattr(m, "text", None) or getattr(m, "caption", "") for m in sess.calls]
    assert any("Ждут ревью: 2" in t for t in texts)
    assert any("Нужен человек" in t and "H1" in t for t in texts)
    assert all(getattr(m, "chat_id", REV) == REV for m in sess.calls)


def test_send_failure_keeps_queue_across_restart(tg, tmp_path):
    env, app, sess = tg
    env.shot("V1", S.VIDEO_REVIEW)
    va = env.asset("V1", "video")
    sess.fail = RuntimeError("network down")
    asyncio.run(app.notify_once())
    (it,) = env.state.unsent()
    assert it["tries"] == 1 and "network" in it["last_error"]
    # рестарт: новое состояние, новый бот, сеть вернулась
    env.state.close()
    state2 = BotState(env.cfg.state_path)
    sess2 = FakeSession()
    app2 = App(env.cfg, Bot("42:TEST", session=sess2), ctl=env.ctl, store=FarmStore(env.db), state=state2)
    app2.notifier.clock = env.clock
    env.clock.advance(minutes=20)
    asyncio.run(app2.notify_once())
    assert state2.unsent() == [] and len(state2.cards(va)) == 2
    assert sess2.names().count("sendAnimation") == 2
    env.state = state2


def test_partial_failure_no_duplicates(tg):
    env, app, sess = tg
    env.shot("V1", S.VIDEO_REVIEW)
    va = env.asset("V1", "video")
    orig = sess.make_request
    seen = {"n": 0}

    async def flaky(bot, method, timeout=None):
        if method.__api_method__ == "sendAnimation":
            seen["n"] += 1
            if seen["n"] == 2:                       # второй чат упал
                raise RuntimeError("boom")
        return await orig(bot, method, timeout)
    sess.make_request = flaky
    asyncio.run(app.notify_once())
    assert len(env.state.cards(va)) == 1 and env.state.unsent()
    env.clock.advance(minutes=20)
    asyncio.run(app.notify_once())
    assert len(env.state.cards(va)) == 2 and not env.state.unsent()
    assert sess.names().count("sendAnimation") == 2   # первый чат не получил дубль


def test_big_video_is_compressed(tg):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("нет ffmpeg")
    import subprocess
    env, app, sess = tg
    env.cfg.preview["video_max_mb"] = 0.05
    env.shot("V1", S.VIDEO_REVIEW)
    big = env.home / "big.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=2:r=24",
                    "-c:v", "libx264", "-crf", "5", "-pix_fmt", "yuv420p", str(big)], check=True)
    aid = env.asset("V1", "video", data=big.read_bytes())
    asyncio.run(app.notify_once())
    anim = next(m for m in sess.calls if m.__api_method__ == "sendAnimation")
    assert "пережато" in anim.caption and str(env.cfg.cache_dir) in anim.animation.path
    assert aid in anim.animation.path


# ------------------------------------------------------------------ бот → воркер (CONTRACTS §3, §4, §9)
def _bot_for(worker, tmp_path):
    from bot_helpers import FakeCtl, make_cfg
    cfg = make_cfg(tmp_path / "botfarm", db_path=str(worker.paths.db))
    store = FarmStore(worker.db)
    return Service(cfg, store, BotState(cfg.state_path), FakeCtl()), store


def test_bot_verdicts_drive_worker(make_worker, tmp_path):
    qa = ScriptedQA({("S3", "video"): ["SLOWMO"]})
    w = make_worker(qa=qa)
    for sid in ("S1", "S2", "S3"):
        add_shot(w.db, sid)
    w.run_until_idle()
    svc, store = _bot_for(w, tmp_path)
    s1, s2 = w.db.shot("S1"), w.db.shot("S2")
    assert s1["state"] == S.VIDEO_REVIEW and w.db.shot("S3")["state"] == S.NEEDS_HUMAN

    cands = {c["shot_id"]: c["asset_id"] for c in store.candidates(svc.cfg.allows) if c["stage"] == "video"}
    assert svc.verdict(REV, "i", cands["S1"], "accept", "k1").done
    svc.ask_comment(REV, cands["S2"])
    assert svc.text(REV, "i", "медленно, ускорь", "k2").done
    (h,) = store.needs_human(svc.cfg.allows)
    assert h["shot_id"] == "S3" and h["asset_id"]
    assert svc.verdict(REV, "i", h["asset_id"], "redo", "k3").done

    w.consume_reviews()
    assert w.db.shot("S1")["state"] == S.APPROVED
    s2 = w.db.shot("S2")
    assert s2["state"] == S.VIDEO_GEN and s2["review_comment"] == "медленно, ускорь" and s2["stage_try"] == 0
    s3 = w.db.shot("S3")
    assert s3["state"] == S.VIDEO_GEN and s3["stage_try"] == 0      # v0.5: redo из needs_human
    assert w.db.one("SELECT COUNT(*) c FROM reviews WHERE consumed=0")["c"] == 0
    # после применения кнопки снова неактивны
    assert "уже" in svc.verdict(REV, "i", cands["S1"], "redo", "k4").text


def test_bot_accept_from_needs_human(make_worker, tmp_path):
    w = make_worker(qa=ScriptedQA({("S1", "frame"): ["FACE_ID"]}))
    add_shot(w.db, "S1")
    w.run_until_idle()
    assert w.db.shot("S1")["state"] == S.NEEDS_HUMAN
    svc, store = _bot_for(w, tmp_path)
    (h,) = store.needs_human(svc.cfg.allows)
    assert svc.verdict(REV, "i", h["asset_id"], "accept", "k").done
    w.consume_reviews()
    s = w.db.shot("S1")
    assert s["state"] == S.VIDEO_GEN and s["frame_asset_id"] == h["asset_id"]


def test_worker_batch_filter(make_worker):
    w = make_worker()
    for sid, prio in (("A", 1), ("B", 5), ("C", 9)):
        add_shot(w.db, sid, priority=prio)
    w.control.set_filter({"shot_ids": ["B"]})
    w.run_until_idle()
    assert w.db.shot("B")["state"] == S.VIDEO_REVIEW
    assert w.db.shot("A")["state"] == w.db.shot("C")["state"] == S.PENDING
    w.control.set_filter({"min_priority": 9})
    w.run_until_idle()
    assert w.db.shot("C")["state"] == S.VIDEO_REVIEW and w.db.shot("A")["state"] == S.PENDING
    w.control.set_filter(None)
    w.run_until_idle()
    assert w.db.shot("A")["state"] == S.VIDEO_REVIEW
    assert w.status()["filter"] is None


def test_runtime_control_filter_semantics(monkeypatch, make_worker):
    from farm import runtime
    w = make_worker()
    monkeypatch.setattr(runtime, "_worker", w)
    monkeypatch.setattr(w, "start", lambda: None)
    st = runtime.control("start", ["test_sfw"], {"min_priority": 2})
    assert st["filter"] == {"min_priority": 2} and st["profiles"] == ["test_sfw"]
    assert runtime.control("start")["filter"] == {"min_priority": 2}      # ключа нет — не меняется
    assert runtime.control("start", None, None)["filter"] is None         # null — снять
    assert runtime.control("pause")["running"] is False
