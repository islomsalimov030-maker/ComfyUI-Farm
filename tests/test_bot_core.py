"""WS-08: кнопки бота меняют БД правильно, очередь уведомлений переживает рестарт (без Telegram)."""
import pytest
from bot_helpers import OP, REV, STRANGER, Env

from bot import cards as C
from bot.config import Access, read_env_file
from bot.store import review_id_for
from farm import states as S


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path / "farm")
    yield e
    e.state.close()
    e.db.close()


# ------------------------------------------------------------------ ✅ 🔁 ✍️
def test_accept_writes_review_once(env):
    env.shot("S1", S.VIDEO_REVIEW)
    aid = env.asset("S1", "video")
    r = env.svc.verdict(REV, "islam", aid, "accept", "cb:1")
    assert r.done and r.done["verdict"] == "accept"
    rows = env.reviews()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "accept" and rows[0]["consumed"] == 0 and rows[0]["asset_id"] == aid
    assert rows[0]["reviewer"] == f"islam#{REV}" and rows[0]["review_id"] == review_id_for("main", "cb:1")
    # тот же callback повторно (Telegram прислал апдейт ещё раз) — без дубля, без ошибки
    assert env.svc.verdict(REV, "islam", aid, "accept", "cb:1").done
    # другой человек/другое нажатие — уже решено, ждёт оркестратора
    r2 = env.svc.verdict(OP, "colleague", aid, "redo", "cb:2")
    assert r2.alert and "уже решено" in r2.text
    assert len(env.reviews()) == 1


def test_redo_writes_redo(env):
    env.shot("S1", S.FRAME_REVIEW, kind="frame")
    aid = env.asset("S1", "frame")
    assert env.svc.verdict(REV, "islam", aid, "redo", "cb:9").done
    (r,) = env.reviews()
    assert r["verdict"] == "redo" and r["comment"] is None


def test_redo_with_comment_next_message(env):
    env.shot("S1", S.VIDEO_REVIEW)
    aid = env.asset("S1", "video")
    r = env.svc.ask_comment(REV, aid)
    assert "комментарий" in r.text and not env.reviews()
    env.reopen_state()                                   # бот перезапустился между нажатием и текстом
    r = env.svc.text(REV, "islam", "  движение медленное, убери слоумо  ", "msg:1:10")
    assert r.done and r.done["verdict"] == "redo_comment"
    (row,) = env.reviews()
    assert row["verdict"] == "redo_comment" and row["comment"] == "движение медленное, убери слоумо"
    assert env.state.peek_awaiting(REV) is None
    assert env.svc.text(REV, "islam", "просто текст", "msg:1:11") is None   # больше ничего не ждём


def test_comment_cancel_and_ttl(env):
    env.shot("S1", S.VIDEO_REVIEW)
    aid = env.asset("S1", "video")
    env.svc.ask_comment(REV, aid)
    assert "Отменено" in env.svc.text(REV, "i", "/cancel", "m1").text
    assert env.svc.text(REV, "i", "комментарий", "m2") is None
    env.svc.ask_comment(REV, aid)
    env.state._exec("UPDATE awaiting SET created_at='2000-01-01T00:00:00+00:00'")
    assert "вышло" in env.svc.text(REV, "i", "поздно", "m3").text
    assert env.reviews() == []


def test_menu_buttons_are_not_comments(env):
    env.shot("S1", S.VIDEO_REVIEW)
    env.svc.ask_comment(REV, env.asset("S1", "video"))
    assert env.svc.text(REV, "i", C.BTN_STATUS, "m1") is None
    assert env.state.peek_awaiting(REV) is not None


def test_access_whitelist(env):
    env.shot("S1", S.VIDEO_REVIEW)
    aid = env.asset("S1", "video")
    for r in (env.svc.verdict(STRANGER, "x", aid, "accept", "cb:x"), env.svc.ask_comment(STRANGER, aid),
              env.svc.batch(STRANGER, {}), env.svc.pause(STRANGER), env.svc.status(STRANGER)):
        assert "Нет доступа" in r.text and str(STRANGER) in r.text
    assert env.reviews() == [] and env.ctl.calls == []
    # ревьюер без прав оператора не запускает пакеты
    assert "Нет доступа" in env.svc.batch(REV, {}).text
    # оператор — тоже ревьюер
    assert env.cfg.access.can_review(OP)


def test_not_reviewable_cases(env):
    env.shot("S1", S.VIDEO_GEN)
    old = env.asset("S1", "video", superseded=1)
    done = env.asset("S1", "video", approved=1)
    env.shot("S2", S.VIDEO_REVIEW)
    assert "устарел" in env.svc.verdict(REV, "i", old, "accept", "a").text
    assert "уже принят" in env.svc.verdict(REV, "i", done, "redo", "b").text
    assert "не найден" in env.svc.verdict(REV, "i", "as_nope", "accept", "c").text
    assert "пустой" in env.store.record("rv_x", old, "i", "redo_comment", "  ")[1]
    assert env.reviews() == []


def test_profile_scope_second_bot(tmp_path):
    e = Env(tmp_path / "farm", name="team_b", profiles=["team_b"])
    e.shot("S1", S.VIDEO_REVIEW, profile="test_sfw")
    e.shot("B1", S.VIDEO_REVIEW, profile="team_b")
    a1, b1 = e.asset("S1", "video"), e.asset("B1", "video")
    assert [c["asset_id"] for c in e.store.candidates(e.cfg.allows)] == [b1]
    assert "не в профилях" in e.svc.verdict(REV, "i", a1, "accept", "x").text
    assert e.svc.verdict(REV, "i", b1, "accept", "y").done
    assert e.cfg.state_path.name == "bot_team_b.sqlite"
    assert "вне этого бота" in e.svc.batch(OP, {"profiles": ["test_sfw"]}).text


# ------------------------------------------------------------------ ▶️ ⏸ 📊
def test_batch_filter_and_control(env):
    # в БД больше = раньше (таблица WS-03: 1 — срочно → БД 3)
    env.shot("P1", S.PENDING, priority=3)
    env.shot("P2", S.READY, priority=1)
    env.shot("P3", S.PENDING, priority=3, profile="other")
    env.shot("R1", S.VIDEO_GEN, priority=3)                 # уже идёт — не «новый»
    r = env.svc.batch_text(OP, "профиль=test_sfw приоритет=1")
    assert "Новых шотов к старту: 1 (P1)" in r.text and "приоритет 1" in r.text
    assert env.ctl.calls[-1] == ("start", ["test_sfw"], {"min_priority": 3}, True)
    env.svc.batch_text(OP, "приоритет<=2")
    assert env.ctl.calls[-1][2] == {"min_priority": 2}
    r = env.svc.batch(OP, {})
    assert "Новых шотов к старту: 3" in r.text
    assert env.ctl.calls[-1] == ("start", ["*"], None, True)  # «всё» снимает фильтр
    r = env.svc.batch_text(OP, "id=P2,P3")
    assert env.ctl.calls[-1][2] == {"shot_ids": ["P2", "P3"]}
    assert "P2" in r.text and "P3" in r.text


def test_batch_filter_text_flow(env):
    r = env.svc.ask_batch_filter(OP)
    assert "фильтр" in r.text.lower()
    r = env.svc.text(OP, "op", "мусор", "m1")
    assert "не понят" in r.text and env.state.peek_awaiting(OP)["kind"] == "batch"
    r = env.svc.text(OP, "op", "все", "m2")
    assert "Ферма запущена" in r.text and env.ctl.calls[-1][0] == "start"


def test_batch_warns_old_worker(env):
    env.ctl.supports_filter = False
    assert "не знает фильтр" in env.svc.batch_text(OP, "prio=1").text
    assert "не знает" not in env.svc.batch(OP, {}).text


def test_pause_resume_status(env):
    env.shot("S1", S.VIDEO_REVIEW)
    env.shot("S2", S.NEEDS_HUMAN)
    r = env.svc.pause(OP)
    assert "паузе" in r.text and env.ctl.calls[-1] == ("pause",)
    assert env.svc.resume(OP).text.startswith("▶️")
    assert env.ctl.calls[-1] == ("start", None, None, False)  # без смены профилей и фильтра
    st = env.svc.status(REV).text
    assert "видео на ревью: 1" in st and "нужен человек: 1" in st and "работает" in st
    env.ctl.down = True
    st = env.svc.status(REV).text
    assert "нет связи" in st and "Шотов: 2" in st
    assert "нет связи" in env.svc.pause(OP).text


def test_parse_filter():
    assert C.parse_filter("") == {} and C.parse_filter("Все") == {}
    assert C.parse_filter("profile=a,b prio=1,2 ids=X-1") == {"profiles": ["a", "b"], "priority_max": 2,
                                                              "shot_ids": ["X-1"]}
    assert C.parse_filter("приоритет<=1") == {"priority_max": 1}
    assert C.fmt_filter({"min_priority": 3}) == "приоритет 1" and C.fmt_filter({"min_priority": 2}) == "приоритет 1–2"
    with pytest.raises(ValueError):
        C.parse_filter("цвет=синий")


# ------------------------------------------------------------------ уведомления
def test_digest_and_urgent(env):
    for i in range(3):
        env.shot(f"S{i}", S.VIDEO_REVIEW)
        env.asset(f"S{i}", "video")
    env.shot("H1", S.NEEDS_HUMAN, last_error="исчерпаны ретраи (3): QA: SLOWMO")
    env.asset("H1", "video", candidate=0, qa_score=0.4)
    assert env.notifier.scan() == 4 and env.notifier.scan() == 0          # идемпотентно

    p = env.notifier.plan()
    assert p.items[0]["kind"] == "needs_human" and p.items[0]["urgent"] == 1
    assert len(p.items) == 3 and "ещё 1" in p.header                     # 1 срочный + 2 из 3 в дайджесте
    for it in p.items:
        env.state.mark_sent(it["key"])
    env.notifier.committed(p)

    env.clock.advance(minutes=5)
    env.shot("H2", S.NEEDS_HUMAN)
    env.notifier.scan()
    p = env.notifier.plan()
    assert [i["shot_id"] for i in p.items] == ["H2"] and p.header is None   # срочный — сразу, дайджест ждёт
    env.notifier.committed(p)
    env.state.mark_sent(p.items[0]["key"])

    env.clock.advance(minutes=11)
    p = env.notifier.plan()
    assert [i["shot_id"] for i in p.items] == ["S2"] and p.header


def test_urgent_debounce(env):
    env.shot("H1", S.NEEDS_HUMAN)
    env.notifier.scan()
    p = env.notifier.plan()
    env.notifier.committed(p)
    env.state.mark_sent(p.items[0]["key"])
    env.shot("H2", S.NEEDS_HUMAN)
    env.notifier.scan()
    assert env.notifier.plan().items == []
    env.clock.advance(seconds=61)
    assert [i["shot_id"] for i in env.notifier.plan().items] == ["H2"]


def test_outbox_survives_restart(env):
    env.shot("S1", S.VIDEO_REVIEW)
    env.asset("S1", "video")
    env.shot("H1", S.NEEDS_HUMAN)
    env.notifier.scan()
    first = env.notifier.plan()
    assert len(first.items) == 2
    # бот упал до отправки: ничего не отмечено
    env.reopen_state()
    again = env.notifier.plan()
    assert {i["key"] for i in again.items} == {i["key"] for i in first.items}
    env.state.mark_sent(again.items[0]["key"])
    env.notifier.committed(again)
    env.reopen_state()
    env.clock.advance(minutes=20)
    left = env.notifier.plan()
    assert [i["key"] for i in left.items] == [again.items[1]["key"]]   # отправленное не повторяется
    assert env.notifier.scan() == 0                                     # повторное сканирование — без дублей


def test_stale_items_skipped(env):
    env.shot("S1", S.VIDEO_REVIEW)
    aid = env.asset("S1", "video")
    env.shot("H1", S.NEEDS_HUMAN)
    env.notifier.scan()
    env.svc.verdict(REV, "i", aid, "accept", "cb")               # решили через /pending до дайджеста
    env.db.update("shots", "shot_id", "H1", {"state": S.VIDEO_GEN})
    p = env.notifier.plan()
    assert p.items == [] and p.skipped == 2
    assert env.state.unsent() == []


def test_needs_human_card_points_to_best_asset(env):
    env.shot("H1", S.NEEDS_HUMAN)
    env.asset("H1", "video", candidate=0, qa_score=0.2, try_no=3)
    best = env.asset("H1", "video", candidate=0, qa_score=0.6, try_no=3)
    env.asset("H1", "video", candidate=0, qa_score=0.9, try_no=2)
    (h,) = env.store.needs_human(env.cfg.allows)
    assert h["asset_id"] == best
    assert env.svc.verdict(REV, "i", best, "redo", "cb").done          # из needs_human можно


def test_card_text(env):
    env.shot("S1", S.VIDEO_REVIEW)
    aid = env.asset("S1", "video", try_no=1)
    card = env.store.card(aid)
    t = C.card_text(card)
    assert "S1" in t and "попытка 2" in t and "model_a.safetensors" in t and "stacy_v3@0.8" in t
    assert "❌" not in t and "⚠️SEAM 0.31" in t and "✅face_id 0.82" in t and "Сид: 4242" in t
    assert len(t) < 1024
    assert "&lt;b&gt;" in C.nh_text("<b>", None, "x")


def test_access_and_env_file(tmp_path):
    a = Access.from_dict({"reviewers": [1, "2"]})
    assert a.operators == {1, 2} and a.notify_chats == [1, 2]
    a = Access.from_dict({"reviewers": [1], "operators": [3]})
    assert a.can_review(3) and not a.can_operate(1)
    f = tmp_path / ".env"
    f.write_text('# c\nexport FARM_BOT_TOKEN="1:abc"\nOTHER=x\n')
    assert read_env_file(f)["FARM_BOT_TOKEN"] == "1:abc"
    e = Env(tmp_path / "farm", env_file=str(f))
    assert e.cfg.token(environ={}) == "1:abc"
    assert e.cfg.token(environ={"FARM_BOT_TOKEN": "2:env"}) == "2:env"
    with pytest.raises(RuntimeError):
        Env(tmp_path / "farm2").cfg.token(environ={})
