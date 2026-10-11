"""Сквозной прогон на мок-ComfyUI: 3 шота «по кругу» с отказами, ретраями, post_fix и ревью."""
import threading
import time
import uuid

from conftest import ScriptedQA, add_shot, visited

from farm import states as S
from farm.db import now


def _review(db, asset_id, verdict="accept", comment=None):
    db.insert("reviews", {"review_id": f"rv_{uuid.uuid4().hex[:8]}", "asset_id": asset_id,
                          "reviewer": "test", "verdict": verdict, "comment": comment,
                          "created_at": now(), "consumed": 0})


def test_three_shots_full_cycle(make_worker, mock):
    calls = {"n": 0}

    def behavior(pid, body):
        farm = (body.get("extra_data") or {}).get("farm") or {}
        if farm.get("shot_id") == "S3" and "OUT video" in str(body["prompt"]) and calls["n"] == 0:
            calls["n"] += 1
            return "error"            # первая видео-генерация S3 падает в ComfyUI → TECH-ретрай
        return "ok"
    mock.behavior = behavior

    qa = ScriptedQA({("S2", "frame"): ["FACE_ID", None],     # кадр: отказ → ретрай → ок
                     ("S2", "video"): ["SEAM"],              # видео: шов → post_fix → ок
                     ("S3", "video"): ["SLOWMO"]})           # видео: всегда slowmo → needs_human
    w = make_worker(qa=qa)
    for sid in ("S1", "S2", "S3"):
        add_shot(w.db, sid)

    w.run_until_idle()

    s1, s2, s3 = (w.db.shot(s) for s in ("S1", "S2", "S3"))
    assert s1["state"] == S.VIDEO_REVIEW
    assert s2["state"] == S.VIDEO_REVIEW
    assert s3["state"] == S.NEEDS_HUMAN

    # S2 прошёл ретрай кадра и post_fix
    v2 = visited(w.db, "S2")
    assert S.FRAME_RETRY in v2 and S.POST_FIX in v2
    fixed = w.db.one("SELECT * FROM assets WHERE asset_id=?", (s2["video_asset_id"],))
    assert fixed["parent_asset_id"] is not None and fixed["candidate"] == 1

    # S3: ретраи ограничены 3 (try 0 — TECH, try 1..3 — SLOWMO)
    tries = w.db.all("SELECT DISTINCT try_no FROM attempts WHERE shot_id='S3' AND stage='video'")
    assert sorted(t["try_no"] for t in tries) == [0, 1, 2, 3]
    assert visited(w.db, "S3").count(S.VIDEO_RETRY) == 3
    assert "исчерпаны ретраи" in s3["last_error"]

    # кадр: 2 сида на попытку, выбран лучший; файлы неизменяемые и с sha256
    fr = w.db.one("SELECT * FROM assets WHERE asset_id=?", (s1["frame_asset_id"],))
    assert fr["candidate"] == 1 and fr["w"] == 64 and len(fr["sha256"]) == 64
    assert w.db.one("SELECT COUNT(*) c FROM attempts WHERE shot_id='S1' AND stage='frame'")["c"] == 2

    # ни одного дубля отправки в ComfyUI
    assert max(mock.submissions.values()) == 1

    # ревью: принять S1 и S2 → approved, выгрузка и запись в таблицу
    _review(w.db, s1["video_asset_id"])
    _review(w.db, s2["video_asset_id"])
    w.run_until_idle()
    assert w.db.shot("S1")["state"] == S.APPROVED
    assert w.db.shot("S2")["state"] == S.APPROVED
    assert {u[0] for u in w.recorder.uploads} == {"S1", "S2"}
    assert any(u[1] == S.APPROVED and u[2] for u in w.recorder.updates)

    # S3 человек отклоняет
    w.reject("S3", "не годится")
    assert w.db.shot("S3")["state"] == S.REJECTED

    # все состояния машины пройдены хотя бы одним шотом
    seen = {S.PENDING} | {r["to_state"] for r in w.db.all("SELECT to_state FROM events")}
    assert seen == set(S.ALL_STATES)

    # бандит набрал статистику по рукам кадра
    assert w.db.one("SELECT SUM(n) n FROM arm_stats")["n"] > 0


def test_frame_review_does_not_block_and_redo_returns(make_worker, mock):
    w = make_worker()
    add_shot(w.db, "R1")
    w.run_until_idle()
    shot = w.db.shot("R1")
    assert shot["state"] == S.VIDEO_REVIEW          # кадр не ждал человека
    first_frame = shot["frame_asset_id"]

    _review(w.db, first_frame, "redo_comment", "руки в кадре")
    w.tick()
    shot = w.db.shot("R1")
    assert shot["state"] in (S.FRAME_GEN, S.FRAME_QA, S.FRAME_REVIEW, S.VIDEO_GEN)
    assert shot["review_comment"] == "руки в кадре"
    w.run_until_idle()
    shot = w.db.shot("R1")
    assert shot["state"] == S.VIDEO_REVIEW and shot["frame_asset_id"] != first_frame
    assert w.db.one("SELECT approved FROM assets WHERE asset_id=?", (first_frame,))["approved"] == 0
    assert w.db.one("SELECT COUNT(*) c FROM attempts WHERE shot_id='R1' AND superseded=1")["c"] >= 3


def test_dependencies_wait_for_approved(make_worker):
    w = make_worker()
    add_shot(w.db, "A")
    add_shot(w.db, "B", depends_on_json=["A"], kind="frame")
    w.run_until_idle()
    assert w.db.shot("B")["state"] == S.PENDING
    _review(w.db, w.db.shot("A")["video_asset_id"])
    w.run_until_idle()
    assert w.db.shot("A")["state"] == S.APPROVED
    assert w.db.shot("B")["state"] == S.FRAME_REVIEW       # kind=frame ждёт человека на кадре
    _review(w.db, w.db.shot("B")["frame_asset_id"])
    w.run_until_idle()
    assert w.db.shot("B")["state"] == S.APPROVED


def test_end_from_skips_frame(make_worker):
    w = make_worker()
    add_shot(w.db, "X")
    w.run_until_idle()
    x = w.db.shot("X")
    last = w.db.one("SELECT a.asset_id FROM assets a JOIN attempts t ON t.attempt_id=a.attempt_id "
                    "WHERE t.shot_id='X' AND a.kind='last_frame'")["asset_id"]
    add_shot(w.db, "Y", depends_on_json=["X"], end_from=last)
    _review(w.db, x["video_asset_id"])
    w.run_until_idle()
    assert w.db.shot("Y")["state"] == S.PENDING     # last_frame ещё не утверждён человеком
    _review(w.db, last)
    w.run_until_idle()
    y = w.db.shot("Y")
    assert y["state"] == S.VIDEO_REVIEW and y["frame_asset_id"] == last
    assert S.FRAME_GEN not in visited(w.db, "Y")


def test_background_thread_runs_and_pauses(make_worker):
    w = make_worker()
    add_shot(w.db, "T1")
    w.start()
    t0 = time.time()
    while w.db.shot("T1")["state"] != S.VIDEO_REVIEW and time.time() - t0 < 20:
        time.sleep(0.05)
    assert w.db.shot("T1")["state"] == S.VIDEO_REVIEW
    w.control.set_running(False)
    add_shot(w.db, "T2")
    time.sleep(0.4)
    assert w.db.shot("T2")["state"] == S.PENDING     # на паузе новое не берём
    assert w.status()["thread_alive"]
    w.stop()
    assert not w.status()["thread_alive"]
