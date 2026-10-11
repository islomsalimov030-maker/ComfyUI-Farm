"""Рестарты, идемпотентность, таймауты, ошибки графа, OOM, VRAM-гейт."""
import threading
import time

from conftest import add_shot, visited

from farm import states as S
from farm.db import DB


def _wait(pred, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_worker_restart_mid_run_loses_nothing(make_worker, mock, farm_home):
    mock.exec_s = 0.3
    w1 = make_worker()
    for sid in ("A", "B", "C"):
        add_shot(w1.db, sid)
    w1.start()
    assert _wait(lambda: w1.db.one("SELECT COUNT(*) c FROM attempts WHERE status='running'")["c"] > 0)
    w1.stop(timeout=5)                       # «процесс умер» посреди ожидания ComfyUI
    assert not w1.status()["thread_alive"]
    w1.db.close()

    db2 = DB(farm_home / "db" / "farm.sqlite")   # новый процесс — новое соединение
    w2 = make_worker(db=db2)
    w2.run_until_idle()
    for sid in ("A", "B", "C"):
        assert db2.shot(sid)["state"] == S.VIDEO_REVIEW, sid
        # 2 сида кадра + 1 видео, без лишних попыток
        assert db2.one("SELECT COUNT(*) c FROM attempts WHERE shot_id=?", (sid,))["c"] == 3
    assert max(mock.submissions.values()) == 1, "задание отправлено в ComfyUI дважды"


def test_crash_after_plan_before_submit(make_worker, mock):
    """Попытки записаны в БД, но процесс умер до POST /prompt — новый воркер отправит каждую ровно раз."""
    w = make_worker()
    add_shot(w.db, "Q")
    S.transition(w.db, "Q", S.READY)
    S.transition(w.db, "Q", S.FRAME_GEN)
    w.plan_attempts(w.db.shot("Q"), "frame", w.profile("test_sfw"))
    rows = w.db.all("SELECT * FROM attempts WHERE shot_id='Q'")
    assert len(rows) == 2 and all(r["status"] == "queued" for r in rows)
    w2 = make_worker(db=w.db)                  # «новый процесс»
    w2.run_until_idle()
    assert w.db.shot("Q")["state"] == S.VIDEO_REVIEW
    assert sum(mock.submissions.values()) == 3 and max(mock.submissions.values()) == 1


def test_same_idem_key_no_duplicate(make_worker, mock):
    w = make_worker()
    add_shot(w.db, "I")
    S.transition(w.db, "I", S.READY)
    S.transition(w.db, "I", S.FRAME_GEN)
    w.plan_attempts(w.db.shot("I"), "frame", w.profile("test_sfw"))
    a = w.db.one("SELECT * FROM attempts WHERE shot_id='I' ORDER BY attempt_id LIMIT 1")
    again = w.enqueue_attempt({k: a[k] for k in ("shot_id", "stage", "arm_id", "workflow", "workflow_ver",
                                                 "params_json", "prompt_text", "seed", "idem_key", "try_no")}
                              | {"attempt_id": "at_other"})
    assert again["attempt_id"] == a["attempt_id"]
    assert w.db.one("SELECT COUNT(*) c FROM attempts WHERE idem_key=?", (a["idem_key"],))["c"] == 1
    w.run_until_idle()
    assert mock.submissions[a["comfy_prompt_id"]] == 1


def test_comfy_restart_resubmits_same_prompt_id(make_worker, mock):
    first = {}

    def behavior(pid, body):
        if not first:
            first["pid"] = pid
            threading.Timer(0.3, mock.restart_runtime).start()
            return "hang"
        return "ok"
    mock.behavior = behavior
    w = make_worker()
    add_shot(w.db, "R")
    w.run_until_idle()
    assert w.db.shot("R")["state"] == S.VIDEO_REVIEW
    pid = first["pid"]
    assert mock.submissions[pid] == 2                      # переотправлен тем же prompt_id
    assert w.db.one("SELECT COUNT(*) c FROM attempts WHERE comfy_prompt_id=?", (pid,))["c"] == 1
    assert w.db.one("SELECT status FROM attempts WHERE comfy_prompt_id=?", (pid,))["status"] == "done"


def test_timeout_interrupts_and_limits_retries(make_worker, mock):
    mock.behavior = lambda pid, body: "hang" if (body.get("extra_data") or {}).get("farm", {}).get(
        "shot_id") == "TO" else "ok"
    w = make_worker(cfg_over={"timeouts": {"frame": 0.3}})
    add_shot(w.db, "TO")
    w.run_until_idle()
    shot = w.db.shot("TO")
    assert shot["state"] == S.NEEDS_HUMAN
    assert "таймаут" in shot["last_error"]
    assert mock.interrupts, "не было /interrupt"
    assert visited(w.db, "TO").count(S.FRAME_RETRY) == 1   # после таймаута — не больше одного ретрая


def test_node_errors_go_to_human_without_retry(make_worker, mock):
    w = make_worker()
    add_shot(w.db, "NE", profile="test_bad")
    w.run_until_idle()
    shot = w.db.shot("NE")
    assert shot["state"] == S.NEEDS_HUMAN
    assert S.FRAME_RETRY not in visited(w.db, "NE")
    assert "граф отклонён" in shot["last_error"]
    assert sum(mock.submissions.values()) == 0


def test_oom_frees_vram_and_retries(make_worker, mock):
    seen = {"n": 0}

    def behavior(pid, body):
        seen["n"] += 1
        return "oom" if seen["n"] <= 2 else "ok"           # обе попытки (2 сида) первого try — OOM
    mock.behavior = behavior
    w = make_worker()
    add_shot(w.db, "OOM")
    w.run_until_idle()
    assert w.db.shot("OOM")["state"] == S.VIDEO_REVIEW
    assert mock.free_calls >= 1
    assert S.FRAME_RETRY in visited(w.db, "OOM")


def test_vram_gate_waits_then_proceeds(make_worker, mock):
    mock.vram_used = 0.95
    mock.free_lowers = False
    w = make_worker(cfg_over={"vram": {"wait_s": 0.2}})
    add_shot(w.db, "V")
    w.tick()
    w.tick()
    assert w.db.shot("V")["state"] == S.FRAME_GEN          # гейт не пустил
    assert sum(mock.submissions.values()) == 0 and mock.free_calls >= 1
    mock.vram_used = 0.5
    w.run_until_idle()
    assert w.db.shot("V")["state"] == S.VIDEO_REVIEW
