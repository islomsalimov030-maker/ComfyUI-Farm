"""Модульные тесты: машина состояний, миграции БД, привязка по title, idem_key, бандит, хранилище, клиент."""
import importlib.util
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from conftest import FIX, add_shot
from mock_comfy import MockComfy, tiny_png

from farm import states as S
from farm.bandit import Bandit, category, pose_class
from farm.comfy_client import (ComfyClient, NodeErrors, bind, collect_outputs, load_workflow,
                               make_idem_key, prompt_id_for)
from farm.db import DB
from farm.storage import ImmutableError, image_size, write_once

# ---------------------------------------------------------------- состояния

def test_happy_path_is_legal():
    chain = [S.PENDING, S.READY, S.FRAME_GEN, S.FRAME_QA, S.FRAME_RETRY, S.FRAME_GEN, S.FRAME_QA,
             S.FRAME_REVIEW, S.VIDEO_GEN, S.VIDEO_QA, S.POST_FIX, S.VIDEO_QA, S.VIDEO_RETRY, S.VIDEO_GEN,
             S.VIDEO_QA, S.VIDEO_REVIEW, S.APPROVED]
    for a, b in zip(chain, chain[1:]):
        assert S.can(a, b), (a, b)


@pytest.mark.parametrize("a,b", [(S.PENDING, S.VIDEO_GEN), (S.FRAME_GEN, S.VIDEO_GEN),
                                 (S.APPROVED, S.READY), (S.REJECTED, S.FRAME_GEN),
                                 (S.VIDEO_REVIEW, S.FRAME_REVIEW), (S.READY, S.APPROVED)])
def test_illegal_transitions(a, b):
    assert not S.can(a, b)


def test_transition_db_and_stale(tmp_path):
    db = DB(tmp_path / "f.sqlite")
    add_shot(db, "s")
    S.transition(db, "s", S.READY, "r")
    with pytest.raises(S.IllegalTransition):
        S.transition(db, "s", S.APPROVED)
    with pytest.raises(S.StaleState):
        S.transition(db, "s", S.FRAME_GEN, expect=S.PENDING)
    assert db.shot("s")["state"] == S.READY
    ev = db.all("SELECT * FROM events WHERE shot_id='s'")
    assert [(e["from_state"], e["to_state"]) for e in ev] == [(S.PENDING, S.READY)]


def test_every_state_reachable_and_terminal_closed():
    reach, todo = {S.PENDING}, [S.PENDING]
    while todo:
        for n in S.TRANSITIONS[todo.pop()]:
            if n not in reach:
                reach.add(n)
                todo.append(n)
    assert reach == set(S.ALL_STATES)
    assert S.TRANSITIONS[S.REJECTED] == set()


# ---------------------------------------------------------------- БД

def test_migration_adds_columns(tmp_path):
    p = tmp_path / "old.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE shots (shot_id TEXT PRIMARY KEY, state TEXT)")
    c.execute("INSERT INTO shots VALUES ('old', 'pending')")
    c.commit()
    c.close()
    db = DB(p)
    cols = {r["name"] for r in db.all("PRAGMA table_info(shots)")}
    assert {"stage_try", "frame_asset_id", "kind", "priority"} <= cols
    assert db.shot("old")["state"] == "pending"
    assert db.one("PRAGMA user_version")["user_version"] == 2
    DB(p)  # повторная миграция — без ошибок


def test_json_roundtrip_and_idem_unique(tmp_path):
    db = DB(tmp_path / "f.sqlite")
    add_shot(db, "j", depends_on_json=["a", "b"], objects_json={"x": 1})
    s = db.shot("j")
    assert s["depends_on_json"] == ["a", "b"] and s["objects_json"] == {"x": 1}
    db.insert("attempts", {"attempt_id": "a1", "shot_id": "j", "idem_key": "k"})
    with pytest.raises(sqlite3.IntegrityError):
        db.insert("attempts", {"attempt_id": "a2", "shot_id": "j", "idem_key": "k"})


# ---------------------------------------------------------------- воркфлоу и idem

def test_bind_by_title_and_outputs():
    graph, params = load_workflow(FIX / "workflows", "video_test@1")
    g, bound = bind(graph, params, {"prompt": "hi", "seed": 5, "duration_s": 7, "shift": 13,
                                    "keyframe": "/x.png", "unknown": 1}, upload=lambda p: "farm/x.png")
    assert g["3"]["inputs"]["seed"] == 5 and g["3"]["inputs"]["duration_sec"] == 7.0
    assert g["3"]["inputs"]["shift_video"] == 13.0 and g["1"]["inputs"]["image"] == "farm/x.png"
    assert graph["3"]["inputs"]["seed"] == 0          # исходный граф не тронут
    assert "unknown" not in bound
    with pytest.raises(ValueError):
        bind(graph, params, {"prompt": "x", "keyframe": "/x.png", "duration_s": 30}, upload=lambda p: p)
    with pytest.raises(ValueError):                  # обязательный вход не задан
        bind(graph, params, {"prompt": "x"}, upload=lambda p: p)
    entry = {"outputs": {"8": {"gifs": [{"filename": "v.mp4", "subfolder": "", "type": "output"}]},
                         "9": {"images": [{"filename": "l.png", "subfolder": "", "type": "output"}]}}}
    outs = collect_outputs(entry, graph, params)
    assert {(o["key"], o["kind"]) for o in outs} == {("video_fullhd", "video"), ("last_frame", "image")}


def test_bind_refuses_linked_input_and_dup_title():
    graph, params = load_workflow(FIX / "workflows", "frame_test@1")
    params = {**params, "inputs": {**params["inputs"], "bad": {"title": "PANEL", "input": "model",
                                                               "type": "string"}}}
    with pytest.raises(ValueError, match="линком"):
        bind(graph, params, {"prompt": "p", "bad": "x"})
    g2 = json.loads(json.dumps(graph))
    g2["99"] = {"class_type": "X", "inputs": {}, "_meta": {"title": "PANEL"}}
    with pytest.raises(KeyError, match="не уникален"):
        bind(g2, params, {"prompt": "p", "seed": 1})


def test_idem_key_deterministic():
    k1 = make_idem_key("wf@1", {"a": 1, "b": "x"}, ["s2", "s1"], 7)
    k2 = make_idem_key("wf@1", {"b": "x", "a": 1}, ["s1", "s2"], 7)
    assert k1 == k2
    assert k1 != make_idem_key("wf@1", {"a": 1, "b": "x"}, ["s1", "s2"], 8)
    assert k1 != make_idem_key("wf@2", {"a": 1, "b": "x"}, ["s1", "s2"], 7)
    assert prompt_id_for(k1) == prompt_id_for(k2) and len(prompt_id_for(k1)) == 36


# ---------------------------------------------------------------- бандит

def test_pose_category():
    assert pose_class("Лежит на диване") == "lying"
    assert pose_class("сидит за столом") == "sitting"
    assert pose_class("стоит у окна") == "standing"
    assert pose_class("танцует") == "other"
    assert category({"pose_ru": "сидя", "location": "cafe"}) == "sitting:cafe"


def test_bandit_cold_start_equal_then_exploits(tmp_path):
    db = DB(tmp_path / "f.sqlite")
    b = Bandit(db, cold_start_n=3, rng=random.Random(0))
    arms = ["a", "b", "c"]
    picks = []
    for _ in range(9):
        a = b.choose(arms, "cat")
        picks.append(a)
        b.update(a, "cat", success=(a == "b"))
    assert sorted(picks.count(x) for x in arms) == [3, 3, 3]
    for _ in range(60):
        a = b.choose(arms, "cat")
        b.update(a, "cat", success=(a == "b"))
    st = b.stats(arms, "cat")
    assert st["b"]["n"] > st["a"]["n"] and st["b"]["n"] > st["c"]["n"]
    db.execute("INSERT INTO arms(arm_id, enabled) VALUES ('b', 0)")
    assert {b.choose(arms, "cat") for _ in range(20)} <= {"a", "c"}


# ---------------------------------------------------------------- хранилище

def test_write_once_immutable(tmp_path):
    p = tmp_path / "r" / "x.png"
    data = tiny_png(10, 7)
    d = write_once(p, data)
    assert write_once(p, data) == d
    with pytest.raises(ImmutableError):
        write_once(p, data + b"1")
    assert image_size(data) == (10, 7)


# ---------------------------------------------------------------- клиент

def test_client_ws_wakeup_and_upload(tmp_path):
    m = MockComfy(ws=True, exec_s=0.2).start()
    try:
        c = ComfyClient(m.url, ws_url=m.ws_url, poll_s=5.0)
        graph, _ = load_workflow(FIX / "workflows", "frame_test@1")
        f = tmp_path / "in.png"
        f.write_bytes(tiny_png(8, 8))
        assert c.upload_image(f, name="abc.png") == "farm/abc.png" and "farm/abc.png" in m.uploads
        pid = c.submit(graph, prompt_id="00000000-0000-0000-0000-000000000001")
        t0 = time.time()
        entry = c.wait(pid, timeout_s=10)
        assert time.time() - t0 < 2.5, "WS не разбудил ожидание (опрос раз в 5 с)"
        assert entry["status"]["status_str"] == "success"
        bad = {**graph, "3": {**graph["3"], "class_type": "BadNode"}}
        with pytest.raises(NodeErrors):
            c.submit(bad)
        c.close()
    finally:
        m.stop()


def test_vram_gate(mock):
    c = ComfyClient(mock.url, poll_s=0.05)
    mock.vram_used = 0.95
    assert c.wait_vram(0.85, wait_s=1, poll_s=0.05) is True and mock.free_calls == 1
    mock.vram_used, mock.free_lowers = 0.95, False
    assert c.wait_vram(0.85, wait_s=0.2, poll_s=0.05) is False
    c.close()


# ---------------------------------------------------------------- загрузка пакета как в ComfyUI

def test_package_loads_like_comfyui(monkeypatch, farm_home):
    monkeypatch.setenv("FARM_AUTOSTART", "0")
    root = Path(__file__).resolve().parents[1]
    name = "ComfyUI-Farm-test"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        assert "NGFarmControl" in mod.NODE_CLASS_MAPPINGS
        node_cls = mod.NODE_CLASS_MAPPINGS["NGFarmControl"]
        assert "action" in node_cls.INPUT_TYPES()["required"]
        out = node_cls().run("pause", "* (все)")
        status = json.loads(out["result"][0])
        assert status["running"] is False and "states" in status
        runtime = sys.modules[f"{name}.farm.runtime"]
        assert runtime.get_worker().paths.home == farm_home
    finally:
        for k in [k for k in sys.modules if k.startswith(name)]:
            sys.modules.pop(k, None)
