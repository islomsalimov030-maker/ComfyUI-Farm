"""WS-04: сборщик и линтер промптов H3 — golden-тесты на 20 SFW-строках, линтер, тайминг, LLM-клиент (мок).

Перегенерировать эталоны после осознанной правки шаблонов: UPDATE_GOLDEN=1 pytest tests/test_ws04_prompts.py
"""
from __future__ import annotations

import json
import os
import re

import httpx
import pytest
import yaml

from prompts_helpers import GOLDEN, PROFILE, load_fields, make_env

from farm.prompts import builder as B
from farm.prompts import build, lint
from farm.prompts.lint import errors, word_count, split_sections
from farm.prompts.llm import LLMUnavailable, OllamaClient
from farm.prompts.schema import check_fields
from farm.prompts.scale import anchor_from_card, body_part_at
from farm.prompts.timing import actual_seconds, fmt, grid_frames, schedule
from farm.prompts.config import load_config

STAGES = ("frame", "video")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FARM_HOME", str(tmp_path))
    lib, shots = make_env(tmp_path)
    return lib, shots, load_fields()


def _build(env, sid, stage, **ctx):
    lib, shots, F = env
    c = {"fields": F[sid][stage], **ctx}
    return build(stage, shots[sid], lib, PROFILE, ctx=c)


# ============================================================================ golden: 20 строк → 40 промптов
def test_golden_20_rows_40_prompts_zero_lint_errors(env):
    lib, shots, F = env
    assert len(shots) == 20 and set(shots) == set(F)
    update = bool(os.environ.get("UPDATE_GOLDEN"))
    GOLDEN.mkdir(parents=True, exist_ok=True)
    report, n = {}, 0
    for sid in shots:
        for st in STAGES:
            r = _build(env, sid, st)
            n += 1
            assert r.text, (sid, st, [(i.code, i.msg) for i in r.lint])
            errs = [(i.code, i.msg) for i in errors(r.lint)]
            assert not errs, (sid, st, errs)
            report[f"{sid}.{st}"] = sorted(i.code for i in r.lint)
            p = GOLDEN / f"{sid}.{st}.txt"
            body = r.text + "\n# inputs: " + json.dumps(r.fields_json["inputs"], ensure_ascii=False, sort_keys=True) + "\n"
            if update or not p.exists():
                p.write_text(body, encoding="utf-8")
            assert p.read_text(encoding="utf-8") == body, f"{p.name}: вывод изменился (UPDATE_GOLDEN=1 — если так и надо)"
    assert n == 40
    (GOLDEN / "_lint_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True),
                                              encoding="utf-8") if update else None


def test_golden_structure_and_user_rules(env):
    lib, shots, F = env
    names = load_config()["lint"]["sections"]
    for sid in shots:
        for st in STAGES:
            t = _build(env, sid, st).text
            order, sec = split_sections(t, names)
            assert order == names
            assert "studio" not in t.lower()
            assert "Colors and saturation stay unchanged from the reference image." in t
            assert "<Subject 1> is St4cy_H3, the young woman in" in t
            if st == "video":
                assert "the camera holds a completely static, locked-off shot throughout" in t
                assert "The lighting is preserved unchanged throughout." in t
                assert 300 <= word_count(sec["detailed_description"]) <= 560
            else:
                assert "One single photograph. <Picture 2> is only an identity reference sheet, not a layout." in t
                assert "True-to-life scale:" in t and "never enlarged to fill the frame" in t


def test_modes_labels_and_variants(env):
    # переход: STD-04 → кадр STD-05 (<Picture 3>), петля выкл
    r = _build(env, "STD-04", "video")
    assert r.template_ver == "video_transition_ref_v1"
    assert "<Picture 3> is the last frame of [Shot 1]." in r.text and "the shot ends on <Picture 3>" in r.text
    assert "<Picture 3>" in r.fields_json["meta"]["connected"]
    assert "Стоит ровно, руки скрещены" in r.fields_json["llm_input"]["end_pose_ru"]
    # видео-реф: STD-13 → <Video 1> и составной тег
    r = _build(env, "STD-13", "video")
    assert "[keyframe completion + reference generation]" in r.text
    assert "<Video 1>: weak_reference" in r.text
    # без видео-рефа <Video 1> нет
    assert "<Video 1>" not in _build(env, "STD-01", "video").text
    # кадр: одежда <Picture 3> и поза <Picture 4> только когда входы есть
    t = _build(env, "SIT-08", "frame").text
    assert "<Picture 3>" in t and "<Picture 4>" in t and "<Subject 3>: attribute_transfer" in t
    t = _build(env, "STD-01", "frame").text
    assert "<Picture 3>" not in t and "<Picture 4>" not in t
    assert "wearing exactly the clothes shown in <Picture 2>" in t


def test_face_prompts_for_workflow_inputs(env):
    r = _build(env, "SIT-08", "video")
    fp = r.fields_json["inputs"]["face_prompt"]
    assert fp.startswith("St4cy_H3, a close-up of the face") and "warm natural smile" in fp
    r = _build(env, "SIT-08", "frame")
    assert r.fields_json["inputs"]["face_prompt_h3"].startswith("St4cy_H3, a tight close-up portrait")


def test_scale_anchor_from_plate_card(env):
    assert "Scale anchor: the ballet barre reaches her hip." in _build(env, "STD-11", "frame").text
    assert "Scale anchor: the reception desk reaches her waist." in _build(env, "STD-06", "frame").text
    r = _build(env, "SIT-01", "frame")
    assert r.fields_json["meta"]["scale_anchor"]["source"] == "card"
    assert "Scale anchor: the pink velvet sofa reaches her knees." in r.text
    # нет якорей в карточке — запасной якорь от LLM, нет и его — предупреждение
    r = _build(env, "SIT-07", "frame")
    assert "Scale anchor: she fits on a single yoga mat." in r.text
    assert "W_SCALE_ANCHOR" in {i.code for i in _build(env, "STD-01", "frame").lint}


def test_scale_body_parts():
    assert body_part_at(92, 168, "standing") == "hip"
    assert body_part_at(105, 168, "standing") == "waist"
    assert body_part_at(50, 168, "standing") == "knees"
    assert body_part_at(5, 168, "standing") is None
    card = {"anchors": [{"name": "table", "top_cm": 48, "bbox": [0.1, 0.6, 0.3, 0.7]},
                        {"name": "door_handle", "top_cm": 100, "bbox": [0.8, 0.4, 0.85, 0.45]}],
            "slots": [{"id": "A", "pose": "standing", "feet_xy": [0.2, 0.9]}]}
    a = anchor_from_card(card, "standing", 168)
    assert a["text"] == "the table reaches her knees"            # ближайший к слоту предмет


# ============================================================================ тайминг
def test_grid_and_end_mark():
    assert grid_frames(8) == 192 and fmt(actual_seconds(8)) == "7.96"   # образец Sitting-8
    assert grid_frames(7) == 175 and fmt(actual_seconds(7)) == "7.25"   # WS-01 V11
    assert fmt(actual_seconds(5)) == "5.12"                             # WS-09: 7.96 → 5.12
    assert fmt(actual_seconds(10)) == "10.08"


def test_schedule_window_and_rest():
    cfg = load_config()
    end = actual_seconds(10)
    ph = [{"kind": "move", "sec": 3}, {"kind": "hold", "sec": 3}, {"kind": "move", "sec": 2}, {"kind": "return", "sec": 2}]
    sched, rest = schedule(ph, end, cfg, "loop")
    assert sched[0][:2] == (0.0, 0.8)
    assert rest <= end * cfg["timing"]["loop_window"] + 1e-6 and end - rest >= cfg["timing"]["min_rest_s"]
    assert all(b > a for a, b, _ in sched)
    with pytest.raises(Exception):
        schedule([{"kind": "move", "sec": 1}] * 12, actual_seconds(5), cfg, "loop")


def test_repair_hints_from_fail_codes(env):
    base = _build(env, "SIT-01", "video")
    r = _build(env, "SIT-01", "video", fail_codes=["SLOWMO", "CAMERA_MOVE"])
    assert "All movements are brisk and quick" in r.text
    assert "The frame never moves or reframes" in r.text
    last = lambda t: re.findall(r"From \d+\.\d+ to (\d+\.\d+) seconds", t)[-2]  # noqa: E731
    assert float(last(r.text)) <= float(last(base.text))                     # фазы короче
    assert not errors(r.lint)


# ============================================================================ линтер: каждое правило ловится
@pytest.fixture
def good(env):
    r = _build(env, "SIT-08", "video")
    ctx = {"mode": "loop", "end_s": 7.96, "trigger": "St4cy_H3", "connected": {"<Picture 1>", "<Picture 2>"}}
    assert not errors(lint(r.text, "video", ctx))
    return r.text, ctx


@pytest.mark.parametrize("mutate,code", [
    (lambda t: t.replace("stretching class hall", "stretching studio"), "E_STUDIO"),
    (lambda t: t.replace("From 0.80 to", "The camera pans left. From 0.80 to"), "E_CAMERA_MOVE"),
    (lambda t: t.replace("From 0.80 to", "The camera slowly zooms in. From 0.80 to"), "E_CAMERA_MOVE"),
    (lambda t: t.replace("with no camera movement, no zoom and no pan", "with gentle motion"), "E_MUST_CAMERA_STATIC"),
    (lambda t: t.replace("The lighting is preserved unchanged throughout. ", ""), "E_MUST_LIGHT_KEEP"),
    (lambda t: t.replace("Colors and saturation stay unchanged from the reference image.", ""), "E_MUST_COLOR_KEEP"),
    (lambda t: t.replace("no close-up inserts", "no inserts"), "E_MUST_ONE_SHOT"),
    (lambda t: t.replace("St4cy_H3", "Stacy_v1"), "E_WRONG_TRIGGER"),
    (lambda t: t.replace("summary:\n", "summary_x:\n"), "E_SECTIONS"),
    (lambda t: t.replace("[Shot 1] The shot", "[Shot 1] At 00:00.000 the shot"), "E_SHOT"),
    (lambda t: t.replace("[keyframe completion]", "[keyframe magic]"), "E_TAG"),
    (lambda t: t.replace("smiling at the camera.", "smiling at <Picture 3>.", 1), "E_LABEL_UNDEFINED"),
    (lambda t: t.replace("ends on <Picture 1>: <Subject 1>", "ends on <Picture 1>: <Subject 3>"), "E_SUMMARY_LABEL"),
    (lambda t: t.replace("<Picture 2>: fully_preserved", "<Picture 2>: kept_mostly"), "E_RETENTION"),
    (lambda t: t.replace("At 7.96 seconds", "At 8.50 seconds"), "E_END_MARK"),
    (lambda t: t.replace("From 4.60 to 7.96", "From 4.60 to 8.40"), "E_TIMELINE"),
    (lambda t: t.replace("From 1.80 to 3.40", "From 1.50 to 3.40"), "E_TIMELINE"),
    (lambda t: t.replace("From 0.80 to 1.80 seconds, her left hand", "From 0.80 to 1.80 seconds, she walks to the window and her left hand"), "E_DIRECTED_MOVE"),
    (lambda t: t.replace("smiling at the camera.", "улыбается.", 1), "E_CYRILLIC"),
    (lambda t: t.replace("never in slow motion.", "never in slow motion. The tug happens in slow motion."), "E_SLOWMO_WORD"),
    (lambda t: t.replace("so the last frame matches the first frame", "and holds"), "E_MUST_LOOP_END"),
    (lambda t: t.replace("From 4.60 to 7.96", "From 4.60 to 7.00").replace("At 7.96", "At 7.00"), "E_END_MARK"),
])
def test_lint_catches(good, mutate, code):
    text, ctx = good
    bad = mutate(text)
    assert bad != text
    codes = {i.code for i in lint(bad, "video", ctx)}
    assert code in codes, codes


def test_lint_labels_vs_connected_inputs(good):
    text, ctx = good
    assert "E_LABEL_UNCONNECTED" in {i.code for i in lint(text, "video", {**ctx, "connected": {"<Picture 1>"}})}
    w = lint(text, "video", {**ctx, "connected": ctx["connected"] | {"<Picture 3>"}})
    assert "W_LABEL_UNUSED" in {i.code for i in w}


def test_lint_warnings_not_errors(good):
    text, ctx = good
    t = text.replace("she pinches a strap", "she slowly pinches a strap")
    t = t.replace("her left hand lifts from her left thigh", "her hand lifts from her left thigh")
    issues = lint(t, "video", ctx)
    assert not errors(issues)
    assert {"W_SLOWLY", "W_SIDE"} <= {i.code for i in issues}


def test_lint_accepts_users_approved_sample():
    """Образец Sitting-8 из Stacy_H3_compact (утверждён пользователем) — без ошибок линтера."""
    sample = (GOLDEN.parent.parent / "fixtures" / "prompts" / "sample_sitting8.txt").read_text(encoding="utf-8")
    ctx = {"mode": "loop", "end_s": 7.96, "trigger": "St4cy_H3", "connected": {"<Picture 1>", "<Picture 2>"}}
    assert not errors(lint(sample, "video", ctx))


def test_lint_frame_rules(env):
    t = _build(env, "SIT-08", "frame").text
    ctx = {"trigger": "St4cy_H3", "connected": {"<Picture 1>", "<Picture 2>", "<Picture 3>", "<Picture 4>"}}
    assert not errors(lint(t, "frame", ctx))
    bad = t.replace("One single photograph. <Picture 2> is only an identity reference sheet, not a layout. ", "")
    assert "E_MUST_SINGLE_PHOTO" in {i.code for i in lint(bad, "frame", ctx)}
    bad = t.replace("she is never enlarged to fill the frame", "she fills the frame")
    assert "E_MUST_SCALE_NEVER_ENLARGED" in {i.code for i in lint(bad, "frame", ctx)}
    assert "E_LABEL_UNCONNECTED" in {i.code for i in lint(t, "frame", {**ctx, "connected": {"<Picture 1>", "<Picture 2>"}})}


def test_face_prompt_lint():
    assert not lint("St4cy_H3, a close-up of a face.", "video", {"kind": "face", "trigger": "St4cy_H3"})
    codes = {i.code for i in lint("a portrait in a photo studio", "video", {"kind": "face", "trigger": "St4cy_H3"})}
    assert {"E_STUDIO", "E_TRIGGER"} <= codes


# ============================================================================ схема полей
def test_fields_schema(env):
    F = env[2]
    for sid, d in F.items():
        mode = "transition" if sid in ("STD-04", "STD-09") else "loop"
        assert check_fields("video", d["video"], mode) == [], sid
        assert check_fields("frame", d["frame"]) == [], sid
    v = dict(F["SIT-08"]["video"])
    assert any("return" in e for e in check_fields("video", {**v, "phases": v["phases"][:-1]}, "loop"))
    assert any("лишнее" in e for e in check_fields("video", {**v, "camera": "pan"}, "loop"))
    assert check_fields("video", {**v, "phases": [{"kind": "jump", "text": "x" * 10, "sec": 1}]}, "loop")


def test_mode_without_end_frame_is_error(env):
    lib, shots, F = env
    s = dict(shots["STD-01"], loop=0)
    r = build("video", s, lib, PROFILE, ctx={"fields": F["STD-01"]["video"]})
    assert "E_MODE" in {i.code for i in errors(r.lint)}


def test_no_llm_no_fields_is_error(env):
    lib, shots, _ = env
    r = build("video", shots["STD-01"], lib, PROFILE, ctx={})
    assert "E_NO_FIELDS" in {i.code for i in errors(r.lint)} and r.text == ""


def test_profile_can_swap_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("FARM_HOME", str(tmp_path))
    p = tmp_path / "my_video.md"
    p.write_text("SFW test system prompt", encoding="utf-8")
    name, text = B.system_prompt("video", {"llm": {"system_prompt": {"video": str(p)}}}, load_config())
    assert text == "SFW test system prompt"
    name, text = B.system_prompt("frame", {}, load_config())
    assert name == "sfw_v1_frame" and "safe for work" in text


# ============================================================================ LLM (мок Ollama)
class FakeOllama:
    """Отвечает очередью ответов; пишет тела запросов."""

    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.bodies.append((request.url.path, body))
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        content = self.replies.pop(0)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}, "eval_count": 10})


def _llm_profile():
    return {**PROFILE, "llm": {"prompt_model": "qwen3-vl:8b-instruct", "keep_alive": 0, "url": "http://ollama.test"}}


def _install(fake, monkeypatch):
    prof = _llm_profile()
    cl = OllamaClient.from_profile(prof, transport=httpx.MockTransport(fake))
    monkeypatch.setattr(B, "_clients", {("qwen3-vl:8b-instruct", "http://ollama.test"): cl})
    return prof, cl


def test_llm_retry_on_invalid_json_then_cache(env, monkeypatch):
    lib, shots, F = env
    good = json.dumps(F["SIT-08"]["video"])
    fake = FakeOllama(["{not json", good])
    prof, _ = _install(fake, monkeypatch)
    r = build("video", shots["SIT-08"], lib, prof, ctx={})
    assert not errors(r.lint), [(i.code, i.msg) for i in r.lint]
    m = r.fields_json["meta"]
    assert m["fields_source"] == "llm" and m["llm"]["attempts"] == 2 and m["llm_first_json_ok"] is False
    chat = [b for pth, b in fake.bodies if pth == "/api/chat"]
    assert chat[0]["format"]["type"] == "object" and chat[0]["keep_alive"] == 0 and chat[0]["stream"] is False
    assert "Your JSON has these problems" in chat[1]["messages"][-1]["content"]
    # второй раз — из кэша, LLM не зовётся
    r2 = build("video", shots["SIT-08"], lib, prof, ctx={})
    assert r2.fields_json["meta"]["fields_source"] == "cache" and r2.text == r.text and not fake.replies


def test_llm_lint_errors_go_back_to_llm(env, monkeypatch):
    lib, shots, F = env
    bad = dict(F["SIT-08"]["video"], location_en="the stretching studio by the window")
    fake = FakeOllama([json.dumps(bad), json.dumps(F["SIT-08"]["video"])])
    prof, _ = _install(fake, monkeypatch)
    r = build("video", shots["SIT-08"], lib, prof, ctx={"no_cache": True})
    assert not errors(r.lint)
    m = r.fields_json["meta"]
    assert m["llm_first_json_ok"] is True and m["lint_rounds"] == 1
    last_user = [b for pth, b in fake.bodies if pth == "/api/chat"][-1]["messages"][-1]["content"]
    assert "E_STUDIO" in last_user


def test_llm_gives_up_after_retries(env, monkeypatch):
    lib, shots, _ = env
    fake = FakeOllama(["nope", "{}", "[]"])
    prof, _ = _install(fake, monkeypatch)
    r = build("video", shots["SIT-08"], lib, prof, ctx={"no_cache": True})
    assert "E_LLM_JSON" in {i.code for i in errors(r.lint)}
    assert r.fields_json["meta"]["llm"]["attempts"] == 3


def test_llm_unavailable_is_lint_error(env, monkeypatch):
    lib, shots, _ = env

    def down(request):
        raise httpx.ConnectError("refused", request=request)
    prof, _ = _install(down, monkeypatch)
    r = build("frame", shots["SIT-08"], lib, prof, ctx={"no_cache": True})
    assert "E_LLM_UNAVAILABLE" in {i.code for i in errors(r.lint)}


def test_batch_unloads_model_after(monkeypatch):
    fake = FakeOllama(['{"a": 1}', '{"a": 2}'])
    cl = OllamaClient("m", url="http://o.test", transport=httpx.MockTransport(fake))
    with cl.batch():
        cl.chat_json([{"role": "user", "content": "x"}])
        cl.chat_json([{"role": "user", "content": "y"}])
    kinds = [(p, b.get("keep_alive")) for p, b in fake.bodies]
    assert kinds == [("/api/chat", "10m"), ("/api/chat", "10m"), ("/api/generate", 0)]


def test_comment_from_review_goes_to_llm(env, monkeypatch):
    lib, shots, F = env
    fake = FakeOllama([json.dumps(F["SIT-08"]["video"])])
    prof, _ = _install(fake, monkeypatch)
    build("video", shots["SIT-08"], lib, prof, ctx={"comment": "движение руки живее", "fail_codes": ["SLOWMO"],
                                                    "no_cache": True})
    user = json.loads([b for p, b in fake.bodies if p == "/api/chat"][0]["messages"][1]["content"])
    assert user["reviewer_note"] == "движение руки живее"
    assert any("slow motion" in h for h in user["repair_hints"])


# ============================================================================ связка с воркером
def test_plugins_pick_real_builder():
    from farm.plugins import Plugins
    assert Plugins().describe()["prompts"] == "real"


def test_worker_passes_params_and_lint_error_stops_gpu(make_worker):
    from conftest import add_shot
    from farm import states as S
    w = make_worker()
    seen = []

    class Rec:
        def build(self, stage, shot, lib, profile, ctx=None):
            seen.append(ctx)
            return build(stage, shot, lib, profile, ctx={**ctx, "fields": {"bad": 1}})
    w.plugins.prompts = Rec()
    add_shot(w.db, "W-1", kind="frame")
    w.promote_pending()
    for _ in range(5):
        w.tick()
    assert seen and seen[0]["workflow"] == "frame_test@1" and "inputs" in seen[0]["params"]
    sh = w.db.shot("W-1")
    assert sh["state"] == S.NEEDS_HUMAN and "PROMPT_LINT" in sh["last_error"]
    assert w.db.all("SELECT * FROM attempts WHERE shot_id='W-1'") == []
