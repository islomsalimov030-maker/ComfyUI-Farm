"""25.10: видимая нода LLM (NGFarmPromptLLM) → «Шот из таблицы» собирает промпты из её полей без LLM."""
import json
from types import SimpleNamespace

import pytest

from farm import shot_node as S


class FakeLib:
    def resolve_inputs(self, shot, stage):
        return {"char_face_mod": "farm/stacy_face", "char_body_mod": "farm/stacy_body"}


class FakePrompts:
    def __init__(self):
        self.calls = []

    def build(self, stage, shot, lib, prof, ctx):
        self.calls.append((stage, dict(ctx), prof.get("llm")))
        f = ctx.get("fields")
        src = "ctx" if f is not None else "llm"
        f = f if f is not None else {"pose": f"{stage} pose"}
        return SimpleNamespace(text=f"{stage}: {f['pose']}", lint=[],
                               fields_json={"fields": f, "inputs": {"face_prompt": f"{stage} face"},
                                            "meta": {"fields_source": src, "template": "t",
                                                     "llm": {"model": (prof.get("llm") or {}).get("prompt_model")}}})


@pytest.fixture
def fake(monkeypatch, tmp_path):
    tbl = tmp_path / "t.xlsx"
    tbl.write_bytes(b"")
    pr = FakePrompts()
    w = SimpleNamespace(profile=lambda name: {"llm": {"prompt_model": "qwen3-vl:8b-instruct"},
                                             "frame": {"workflow": "frame_ls@1"}, "video": {"workflow": "video_h3_loop@1"}},
                        workflow=lambda wf: ({}, {}), plugins=SimpleNamespace(prompts=pr))
    import farm.runtime
    monkeypatch.setattr(farm.runtime, "get_worker", lambda: w)
    shot = {"shot_id": "STD-01", "profile": "sfw_base", "duration_s": 6, "loop": True, "extra_json": {"zone": "loft"}}
    monkeypatch.setattr(S, "_img", lambda path: None)                     # без torch
    monkeypatch.setattr(S, "load_row", lambda table, row, sid: (dict(shot), FakeLib(), ["STD-01"]))
    return pr, str(tbl)


def test_llm_node_then_shot(fake):
    pr, tbl = fake
    out = S.FarmPromptLLM().run(tbl, 1, "", "qwen3-vl:32b-instruct", "", False)
    fields, table, sid, report = out["result"]
    assert sid == "STD-01" and table == tbl and "источник: llm" in report
    assert pr.calls[0][2]["prompt_model"] == "qwen3-vl:32b-instruct"     # модель из ноды перекрыла профиль
    n = len(pr.calls)
    res = S.FarmShotFromTable().run(table, 1, sid, fields)["result"]
    assert res[3] == "frame: frame pose" and res[6] == "video: video pose"
    assert all("fields" in c[1] for c in pr.calls[n:])                     # второй раз LLM не вызывается


def test_shot_without_llm_node(fake):
    pr, tbl = fake
    res = S.FarmShotFromTable().run(tbl, 1, "")["result"]
    assert res[3] == "frame: frame pose" and res[10] == "STD-01"
    assert all("fields" not in c[1] for c in pr.calls)


def test_fresh_skips_cache(fake):
    pr, tbl = fake
    S.FarmPromptLLM().run(tbl, 1, "", "", "", True)
    assert all(c[1]["no_cache"] for c in pr.calls)
    assert json.loads(S.FarmPromptLLM().run(tbl, 1, "", "", "", False)["result"][0])["frame"] == {"pose": "frame pose"}
