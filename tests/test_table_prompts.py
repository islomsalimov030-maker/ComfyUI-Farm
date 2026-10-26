"""26.10: нода «Таблица → промпты» — xlsx сцен → та же таблица с промптами; LLM llama.cpp внутри ComfyUI (фейк)."""
from __future__ import annotations

import json

import pytest

from farm import table_prompts as T
from farm.prompts import builder as B
from farm.prompts.llm import LLMUnavailable, LlamaCppClient, OllamaClient, make_client, resolve_gguf

from prompts_helpers import load_fields
from test_ws03_sheets import S_HDR, TEN, lib_tables

openpyxl = pytest.importorskip("openpyxl")
F = load_fields()


class FakeLlama:
    """Вместо llama_cpp.Llama: отвечает полями STD-01 из фикстуры по стадии (её видно по схеме)."""
    calls: list = []
    closed = 0

    def __init__(self, path, n_ctx):
        self.path = path

    def create_chat_completion(self, messages, response_format, **kw):
        FakeLlama.calls.append((response_format, kw))
        stage = "video" if "start_pose" in json.dumps(response_format.get("schema")) else "frame"
        return {"choices": [{"message": {"content": json.dumps(F["STD-01"][stage])}}],
                "usage": {"completion_tokens": 5}}

    def close(self):
        FakeLlama.closed += 1


@pytest.fixture
def env(tmp_path, monkeypatch):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    t = lib_tables()
    t["Сцены"] = [S_HDR] + TEN
    for name, rows in t.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(list(r))
    src = tmp_path / "farm.xlsx"
    wb.save(src)
    gguf = tmp_path / "LLM" / "sub" / "m.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"x")
    monkeypatch.setattr(B, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(B, "_clients", {})
    FakeLlama.calls, FakeLlama.closed = [], 0
    orig = LlamaCppClient.__init__
    monkeypatch.setattr(LlamaCppClient, "__init__", lambda self, *a, **kw: orig(
        self, *a, **{**kw, "llama_factory": FakeLlama, "models_dirs": [tmp_path / "LLM"]}))
    return src, tmp_path


def test_table_to_prompts_writes_columns(env):
    src, tmp = env
    dst = tmp / "out" / "farm_prompts.xlsx"
    seen = []
    r = T.table_to_prompts(src, dst, {"backend": "llamacpp", "prompt_model": "m.gguf"},
                           progress=lambda i, n: seen.append((i, n)))
    st = dict(x.split(": ", 1) for x in r["lines"])
    assert r["rows"] == 10 and seen[-1] == (10, 10)
    assert st["STD-01"] == "ok" and st["SIT-02"] == "ok"
    assert st["СТ-01"].startswith("строка не прошла проверку")                 # невалидная строка — без LLM
    assert st["ENT-01"].startswith("ошибка — video: E_LLM_JSON")                # фейк отвечает петлёй на переход
    assert FakeLlama.closed == 1                                                # модель выгружена один раз, в конце
    assert all(rf["type"] == "json_object" and rf["schema"]["type"] == "object" for rf, _ in FakeLlama.calls)

    ws = openpyxl.load_workbook(dst)["Сцены"]
    head = {c.value: c.column for c in ws[1]}
    assert all(title in head for _, title in T.OUT_COLS)
    assert ws.max_column == len(S_HDR) + len(T.OUT_COLS)                       # исходные колонки не тронуты
    row = {ws.cell(i, 1 + S_HDR.index("ID")).value: i for i in range(2, ws.max_row + 1)}
    get = lambda sid, key: ws.cell(row[sid], head[dict(T.OUT_COLS)[key]]).value  # noqa: E731
    assert get("STD-01", "frame") and get("STD-01", "video") and get("STD-01", "video_face")
    assert get("SIT-02", "frame") and not get("SIT-02", "video")               # кадр — без промпта видео
    assert not openpyxl.load_workbook(src)["Сцены"].cell(1, len(S_HDR) + 1).value   # исходник не изменён

    # повтор — удачные поля из кэша; LLM спрашивается только по ENT-01 (3 попытки); колонки не дублируются
    n = len(FakeLlama.calls)
    T.table_to_prompts(dst, dst, {"backend": "llamacpp", "prompt_model": "m.gguf"})
    assert len(FakeLlama.calls) == n + 3
    assert openpyxl.load_workbook(dst)["Сцены"].max_column == len(S_HDR) + len(T.OUT_COLS)


def test_only_ready(env):
    src, tmp = env
    r = T.table_to_prompts(src, tmp / "o.xlsx", {"backend": "llamacpp", "prompt_model": "m.gguf"}, only_ready=True)
    assert r["rows"] == 10                                                      # черновиков в TEN нет


def test_make_client_and_resolve_gguf(tmp_path):
    assert isinstance(make_client({"llm": {"prompt_model": "q"}}), OllamaClient)
    assert not isinstance(make_client({"llm": {"prompt_model": "q"}}), LlamaCppClient)
    assert isinstance(make_client({"llm": {"backend": "llama.cpp", "prompt_model": "q.gguf"}}), LlamaCppClient)
    f = tmp_path / "a" / "b" / "q.gguf"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    assert resolve_gguf("q.gguf", [tmp_path]) == f                             # поиск в подпапках
    assert resolve_gguf(str(f), []) == f                                       # абсолютный путь
    with pytest.raises(LLMUnavailable):
        resolve_gguf("unknown.gguf", [tmp_path])                               # не из KNOWN_GGUF — без докачки
    assert not LlamaCppClient("nope.gguf", models_dirs=[tmp_path]).ping()


def test_node_inputs_and_missing_table(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "_in_dir", lambda: tmp_path)
    (tmp_path / T.TABLE_SUBDIR).mkdir()
    (tmp_path / T.TABLE_SUBDIR / "scenes.xlsx").write_bytes(b"")
    (tmp_path / "x_prompts.xlsx").write_bytes(b"")
    assert T.table_files() == [f"{T.TABLE_SUBDIR}/scenes.xlsx"]                # свои итоги в список не попадают
    with pytest.raises(ValueError, match="нет таблицы"):
        T.FarmTablePrompts().run("nope.xlsx", False, "ollama", "m", "", False)
