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


def test_studio_profiles_dir(env):
    """Папка студии: образец sfw_base, она главнее репо, промпт — по имени файла рядом, модель профиля."""
    import yaml
    from farm.config import repo_dir
    src, tmp = env
    fp = tmp / "input" / T.PROFILES_SUBDIR
    assert T.seed_profiles(fp) and not T.seed_profiles(fp)                       # второй раз не трогает
    assert sorted(f.name for f in fp.iterdir()) == ["sfw_base.yaml", "sfw_v1_frame.md", "sfw_v1_video.md"]
    prof = yaml.safe_load((fp / "sfw_base.yaml").read_text(encoding="utf-8"))
    assert prof["frame"]["workflow"] and prof["llm"]["system_prompt"] == {"frame": "sfw_v1_frame.md",
                                                                          "video": "sfw_v1_video.md"}
    dirs = [fp, repo_dir() / "profiles"]

    # образец без правок = тот же промпт, что из репо
    a = T.table_to_prompts(src, tmp / "a.xlsx", {"backend": "llamacpp", "prompt_model": "m.gguf"}, profiles_dirs=dirs)
    assert str(fp / "sfw_base.yaml") in a["profiles"]["sfw_base"] and a["ok"] == 4

    # свой промпт кадра и модель профиля (поле model ноды пустое, бэкенд ноды — другой)
    (fp / "sfw_v1_frame.md").write_text((fp / "sfw_v1_frame.md").read_text(encoding="utf-8") + "\nCUSTOM-MARK\n",
                                        encoding="utf-8")
    prof["llm"].update(backend="llamacpp", prompt_model="m.gguf")
    (fp / "sfw_base.yaml").write_text(yaml.safe_dump(prof, allow_unicode=True), encoding="utf-8")
    FakeLlama.calls.clear()
    seen = []
    orig = FakeLlama.create_chat_completion
    FakeLlama.create_chat_completion = lambda self, messages, **kw: seen.append(messages[0]["content"]) or \
        orig(self, messages, **kw)
    try:
        b = T.table_to_prompts(src, tmp / "b.xlsx", {"backend": "transformers", "prompt_model": ""},
                               profiles_dirs=dirs)
    finally:
        FakeLlama.create_chat_completion = orig
    assert b["profiles"]["sfw_base"].endswith("llamacpp: m.gguf") and FakeLlama.calls
    assert any("CUSTOM-MARK" in s for s in seen)                                # кэш не помешал: другой промпт

    # профиля нет нигде → ошибка в статусе строки, таблица не падает
    c = T.table_to_prompts(src, tmp / "c.xlsx", {"backend": "llamacpp", "prompt_model": "m.gguf"},
                           profiles_dirs=[tmp / "empty"])
    assert all("нет профиля sfw_base.yaml" in x for x in c["lines"] if not x.startswith("СТ-01")
               and not x.startswith("BAD"))


# ----------------------------------------------------------------------------- transformers (без torch: заглушки)
class _Ids(dict):
    def to(self, device):
        return self


class FakeTok:
    """Токены = символы; модель «пишет» поля STD-01 по стадии из системного промпта."""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return json.dumps(messages, ensure_ascii=False) + "<assistant>"

    def __call__(self, text, return_tensors, add_special_tokens):
        assert text.endswith("<assistant>{")                                   # ответ начат с «{»
        ids = list(text)
        return _Ids(input_ids=type("T", (), {"shape": (1, len(ids))})(), _ids=ids)

    def decode(self, ids, skip_special_tokens):
        return "".join(ids)


class FakeModel:
    device = "cuda"
    calls = 0

    def eval(self):
        return self

    def generate(self, input_ids, _ids, max_new_tokens, do_sample):
        FakeModel.calls += 1
        stage = "video" if "start_pose" in "".join(_ids) else "frame"
        reply = "```json\n" + json.dumps(F["STD-01"][stage]) + "\n```" if FakeModel.calls == 1 else \
            json.dumps(F["STD-01"][stage])
        return [_ids + list(reply[1:] if reply.startswith("{") else reply)]


@pytest.fixture
def fake_torch(monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda s: None,
                                                              inference_mode=contextlib.nullcontext))
    FakeModel.calls = 0


def test_transformers_backend(env, fake_torch, monkeypatch):
    from farm.prompts.llm import TransformersClient, resolve_hf
    src, tmp = env
    hf = tmp / "LLM" / "Qwen3-VL-8B-Instruct"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    (hf / "model.safetensors").write_bytes(b"x")
    assert resolve_hf("Qwen/Qwen3-VL-8B-Instruct", [tmp / "LLM"]) == hf             # папка по имени репозитория
    with pytest.raises(LLMUnavailable):
        resolve_hf("Qwen/Other", [tmp / "LLM"], download=False)
    assert isinstance(make_client({"llm": {"backend": "transformers"}}), TransformersClient)
    loads = []
    orig = TransformersClient.__init__
    monkeypatch.setattr(TransformersClient, "__init__", lambda self, *a, **kw: orig(
        self, *a, **{**kw, "loader": lambda p: loads.append(p) or (FakeModel(), FakeTok())}))
    r = T.table_to_prompts(src, tmp / "o.xlsx", {"backend": "transformers", "prompt_model": "Qwen/Qwen3-VL-8B-Instruct"})
    st = dict(x.split(": ", 1) for x in r["lines"])
    assert st["STD-01"] == "ok" and st["SIT-02"] == "ok"                       # первый ответ не JSON — ушёл на переделку
    assert loads == [hf]                                                        # загрузка один раз за таблицу
    assert not FakeLlama.calls                                                  # llama.cpp не трогали


def test_hf_fetch_resume_and_complete(tmp_path):
    import httpx
    from farm.prompts.llm import _hf_complete, hf_fetch
    blobs = {"config.json": b"{}", "model-00001-of-00002.safetensors": b"A" * 100,
             "model-00002-of-00002.safetensors": b"B" * 50, "README.md": b"skip",
             "model.safetensors.index.json": json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                                                       "b": "model-00002-of-00002.safetensors"}}).encode()}
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/tree/main"):
            return httpx.Response(200, json=[{"type": "file", "path": k, "size": len(v)} for k, v in blobs.items()])
        name = req.url.path.rsplit("/", 1)[-1]
        rng = req.headers.get("range")
        seen.append((name, rng))
        data = blobs[name]
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return httpx.Response(206, content=data[start:])
        return httpx.Response(200, content=data)

    dst = tmp_path / "Qwen3-VL-8B-Instruct"
    dst.mkdir()
    (dst / "config.json").write_bytes(b"{}")                                        # уже есть — не качается
    (dst / "model-00001-of-00002.safetensors.part").write_bytes(b"A" * 40)          # обрыв на 40 байтах
    (dst / "model.safetensors.index.json").write_bytes(blobs["model.safetensors.index.json"])
    assert not _hf_complete(dst)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    got = hf_fetch("Qwen/Qwen3-VL-8B-Instruct", dst, http=http)
    assert sorted(got) == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    assert ("model-00001-of-00002.safetensors", "bytes=40-") in seen and not any(n == "README.md" for n, _ in seen)
    assert (dst / "model-00001-of-00002.safetensors").read_bytes() == b"A" * 100 and _hf_complete(dst)
    assert hf_fetch("Qwen/Qwen3-VL-8B-Instruct", dst, http=http) == []                  # повтор — ничего не качает
