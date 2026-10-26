"""Клиент Ollama для полей промпта (WS-04): /api/chat + structured outputs, ретраи, выгрузка модели.
26.10: `LlamaCppClient` — то же без сервера (GGUF в процессе ComfyUI), `TransformersClient` — модель HF на GPU;
выбор — profile.llm.backend.

- `format` = JSON-схема полей (Ollama ≥ 0.5) или просто "json" (`format_mode: json`).
- Невалидный JSON или схема не сошлась → ошибки уходят обратно в чат, не больше `max_retries` (2) раз.
- Внутри `with client.batch():` модель держится в памяти, на выходе — `keep_alive: 0` (VRAM свободна
  для ComfyUI). Вне пакета каждый запрос идёт с keep_alive профиля (по умолчанию 0).
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


class LLMUnavailable(RuntimeError):
    """Ollama не отвечает или модели нет."""


class LLMInvalid(RuntimeError):
    """После всех попыток JSON так и не прошёл проверку."""

    def __init__(self, msg: str, stats: "CallStats"):
        super().__init__(msg)
        self.stats = stats


@dataclass
class CallStats:
    model: str = ""
    attempts: int = 0
    first_try_ok: bool = False
    ok: bool = False
    ms: int = 0
    errors: list = field(default_factory=list)       # по попыткам: список строк ошибок
    eval_tokens: int = 0

    def as_dict(self) -> dict:
        return {"model": self.model, "attempts": self.attempts, "first_try_ok": self.first_try_ok,
                "ok": self.ok, "ms": self.ms, "errors": self.errors[-3:], "eval_tokens": self.eval_tokens}


class OllamaClient:
    def __init__(self, model: str, url: str = DEFAULT_URL, keep_alive: Any = 0, timeout_s: float = 180,
                 options: Optional[dict] = None, format_mode: str = "schema", max_retries: int = 2,
                 transport: Optional[httpx.BaseTransport] = None, think: Optional[bool] = None):
        self.model = model
        self.url = url.rstrip("/")
        self.keep_alive = keep_alive
        self.options = {"temperature": 0, "num_ctx": 8192, **(options or {})}   # гайд Ollama: 0 для схемы
        self.format_mode = format_mode
        self.max_retries = max_retries
        self.think = think
        self._batch_keep: Optional[str] = None
        self._http = httpx.Client(timeout=timeout_s, transport=transport)

    @classmethod
    def from_profile(cls, profile: dict, **kw) -> "OllamaClient":
        llm = (profile or {}).get("llm") or {}
        opts = dict(llm.get("options") or {})
        return cls(model=llm.get("prompt_model") or "qwen3-vl:8b-instruct",
                   url=llm.get("url") or DEFAULT_URL, keep_alive=llm.get("keep_alive", 0),
                   timeout_s=float(llm.get("timeout_s", 180)), options=opts,
                   format_mode=llm.get("format_mode", "schema"),
                   max_retries=int(llm.get("max_retries", 2)), think=llm.get("think"), **kw)

    # ---- пакет: модель живёт, пока идёт пакет; в конце — выгрузить -----------------------
    @contextlib.contextmanager
    def batch(self, keep: str = "10m"):
        prev, self._batch_keep = self._batch_keep, keep
        try:
            yield self
        finally:
            self._batch_keep = prev
            if prev is None:
                self.unload()

    def unload(self) -> None:
        try:
            self._http.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass

    def ping(self) -> bool:
        try:
            r = self._http.get(f"{self.url}/api/tags", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    # ---- запрос ---------------------------------------------------------------------------------
    def _post(self, messages: list, schema: Optional[dict], seed: Optional[int]) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False,
                "format": schema if (schema is not None and self.format_mode == "schema") else "json",
                "keep_alive": self._batch_keep if self._batch_keep is not None else self.keep_alive,
                "options": dict(self.options, **({"seed": seed} if seed is not None else {}))}
        if self.think is not None:
            body["think"] = self.think
        try:
            r = self._http.post(f"{self.url}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"Ollama {self.url}: {e}") from e
        if r.status_code == 404:
            raise LLMUnavailable(f"Ollama: модели {self.model} нет ({r.text[:200]})")
        if r.status_code >= 400:
            raise LLMUnavailable(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def chat_json(self, messages: list, schema: Optional[dict] = None,
                  validator: Optional[Callable[[Any], list]] = None, seed: Optional[int] = None,
                  stats: Optional[CallStats] = None) -> tuple[dict, list, CallStats]:
        """Возвращает (данные, история сообщений с ответом, статистика). Ошибки → обратно в чат ≤ max_retries."""
        st = stats or CallStats(model=self.model)
        msgs = list(messages)
        t0 = time.monotonic()
        try:
            for i in range(self.max_retries + 1):
                st.attempts += 1
                resp = self._post(msgs, schema, None if seed is None else seed + i)
                st.eval_tokens += int(resp.get("eval_count") or 0)
                content = ((resp.get("message") or {}).get("content") or "").strip()
                errs: list[str]
                data: Any = None
                try:
                    data = json.loads(_strip_fences(content))
                    errs = list(validator(data)) if validator else []
                except json.JSONDecodeError as e:
                    errs = [f"это не JSON: {e.msg} (позиция {e.pos})"]
                st.errors.append(errs)
                msgs.append({"role": "assistant", "content": content})
                if not errs:
                    st.ok = True
                    if st.attempts == 1:
                        st.first_try_ok = True
                    return data, msgs, st
                msgs.append({"role": "user", "content": fix_request(errs)})
        finally:
            st.ms += int((time.monotonic() - t0) * 1000)
        raise LLMInvalid(f"JSON не прошёл проверку за {st.attempts} попыток: {st.errors[-1][:3]}", st)


class LlamaCppClient(OllamaClient):
    """26.10: тот же клиент без сервера Ollama — GGUF грузится прямо в процесс ComfyUI (llama-cpp-python).

    Нужен там, где Ollama нет (сервер студии). JSON-схема → грамматика llama.cpp (`response_format.schema`) —
    аналог structured outputs Ollama. Модель: имя файла в models/LLM (ищется рекурсивно) или абсолютный путь;
    известные официальные GGUF (`KNOWN_GGUF`) докачиваются с Hugging Face при первом запуске.
    """

    def __init__(self, model: str, keep_alive: Any = 0, options: Optional[dict] = None,
                 format_mode: str = "schema", max_retries: int = 2, llama_factory: Optional[Callable] = None,
                 models_dirs: Optional[list] = None, **_):
        super().__init__(model, url="llama.cpp", keep_alive=keep_alive, options=options,
                         format_mode=format_mode, max_retries=max_retries)
        self._factory = llama_factory
        self._dirs = models_dirs
        self._llm = None

    @classmethod
    def from_profile(cls, profile: dict, **kw) -> "LlamaCppClient":
        llm = (profile or {}).get("llm") or {}
        return cls(model=llm.get("prompt_model") or DEFAULT_GGUF, keep_alive=llm.get("keep_alive", 0),
                   options=dict(llm.get("options") or {}), format_mode=llm.get("format_mode", "schema"),
                   max_retries=int(llm.get("max_retries", 2)), **kw)

    def _load(self):
        if self._llm is None:
            path = resolve_gguf(self.model, self._dirs)
            if self._factory is not None:
                self._llm = self._factory(path, n_ctx=int(self.options.get("num_ctx", 8192)))
            else:
                try:
                    from llama_cpp import Llama
                except ImportError as e:
                    raise LLMUnavailable("нет llama-cpp-python в окружении ComfyUI") from e
                self._llm = Llama(model_path=str(path), n_gpu_layers=-1, n_ctx=int(self.options.get("num_ctx", 8192)),
                                  verbose=False)
        return self._llm

    def unload(self) -> None:
        llm, self._llm = self._llm, None
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
            del llm
            import gc
            gc.collect()

    def ping(self) -> bool:
        try:
            resolve_gguf(self.model, self._dirs, download=False)
            return True
        except LLMUnavailable:
            return False

    def _post(self, messages: list, schema: Optional[dict], seed: Optional[int]) -> dict:
        llm = self._load()
        rf = {"type": "json_object"}
        if schema is not None and self.format_mode == "schema":
            rf["schema"] = schema
        try:
            r = llm.create_chat_completion(messages=messages, response_format=rf,
                                           temperature=float(self.options.get("temperature", 0)),
                                           seed=seed, max_tokens=int(self.options.get("num_predict", 2048)))
        except Exception as e:   # noqa: BLE001 — ошибки llama.cpp не типизированы
            raise LLMUnavailable(f"llama.cpp {self.model}: {e}") from e
        return {"message": {"content": r["choices"][0]["message"].get("content") or ""},
                "eval_count": (r.get("usage") or {}).get("completion_tokens", 0)}

    def chat_json(self, *a, **kw):
        try:
            return super().chat_json(*a, **kw)
        finally:
            if self._batch_keep is None and str(self.keep_alive) == "0":
                self.unload()


class TransformersClient(LlamaCppClient):
    """26.10: модель HF (safetensors) на GPU через transformers — там, где llama.cpp собран без CUDA (студия).

    Грамматики нет: схема только в системном промпте, ответ начинается с «{», невалидный JSON уходит
    обратно в чат (общий цикл `chat_json`). Модель: id репозитория HF (`Qwen/Qwen3-VL-8B-Instruct` —
    папка `models/LLM/Qwen3-VL-8B-Instruct`, докачивается) или путь к папке с config.json.
    """

    def __init__(self, model: str, loader: Optional[Callable] = None, **kw):
        kw.pop("llama_factory", None)
        super().__init__(model, **kw)
        self.url = "transformers"
        self._loader = loader          # тесты: (path) → (model, tokenizer)

    @classmethod
    def from_profile(cls, profile: dict, **kw) -> "TransformersClient":
        llm = (profile or {}).get("llm") or {}
        return cls(model=llm.get("prompt_model") or DEFAULT_HF, keep_alive=llm.get("keep_alive", 0),
                   options=dict(llm.get("options") or {}), format_mode="json",
                   max_retries=int(llm.get("max_retries", 2)), **kw)

    def _load(self):
        if self._llm is None:
            path = resolve_hf(self.model, self._dirs)
            if self._loader is not None:
                self._llm = self._loader(path)
            else:
                try:
                    import torch
                    import transformers
                    from transformers import AutoModelForImageTextToText, AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(str(path))
                    m = AutoModelForImageTextToText.from_pretrained(str(path), dtype=torch.bfloat16,
                                                                    device_map="cuda", attn_implementation="sdpa")
                except Exception as e:   # noqa: BLE001 — нет пакета, старая версия, нет CUDA
                    v = getattr(locals().get("transformers"), "__version__", "?")
                    raise LLMUnavailable(f"transformers {v}: не загрузилась {path.name}: {e}") from e
                self._llm = (m.eval(), tok)
        return self._llm

    def unload(self) -> None:
        had, self._llm = self._llm is not None, None
        if had:
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:   # noqa: BLE001
                pass

    def ping(self) -> bool:
        try:
            resolve_hf(self.model, self._dirs, download=False)
            return True
        except LLMUnavailable:
            return False

    def _post(self, messages: list, schema: Optional[dict], seed: Optional[int]) -> dict:
        m, tok = self._load()
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + "{"
        try:
            import torch
            if seed is not None:
                torch.manual_seed(int(seed))
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to(m.device)
            temp = float(self.options.get("temperature", 0))
            gen = {"do_sample": True, "temperature": temp} if temp > 0 else {"do_sample": False}
            with torch.inference_mode():
                out = m.generate(**ids, max_new_tokens=int(self.options.get("num_predict", 2048)), **gen)
        except Exception as e:   # noqa: BLE001
            raise LLMUnavailable(f"transformers {self.model}: {e}") from e
        new = out[0][ids["input_ids"].shape[1]:]
        return {"message": {"content": "{" + tok.decode(new, skip_special_tokens=True)}, "eval_count": len(new)}


DEFAULT_HF = "Qwen/Qwen3-VL-8B-Instruct"   # Apache-2.0, bf16 ~17 ГБ VRAM
HF_FILES = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]


def resolve_hf(name: str, dirs: Optional[list] = None, download: bool = True):
    """Папка модели HF: абсолютный путь, models/LLM/<имя> (рекурсивно, с config.json) или докачка репозитория."""
    from pathlib import Path
    p = Path(name)
    if p.is_absolute():
        if (p / "config.json").is_file():
            return p
        raise LLMUnavailable(f"нет config.json в {p}")
    dirs = [Path(d) for d in (dirs if dirs is not None else _llm_dirs())]
    base = p.name
    for d in dirs:
        if d.is_dir():
            hit = next((c.parent for c in d.rglob("config.json") if c.parent.name == base), None)
            if hit is not None and any(hit.glob("*.safetensors")):
                return hit
    if not download or "/" not in name or not dirs:
        raise LLMUnavailable(f"нет модели {name} в {', '.join(map(str, dirs)) or 'models/LLM'}")
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(name, local_dir=str(dirs[-1] / base), allow_patterns=HF_FILES))
    except Exception as e:   # noqa: BLE001
        raise LLMUnavailable(f"не скачалась {name}: {e}") from e


DEFAULT_GGUF = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"   # тот же квант, что у Ollama qwen3-vl:8b-instruct
KNOWN_GGUF = {   # официальные GGUF (Apache-2.0) — докачка в models/LLM/<repo>/
    "Qwen3VL-8B-Instruct-Q4_K_M.gguf": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
    "Qwen3VL-8B-Instruct-Q8_0.gguf": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
}


def _llm_dirs() -> list:
    from pathlib import Path
    try:
        import folder_paths
        dirs = [Path(d) for d in folder_paths.get_folder_paths("LLM")] \
            if "LLM" in folder_paths.folder_names_and_paths else []
        return dirs + [Path(folder_paths.models_dir) / "LLM"]
    except ImportError:   # вне ComfyUI
        return [Path(os.environ.get("FARM_LLM_DIR", "models/LLM"))]


def resolve_gguf(name: str, dirs: Optional[list] = None, download: bool = True):
    """Путь к GGUF: абсолютный путь, файл в models/LLM (рекурсивно) или докачка известного файла с HF."""
    from pathlib import Path
    p = Path(name)
    if p.is_absolute():
        if p.is_file():
            return p
        raise LLMUnavailable(f"нет файла {p}")
    dirs = [Path(d) for d in (dirs if dirs is not None else _llm_dirs())]
    for d in dirs:
        if d.is_dir():
            hit = next((f for f in d.rglob(p.name) if f.is_file()), None)
            if hit is not None:
                return hit
    repo = KNOWN_GGUF.get(p.name)
    if not download or repo is None or not dirs:
        raise LLMUnavailable(f"нет модели {name} в {', '.join(map(str, dirs)) or 'models/LLM'}")
    try:
        from huggingface_hub import hf_hub_download
        dst = dirs[-1] / repo.split("/")[-1]
        return Path(hf_hub_download(repo, p.name, local_dir=str(dst)))
    except Exception as e:   # noqa: BLE001
        raise LLMUnavailable(f"не скачалась {repo}/{p.name}: {e}") from e


def make_client(profile: dict):
    """Клиент по profile.llm.backend: ollama (по умолчанию) | llamacpp | transformers."""
    backend = str(((profile or {}).get("llm") or {}).get("backend") or "ollama").lower()
    if backend in ("transformers", "hf"):
        return TransformersClient.from_profile(profile)
    return (LlamaCppClient if backend in ("llamacpp", "llama.cpp", "llama_cpp") else OllamaClient).from_profile(profile)


def fix_request(errs: list[str]) -> str:
    return ("Your JSON has these problems:\n- " + "\n- ".join(errs[:12]) +
            "\nReturn the corrected, complete JSON object only.")


def _strip_fences(s: str) -> str:
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    return s.strip()
