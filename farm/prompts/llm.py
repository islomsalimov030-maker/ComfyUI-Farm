"""Клиент Ollama для полей промпта (WS-04): /api/chat + structured outputs, ретраи, выгрузка модели.

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


def fix_request(errs: list[str]) -> str:
    return ("Your JSON has these problems:\n- " + "\n- ".join(errs[:12]) +
            "\nReturn the corrected, complete JSON object only.")


def _strip_fences(s: str) -> str:
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    return s.strip()
