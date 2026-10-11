"""Пульт фермы через HTTP пакета ng-farm (`GET /farm/status`, `POST /farm/control`) на 127.0.0.1:8188.

Порт наружу не открыт — бот живёт на том же поде (или за SSH-туннелем). Файл control.json
напрямую не трогаем: воркер держит состояние пульта в памяти процесса ComfyUI.
"""
from __future__ import annotations

from typing import Optional

import httpx


class FarmUnavailable(RuntimeError):
    pass


class FarmControl:
    def __init__(self, base_url: str, timeout: float = 5.0, client: Optional[httpx.Client] = None):
        self.base = base_url.rstrip("/")
        self.http = client or httpx.Client(timeout=timeout)

    def _req(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        try:
            r = self.http.request(method, self.base + path, json=body)
        except httpx.HTTPError as e:
            raise FarmUnavailable(f"{type(e).__name__}: {e}") from e
        if r.status_code == 404:
            raise FarmUnavailable("пакет фермы не загружен в ComfyUI (404 /farm/*)")
        if r.status_code >= 400:
            raise FarmUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def status(self) -> dict:
        return self._req("GET", "/farm/status")

    def start(self, profiles: Optional[list[str]] = None, flt: Optional[dict] = None,
              set_filter: bool = False) -> dict:
        body: dict = {"action": "start"}
        if profiles:
            body["profiles"] = profiles
        if set_filter:
            body["filter"] = flt or None      # ключ есть → фильтр заменяется (None/{} — снять)
        return self._req("POST", "/farm/control", body)

    def pause(self) -> dict:
        return self._req("POST", "/farm/control", {"action": "pause"})

    def close(self) -> None:
        self.http.close()
