"""Синглтон воркера внутри процесса ComfyUI + служебные HTTP-маршруты (/farm/status, /farm/control).

Маршруты висят на том же порту 8188, наружу он не открыт (только SSH-туннель) — CONTRACTS §2.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Optional

log = logging.getLogger("farm")

_worker = None
_lock = threading.Lock()


def get_worker(create: bool = True):
    global _worker
    with _lock:
        if _worker is None and create:
            from .worker import Worker
            _worker = Worker()
        return _worker


def start_background() -> None:
    """Запустить фоновый поток (он сам ждёт ComfyUI и работает только после «start» на пульте)."""
    try:
        w = get_worker()
        if w.cfg["worker"].get("autostart", True):
            w.start()
    except Exception:
        log.exception("ферма: воркер не запущен (ComfyUI продолжает работу)")


_KEEP = object()


def control(action: str, profiles: Optional[list[str]] = None, flt=_KEEP) -> dict:
    """flt (v0.5): не передан — фильтр пакета не меняется; None/{} — снять; dict — заменить."""
    w = get_worker()
    if profiles:
        w.control.set_profiles(profiles)
    if flt is not _KEEP:
        w.control.set_filter(flt)
    if action == "start":
        w.control.set_running(True)
        w.start()
        w.wake()
    elif action == "pause":
        w.control.set_running(False)
    elif action != "status":
        raise ValueError(f"неизвестное действие {action}")
    return w.status()


def list_profiles() -> list[str]:
    from .config import repo_dir
    d = repo_dir() / "profiles"
    return sorted(p.stem for p in d.glob("*.yaml")) if d.exists() else []


def register_routes() -> bool:
    try:
        from aiohttp import web
        from server import PromptServer  # есть только внутри ComfyUI
    except Exception:
        return False
    routes = PromptServer.instance.routes

    @routes.get("/farm/status")
    async def _status(request):
        return web.json_response(control("status"))

    @routes.post("/farm/control")
    async def _control(request):
        try:
            body = await request.json()
            flt = body["filter"] if "filter" in body else _KEEP
            return web.json_response(control(body.get("action", "status"), body.get("profiles"), flt))
        except (ValueError, json.JSONDecodeError) as e:
            return web.json_response({"error": str(e)}, status=400)

    return True
