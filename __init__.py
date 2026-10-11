"""ComfyUI-Farm: пакет нод + фоновый воркер-оркестратор (CONTRACTS §1).

ComfyUI импортирует этот файл при старте → регистрируем пульт и маршруты, запускаем воркер.
FARM_AUTOSTART=0 — не запускать фоновый поток (например, на машине без фермы).
"""
import logging
import os

log = logging.getLogger("farm")

if __package__:  # загружен ComfyUI как пакет custom_nodes/<папка>
    from .farm.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    try:
        from .farm import runtime as _runtime
        _runtime.register_routes()
        if os.environ.get("FARM_AUTOSTART", "1") != "0":
            _runtime.start_background()
    except Exception:  # ферма не должна ронять загрузку ComfyUI
        log.exception("ферма: ошибка инициализации")
else:  # импорт как отдельного файла (pytest при сборе тестов) — без побочных эффектов
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
