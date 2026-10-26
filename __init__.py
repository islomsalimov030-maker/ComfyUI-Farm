"""ComfyUI-Farm: пакет нод + фоновый воркер-оркестратор (CONTRACTS §1).

ComfyUI импортирует этот файл при старте → регистрируем пульт и маршруты, запускаем воркер.
FARM_AUTOSTART=0 — не запускать фоновый поток. Нет папки $FARM_HOME (сервер студии) — воркер тоже не запускается.
"""
import logging
import os

log = logging.getLogger("farm")

WEB_DIRECTORY = "./web"   # 26.10: кнопка «Загрузить таблицу» у ноды «Таблица → промпты»

if __package__:  # загружен ComfyUI как пакет custom_nodes/<папка>
    from .farm.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    try:
        from .farm import runtime as _runtime
        _runtime.register_routes()
        from .farm.config import farm_home as _fh
        if os.environ.get("FARM_AUTOSTART", "1") != "0" and _fh().is_dir():
            _runtime.start_background()
        else:   # 23.10: сервер студии и т.п. — только ноды воркфлоу, без фонового воркера фермы
            log.info("ферма: воркер не запущен (нет %s или FARM_AUTOSTART=0) — доступны только ноды", _fh())
    except Exception:  # ферма не должна ронять загрузку ComfyUI
        log.exception("ферма: ошибка инициализации")
else:  # импорт как отдельного файла (pytest при сборе тестов) — без побочных эффектов
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
