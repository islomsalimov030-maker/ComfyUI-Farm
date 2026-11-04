"""Путь к данным Стейси на ПК (WS-15, 04.11).

Данные переехали с диска C на диск D, чтобы освободить место на C. Чтобы следующий перенос не ломал
одноразовые утилиты `tools/`, путь берётся из переменной окружения STACY_DATA, а не зашит в код.

    set STACY_DATA=D:/stacy-lora        # PowerShell: $env:STACY_DATA = "D:/stacy-lora"

Если STACY_DATA не задан, берётся D:/stacy-lora; если такой папки нет, но есть старая C:/stacy-lora —
используется она. Иначе возвращается путь по умолчанию, и скрипт сам скажет, что данных нет.
"""
import os
from pathlib import Path

_DEFAULT = Path("D:/stacy-lora")
_LEGACY = Path("C:/stacy-lora")


def stacy_data() -> Path:
    """Папка с данными Стейси (кадры, датасеты, face_metrics_final.csv, моды)."""
    env = os.environ.get("STACY_DATA")
    if env:
        return Path(env)
    if _DEFAULT.exists():
        return _DEFAULT
    if _LEGACY.exists():
        return _LEGACY
    return _DEFAULT


def require() -> Path:
    """Как stacy_data(), но с понятной ошибкой, если папки нет."""
    p = stacy_data()
    if not p.exists():
        raise SystemExit(
            "данные Стейси не найдены: %s\n"
            "задай путь переменной окружения STACY_DATA (напр. D:/stacy-lora)" % p
        )
    return p
