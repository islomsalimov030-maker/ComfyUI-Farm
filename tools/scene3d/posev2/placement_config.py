"""Единый конфиг постановки: только config/placement.yaml от корня репозитория.

Нет cwd/env-поиска и значений по умолчанию: отсутствующий файл/раздел/ключ — ошибка.
"""
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "placement.yaml"


@lru_cache(maxsize=1)
def _load():
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{CONFIG_PATH}: ожидается mapping разделов")
    return data


def section(name):
    """Обязательный раздел YAML; KeyError при неизвестном имени, без fallback."""
    value = _load()[name]
    if not isinstance(value, dict):
        raise ValueError(f"{CONFIG_PATH}: {name!r} должен быть mapping")
    return value
