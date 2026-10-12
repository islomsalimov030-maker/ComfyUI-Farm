"""Настройки сборщика промптов: farm/prompts/config.yaml ← $FARM_HOME/prompts.yaml."""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import yaml

from ..config import deep_merge, farm_home

PKG = Path(__file__).resolve().parent


@lru_cache(maxsize=4)
def _load(home: str) -> dict:
    with open(PKG / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    over = Path(home) / "prompts.yaml"
    if over.is_file():
        with open(over, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def load_config() -> dict:
    return copy.deepcopy(_load(str(farm_home())))
