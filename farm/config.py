"""Загрузка настроек: farm/defaults.yaml ← $FARM_HOME/farm.yaml ← явные overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import yaml

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent


def farm_home() -> Path:
    return Path(os.environ.get("FARM_HOME", "/workspace/farm"))


def repo_dir() -> Path:
    return Path(os.environ.get("FARM_REPO", str(REPO_DIR)))


def deep_merge(base: dict, over: Optional[dict]) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(overrides: Optional[dict] = None, home: Optional[Path] = None) -> dict:
    cfg = load_yaml(PKG_DIR / "defaults.yaml")
    user = (home or farm_home()) / "farm.yaml"
    if user.exists():
        cfg = deep_merge(cfg, load_yaml(user) or {})
    return deep_merge(cfg, overrides)


def load_routing(cfg: dict) -> dict:
    """Маршрут отказов: дефолт пакета + перекрытие из farm/qa/routing.yaml (если есть)."""
    qa = cfg.get("qa", {})
    base = load_yaml(PKG_DIR / qa.get("routing_default", "routing_default.yaml")) or {}
    over_path = PKG_DIR / qa.get("routing_override", "qa/routing.yaml")
    if over_path.exists():
        over = load_yaml(over_path) or {}
        routes = dict(base.get("routes", {}))
        # ключи перекрытия идут первыми — у них приоритет
        merged = {k: {**routes.get(k, {}), **v} for k, v in (over.get("routes") or {}).items()}
        for k, v in routes.items():
            merged.setdefault(k, v)
        base = {**base, **{k: v for k, v in over.items() if k != "routes"}, "routes": merged}
    return base
