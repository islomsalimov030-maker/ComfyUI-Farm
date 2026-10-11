"""Конфиг бота: YAML (bot/config.example.yaml) + доступы и токен из secrets/. Код под каждого бота не меняется."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

PKG_DIR = Path(__file__).resolve().parent
EXAMPLE = PKG_DIR / "config.example.yaml"


def _merge(base: dict, over: Optional[dict]) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _ids(v: Any) -> set[int]:
    if v is None:
        return set()
    if isinstance(v, (int, str)):
        v = [v]
    return {int(x) for x in v}


@dataclass
class Access:
    reviewers: set[int] = field(default_factory=set)
    operators: set[int] = field(default_factory=set)
    notify_chats: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Access":
        rev = _ids(d.get("reviewers"))
        ops = _ids(d.get("operators")) or set(rev)
        chats = [int(x) for x in (d.get("notify_chats") or [])] or sorted(rev | ops)
        return cls(rev | ops, ops, chats)

    def can_review(self, uid: int) -> bool:
        return uid in self.reviewers

    def can_operate(self, uid: int) -> bool:
        return uid in self.operators


@dataclass
class BotConfig:
    raw: dict
    name: str
    farm_home: Path
    db_path: Path
    state_path: Path
    cache_dir: Path
    farm_url: str
    profiles: list[str]
    access: Access
    notify: dict
    preview: dict

    def allows(self, profile: Optional[str]) -> bool:
        return "*" in self.profiles or (profile or "") in self.profiles

    def token(self, environ: Optional[dict] = None) -> str:
        env = os.environ if environ is None else environ
        name = self.raw.get("token_env") or "FARM_BOT_TOKEN"
        if env.get(name):
            return env[name].strip()
        env_file = self._path(self.raw.get("env_file"))
        if env_file and env_file.exists():
            vals = read_env_file(env_file)
            if vals.get(name):
                return vals[name]
        tf = self._path(self.raw.get("token_file"))
        if tf and tf.exists():
            return tf.read_text(encoding="utf-8").strip()
        raise RuntimeError(f"нет токена бота: задайте {name} в окружении, в {env_file} или token_file")

    def _path(self, p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        p = Path(p)
        return p if p.is_absolute() else self.farm_home / p


def read_env_file(path: Path) -> dict[str, str]:
    """Минимальный разбор .env: KEY=VALUE, # комментарии, кавычки снимаются."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        out[k] = v.strip().strip('"').strip("'")
    return out


def load(path: Optional[str | Path] = None, overrides: Optional[dict] = None) -> BotConfig:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")) or {}
    if path:
        raw = _merge(raw, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    raw = _merge(raw, overrides)
    name = str(raw.get("name") or "main")
    home = Path(raw.get("farm_home") or os.environ.get("FARM_HOME", "/workspace/farm"))
    access = dict(raw.get("access") or {})
    af = raw.get("access_file")
    if af:
        afp = Path(af) if Path(af).is_absolute() else home / af
        if afp.exists():
            access = _merge(access, yaml.safe_load(afp.read_text(encoding="utf-8")) or {})
    return BotConfig(
        raw=raw, name=name, farm_home=home,
        db_path=Path(raw.get("db_path") or home / "db" / "farm.sqlite"),
        state_path=Path(raw.get("state_path") or home / "state" / f"bot_{name}.sqlite"),
        cache_dir=home / "cache" / f"bot_{name}",
        farm_url=str(raw.get("farm_url") or "http://127.0.0.1:8188").rstrip("/"),
        profiles=list(raw.get("profiles") or ["*"]),
        access=Access.from_dict(access),
        notify=dict(raw.get("notify") or {}),
        preview=dict(raw.get("preview") or {}),
    )
