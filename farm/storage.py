"""Пути фермы (CONTRACTS §2) и неизменяемое хранилище результатов.

Ассет пишется один раз: tmp → fsync → rename → chmod 0444. Повторная запись того же содержимого
возвращает существующий файл; другого содержимого по тому же пути — ошибка (неизменяемость).
"""
from __future__ import annotations

import hashlib
import logging
import os
import struct
from pathlib import Path
from typing import Optional

from .config import farm_home

log = logging.getLogger("farm.storage")


class Paths:
    def __init__(self, home: Optional[Path] = None):
        self.home = Path(home or farm_home())
        self.db = self.home / "db" / "farm.sqlite"
        self.refs = self.home / "refs"
        self.results = self.home / "results"
        self.cache = self.home / "cache"
        self.logs = self.home / "logs"
        self.secrets = self.home / "secrets"
        self.state = self.home / "state"

    def ensure(self) -> "Paths":
        for p in (self.db.parent, self.refs, self.results, self.cache, self.logs, self.state):
            p.mkdir(parents=True, exist_ok=True)
        self.secrets.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.secrets, 0o700)
        except OSError:
            pass
        return self

    def attempt_dir(self, shot_id: str, attempt_id: str) -> Path:
        return self.results / safe_name(shot_id) / safe_name(attempt_id)


def safe_name(s: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))
    return keep.strip(".") or "_"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class ImmutableError(RuntimeError):
    pass


def write_once(dst: Path, data: bytes) -> str:
    """Записать файл атомарно и сделать только для чтения. Возвращает sha256."""
    digest = sha256_bytes(data)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256_file(dst) == digest:
            return digest
        raise ImmutableError(f"{dst} уже существует с другим содержимым")
    tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    try:
        os.chmod(dst, 0o444)
    except OSError:
        pass
    return digest


def image_size(data: bytes) -> tuple[Optional[int], Optional[int]]:
    """Размер PNG/JPEG/WebP без PIL (на поде PIL есть, но ядро от него не зависит)."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", data[16:24])
            return w, h
        if data[:2] == b"\xff\xd8":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return w, h
                seg = struct.unpack(">H", data[i + 2:i + 4])[0]
                i += 2 + seg
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and data[12:16] == b"VP8X":
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return w, h
    except (struct.error, IndexError):
        pass
    return None, None


# --- выгрузка финалов (WS-03 даст drive-клиент; пока заглушка) -----------------

class NullUploader:
    """Заглушка: ничего не выгружает. Реальная — farm.sheets.upload_final (WS-03)."""
    is_stub = True

    def upload_final(self, asset_path: str, shot: dict) -> Optional[str]:
        log.info("upload_final(stub): %s → (нет drive-клиента)", asset_path)
        return None


def verified_upload(uploader, asset_path: str, expected_sha: str, shot: dict) -> Optional[str]:
    """Перед выгрузкой сверить sha256 файла с записью в БД (защита от подмены/порчи)."""
    actual = sha256_file(asset_path)
    if actual != expected_sha:
        raise ImmutableError(f"sha256 не совпал для {asset_path}: {actual} != {expected_sha}")
    return uploader.upload_final(asset_path, shot)
