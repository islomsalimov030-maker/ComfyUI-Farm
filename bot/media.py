"""Превью для Telegram: кадр — фото, видео — mp4 ≤ лимита (иначе пережатие ffmpeg), иначе документ/текст.

ffmpeg вызывается внешним процессом (бинарь уже есть на поде, LGPL/GPL сборка — как инструмент,
не линкуется). Пережатые превью кэшируются в <farm_home>/cache/bot_<name>/.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("farm.bot.media")

MB = 1024 * 1024
TG_UPLOAD_MAX = 50 * MB          # лимит Bot API на загрузку файла
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class Preview:
    method: str                      # photo | animation | video | document | none
    path: Optional[str] = None
    note: str = ""


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return -1


def _ffmpeg(cfg: dict) -> Optional[str]:
    return shutil.which(cfg.get("ffmpeg") or "ffmpeg")


def _run(cmd: list[str], timeout: float = 300) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("ffmpeg не отработал: %s", e)
        return False
    if r.returncode != 0:
        log.warning("ffmpeg код %s: %s", r.returncode, r.stderr[-300:].decode(errors="replace"))
    return r.returncode == 0


def prepare(path: str, kind: str, cfg: dict, cache_dir: Path, key: str) -> Preview:
    src = Path(path)
    size = _size(src)
    if size < 0:
        return Preview("none", note="файл не найден")
    is_image = kind in ("frame", "last_frame", "crop") or src.suffix.lower() in IMG_EXT
    if is_image:
        return _image(src, size, cfg, cache_dir, key)
    return _video(src, size, cfg, cache_dir, key)


def _image(src: Path, size: int, cfg: dict, cache: Path, key: str) -> Preview:
    if size <= float(cfg.get("photo_max_mb", 9.5)) * MB:
        return Preview("photo", str(src))
    ff = _ffmpeg(cfg)
    if ff:
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / f"{key}.jpg"
        if out.exists() or _run([ff, "-y", "-v", "error", "-i", str(src), "-vf",
                                 "scale='min(2048,iw)':-2", "-q:v", "3", str(out)]):
            return Preview("photo", str(out), note="JPEG-превью")
    if size <= TG_UPLOAD_MAX:
        return Preview("document", str(src), note="без превью")
    return Preview("none", note=f"файл {size // MB} МБ, превью не сделать")


def _video(src: Path, size: int, cfg: dict, cache: Path, key: str) -> Preview:
    as_anim = "animation" if cfg.get("video_as_animation", True) else "video"
    limit = float(cfg.get("video_max_mb", 20)) * MB
    if size <= limit:
        return Preview(as_anim, str(src))
    ff = _ffmpeg(cfg)
    if ff:
        cache.mkdir(parents=True, exist_ok=True)
        side = int(cfg.get("video_max_side", 720))
        for scale, crf in ((side, 28), (side * 2 // 3, 32), (side // 2, 36)):
            out = cache / f"{key}_{scale}.mp4"
            vf = (f"scale='if(gt(iw,ih),min({scale},iw),-2)':'if(gt(iw,ih),-2,min({scale},ih))',"
                  "pad=ceil(iw/2)*2:ceil(ih/2)*2")
            if not out.exists():
                if not _run([ff, "-y", "-v", "error", "-i", str(src), "-an", "-vf", vf, "-c:v", "libx264",
                             "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(out)]):
                    break
            if _size(out) <= limit:
                return Preview(as_anim, str(out), note=f"пережато до {scale}px")
    if size <= TG_UPLOAD_MAX:
        return Preview("document", str(src), note="без пережатия")
    return Preview("none", note=f"видео {size // MB} МБ, превью не сделать")
