"""Клиент MiVOLO v2: держит подпроцесс `mivolo_worker.py` в отдельном venv (torch/timm 0.8.13 не ставим в venv фермы).

Питон venv: `FARM_MIVOLO_PY` или `frame.age.python` в порогах; по умолчанию
`/workspace/venv_mivolo/bin/python` (под) или `~/.cache/farm_qa/venv_mivolo/Scripts/python.exe` (ПК).
Кропы: лицо — рамка детектора, расширенная `face_pad` (MiVOLO учили на рамках YOLO, они шире MediaPipe);
тело — по точкам MediaPipe Pose с полями.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

WORKER = Path(__file__).with_name("mivolo_worker.py")
_proc: Optional[subprocess.Popen] = None
_lock = threading.Lock()


class MivoloUnavailable(RuntimeError):
    pass


def python_path(cfg_py: Optional[str] = None) -> Optional[Path]:
    cands = [os.environ.get("FARM_MIVOLO_PY"), cfg_py, "/workspace/venv_mivolo/bin/python",
             str(Path.home() / ".cache" / "farm_qa" / "venv_mivolo" / "Scripts" / "python.exe"),
             str(Path.home() / ".cache" / "farm_qa" / "venv_mivolo" / "bin" / "python")]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


def _start(cfg_py: Optional[str]) -> subprocess.Popen:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc
    py = python_path(cfg_py)
    if py is None:
        raise MivoloUnavailable("нет venv MiVOLO (FARM_MIVOLO_PY / frame.age.python)")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    _proc = subprocess.Popen([str(py), "-I", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=env, bufsize=1)
    ready = _proc.stdout.readline()
    if not ready or not json.loads(ready).get("ready"):
        stop()
        raise MivoloUnavailable(f"MiVOLO не стартовал: {ready!r}")
    return _proc


def stop() -> None:
    """Выгрузить модель (процесс) — группировка задач по модели."""
    global _proc
    if _proc is not None:
        try:
            _proc.stdin.close()
            _proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            _proc.kill()
        _proc = None


atexit.register(stop)


def pad_box(b, k: float, w: int, h: int) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in b)
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * k / 2, (y1 - y0) * k / 2
    return [max(cx - hw, 0), max(cy - hh, 0), min(cx + hw, w), min(cy + hh, h)]


def body_box(pose: Optional[dict], w: int, h: int, pad: float = 0.12) -> Optional[list[float]]:
    """Рамка тела по видимым точкам MediaPipe Pose (+поля, голова сверху)."""
    if not pose:
        return None
    pts = np.array([v[:2] for k, v in pose.items() if not k.startswith("_") and v[2] >= 0.3])
    if len(pts) < 4:
        return None
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return [max(x0 - px, 0), max(y0 - 2 * py, 0), min(x1 + px, w), min(y1 + py, h)]


def predict(path: str, face: Optional[list], body: Optional[list], cfg_py: Optional[str] = None) -> dict:
    with _lock:
        p = _start(cfg_py)
        try:
            p.stdin.write(json.dumps({"path": str(path), "face": face, "body": body}) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
        except (BrokenPipeError, OSError) as e:
            stop()
            raise MivoloUnavailable(f"MiVOLO упал: {e}") from e
    if not line:
        stop()
        raise MivoloUnavailable("MiVOLO не ответил")
    res = json.loads(line)
    if "error" in res:
        raise MivoloUnavailable(res["error"])
    return res


if __name__ == "__main__":  # быстрая проверка: python -m farm.qa.age_mivolo <картинка>
    from . import face as F, mp
    img = F.load_bgr(sys.argv[1])
    fs = F.detect(img)
    ps = mp.pose(img)
    h, w = img.shape[:2]
    print(predict(sys.argv[1], pad_box(fs[0].bbox, 1.25, w, h) if fs else None, body_box(ps[0] if ps else None, w, h)))
