#!/usr/bin/env python3
"""WS-07: скачивание моделей серий на под (в /workspace/models/<folder>).

    python tools/ws07_fetch.py --need s1 [--dry]
    python tools/ws07_fetch.py --config tools/ws07_round_a_models.yaml --need round_a --dry

CIVITAI_TOKEN — из окружения (secrets/.env на поде, не в репо и не в логах).
Каждая загрузка идёт в отдельную временную папку; размер и, если задан, SHA256
проверяются до переноса. Существующие файлы не перезаписываются и не удаляются.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "tools" / "ws07_models.yaml"
MODELS = Path(os.environ.get("WS07_MODELS", "/workspace/models"))


def find(name: str) -> Path | None:
    roots = [MODELS, Path("/workspace/ComfyUI/models")]
    for r in roots:
        if r.exists():
            for p in r.rglob(name):
                if p.is_file():
                    return p
    return None


def items(cfg: dict, need: set[str]) -> list[dict]:
    return [m for m in cfg["models"] if m["need"] in need or m["need"] == "have"]


def verify(m: dict, path: Path) -> None:
    size = path.stat().st_size
    if not size or (m.get("bytes") is not None and size != m["bytes"]):
        raise ValueError(f"неверный размер {m['file']}: {size}, ожидали {m.get('bytes')}")
    if m.get("gb") and abs(size / 1e9 - m["gb"]) > 0.01 * m["gb"] + 0.05:
        raise ValueError(f"неверный размер {m['file']}: {size / 1e9:.2f} ГБ, ожидали {m['gb']}")
    if m.get("sha256"):
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != m["sha256"]:
            raise ValueError(f"не совпал SHA256: {m['file']}")


def fetch_hf(m: dict, dst: Path) -> None:
    from huggingface_hub import hf_hub_download
    p = Path(hf_hub_download(m["repo"], m["path"], revision=m.get("revision"), local_dir=str(dst.parent / "hf")))
    shutil.move(str(p), dst)


def fetch_civitai(url: str, dst: Path) -> None:
    tok = os.environ.get("CIVITAI_TOKEN")
    if not tok:
        raise RuntimeError("нет CIVITAI_TOKEN в окружении (Civitai отдаёт 401 без токена)")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "User-Agent": "ws07-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, length=16 << 20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--need", default="s1", help="s1 | s2 | s1,s2 | round_a")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--alt", action="store_true", help="использовать явно заданную запасную загрузку")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    need = set(a.need.split(","))
    unknown = need - {m["need"] for m in cfg["models"]}
    if unknown:
        ap.error(f"нет такой группы загрузок в манифесте: {sorted(unknown)}")
    todo, total, bad = [], 0.0, 0
    for m in items(cfg, need):
        have = find(m["file"])
        if have:
            try:
                verify(m, have)
            except (OSError, ValueError) as e:
                print(f"!! {e}; существующий файл не трогаем")
                bad += 1
            else:
                print(f"есть   {m['file']}  ({have.stat().st_size / 1e9:.2f} ГБ)  {have.parent}")
            continue
        if m["src"] == "local":
            alt = m.get("alt")
            if alt and a.alt and not find(alt["file"]):
                m = {**alt, "folder": m["folder"], "need": m["need"]}
            else:
                print(f"НЕТ    {m['file']} — локальный файл, скачать неоткуда" + (" (есть --alt)" if alt else ""))
                bad += 1
                continue
        todo.append(m)
        total += m.get("bytes", m.get("gb", 0) * 1e9) / 1e9
        print(f"качать {m['file']}  {m.get('gb')} ГБ  [{m['src']}]  {m['license']}")
    free = shutil.disk_usage(MODELS if MODELS.exists() else "/").free / 1e9
    print(f"\nк скачиванию {len(todo)} файлов, {total:.1f} ГБ; свободно на томе {free:.0f} ГБ")
    if bad:
        print(f"!! {bad} обязательных файлов отсутствуют или не прошли проверку — стоп")
        return 1
    if total > free - 20:
        print("!! места мало (оставляем запас 20 ГБ) — удаление только после согласования")
        return 1
    if a.dry:
        return 0
    for m in todo:
        dst = MODELS / m["folder"] / m["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"→ {dst}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix=".ws07-", dir=dst.parent) as tmp:
                part = Path(tmp) / m["file"]
                fetch_civitai(m["url"], part) if m["src"] == "civitai" else fetch_hf(m, part)
                verify(m, part)
                if dst.exists():
                    raise FileExistsError(f"файл появился во время загрузки, не перезаписываем: {m['file']}")
                part.rename(dst)
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            status = f" HTTP {code}" if code is not None else ""
            print(f"!! {m['file']}: {type(e).__name__}{status}; загрузка не принята")
            bad += 1
            if code == 403:
                print("403 — стоп; перед повтором сделать паузу, не опрашивать источник подряд")
                break
            continue
        print(f"   OK {dst.stat().st_size / 1e9:.2f} ГБ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
