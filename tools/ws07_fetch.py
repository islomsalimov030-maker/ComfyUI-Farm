#!/usr/bin/env python3
"""WS-07: скачивание моделей серий на под (в /workspace/models/<folder>) по tools/ws07_models.yaml.

    python tools/ws07_fetch.py --need s1 [--dry]      # что нужно, что уже есть, сколько ГБ, свободно ли на томе
    python tools/ws07_fetch.py --need s1              # качать (HF — huggingface_hub; Civitai — нужен CIVITAI_TOKEN)

CIVITAI_TOKEN — из окружения (secrets/.env на поде, не в репо и не в логах). Файл качается во временное имя
`.part` и переименовывается после проверки размера (±1 %). Уже лежащие файлы не трогает. Ничего не удаляет —
удаление проигравших после серии делается вручную по списку в STATUS.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
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
    out = []
    for m in cfg["models"]:
        if m["need"] in need or m["need"] == "have":
            out.append(m)
    return out


def fetch_hf(m: dict, dst: Path) -> None:
    from huggingface_hub import hf_hub_download
    tmp = dst.parent / ".hf_tmp"
    p = Path(hf_hub_download(m["repo"], m["path"], local_dir=str(tmp)))
    shutil.move(str(p), dst)
    shutil.rmtree(tmp, ignore_errors=True)


def fetch_civitai(url: str, dst: Path) -> None:
    tok = os.environ.get("CIVITAI_TOKEN")
    if not tok:
        raise RuntimeError("нет CIVITAI_TOKEN в окружении (Civitai отдаёт 401 без токена)")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "User-Agent": "ws07-fetch"})
    part = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, length=16 << 20)
    part.rename(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", default="s1", help="s1 | s2 | s1,s2")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--alt", action="store_true", help="10Eros: качать Civitai v1.1, если локального файла нет")
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO / "tools" / "ws07_models.yaml").read_text(encoding="utf-8"))
    need = set(a.need.split(","))
    todo, total = [], 0.0
    for m in items(cfg, need):
        have = find(m["file"])
        if have:
            print(f"есть   {m['file']}  ({have.stat().st_size / 1e9:.2f} ГБ)  {have.parent}")
            continue
        if m["src"] == "local":
            alt = m.get("alt")
            if alt and a.alt and not find(alt["file"]):
                todo.append({**alt, "folder": m["folder"], "need": m["need"]})
                total += alt["gb"]
                print(f"НЕТ    {m['file']} → вместо него {alt['file']} ({alt['gb']} ГБ)  ⚠ {alt['license']}")
            else:
                print(f"НЕТ    {m['file']} — локальный файл, скачать неоткуда" + (" (есть --alt)" if alt else ""))
            continue
        todo.append(m)
        total += m.get("gb", 0)
        print(f"качать {m['file']}  {m.get('gb')} ГБ  [{m['src']}]  {m['license']}")
    free = shutil.disk_usage(MODELS if MODELS.exists() else "/").free / 1e9
    print(f"\nк скачиванию {len(todo)} файлов, {total:.1f} ГБ; свободно на томе {free:.0f} ГБ")
    if total > free - 20:
        print("!! места мало (оставляем запас 20 ГБ) — сначала удалить ненужное")
        return 1
    if a.dry:
        return 0
    for m in todo:
        dst = MODELS / m["folder"] / m["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"→ {dst}", flush=True)
        try:
            fetch_civitai(m["url"], dst) if m["src"] == "civitai" else fetch_hf(m, dst)
        except Exception as e:  # noqa: BLE001
            print(f"!! {m['file']}: {e}")
            continue
        gb = dst.stat().st_size / 1e9
        if m.get("gb") and abs(gb - m["gb"]) > 0.01 * m["gb"] + 0.05:
            print(f"!! размер {gb:.2f} ГБ, ожидали {m['gb']} — проверить файл")
        else:
            print(f"   OK {gb:.2f} ГБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
