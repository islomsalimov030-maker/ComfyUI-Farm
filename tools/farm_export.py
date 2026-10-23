#!/usr/bin/env python3
"""Выгрузка итогов фермы для ПК (23.10): только итоговые результаты, без raw.

    python tools/farm_export.py OUT_DIR [--approved] [--shots ID ...] [--zip]

На каждый шот: `<ID>.mp4` — итоговое видео FullHD (текущий кандидат после постобработки, иначе последнее видео),
`<ID>.jpg` — итоговый кадр с проходом лица, приведённый к 1920×1080. `--approved` — только принятые.
`--zip` — дополнительно OUT_DIR.zip (одним файлом удобнее скачивать через JupyterLab).
Уже выгруженный файл с тем же sha256 не переписывается — запускать можно повторно по мере готовности.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from farm.db import DB  # noqa: E402

FULLHD = (1920, 1080)


def _db() -> DB:
    home = Path(os.environ.get("FARM_HOME", "/workspace/farm"))
    return DB(home / "db" / "farm.sqlite")


def _asset(db: DB, sid: str, kind: str, current: str | None) -> dict | None:
    if current:
        a = db.one("SELECT * FROM assets WHERE asset_id=?", (current,))
        if a:
            return a
    return db.one("SELECT a.* FROM assets a JOIN attempts t USING(attempt_id) WHERE t.shot_id=? AND a.kind=? "
                  "AND a.path NOT LIKE '%raw%' ORDER BY a.created_at DESC LIMIT 1", (sid, kind))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _frame_fullhd(src: Path, dst: Path) -> None:
    from PIL import Image
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = max(FULLHD[0] / w, FULLHD[1] / h)          # заполнить 16:9 и обрезать по центру
    im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    x, y = (im.width - FULLHD[0]) // 2, (im.height - FULLHD[1]) // 2
    im.crop((x, y, x + FULLHD[0], y + FULLHD[1])).save(dst, quality=95)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--approved", action="store_true")
    ap.add_argument("--shots", nargs="*")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    db, out = _db(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q, args = "SELECT * FROM shots", []
    if a.shots:
        q += f" WHERE shot_id IN ({','.join('?' * len(a.shots))})"; args = [s.upper() for s in a.shots]
    n = 0
    for s in db.all(q + " ORDER BY shot_id", args):
        if a.approved and s["state"] != "approved":
            continue
        sid = s["shot_id"]
        v = _asset(db, sid, "video", s.get("video_asset_id"))
        f = _asset(db, sid, "frame", s.get("frame_asset_id"))
        if v and Path(v["path"]).is_file():
            dst = out / f"{sid}.mp4"
            if not dst.exists() or _sha(dst) != _sha(Path(v["path"])):
                shutil.copyfile(v["path"], dst); n += 1
        if f and Path(f["path"]).is_file():
            dst = out / f"{sid}.jpg"
            stamp = out / f".{sid}.src"
            if not dst.exists() or not stamp.exists() or stamp.read_text() != f["asset_id"]:
                _frame_fullhd(Path(f["path"]), dst); stamp.write_text(f["asset_id"]); n += 1
        print(f"{sid:10} {s['state']:13} видео {'да' if v else '—'}  кадр {'да' if f else '—'}")
    if a.zip:
        z = out.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            for p in sorted(out.iterdir()):
                if not p.name.startswith("."):
                    zf.write(p, p.name)
        print(z)
    print(f"обновлено файлов: {n} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
