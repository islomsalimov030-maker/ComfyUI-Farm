#!/usr/bin/env python3
"""Ревью шотов без Telegram-бота (WS-10): список кандидатов, контакт-лист, вердикты в `reviews`.

    python tools/farm_review.py list                      # шоты: состояние, текущие кадр/видео, причина
    python tools/farm_review.py sheet out.jpg [--all]     # контакт-лист текущих кадров (или всех кандидатов)
    python tools/farm_review.py accept STD-01 frame       # вердикт по текущему кадру/видео шота
    python tools/farm_review.py redo STD-01 video "рука уходит за край кадра"

Вердикт пишется так же, как из бота (`bot.store.FarmStore.record`): воркер подхватит его на следующем тике
(CONTRACTS §4: accept / redo / redo_comment). Ревьюер по умолчанию — `cli`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bot.store import FarmStore  # noqa: E402
from farm.db import DB, new_id  # noqa: E402


def _db() -> DB:
    home = Path(os.environ.get("FARM_HOME", "/workspace/farm"))
    return DB(home / "db" / "farm.sqlite")


def _reason(db: DB, sid: str) -> str:
    r = db.one("SELECT reason FROM events WHERE shot_id=? ORDER BY ts DESC LIMIT 1", (sid,))
    return (r["reason"] if r else "") or ""


def _asset(db: DB, aid):
    return db.one("SELECT * FROM assets WHERE asset_id=?", (aid,)) if aid else None


def cmd_list(db: DB) -> int:
    for s in db.all("SELECT * FROM shots ORDER BY shot_id"):
        fa, va = _asset(db, s.get("frame_asset_id")), _asset(db, s.get("video_asset_id"))
        print(f"{s['shot_id']:8} {s['state']:13} try={s.get('stage_try') or 0}  {_reason(db, s['shot_id'])[:90]}")
        for kind, a in (("кадр", fa), ("видео", va)):
            if a:
                ok = {1: "✓", 0: "·"}.get(a.get("approved"), "·")
                print(f"           {kind}: {ok} qa={a.get('qa_passed')} {a['path']}")
    return 0


def cmd_sheet(db: DB, out: str, all_: bool) -> int:
    from PIL import Image, ImageDraw
    if all_:
        rows = db.all("SELECT a.path, t.shot_id FROM assets a JOIN attempts t ON t.attempt_id=a.attempt_id "
                      "WHERE t.stage='frame' AND a.path LIKE '%.png' AND a.path NOT LIKE '%frame_raw%' ORDER BY t.shot_id")
    else:
        rows = [{"path": a["path"], "shot_id": s["shot_id"]} for s in db.all("SELECT * FROM shots ORDER BY shot_id")
                if (a := _asset(db, s.get("frame_asset_id")))]
    if not rows:
        print("нет кадров"); return 1
    w, tiles = 640, []
    for r in rows:
        im = Image.open(r["path"]).convert("RGB"); im.thumbnail((w, w))
        d = ImageDraw.Draw(im); d.rectangle([0, 0, w, 22], fill=(0, 0, 0))
        d.text((6, 4), f"{r['shot_id']}  {Path(r['path']).parent.name}", fill=(255, 255, 0)); tiles.append(im)
    h = max(t.height for t in tiles); cols = 2
    sheet = Image.new("RGB", (cols * w, (len(tiles) + cols - 1) // cols * h), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    sheet.save(out, quality=88); print(out, len(tiles))
    return 0


def cmd_verdict(db: DB, verdict: str, sid: str, stage: str, comment: str, reviewer: str) -> int:
    s = db.shot(sid)
    aid = s.get("frame_asset_id") if stage == "frame" else s.get("video_asset_id")
    if not aid:
        print(f"у {sid} нет текущего ассета стадии {stage} (состояние {s['state']})"); return 1
    if verdict == "redo" and comment.strip():
        verdict = "redo_comment"
    ok, why = FarmStore(db).record(new_id("rv"), aid, reviewer, verdict, comment or None)
    print("записано" if ok else f"не записано: {why}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "sheet", "accept", "redo"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--reviewer", default="cli")
    a = ap.parse_args()
    db = _db()
    if a.cmd == "list":
        return cmd_list(db)
    if a.cmd == "sheet":
        return cmd_sheet(db, a.args[0] if a.args else "review_sheet.jpg", a.all)
    if len(a.args) < 2 or a.args[1] not in ("frame", "video"):
        ap.error("нужно: <shot_id> frame|video [комментарий]")
    return cmd_verdict(db, a.cmd, a.args[0], a.args[1], " ".join(a.args[2:]), a.reviewer)


if __name__ == "__main__":
    sys.exit(main())
