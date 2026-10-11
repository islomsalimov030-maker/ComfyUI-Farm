"""Thompson sampling по рукам (модель × LoRA × сэмплер) в разрезе категорий шота.

Категория = поза (лежа/сидя/стоя/other) × локация. Холодный старт: пока у какой-то руки в категории
меньше cold_start_n попыток, выдаём руку с наименьшим n (поровну). Дальше — Beta(1+s, 1+f).
Успех = лучший кандидат попытки прошёл QA; позже вердикт человека корректирует статистику.
"""
from __future__ import annotations

import random
import re
from typing import Optional

from .db import DB, dumps

# Порядок важен: «сидит за столом» → sitting раньше, чем standing по « сто».
POSE_KEYS = [("lying", (" леж", " лёж", " lying", " lies", " lie ")),
             ("sitting", (" сид", " сел", " присе", " sitting", " seated", " sits")),
             ("standing", (" сто", " standing", " stands", " stand "))]


def pose_class(pose_ru: Optional[str]) -> str:
    t = " " + re.sub(r"[^\w]+", " ", (pose_ru or "").lower()) + " "
    for cls, keys in POSE_KEYS:
        if any(k in t for k in keys):
            return cls
    return "other"


def category(shot: dict) -> str:
    return f"{pose_class(shot.get('pose_ru'))}:{shot.get('location') or '-'}"


def arm_key(profile: str, stage: str, arm_local_id: str) -> str:
    return f"{profile}/{stage}/{arm_local_id}"


def sync_arms(db: DB, profile: dict) -> None:
    """Записать руки профиля в arms (новые — enabled=1; существующие не включаем принудительно)."""
    name = profile["name"]
    for stage in ("frame", "video"):
        for arm in (profile.get(stage) or {}).get("arms") or []:
            if not isinstance(arm, dict) or "id" not in arm:
                continue
            aid = arm_key(name, stage, arm["id"])
            db.execute(
                "INSERT INTO arms(arm_id, profile, model, loras_json, sampler_json, enabled, stage) "
                "VALUES (?,?,?,?,?,1,?) ON CONFLICT(arm_id) DO UPDATE SET model=excluded.model, "
                "loras_json=excluded.loras_json, sampler_json=excluded.sampler_json, stage=excluded.stage",
                (aid, name, arm.get("model"), dumps(arm.get("loras") or []),
                 dumps(arm.get("sampler") or {}), stage))


class Bandit:
    def __init__(self, db: DB, cold_start_n: int = 3, rng: Optional[random.Random] = None):
        self.db = db
        self.cold_start_n = cold_start_n
        self.rng = rng or random.Random()

    def stats(self, arm_ids: list[str], cat: str) -> dict[str, dict]:
        out = {a: {"n": 0, "successes": 0, "score_sum": 0.0} for a in arm_ids}
        if not arm_ids:
            return out
        qs = ",".join("?" for _ in arm_ids)
        for r in self.db.all(f"SELECT * FROM arm_stats WHERE category=? AND arm_id IN ({qs})",
                             [cat, *arm_ids]):
            out[r["arm_id"]] = {"n": r["n"], "successes": r["successes"], "score_sum": r["score_sum"]}
        return out

    def enabled(self, arm_ids: list[str]) -> list[str]:
        if not arm_ids:
            return []
        qs = ",".join("?" for _ in arm_ids)
        off = {r["arm_id"] for r in self.db.all(
            f"SELECT arm_id FROM arms WHERE enabled=0 AND arm_id IN ({qs})", arm_ids)}
        return [a for a in arm_ids if a not in off]

    def choose(self, arm_ids: list[str], cat: str) -> str:
        arms = self.enabled(arm_ids)
        if not arms:
            raise ValueError("нет включённых рук")
        if len(arms) == 1:
            return arms[0]
        st = self.stats(arms, cat)
        min_n = min(st[a]["n"] for a in arms)
        if min_n < self.cold_start_n:
            cands = [a for a in arms if st[a]["n"] == min_n]
            return self.rng.choice(cands)
        draws = {a: self.rng.betavariate(1 + st[a]["successes"],
                                         1 + st[a]["n"] - st[a]["successes"]) for a in arms}
        return max(draws, key=draws.get)

    def update(self, arm_id: str, cat: str, success: bool, score: float = 0.0) -> None:
        self.db.execute(
            "INSERT INTO arm_stats(arm_id, category, n, successes, score_sum) VALUES (?,?,1,?,?) "
            "ON CONFLICT(arm_id, category) DO UPDATE SET n=n+1, successes=successes+excluded.successes, "
            "score_sum=score_sum+excluded.score_sum",
            (arm_id, cat, int(bool(success)), float(score)))

    def correct(self, arm_id: str, cat: str, success_delta: int) -> None:
        """Поправка по вердикту человека (принял провал QA / отверг прошедшее), без изменения n."""
        self.db.execute(
            "UPDATE arm_stats SET successes=MAX(0, MIN(n, successes+?)) WHERE arm_id=? AND category=?",
            (int(success_delta), arm_id, cat))
