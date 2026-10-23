"""Нода-пульт «Farm Control»: старт / пауза / статус очереди, выбор профиля."""
from __future__ import annotations

import json

from . import runtime


class FarmControl:
    CATEGORY = "Next Genium/Farm"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = ("Пульт фермы. status — показать очередь; start — запустить фоновый воркер "
                   "(переживает рестарт пода); pause — остановить взятие новых задач.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "action": (["status", "start", "pause"], {"default": "status"}),
            "profile": (["* (все)"] + runtime.list_profiles(), {"default": "* (все)"}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")   # всегда выполнять: это пульт, а не вычисление

    def run(self, action: str, profile: str):
        profiles = ["*"] if profile.startswith("*") else [profile]
        status = runtime.control(action, profiles if action != "status" else None)
        text = json.dumps(status, ensure_ascii=False, indent=1)
        return {"ui": {"text": [text]}, "result": (text,)}


from .shot_node import FarmShotFromTable  # noqa: E402

from .util_nodes import NAMES as _UN, NODES as _U  # noqa: E402

NODE_CLASS_MAPPINGS = {"NGFarmControl": FarmControl, "NGFarmShotFromTable": FarmShotFromTable, **_U}
NODE_DISPLAY_NAME_MAPPINGS = {"NGFarmControl": "Farm Control (Next Genium)",
                              "NGFarmShotFromTable": "Шот из таблицы (Next Genium)", **_UN}
