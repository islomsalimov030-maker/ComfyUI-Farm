"""Нода «Шот из таблицы» (23.10): строка таблицы сцен → всё, что нужно движкам кадра и видео.

Тот же код, что у автоматики: валидатор таблицы (farm/sheets.py), справочники (farm/library.py), сборщик промптов
(farm/prompts, LLM через Ollama + шаблоны H3 + линтер). Ручной воркфлоу `farm_table@1` и ферма дают одинаковые входы.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("farm")


def _img(path: str | None):
    import numpy as np
    import torch
    from PIL import Image
    if not path or not Path(path).is_file():
        return torch.zeros((1, 64, 64, 3))
    im = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(im).astype("float32") / 255.0)[None]


def _table_path(table: str) -> str:
    if table.strip():
        return table.strip()
    from .config import load_config
    cfg = load_config()
    p = (cfg.get("sheets") or {}).get("xlsx")
    if not p:
        raise ValueError("не задан путь к таблице: впишите его в поле table или в farm.yaml → sheets.xlsx")
    return p


def load_row(table: str, row: int, shot_id: str):
    """(shot, lib, ready_ids): строка «В работу» по номеру (1…N) или по ID."""
    from .config import load_config
    from .sheets import Sheets, XlsxBackend, to_shot
    cfg = load_config()
    sh = Sheets(XlsxBackend(_table_path(table)), cfg)
    lib, checks = sh.check(check_files=False)
    ready = [rc for rc in checks if rc.ready]
    if not ready:
        raise ValueError("в таблице нет строк с галочкой «В работу»")
    if shot_id.strip():
        rc = next((r for r in ready if r.id.upper() == shot_id.strip().upper()), None)
        if rc is None:
            raise ValueError(f"нет строки «В работу» с ID {shot_id}")
    else:
        rc = ready[(max(1, row) - 1) % len(ready)]
    if not rc.ok:
        raise ValueError(f"строка {rc.id}: {rc.status_text()}")
    loc = rc.norm.get("location_image")
    shot = to_shot(rc, location_image=sh._localize_ref(rc, "location_image", {"lib_issues": []}) if loc else None)
    for k in ("objects_json", "depends_on_json"):
        shot[k] = json.dumps(shot.get(k) or [], ensure_ascii=False)
    return shot, lib, [r.id for r in ready]


class FarmShotFromTable:
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "IMAGE", "STRING", "STRING",
                    "FLOAT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("location", "face_mod", "body_mod", "frame_prompt", "frame_face_prompt", "face_ref",
                    "video_prompt", "video_face_prompt", "duration_sec", "loop", "shot_id", "info")
    OUTPUT_TOOLTIPS = (
        "Локация <Picture 1>: «Картинка локации» строки или лист ракурсов зоны.",
        "Мод лица персонажа (farm/<персонаж>_face).", "Мод тела — персонаж в наряде (farm/<вариант>_body).",
        "Промпт кадра H3.", "Промпт прохода лица кадра.", "Эталон лица персонажа для видео <Picture 2>.",
        "Промпт видео-петли H3.", "Промпт прохода лица видео.", "Длительность видео из таблицы, с.",
        "Петля из таблицы.", "ID строки.", "Что взято из таблицы (для проверки глазами).")
    DESCRIPTION = ("Шот из таблицы сцен: номер строки с галочкой «В работу» (1…N) или ID → локация, моды, промпты "
                   "кадра и видео, длительность. Промпты пишет LLM фермы (Ollama) по шаблонам H3. "
                   "Чтобы прогнать N строк подряд: row → control «increment» и Queue ×N.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "table": ("STRING", {"default": "", "tooltip": "Путь к xlsx таблицы. Пусто — таблица из farm.yaml."}),
            "row": ("INT", {"default": 1, "min": 1, "max": 9999, "control_after_generate": "increment",
                            "tooltip": "Номер строки среди отмеченных «В работу» (1 — первая). "
                                       "increment + Queue ×N = N строк подряд."}),
            "shot_id": ("STRING", {"default": "", "tooltip": "ID шота (SAMH-STD01). Заполнено — важнее номера строки."}),
        }}

    def run(self, table: str, row: int, shot_id: str):
        from .runtime import get_worker
        shot, lib, ready = load_row(table, row, shot_id)
        w = get_worker()
        prof = w.profile(shot["profile"])
        out = {}
        for stage in ("frame", "video"):
            sp = prof.get(stage) or {}
            graph, params = w.workflow(sp["workflow"])
            pr = w.plugins.prompts.build(stage, shot, lib, prof,
                                         ctx={"try_no": 0, "fail_codes": [], "knobs": [],
                                              "template": sp.get("prompt_template"),
                                              "workflow": sp["workflow"], "params": params})
            errs = [f"{getattr(i, 'code', '')}: {getattr(i, 'msg', '')}" for i in (getattr(pr, "lint", None) or [])
                    if getattr(i, "severity", "") == "error"]
            if errs:
                raise ValueError(f"{shot['shot_id']} · промпт {stage} не прошёл линтер: " + "; ".join(errs))
            out[stage] = (pr.text, ((pr.fields_json or {}).get("inputs") or {}).get("face_prompt", ""))
        fr = lib.resolve_inputs(shot, "frame")
        vr = lib.resolve_inputs(shot, "video")
        dur = float(shot.get("duration_s") or 8.0)
        loop = True if shot.get("loop") is None else bool(shot["loop"])
        extra = shot.get("extra_json") or {}
        info = (f"{shot['shot_id']} · строка {ready.index(shot['shot_id']) + 1} из {len(ready)}\n"
                f"персонаж: {shot.get('character')} · зона: {extra.get('zone')} · {extra.get('pose_class')} · "
                f"план: {extra.get('plan')} · лицо: {extra.get('face_in_frame')}\n"
                f"локация: {fr.get('location_sheet')}\nмоды: {fr.get('char_face_mod')} + {fr.get('char_body_mod')}\n"
                f"видео: {dur} с, петля: {'да' if loop else 'нет'}")
        res = (_img(fr.get("location_sheet")), fr.get("char_face_mod") or "(none)", fr.get("char_body_mod") or "(none)",
               out["frame"][0], out["frame"][1], _img(vr.get("face_ref")), out["video"][0], out["video"][1],
               dur, loop, shot["shot_id"], info)
        return {"ui": {"text": [info]}, "result": res}

    @classmethod
    def IS_CHANGED(cls, table, row, shot_id):
        try:
            p = Path(_table_path(table))
            return f"{p.stat().st_mtime}:{row}:{shot_id}"
        except Exception:   # noqa: BLE001
            return float("nan")
