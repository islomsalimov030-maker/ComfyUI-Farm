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
    rep = {"lib_issues": []}
    loc = sh._localize_ref(rc, "location_image", rep) if rc.norm.get("location_image") else None
    mr = sh._localize_ref(rc, "motion_ref", rep) if rc.norm.get("motion_ref") else None
    pr = sh._localize_ref(rc, "pose_ref", rep) if rc.norm.get("pose_ref") else None
    if not rc.ok:
        raise ValueError(f"строка {rc.id}: {rc.status_text()}")
    shot = to_shot(rc, pose_ref=pr, motion_ref=mr, location_image=loc)
    for k in ("objects_json", "depends_on_json"):
        shot[k] = json.dumps(shot.get(k) or [], ensure_ascii=False)
    return shot, lib, [r.id for r in ready]


def build_prompts(shot: dict, lib, fields: dict | None = None, llm: dict | None = None,
                  no_cache: bool = False) -> dict:
    """{stage: PromptResult} для кадра и видео. fields = {stage: поля} — готовые поля LLM (без вызова LLM);
    llm — перекрытие profile.llm (prompt_model, url)."""
    from .runtime import get_worker
    w = get_worker()
    prof = w.profile(shot["profile"])
    if llm:
        prof = {**prof, "llm": {**(prof.get("llm") or {}), **{k: v for k, v in llm.items() if v}}}
    out = {}
    for stage in ("frame", "video"):
        sp = prof.get(stage) or {}
        graph, params = w.workflow(sp["workflow"])
        ctx = {"try_no": 0, "fail_codes": [], "knobs": [], "template": sp.get("prompt_template"),
               "workflow": sp["workflow"], "params": params, "no_cache": no_cache}
        if fields and fields.get(stage) is not None:
            ctx["fields"] = fields[stage]
        pr = w.plugins.prompts.build(stage, shot, lib, prof, ctx=ctx)
        errs = [f"{getattr(i, 'code', '')}: {getattr(i, 'msg', '')}" for i in (getattr(pr, "lint", None) or [])
                if getattr(i, "severity", "") == "error"]
        if errs:
            raise ValueError(f"{shot['shot_id']} · промпт {stage} не прошёл линтер: " + "; ".join(errs))
        out[stage] = pr
    return out


class FarmPromptLLM:
    """25.10: LLM вынесен из «Шот из таблицы» в отдельную видимую ноду — видно модель, откуда поля (LLM/кэш),
    время и сам ответ; ответ можно перезапросить (fresh) не трогая остальной граф."""
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("fields", "table", "shot_id", "report")
    OUTPUT_TOOLTIPS = ("Поля промптов кадра и видео (JSON) → вход fields ноды «Шот из таблицы».",
                       "Путь к таблице (тот же, что в поле table).", "ID выбранной строки.",
                       "Модель, источник полей, время, ответ LLM — для проверки глазами.")
    DESCRIPTION = ("LLM фермы (Ollama): строка таблицы сцен → поля промптов H3 (поза, выражение, фазы движения…) "
                   "по системному промпту и JSON-схеме. Текст промпта из полей собирает «Шот из таблицы» "
                   "по шаблонам H3 + линтер. Строку выбирает эта нода: row → control «increment» и Queue ×N.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "table": ("STRING", {"default": "", "tooltip": "Путь к xlsx таблицы. Пусто — таблица из farm.yaml."}),
            "row": ("INT", {"default": 1, "min": 1, "max": 9999, "control_after_generate": "increment",
                            "tooltip": "Номер строки среди отмеченных «В работу» (1 — первая). "
                                       "increment + Queue ×N = N строк подряд."}),
            "shot_id": ("STRING", {"default": "", "tooltip": "ID шота (SAMH-STD01). Заполнено — важнее номера строки."}),
            "model": ("STRING", {"default": "", "tooltip": "Модель Ollama (qwen3-vl:8b-instruct). Пусто — из профиля "
                                                           "(llm.prompt_model)."}),
            "ollama_url": ("STRING", {"default": "", "tooltip": "Адрес Ollama. Пусто — из профиля / http://127.0.0.1:11434."}),
            "fresh": ("BOOLEAN", {"default": False, "tooltip": "Спросить LLM заново, не брать ответ из кэша."}),
        }}

    def run(self, table: str, row: int, shot_id: str, model: str, ollama_url: str, fresh: bool):
        import time
        shot, lib, ready = load_row(table, row, shot_id)
        t0 = time.time()
        prs = build_prompts(shot, lib, llm={"prompt_model": model.strip(), "url": ollama_url.strip()},
                            no_cache=fresh)
        fields = {st: (pr.fields_json or {}).get("fields") for st, pr in prs.items()}
        lines = [f"{shot['shot_id']} · строка {ready.index(shot['shot_id']) + 1} из {len(ready)} · {time.time() - t0:.1f} с"]
        for st, pr in prs.items():
            meta = (pr.fields_json or {}).get("meta") or {}
            llm = meta.get("llm") or {}
            lines.append(f"[{st}] источник: {meta.get('fields_source')} · модель: {llm.get('model') or '—'} · "
                         f"шаблон: {meta.get('template')} · правок линтера: {meta.get('lint_rounds', 0)}")
        lines.append(json.dumps(fields, ensure_ascii=False, indent=1))
        report = "\n".join(lines)
        return {"ui": {"text": [report]},
                "result": (json.dumps(fields, ensure_ascii=False), _table_path(table), shot["shot_id"], report)}

    @classmethod
    def IS_CHANGED(cls, table, row, shot_id, model, ollama_url, fresh):
        if fresh:
            return float("nan")
        try:
            return f"{Path(_table_path(table)).stat().st_mtime}:{row}:{shot_id}:{model}:{ollama_url}"
        except Exception:   # noqa: BLE001
            return float("nan")


class FarmShotFromTable:
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "IMAGE", "STRING", "STRING",
                    "FLOAT", "BOOLEAN", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("location", "face_mod", "body_mod", "frame_prompt", "frame_face_prompt", "face_ref",
                    "video_prompt", "video_face_prompt", "duration_sec", "loop", "shot_id", "info",
                    "zone", "motion_ref", "pose_ref")
    OUTPUT_TOOLTIPS = (
        "Локация <Picture 1>: «Картинка локации» строки или лист ракурсов зоны.",
        "Мод лица персонажа (farm/<персонаж>_face).", "Мод тела — персонаж в наряде (farm/<вариант>_body).",
        "Промпт кадра H3.", "Промпт прохода лица кадра.", "Эталон лица персонажа для видео <Picture 2>.",
        "Промпт видео-петли H3.", "Промпт прохода лица видео.", "Длительность видео из таблицы, с.",
        "Петля из таблицы.", "ID строки.", "Что взято из таблицы (для проверки глазами).",
        "ID зоны из таблицы (по нему выбирается загруженная картинка локации).",
        "Путь к видео-рефу движения строки (пусто — нет).", "Путь к фото-рефу позы строки (пусто — нет).")
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
        }, "optional": {
            "fields": ("STRING", {"forceInput": True, "tooltip": "Поля промптов от ноды «LLM · поля промптов». "
                                                                 "Не подключено — LLM вызывается внутри этой ноды."}),
        }}

    def run(self, table: str, row: int, shot_id: str, fields: str = ""):
        shot, lib, ready = load_row(table, row, shot_id)
        fl = json.loads(fields) if (fields or "").strip() else None
        prs = build_prompts(shot, lib, fields=fl)
        out = {st: (pr.text, ((pr.fields_json or {}).get("inputs") or {}).get("face_prompt", ""))
               for st, pr in prs.items()}
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
               dur, loop, shot["shot_id"], info, str(extra.get("zone") or shot.get("plate") or ""),
               str(shot.get("motion_ref") or ""), str(shot.get("pose_ref") or ""))
        return {"ui": {"text": [info]}, "result": res}

    @classmethod
    def IS_CHANGED(cls, table, row, shot_id, fields=""):
        try:
            p = Path(_table_path(table))
            return f"{p.stat().st_mtime}:{row}:{shot_id}:{fields}"
        except Exception:   # noqa: BLE001
            return float("nan")
