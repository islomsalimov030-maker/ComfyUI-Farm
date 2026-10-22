"""JSON-схемы полей, которые пишет LLM (WS-04), и проверка без внешних библиотек.

LLM отдаёт только эти поля. Строку H3 собирает шаблон: обязательные строки там, а не в LLM.
Та же схема уходит в Ollama как `format` (structured outputs).
"""
from __future__ import annotations

from typing import Any

SCHEMA_VER = "fields_v1"

_S = lambda lo=1, hi=400: {"type": "string", "minLength": lo, "maxLength": hi}  # noqa: E731

VIDEO_FIELDS: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["location_en", "framing_en", "start_pose", "end_pose", "summary_action", "phases",
                 "moving_parts", "still_parts", "extra_rules", "expression", "face_expression",
                 "idle_detail", "sounds"],
    "properties": {
        "location_en": _S(5, 200),      # «the stretching class hall by the ballet barre»
        "framing_en": _S(0, 160),       # «a full-length frontal shot»; пусто — без ракурса
        "start_pose": _S(10, 400),      # стартовая поза, стороны тела — её
        "end_pose": _S(0, 400),         # только переход (loop ВЫКЛ + end_frame): конечная поза
        "summary_action": _S(10, 300),  # действие для summary, без возврата в позу
        "phases": {
            "type": "array", "minItems": 1, "maxItems": 6,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["kind", "text", "sec"],
                "properties": {
                    "kind": {"type": "string", "enum": ["move", "hold", "return"]},
                    "text": _S(5, 300),
                    "sec": {"type": "number", "minimum": 0.3, "maximum": 4.0},
                },
            },
        },
        "moving_parts": _S(3, 160),     # «her left hand, left arm, head and eyes»
        "still_parts": _S(3, 200),      # «her right hand on the stool, legs and feet»
        "extra_rules": {"type": "array", "maxItems": 4, "items": _S(5, 200)},
        "expression": _S(5, 160),       # «a warm, natural smile throughout»
        "face_expression": _S(3, 80),   # для face pass: «warm natural smile»
        "idle_detail": _S(3, 120),      # «smiling at the camera» — в начале и в отдыхе
        "sounds": _S(0, 200),           # физические звуки действия
    },
}

FRAME_FIELDS: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["location_en", "framing_en", "placement", "pose", "expression", "summary_pose",
                 "scale_anchor"],
    "properties": {
        "location_en": _S(5, 200),
        "framing_en": _S(0, 160),
        "placement": _S(5, 200),        # где в комнате: «sitting on the pink velvet sofa, deeper in the room»
        "pose": _S(10, 500),            # поза подробно, стороны тела — её
        "expression": _S(5, 160),       # взгляд и выражение
        "summary_pose": _S(5, 200),     # коротко для summary
        "scale_anchor": _S(0, 160),     # запасной якорь «the sofa seat reaches just above her knees» (если нет карточки)
    },
}

SCHEMAS = {"video": VIDEO_FIELDS, "frame": FRAME_FIELDS}


def validate(data: Any, schema: dict, path: str = "$") -> list[str]:
    """Подмножество JSON Schema: type, required, enum, min/maxLength, min/maxItems, minimum/maximum,
    additionalProperties=false. Возвращает список ошибок (пусто — валидно)."""
    errs: list[str] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(data, dict):
            return [f"{path}: нужен объект"]
        for k in schema.get("required", []):
            if k not in data:
                errs.append(f"{path}.{k}: нет поля")
        props = schema.get("properties", {})
        for k, v in data.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}.{k}: лишнее поле")
    elif t == "array":
        if not isinstance(data, list):
            return [f"{path}: нужен массив"]
        if len(data) < schema.get("minItems", 0):
            errs.append(f"{path}: элементов меньше {schema['minItems']}")
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errs.append(f"{path}: элементов больше {schema['maxItems']}")
        for i, v in enumerate(data):
            errs += validate(v, schema.get("items", {}), f"{path}[{i}]")
    elif t == "string":
        if not isinstance(data, str):
            return [f"{path}: нужна строка"]
        if len(data.strip()) < schema.get("minLength", 0):
            errs.append(f"{path}: короче {schema['minLength']} символов")
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errs.append(f"{path}: длиннее {schema['maxLength']} символов")
        if "enum" in schema and data not in schema["enum"]:
            errs.append(f"{path}: «{data}» не из {schema['enum']}")
    elif t == "number":
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            return [f"{path}: нужно число"]
        if "minimum" in schema and data < schema["minimum"]:
            errs.append(f"{path}: меньше {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            errs.append(f"{path}: больше {schema['maximum']}")
    return errs


def check_fields(stage: str, data: Any, mode: str = "loop") -> list[str]:
    """Схема + смысловые правила (то, что JSON Schema не выразит)."""
    errs = validate(data, SCHEMAS[stage])
    if errs or stage != "video":
        return errs
    phases = data["phases"]
    if mode == "loop":
        # WS-10 (22.10): qwen3-vl:8b в ~40 % петель пишет последний шаг-возврат как "move" (текст — возврат в
        # стартовую позу). Ярлык чиним кодом, без лишнего круга LLM; текст по-прежнему проверяет линтер.
        if len(phases) >= 2 and phases[-1]["kind"] == "move" and not any(p["kind"] == "return" for p in phases):
            phases[-1]["kind"] = "return"
        if phases[-1]["kind"] != "return":
            errs.append("$.phases: в петле последняя фаза — return (возврат в стартовую позу)")
        if sum(1 for p in phases if p["kind"] == "return") > 1:
            errs.append("$.phases: возврат (return) должен быть один")
        if data.get("end_pose", "").strip():
            errs.append("$.end_pose: в петле пусто (конечная поза = стартовая)")
    else:
        if any(p["kind"] == "return" for p in phases):
            errs.append("$.phases: в переходе нет возврата (return) — движение идёт в конечную позу")
        if not data.get("end_pose", "").strip():
            errs.append("$.end_pose: для перехода опишите конечную позу")
    if not any(p["kind"] in ("move", "return") for p in phases):
        errs.append("$.phases: нет ни одного движения")
    return errs
