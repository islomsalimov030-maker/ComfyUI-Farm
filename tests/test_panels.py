"""WS-14 (29.10): пульты фермы — выбор модели, сила turbo-LoRA, число шагов.

Пульт `NGFarmPanelVideo`/`NGFarmPanelPhoto` берёт из таблицы `FARM_MODELS` три вещи:
файл модели, силу turbo-LoRA и число шагов сэмплера. Отсюда — скорость генерации:
turbo-модели (ref2v) идут 8 шагов, дообученные без turbo (Singularity, 10Eros, PinkCherry) — 20,
то есть в 2.5 раза дольше. Ручка `steps_override` (optional) позволяет перекрыть число шагов вручную.

Тесты не требуют torch — пульты работают на чистом Python.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import util_nodes as U  # noqa: E402


# ---------------------------------------------------------------- WS-14 (29.10): пульт — модель, turbo, шаги
def _panel_kwargs(**over):
    """Базовые значения пульта ВИДЕО (все ручки VIDEO_SPEC)."""
    kw = {"use_Singularity": False, "use_ref2v": True, "use_10Eros": False, "use_PinkCherry": False,
          "shift": 12.0, "color_lock": -1.0, "seam_crossfade": -1, "temporal_strength": 1.0,
          "face_confidence": -1.0, "crop_factor": 2.5, "large_face_mult": 0.35, "face_lock": 1.0,
          "hand_strength": 0.2}
    kw.update(over)
    return kw


def test_panel_model_turbo_and_steps():
    """Пульт отдаёт модель, силу turbo-LoRA и число шагов — по таблице FARM_MODELS."""
    pv = U.NODES["NGFarmPanelVideo"]()
    out = pv.run(None, **_panel_kwargs())
    assert out[0] == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"   # ref2v
    assert out[1] == 1.0 and out[2] == 8                                  # turbo-LoRA + 8 шагов
    out = pv.run(None, **_panel_kwargs(use_ref2v=False, use_Singularity=True))
    assert out[0] == "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
    assert out[1] == 0.0 and out[2] == 20                                 # без turbo, 20 шагов


def test_panel_steps_override():
    """WS-14: ручка steps_override — 0 = как у модели, >0 = перекрыть (быстрый путь для turbo)."""
    pv = U.NODES["NGFarmPanelVideo"]()
    assert pv.run(None, **_panel_kwargs())[2] == 8
    assert pv.run(None, steps_override=0, **_panel_kwargs())[2] == 8
    assert pv.run(None, steps_override=4, **_panel_kwargs())[2] == 4
    # модель без turbo: по умолчанию 20, но можно перекрыть (экономия времени ценой качества)
    sing = _panel_kwargs(use_ref2v=False, use_Singularity=True)
    assert pv.run(None, **sing)[2] == 20
    assert pv.run(None, steps_override=8, **sing)[2] == 8


def test_panel_steps_output_index_unchanged():
    """Новая ручка — optional: порядок выходов пульта не сдвигается, старые воркфлоу работают как есть."""
    names = U.NODES["NGFarmPanelVideo"].RETURN_NAMES
    assert names[0] == "model_name" and names[1] == "turbo_lora" and names[2] == "steps"
    assert names[3] == "model_label"
    it = U.NODES["NGFarmPanelVideo"].INPUT_TYPES()
    # required-виджеты не сдвинулись: модельные включатели впереди, ручки в том же порядке
    req = list(it["required"].keys())
    assert req[:4] == ["use_Singularity", "use_ref2v", "use_10Eros", "use_PinkCherry"]
    assert req[4:] == ["shift", "color_lock", "seam_crossfade", "temporal_strength", "face_confidence",
                      "crop_factor", "large_face_mult", "face_lock", "hand_strength"]
    assert list(it["optional"].keys()) == ["model_override", "steps_override"]


def test_builder_models_match_runtime():
    """WS-14: список моделей сборщика и пульта (farm/util_nodes.py) не должен расходиться.

    Расхождение = воркфлоу, который не проходит валидацию пульта (нет обязательного use_*),
    или модель, которая в пульте есть, а в графе нет.
    """
    import importlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "tools"))
    builder = importlib.import_module("build_farm_wf")
    rt = {m[0]: m[1] for m in U.FARM_MODELS}
    bd = {m[0]: m[1] for m in builder.MODELS}
    assert rt == bd, f"модели разошлись: пульт {rt} vs сборщик {bd}"
    assert list(U.UI_MODEL_ORDER) == ["Singularity", "ref2v", "10Eros", "PinkCherry"]
    # каждый API-граф с пультом должен содержать все обязательные включатели
    for f in sorted((root / "workflows").glob("*.api.json")):
        g = json.loads(f.read_text(encoding="utf-8"))
        for nid, n in g.items():
            if n.get("class_type") in ("NGFarmPanelPhoto", "NGFarmPanelVideo"):
                ins = n.get("inputs") or {}
                missing = [f"use_{lab}" for lab in U.UI_MODEL_ORDER if f"use_{lab}" not in ins]
                assert not missing, f"{f.name}: узел {nid} без {missing}"


# ---------------------------------------------------------------- WS-14 (29.10): миграция раскладок пульта (web/farm_panels.js)
# Фронтенд ComfyUI раскладывает widgets_values по позиции — сдвиг списка виджетов ломает
# сохранённые воркфлоу. Ниже — порт логики миграции из JS, чтобы её можно было проверить без браузера.
SPEC = {
    "NGFarmPanelPhoto": ["seed", "face_seed", "shift", "megapixels", "pass2_total", "pass2_steps", "face_denoise"],
    "NGFarmPanelVideo": ["shift", "color_lock", "seam_crossfade", "temporal_strength", "face_confidence",
                         "crop_factor", "large_face_mult", "face_lock", "hand_strength"],
}
OLD_MODEL_ORDER = ["ref2v", "10Eros", "PinkCherry"]
MODEL_ORDER = ["Singularity", "ref2v", "10Eros", "PinkCherry"]


def _leading_bools(v):
    i = 0
    while i < len(v) and isinstance(v[i], bool):
        i += 1
    return i


def _layout(v):
    nb = _leading_bools(v)
    if nb == 4:
        return "C"
    if nb != 3:
        return None
    return "B" if isinstance(v[-1], bool) else "A"


def _value_for(kind, name, v, spec):
    nb = _leading_bools(v)
    models = MODEL_ORDER if kind == "C" else OLD_MODEL_ORDER
    if name.startswith("use_"):
        if name == "use_Singularity":
            return v[0] if kind == "C" else (v[-1] if kind == "B" else False)
        i = models.index(name[4:])
        return v[nb - len(models) + i]
    if name == "model_override":
        for i in range(len(v) - 1, -1, -1):
            if isinstance(v[i], str) and v[i] != "fixed":
                return v[i]
        return ""
    if name == "steps_override":
        return 0
    si = spec.index(name)
    i, seen = nb, 0
    while i < len(v):
        if isinstance(v[i], str) and v[i] == "fixed":
            i += 1
            continue
        if seen == si:
            return v[i]
        seen += 1
        i += 1
    return None


def test_panel_widget_migration_layouts():
    """Раскладки A (без Singularity), B (Singularity в конце), C (новая) — значения ручек не теряются."""
    spec = SPEC["NGFarmPanelVideo"]
    vals = [12, -1, -1, 1, -1, 2.5, 0.35, 1, 0.2]
    names = ["use_Singularity", "use_ref2v", "use_10Eros", "use_PinkCherry"] + spec + ["model_override", "steps_override"]
    cases = {
        "A": ([True, False, False] + vals + [""], False),
        "B": ([True, False, False] + vals + ["", True], True),
        "C": ([False, True, False, False] + vals + ["", 0], False),
    }
    for kind, (v, sing) in cases.items():
        assert _layout(v) == kind, f"раскладка {v} определена как {_layout(v)}"
        out = {n: _value_for(kind, n, v, spec) for n in names}
        assert out["use_Singularity"] is sing
        assert out["use_ref2v"] is True
        assert out["model_override"] == ""
        assert out["steps_override"] == 0
        assert [out[s] for s in spec] == vals, f"{kind}: ручки поехали — {[out[s] for s in spec]}"


def test_panel_widget_migration_photo_seeds():
    """У сидов ComfyUI пишет маркер "fixed" — миграция не должна считать его значением ручки."""
    spec = SPEC["NGFarmPanelPhoto"]
    v = [True, False, False, 1, "fixed", 953552952704593, "fixed", 12, 2, 40, 8, 0.85, ""]
    kind = _layout(v)
    assert kind == "A"
    out = {s: _value_for(kind, s, v, spec) for s in spec}
    assert out == {"seed": 1, "face_seed": 953552952704593,
                   "shift": 12, "megapixels": 2, "pass2_total": 40, "pass2_steps": 8, "face_denoise": 0.85}
