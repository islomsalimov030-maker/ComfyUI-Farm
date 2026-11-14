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
    # WS-15: поля бывшего PANEL — после steps_override, выход settings — последним
    assert list(it["optional"].keys())[:2] == ["model_override", "steps_override"]
    assert names[-1] == "settings"


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
# SPEC — ручки СТАРЫХ раскладок A/B/C (по ним читаются сохранённые значения); раскладка D пульта фото — PHOTO_D.
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


# ---------------------------------------------------------------- 3D-A (15.11): ПУЛЬТ ФОТО, раскладка D
PHOTO_D = ["use_Singularity", "use_ref2v", "use_10Eros", "use_PinkCherry", "turbo", "steps", "seed",
           "megapixels", "face_pass", "face_seed", "face_denoise", "shift", "pass2_total", "pass2_steps"]
PHOTO_OUT_OLD = ["model_name", "turbo_lora", "steps", "model_label", "seed", "face_seed", "shift", "megapixels",
                 "pass2_total", "pass2_steps", "face_denoise"]
PP = U.NODES["NGFarmPanelPhoto"]


def _photo(**over):
    kw = {"use_Singularity": False, "use_ref2v": True, "use_10Eros": False, "use_PinkCherry": False}
    kw.update(over)
    return kw


def test_photo_widget_order():
    it = PP.INPUT_TYPES()
    assert list(it["required"]) + list(it.get("optional", {})) == PHOTO_D
    assert "model_override" not in it["optional"] and "steps_override" not in it["optional"]
    assert it["hidden"]["extra_pnginfo"] == "EXTRA_PNGINFO"
    o = it["optional"]
    assert o["turbo"][0] == "BOOLEAN" and o["turbo"][1]["default"] is True
    st = o["steps"][1]
    assert (st["default"], st["min"], st["max"], st["step"]) == (20, 15, 30, 1)
    assert o["face_pass"][0] == "BOOLEAN" and o["face_pass"][1]["default"] is True
    assert (o["face_denoise"][1]["max"], o["face_denoise"][1]["default"]) == (0.9, 0.85)
    assert o["seed"][1]["control_after_generate"] == "fixed"
    assert o["face_seed"][1]["control_after_generate"] == "fixed"


def test_photo_outputs_unchanged_face_pass_last():
    assert list(PP.RETURN_NAMES[:11]) == PHOTO_OUT_OLD
    assert PP.RETURN_NAMES[-1] == "face_pass" and len(PP.RETURN_NAMES) == 12
    assert PP.RETURN_TYPES[1:11] == ("FLOAT", "INT", "STRING", "INT", "INT", "FLOAT", "FLOAT", "INT", "INT", "FLOAT")
    assert PP.RETURN_TYPES[-1] == "BOOLEAN"


def test_photo_run_turbo_and_steps():
    p = PP()
    out = p.run(None, **_photo(turbo=True, steps=22))
    assert (out[2], out[1]) == (8, 1.0) and out[3] == "ref2v"
    out = p.run(None, **_photo(turbo=False, steps=22))
    assert (out[2], out[1]) == (22, 0.0)
    out = p.run(None, **_photo(use_ref2v=False, use_10Eros=True, turbo=True, steps=17))
    assert (out[2], out[1]) == (17, 0.0) and out[3] == "10Eros"
    assert p.run(None, **_photo(turbo=False, steps=40))[2] == 30       # зажато 15…30
    assert p.run(None, **_photo(turbo=False, steps=4))[2] == 15
    assert p.run(None, **_photo(turbo=False))[2] == 20                   # API-граф без steps — по умолчанию
    assert p.run(None, **_photo(face_pass=False))[-1] is False
    assert p.run(None, **_photo())[-1] is True


def test_photo_first_model_by_ui_order(monkeypatch):
    calls = []
    monkeypatch.setattr(U, "_requeue", lambda pr, uid, labels, extra=None: calls.append(labels))
    p = PP()
    assert p.run(None, **_photo(use_Singularity=True))[3] == "Singularity"   # в FARM_MODELS она последняя
    assert p.run(None, **_photo(use_PinkCherry=True, use_ref2v=False, use_10Eros=True))[3] == "10Eros"
    prompt = {"5": {"class_type": "NGFarmPanelPhoto", "inputs": {}},
              "9": {"class_type": "NGFarmPanelVideo", "inputs": {}}}
    allon = _photo(use_Singularity=True, use_10Eros=True, use_PinkCherry=True)
    assert p.run(prompt, "5", None, **allon)[3] == "Singularity"
    assert calls == [["ref2v", "10Eros", "PinkCherry"]]


def test_photo_model_override_from_extra_pnginfo_and_kwargs(monkeypatch):
    calls = []
    monkeypatch.setattr(U, "_requeue", lambda *a, **k: calls.append(a))
    p = PP()
    prompt = {"5": {"class_type": "NGFarmPanelPhoto", "inputs": {}}}
    out = p.run(prompt, "5", {"workflow": {}, "ng_model_override": "10Eros"},
                **_photo(use_10Eros=True, turbo=True, steps=21))
    assert out[3] == "10Eros" and (out[2], out[1]) == (21, 0.0) and not calls   # копия очередь не ставит
    out = p.run(prompt, "5", {"workflow": {}}, ng_model_override="PinkCherry", **_photo(use_PinkCherry=True))
    assert out[3] == "PinkCherry" and not calls


class _Resp:
    def read(self):
        return b"{}"


def _capture_post(monkeypatch):
    import json as _j
    import urllib.request as ur
    sent = []
    monkeypatch.setattr(ur, "urlopen", lambda req, timeout=0: (sent.append(_j.loads(req.data)), _Resp())[1])
    return sent


def test_requeue_puts_override_into_extra_pnginfo(monkeypatch):
    sent = _capture_post(monkeypatch)
    prompt = {"5": {"class_type": "NGFarmPanelPhoto", "inputs": {"use_ref2v": True}},
              "9": {"class_type": "NGFarmPanelVideo", "inputs": {"model_override": ""}}}
    extra = {"workflow": {"nodes": [1]}}
    U._requeue(prompt, "5", ["10Eros", "PinkCherry"], extra)
    assert [b["extra_data"]["extra_pnginfo"]["ng_model_override"] for b in sent] == ["10Eros", "PinkCherry"]
    assert all(b["extra_data"]["extra_pnginfo"]["workflow"] == {"nodes": [1]} for b in sent)   # прежнее сохранено
    assert extra == {"workflow": {"nodes": [1]}}                                            # исходный не тронут
    assert sent[0]["prompt"]["5"]["inputs"]["ng_model_override"] == "10Eros"               # ключ кэша
    assert "model_override" not in sent[0]["prompt"]["5"]["inputs"]
    assert sent[0]["prompt"]["9"]["inputs"]["model_override"] == "10Eros"                  # видео — как раньше
    sent.clear()
    U._requeue(prompt, "5", ["ref2v"], None)
    assert sent[0]["extra_data"]["extra_pnginfo"] == {"ng_model_override": "ref2v"}


def _comfy_signature(node):
    """Порт comfy_execution/caching.py get_immediate_node_signature (без связей): ключ кэша узла — его inputs."""
    return [node["class_type"]] + [(k, node["inputs"][k]) for k in sorted(node["inputs"])]


def _comfy_validate(cls, inputs):
    """Порт execution.validate_inputs: проверяются только объявленные входы, лишние ключи игнорируются."""
    it = cls.INPUT_TYPES()
    missing = [k for k in it["required"] if k not in inputs]
    return not missing, missing


def test_photo_requeue_copies_differ_for_cache_and_validate(monkeypatch):
    sent = _capture_post(monkeypatch)
    base = {"5": {"class_type": "NGFarmPanelPhoto", "inputs": _photo(use_10Eros=True)}}
    U._requeue(base, "5", ["10Eros"], {})
    U._requeue(base, "5", ["ref2v"], {})
    a, b = (s["prompt"]["5"] for s in sent)
    assert _comfy_signature(a) != _comfy_signature(b)
    assert _comfy_signature(a) != _comfy_signature(base["5"])
    for s in sent:
        ok, missing = _comfy_validate(PP, s["prompt"]["5"]["inputs"])
        assert ok, missing
    # ComfyUI передаёт в функцию только объявленные входы; если лишний ключ всё же дойдёт — run() его принимает
    ra = PP().run(a, "5", sent[0]["extra_data"]["extra_pnginfo"], **a["inputs"])
    rb = PP().run(b, "5", sent[1]["extra_data"]["extra_pnginfo"], **b["inputs"])
    assert ra[3] == "10Eros" and rb[3] == "ref2v" and ra[0] != rb[0]


# ---------------------------------------------------------------- JS: миграция C → D, серость, 3DAControl (node)
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_NO_NODE = pytest.mark.skipif(not shutil.which("node"), reason="нет node")


def _run_js(path, body, between=None):
    src = path.read_text(encoding="utf-8")
    if between:   # только кусок файла (у extension.js 3DA свои импорты)
        a, b = between
        src = src[src.index(a):src.index(b)]
    stub = "const app = {registerExtension() {}, graph: null};"
    src = src.replace('import { app } from "../../scripts/app.js";', stub)
    src = src.replace("import { app } from '../../scripts/app.js';", stub)
    tmp = ROOT / ".pytest_panels_d.mjs"
    tmp.write_text(src + "\n" + body, encoding="utf-8")
    try:
        r = subprocess.run(["node", str(tmp)], capture_output=True, text=True, timeout=30, encoding="utf-8")
    finally:
        tmp.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _photo_widgets():
    it = PP.INPUT_TYPES()
    ws = []
    for sec in ("required", "optional"):
        for k, spec in it[sec].items():
            ws.append((k, spec[1].get("default")))
            if spec[1].get("control_after_generate"):
                ws.append(("control_after_generate", "fixed"))
    return ws


def _js_photo(values, props=None, extra=""):
    ws = _photo_widgets()
    body = f"""
const ws = {json.dumps([{"name": n, "value": d} for n, d in ws])};
const node = {{comfyClass: "NGFarmPanelPhoto", id: 1, widgets: ws, properties: {{}}}};
setupPhoto(node);
// фронтенд раскладывает widgets_values по позиции до onConfigure
{json.dumps(values)}.forEach((x, i) => {{ if (ws[i]) ws[i].value = x; }});
node.onConfigure({{widgets_values: {json.dumps(values)}, properties: {json.dumps(props or {})}}});
{extra}
console.log(JSON.stringify({{
  v: Object.fromEntries(ws.map((w, i) => [w.name === "control_after_generate" ? "ctrl" + i : w.name, w.value])),
  dis: ws.filter((w) => w.disabled).map((w) => w.name), labels: ws.map((w) => w.label ?? null),
  props: node.properties}}));
"""
    return _run_js(ROOT / "web" / "farm_panels.js", body)


@_NO_NODE
def test_js_photo_migration_c_to_d():
    c = [False, True, False, False, 295184853228273, "randomize", 276557735836995, "randomize",
         12, 2, 40, 8, 0.45, "", 0]
    r = _js_photo(c)
    v = r["v"]
    assert r["props"]["ng_photo_layout"] == "D"
    assert (v["use_Singularity"], v["use_ref2v"], v["use_10Eros"], v["use_PinkCherry"]) == (False, True, False, False)
    assert (v["seed"], v["face_seed"], v["shift"], v["megapixels"]) == (295184853228273, 276557735836995, 12, 2)
    assert (v["pass2_total"], v["pass2_steps"], v["face_denoise"]) == (40, 8, 0.45)
    assert (v["turbo"], v["steps"], v["face_pass"]) == (True, 20, True)
    assert [x for k, x in v.items() if k.startswith("ctrl")] == ["randomize", "randomize"]
    assert _js_photo(c[:-1] + [24])["v"]["steps"] == 24          # steps_override > 0 → steps
    assert _js_photo(c[:-1] + [40])["v"]["steps"] == 30          # зажато 15…30
    assert _js_photo(c[:-1] + [8])["v"]["steps"] == 15
    # сохранено уже в D — не трогаем
    d = [False, True, False, False, False, 25, 7, "fixed", 3, False, 9, "fixed", 0.5, 11, 30, 6]
    r = _js_photo(d, {"ng_photo_layout": "D"})
    assert r["v"]["turbo"] is False and r["v"]["steps"] == 25 and r["v"]["face_pass"] is False


@_NO_NODE
def test_js_photo_greying_and_labels():
    d = [False, True, False, False, True, 20, 1, "fixed", 2, True, 2, "fixed", 0.85, 12, 40, 8]
    r = _js_photo(d, {"ng_photo_layout": "D"})
    assert r["dis"] == ["steps"]                                  # turbo вкл и включена только ref2v
    labels = dict(zip([n for n, _ in _photo_widgets()], r["labels"]))
    assert labels["shift"] == "⚙ shift" and labels["pass2_steps"] == "⚙ pass2_steps" and labels["turbo"] is None
    r = _js_photo(d[:1] + [False, True] + d[3:], {"ng_photo_layout": "D"})
    assert "turbo" in r["dis"] and "steps" not in r["dis"]       # ref2v выкл
    r = _js_photo(d[:9] + [False] + d[10:], {"ng_photo_layout": "D"})
    assert {"face_seed", "face_denoise"} <= set(r["dis"])
    ctl = ('app.graph = {_nodes: [{comfyClass: "3DAControl", widgets: [{name: "control_3d", value: true}]}]};'
           "greyPhoto(node);")
    r = _js_photo(d, {"ng_photo_layout": "D"}, ctl)
    assert {"pass2_total", "pass2_steps"} <= set(r["dis"])


@_NO_NODE
def test_js_3da_control_migration():
    js = ROOT / "comfyui" / "ComfyUI-3DA" / "web" / "extension.js"
    names = ["control_3d", "character", "step", "prop", "body_margin", "body", "shadows", "shadow_strength",
             "shadow_opacity", "clean_shadows"]
    defaults = [True, "Стейси", "Генерация", "(нет предмета)", 2.0, "Стейси MPFB (пальцы, лицо)", "Тени от H3",
                0.7, 0.6, False]
    old = [True, "Стейси", "stick_1", "Подготовка сцены", "Тени от рендера", 3.0, 0.5, 0.4, "MHR (SAM 3D Body)", True]

    def run(values, props):
        body = f"""
const ws = {json.dumps([{"name": n, "value": d} for n, d in zip(names, defaults)])};
const defaults = Object.fromEntries(ws.map((w) => [w.name, w.value]));
const node = {{comfyClass: "3DAControl", id: 1, widgets: ws, properties: {{}}}};
{json.dumps(values)}.forEach((x, i) => {{ if (ws[i]) ws[i].value = x; }});
migrateControl(node, {{widgets_values: {json.dumps(values)}, properties: {json.dumps(props)}}}, defaults);
console.log(JSON.stringify({{v: Object.fromEntries(ws.map((w) => [w.name, w.value])), p: node.properties}}));
"""
        return _run_js(js, body, ("// 0.3.25: раскладка 3DAControl", "app.registerExtension({\n  name: 'ThreeDA.Control'"))

    r = run(old, {})
    assert r["p"]["ng3da_layout"] == 2
    assert r["v"] == dict(control_3d=True, character="Стейси", step="Подготовка сцены", prop="stick_1",
                          body_margin=3.0, body="MHR (SAM 3D Body)", shadows="Тени от рендера",
                          shadow_strength=0.5, shadow_opacity=0.4, clean_shadows=True)
    r = run(old[:8], {})                                           # 8 значений: body / clean_shadows — по умолчанию
    assert r["v"]["body"] == defaults[5] and r["v"]["clean_shadows"] is False and r["v"]["shadow_opacity"] == 0.4
    new = [False, "Стейси", "Генерация", "stick_2", 1.5, defaults[5], "Тени от H3", 0.3, 0.2, False]
    assert run(new, {"ng3da_layout": 2})["v"] == dict(zip(names, new))    # новое сохранение — без миграции
