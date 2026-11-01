"""WS-15: один пульт на блок — ПУЛЬТ ВИДЕО / ПРОХОДА ЛИЦА отдают STACY_SETTINGS (поля бывшего PANEL);
миграция раскладок проверяется на настоящем web/farm_panels.js (node), а не на порте."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from farm import util_nodes as U  # noqa: E402

VIDEO = {"use_Singularity": False, "use_ref2v": True, "use_10Eros": False, "use_PinkCherry": False,
         "shift": 12.0, "color_lock": -1.0, "seam_crossfade": -1, "temporal_strength": 1.0,
         "face_confidence": -1.0, "crop_factor": 2.5, "large_face_mult": 0.35, "face_lock": 1.0, "hand_strength": 0.2}


def test_video_panel_settings():
    pv = U.NODES["NGFarmPanelVideo"]
    assert pv.RETURN_TYPES[-1] == "STACY_SETTINGS" and len(pv.RETURN_TYPES) == len(pv.RETURN_NAMES)
    out = pv().run(None, seed=5, duration_sec=10.0, loop=False, end_frame=True, face_pass="off",
                   face_lora=0.0, **VIDEO)
    s = out[-1]   # вне ComfyUI — словарь входа StacyPanel
    assert s["mode"] is True and s["seed"] == 5 and s["duration_sec"] == 10.0 and s["loop"] is False
    assert s["end_frame"] is True and s["face_pass"] == "off" and s["steps"] == 8 and s["shift"] == 12.0
    assert s["char_lora"] == 0.0 and s["face_lora"] == 0.0
    d = pv().run(None, **VIDEO)[-1]   # старый граф без новых полей — значения по умолчанию
    assert d["duration_sec"] == 8.0 and d["loop"] is True and d["free_vram"] == "auto"
    assert set(U._settings_inputs(False)) <= set(pv.INPUT_TYPES()["optional"])


def test_face_panel_settings():
    pf = U.NODES["NGFarmPanelFace"]
    kw = {k: v for k, v in VIDEO.items() if not k.startswith("use_")}
    out = pf().run(None, seed=3, loop=True, **kw)
    assert len(out) == len(kw) + 1 and out[-1]["mode"] is False and out[-1]["seed"] == 3
    assert "duration_sec" not in pf.INPUT_TYPES()["optional"]


def _migrate(values, widgets):
    """Запуск migrate() из web/farm_panels.js: возвращает значения виджетов после конфигурации."""
    src = (ROOT / "web" / "farm_panels.js").read_text(encoding="utf-8")
    src = src.replace('import { app } from "../../scripts/app.js";', "const app = {registerExtension() {}};")
    harness = src + f"""
const names = {json.dumps([w[0] for w in widgets])};
const v = {json.dumps(values)};
const ws = names.map((n, i) => ({{name: n, value: v[i] !== undefined ? v[i] : {json.dumps([w[1] for w in widgets])}[i]}}));
const node = {{comfyClass: "NGFarmPanelVideo", id: 1, widgets: ws}};
migrate(node, {{widgets_values: v}});
console.log(JSON.stringify(ws.map((w) => w.value)));
"""
    tmp = ROOT / ".pytest_panels.mjs"
    tmp.write_text(harness, encoding="utf-8")
    try:
        r = subprocess.run(["node", str(tmp)], capture_output=True, text=True, timeout=30)
    finally:
        tmp.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not shutil.which("node"), reason="нет node")
def test_js_migration_video_layouts():
    it = U.NODES["NGFarmPanelVideo"].INPUT_TYPES()
    widgets = []
    for sec in ("required", "optional"):
        for k, spec in it[sec].items():
            d = spec[1].get("default") if len(spec) > 1 else None
            d = spec[0][0] if d is None and isinstance(spec[0], list) else d
            widgets.append((k, d))
            if spec[1].get("control_after_generate"):
                widgets.append(("control_after_generate", "fixed"))
    knobs = [12, -1, -1, 1, -1, 2.5, 0.35, 1, 0.2]
    defaults = [d for _, d in widgets]
    n_old = 4 + len(knobs)
    cases = {   # раскладка → (значения, ожидаемые первые поля, steps_override)
        "A": ([True, False, False] + knobs + [""], [False, True, False, False], 0),
        "B": ([True, False, False] + knobs + ["", True], [True, True, False, False], 0),
        "C": ([False, True, False, False] + knobs + ["", 6], [False, True, False, False], 6),
    }
    for kind, (v, bools, so) in cases.items():
        got = _migrate(v, widgets)
        assert got[:4] == bools and got[4:n_old] == knobs, (kind, got)
        assert got[n_old] == "" and got[n_old + 1] == so, (kind, got)
        assert got[n_old + 2:] == defaults[n_old + 2:], (kind, "новые поля не по умолчанию", got)
    cur = defaults[:]   # текущая раскладка — не трогается
    cur[n_old + 2] = 77
    assert _migrate(cur, widgets) == cur
