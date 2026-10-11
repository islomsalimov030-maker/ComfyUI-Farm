"""Привязка параметров оркестратора совпадает с эталоном WS-01 (tools/wf_params.py) на реальных движках."""
import json
import sys
from pathlib import Path

import pytest

from farm.comfy_client import bind, check_params_map, load_workflow, main_output

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows"
TOOLS = ROOT / "tools"

pytestmark = pytest.mark.skipif(not (WF / "video_h3_loop@1.params.yaml").exists()
                                or not (TOOLS / "wf_params.py").exists(),
                                reason="нет движков WS-01 в репо")


def _ref_apply(ref, values):
    sys.path.insert(0, str(TOOLS))
    try:
        import wf_params
        p, api = wf_params.load(str(WF / f"{ref}.params.yaml"))
        assert wf_params.check(p, api) == []
        return wf_params.apply(p, api, values)
    finally:
        sys.path.remove(str(TOOLS))


CASES = {
    "video_h3_loop@1": dict(json.loads((TOOLS / "values_sitting8_video.json").read_text(encoding="utf-8"))
                            if (TOOLS / "values_sitting8_video.json").exists() else {},
                            prompt="test prompt", model="h3_test.safetensors",
                            lora="stacy_test.safetensors", lora_w=1.0, shift=13.0),
    "frame_h3@1": {"plate": "plate.png", "sheet": "sheet.png", "prompt": "image 1 test", "seed": 5,
                   "face_seed": 5, "face_prompt_h3": "St4cy_H3 face", "model": "m.safetensors",
                   "loras": [{"name": "a.safetensors", "w": 1.0}, {"name": "b.safetensors", "w": 0.5}],
                   "pass1_steps": 8},
}


@pytest.mark.parametrize("ref", list(CASES))
def test_bind_matches_reference(ref):
    graph, params = load_workflow(WF, ref)
    assert check_params_map(graph, params) == []
    values = CASES[ref]
    mine, _ = bind(graph, params, values, upload=lambda p: p)
    assert mine == _ref_apply(ref, values)


def test_main_outputs_of_engines():
    _, pv = load_workflow(WF, "video_h3_loop@1")
    _, pf = load_workflow(WF, "frame_h3@1")
    assert main_output(pv, "video") == "video_fullhd"
    assert main_output(pf, "frame") == "frame"
