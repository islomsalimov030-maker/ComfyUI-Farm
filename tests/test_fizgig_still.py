"""25.10: стилл-ветки H3 → Fizgig (tools/fizgig_still.py) на настоящих API-графах."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fizgig_still import to_fizgig  # noqa: E402


@pytest.mark.parametrize("wf,n", [("frame_h3@1", 2), ("manual_frame@1", 2), ("frame_ls@1", 2)])
def test_convert(wf, n):
    src = json.loads((ROOT / "workflows" / f"{wf}.api.json").read_text(encoding="utf-8"))
    g, rep = to_fizgig(src)
    assert len(rep["latents"]) == n and len(rep["decodes"]) == n
    cls = [x["class_type"] for x in g.values()]
    assert "H3ImageDecode" not in cls and "H3ImageFrameSelector" not in cls
    assert cls.count("FizgigH3StillLatent") == n and cls.count("FizgigH3StillDecode") == n
    for k, x in g.items():                      # нет висячих ссылок, латенты сэмплеров — от Fizgig
        for a, b in x["inputs"].items():
            if isinstance(b, list) and len(b) == 2 and isinstance(b[0], str):
                assert b[0] in g, (k, a, b)
                if a in ("latent_image", "h3_latent") and g[b[0]]["class_type"] in (
                        "H3ReferenceEditPrepare", "H3TextToImagePrepare", "MiniMaxH3ReferenceToVideo"):
                    pytest.fail(f"{k}.{a} всё ещё от prepare")
    assert "H3ImageDecode" in [x["class_type"] for x in src.values()]   # исходник не тронут


def test_video_untouched():
    src = json.loads((ROOT / "workflows" / "video_h3_loop@1.api.json").read_text(encoding="utf-8"))
    g, rep = to_fizgig(src)
    assert g == src and rep == {"latents": [], "decodes": []}
