"""tools/dedup_loaders.py: склейка одинаковых загрузчиков и защитные правила (WS-14, OOM).

ComfyUI не дедуплицирует загрузчики: каждый узел CLIPLoader = свой энкодер (~25 ГБ).
Склейка экономит ОЗУ, но не должна ломать намеренно раздельные загрузки.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import dedup_loaders as D  # noqa: E402

CLIP = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"


def _clip(nid, title="Qwen3-VL", name=CLIP):
    return nid, {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": name, "type": "minimax"},
                 "_meta": {"title": title}}


def _consumer(nid, src, cls="MiniMaxH3ReferenceToVideo", inp="clip"):
    return nid, {"class_type": cls, "inputs": {inp: [src, 0]}}


def test_same_file_clips_merge():
    g = dict([_clip("c1"), _clip("c2"), _consumer("u1", "c1"), _consumer("u2", "c2")])
    new, notes = D.dedup(g)
    assert [k for k, n in new.items() if n["class_type"] == "CLIPLoader"] == ["c1"]
    assert new["u1"]["inputs"]["clip"] == ["c1", 0]
    assert new["u2"]["inputs"]["clip"] == ["c1", 0]
    assert notes


def test_different_files_not_merged():
    g = dict([_clip("c1", name=CLIP), _clip("c2", name="qwen_3_8b_fp8mixed.safetensors"),
              _consumer("u1", "c1"), _consumer("u2", "c2")])
    new, notes = D.dedup(g)
    assert len([k for k, n in new.items() if n["class_type"] == "CLIPLoader"]) == 2
    assert not notes


def test_own_loader_marker_protected():
    """«свой загрузчик» — намеренно раздельная загрузка прохода лица (video_h3_loop@2)."""
    g = dict([_clip("c1"), _clip("c2", title="CLIP face pass · свой загрузчик (с --cache-none …)"),
              _consumer("u1", "c1"), _consumer("u2", "c2")])
    new, notes = D.dedup(g)
    assert len([k for k, n in new.items() if n["class_type"] == "CLIPLoader"]) == 2
    assert not notes


def test_video_prefix_protected():
    """Блок прохода лица готового видео помечен префиксом «[видео]» — не склеиваем."""
    g = dict([_clip("c1"), _clip("c2", title="[видео] Qwen3-VL"),
              _consumer("u1", "c1"), _consumer("u2", "c2")])
    new, _ = D.dedup(g)
    assert len([k for k, n in new.items() if n["class_type"] == "CLIPLoader"]) == 2


def test_facepass_role_protected():
    g = dict([_clip("c1"), _clip("c2"), _consumer("u1", "c1"), _consumer("u2", "c2")])
    g["c2"]["_meta"]["role"] = "facepass:engine"
    new, _ = D.dedup(g)
    assert len([k for k, n in new.items() if n["class_type"] == "CLIPLoader"]) == 2


def test_unload_encoder_protected():
    """UnloadEncoderFromRAM уничтожает энкодер — его загрузчик нельзя делить с кем-то ещё."""
    g = dict([_clip("c1"), _clip("c2"), _consumer("u1", "c1"), _consumer("u2", "c2")])
    g["u2"]["class_type"] = "UnloadEncoderFromRAM"
    g["u2"]["inputs"] = {"anything": ["c2", 0], "clip": ["c2", 0]}
    new, _ = D.dedup(g)
    assert len([k for k, n in new.items() if n["class_type"] == "CLIPLoader"]) == 2


def test_model_name_from_node_not_merged():
    """Имя модели из пульта ([PANEL, 0]) выбирается в рантайме — такой узел не склеиваем."""
    g = {"p": {"class_type": "NGFarmPanelVideo", "inputs": {}},
         "u1": {"class_type": "UNETLoader", "inputs": {"unet_name": ["p", 0]}},
         "u2": {"class_type": "UNETLoader", "inputs": {"unet_name": ["p", 0]}}}
    new, notes = D.dedup(g)
    assert len([k for k, n in new.items() if n["class_type"] == "UNETLoader"]) == 2
    assert not notes


def test_vae_merge():
    g = {"v1": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
         "v2": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
         "d1": {"class_type": "H3ImageDecode", "inputs": {"vae": ["v1", 0]}},
         "d2": {"class_type": "H3ImageDecode", "inputs": {"vae": ["v2", 0]}}}
    new, _ = D.dedup(g)
    assert [k for k, n in new.items() if n["class_type"] == "VAELoader"] == ["v1"]
    assert new["d2"]["inputs"]["vae"] == ["v1", 0]


def test_no_dangling_links():
    g = dict([_clip("c1"), _clip("c2"), _consumer("u1", "c1"), _consumer("u2", "c2")])
    new, _ = D.dedup(g)
    for nid, node in new.items():
        for name, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                assert val[0] in new, (nid, name, val)


@pytest.mark.parametrize("wf", ["farm_manual@1", "farm_auto@1", "farm_table@1"])
def test_repo_workflows_still_valid(wf):
    """Склейка не ломает связи и не оставляет висячих узлов в реальных воркфлоу."""
    import json
    import wfkit as K
    src = ROOT / "workflows" / f"{wf}.api.json"
    g = json.loads(src.read_text(encoding="utf-8"))
    new, _ = D.dedup(g)
    for nid, node in new.items():
        for name, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                assert val[0] in new, (wf, nid, name, val)
    # каждое удалённое загрузчик-дубликат — тот же файл, что оставшийся
    assert len(new) <= len(g)


def test_video_h3_loop2_untouched():
    """У video_h3_loop@2 раздельный загрузчик прохода лица намеренный — не трогаем."""
    import json
    src = ROOT / "workflows" / "video_h3_loop@2.api.json"
    g = json.loads(src.read_text(encoding="utf-8"))
    new, notes = D.dedup(g)
    assert not notes
    assert len(new) == len(g)
