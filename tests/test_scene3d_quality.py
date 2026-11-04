"""Offline checks of quality experiment graphs; no GPU, network or private inputs."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

PLATE = Path(__file__).resolve().parents[1] / 'tools/scene3d/plate'


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(PLATE))
    spec = importlib.util.spec_from_file_location('scene3d_quality_test', PLATE / 'quality_round.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((PLATE / 'quality_round.yaml').read_text())
    scene_spec = json.loads((PLATE / 'scenes_mainroom1_r4.json').read_text())
    return module, config, scene_spec


def test_all_quality_graphs_have_valid_links(runner):
    module, config, scene_spec = runner
    for recipe in config['recipes']:
        graph = module.make_graph(scene_spec, scene_spec['scenes'][1], 101, recipe, config)
        assert 'lora' not in graph
        assert 'save_comp' not in graph
        assert graph['ks']['inputs']['cfg'] == 1.0
        assert graph['r2v']['inputs']['width'] == 1344
        assert graph['r2v']['inputs']['height'] == 768
        assert len({node['_meta']['title'] for node in graph.values()}) == len(graph)
        for node in graph.values():
            for value in node['inputs'].values():
                if isinstance(value, list):
                    assert value[0] in graph
        prompt = graph['r2v']['inputs']['prompt']
        assert 'studio' not in prompt.lower()
        assert '\nThe lighting is preserved unchanged throughout.\n' in prompt
        assert '\nColors and saturation stay unchanged from the reference image.\n' in prompt


def test_clay_only_never_loads_controlnet(runner):
    module, config, scene_spec = runner
    recipe = next(r for r in config['recipes'] if r['name'] == 'R1_clay')
    graph = module.make_graph(scene_spec, scene_spec['scenes'][1], 101, recipe, config)
    assert 'patch' not in graph
    assert not any(n['class_type'] == 'MiniMaxH3FunControlNetApply' for n in graph.values())
    assert graph['ks']['inputs']['model'] == ['shift', 0]
    assert graph['ks']['inputs']['steps'] == 25
    assert graph['r2v']['inputs']['ref_videos.ref_video_0'] == ['clay_batch', 0]


def test_bf16_inpaint_and_masked_canny(runner):
    module, config, scene_spec = runner
    graphs = {r['name']: module.make_graph(scene_spec, scene_spec['scenes'][1], 101, r, config)
              for r in config['recipes']}
    graph = graphs['Q4_inpaint']
    assert graph['patch']['inputs']['name'] == config['bf16_patch']
    assert graph['ap_inpaint']['inputs']['mask'] == ['inpaint_mask', 0]
    assert graph['ap_inpaint']['inputs']['source_video'] == ['source_batch', 0]
    assert 'ap_depth' not in graph
    assert graphs['Q3_edge05']['edge_outside']['inputs']['mask'] == ['edge_mask', 0]
