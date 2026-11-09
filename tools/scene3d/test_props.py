"""CPU-only prop import/render contract tests; no Blender or GPU required."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scene3d.import_props import check_references, inventory, prop_id, source_fingerprint
from tools.scene3d.plate.prop_render import anchors_from_mhr70, prop_transform, combine_strength, aabb_overlap
from tools.scene3d.plate.pose_proxy_v26 import strength_map


def test_inventory_unknown_license(tmp_path):
    (tmp_path / "a.obj").write_text("v 0 0 0\n")
    facts = inventory(tmp_path)
    assert facts["file_count"] == 1
    assert facts["models"][0]["licenses"] == []
    (tmp_path / "LICENSE.txt").write_text("custom")
    assert inventory(tmp_path)["models"][0]["licenses"] == ["LICENSE.txt"]


def test_ids_distinct_safe():
    assert prop_id("a/Head.fbx") != prop_id("b/Head.fbx")
    assert "/" not in prop_id("../../X.fbx")


def test_fingerprint_texture_invalidates(tmp_path):
    model = tmp_path / "x.obj"
    model.write_text("v 0 0 0\n")
    texture = tmp_path / "t.png"
    texture.write_bytes(b"a")
    first = source_fingerprint(model)
    texture.write_bytes(b"b")
    assert source_fingerprint(model) != first


@pytest.mark.parametrize("uri", ["../x.bin", "https://example.com/x.bin", "C:/x.bin", "//host/x.bin"])
def test_external_refs_rejected(tmp_path, uri):
    model = tmp_path / "asset.gltf"
    model.write_text(json.dumps({"buffers": [{"uri": uri}]}))
    with pytest.raises(ValueError):
        check_references(model)


def test_anchors():
    k = np.arange(210).reshape(70, 3)
    anchors = anchors_from_mhr70(k)
    np.testing.assert_allclose(anchors["world"], (k[9] + k[10]) / 2)
    np.testing.assert_allclose(anchors["char_hand_l"], k[62])
    with pytest.raises(ValueError):
        anchors_from_mhr70(k[:69])


def test_transform():
    xyz, angles, scale = prop_transform({"anchor": "world", "x": .3, "z": .4, "yaw": 90, "scale": 2},
                                        {"char_pelvis": [1, 2, 3]})
    np.testing.assert_allclose(xyz, [1.3, 2, 3.4])
    np.testing.assert_allclose(angles, [0, 0, np.pi / 2])
    assert scale == 2
    with pytest.raises(ValueError):
        prop_transform({"scale": 0}, {"char_pelvis": [0, 0, 0]})


def test_strength_same_edge_as_body():
    mask = np.zeros((128, 128), np.uint8)
    mask[64, 64] = 255
    expected = strength_map(mask > 0, .35, 16)
    actual = combine_strength(np.zeros(mask.shape), [(mask, .35)])
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert actual[0, 0] == 0
    body = np.zeros(mask.shape)
    body[64, 64] = .8
    assert combine_strength(body, [(mask, .35)])[64, 64] == pytest.approx(.8)
    assert not combine_strength(body * 0, [(mask * 0, .35)]).any()


def test_aabb_diagnostic():
    assert aabb_overlap([[0, 0, 0], [1, 1, 1]], [[.5, .5, .5], [2, 2, 2]])
    assert not aabb_overlap([[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]])
