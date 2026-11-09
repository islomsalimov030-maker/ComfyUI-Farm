"""CPU-only prop import/render contract tests; no Blender or GPU required."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scene3d.import_props import check_references, inventory, prop_id, source_fingerprint, import_one
from tools.scene3d.import_props_blender import normalization_transform
from tools.scene3d.plate.prop_render import (anchors_from_mhr70, prop_transform, combine_strength, aabb_overlap,
                                           projected_bbox_fraction, load_props)
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


@pytest.mark.parametrize("size,scale", [(.30, 1), (.75, 1), (.30, 2.5), (.10, .5)])
def test_normalization_metres_and_multiplier(size, scale):
    bbox = [[10, -4, 1], [14, -2, 2]]
    centre, factor, final = normalization_transform(bbox, size_m=size, scale=scale)
    np.testing.assert_allclose(centre, [12, -3, 1.5])
    np.testing.assert_allclose((np.asarray(bbox) - centre) * factor, final)
    assert max(np.subtract(final[1], final[0])) == pytest.approx(size * scale)
    np.testing.assert_allclose(np.mean(final, axis=0), 0)
    # Repeat from a normalized GLB: no double application of source units.
    _, _, again = normalization_transform(final, size_m=size, scale=scale)
    np.testing.assert_allclose(again, final)


def test_normalization_default():
    _, _, bbox = normalization_transform([[0, 0, 0], [2, 1, 1]])
    assert max(np.subtract(bbox[1], bbox[0])) == pytest.approx(.30)


@pytest.mark.parametrize("bbox,size,scale", [
    ([[0, 0, 0], [0, 0, 0]], .3, 1),
    ([[1, 0, 0], [0, 2, 2]], .3, 1),
    ([[0, 0, 0], [float('inf'), 1, 1]], .3, 1),
    ([[0, 0], [1, 1]], .3, 1),
    ([[0, 0, 0], [1, 1, 1]], 0, 1),
    ([[0, 0, 0], [1, 1, 1]], float('nan'), 1),
    ([[0, 0, 0], [1, 1, 1]], .3, -1),
    ([[0, 0, 0], [1, 1, 1]], .3, float('inf')),
])
def test_normalization_rejects_invalid(bbox, size, scale):
    with pytest.raises(ValueError):
        normalization_transform(bbox, size_m=size, scale=scale)


@pytest.mark.parametrize("corners,expected", [
    ([(.1, .1, 1), (.9, .9, 2)], 1),
    ([(-.01, .1, 1), (.99, .9, 1)], .99),  # outside corner alone is no longer a flag
    ([(-.05, .1, 1), (.95, .9, 1)], .95),
    ([(-.06, .1, 1), (.94, .9, 1)], .94),
    ([(-.1, -.1, 1), (.9, .9, 1)], .81),
    ([(-1, -1, 1), (2, 2, 1)], 1 / 9),  # frame / full bbox, NOT IoU
    ([(2, .1, 1), (3, .9, 1)], 0),
    ([(1, .1, 1), (2, .9, 1)], 0),
    ([(.1, .1, -1), (.9, .9, 1)], 0),
    ([(.1, .1, 0), (.9, .9, 1)], 0),
    ([(.1, .1, 1), (.1, .9, 1)], 0),
])
def test_projected_bbox_fraction(corners, expected):
    fraction = projected_bbox_fraction(iter(corners))
    assert fraction == pytest.approx(expected)
    assert (fraction < .95) == (expected < .95)


@pytest.mark.parametrize("corners", [[], [(0, 1)], [(0, 0, float('nan'))], [(0, float('inf'), 1)]])
def test_projected_bbox_invalid(corners):
    with pytest.raises(ValueError):
        projected_bbox_fraction(corners)


def test_import_cache_normalization_version_and_size(tmp_path, monkeypatch):
    from tools.scene3d import import_props as importer
    source = tmp_path / "untrusted"
    source.mkdir()
    model = source / "asset.obj"
    model.write_text("v 0 0 0\n")
    library = tmp_path / "library"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == ROOT
        assert "--disable-autoexec" in command and "--factory-startup" in command
        assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
        assert Path(command[command.index("--python") + 1]).parent == Path(importer.__file__).parent
        size = float(command[command.index("--size-m") + 1])
        Path(command[command.index("--out") + 1]).write_bytes(b"test-cache")
        Path(command[command.index("--facts") + 1]).write_text(json.dumps({
            "source_bbox_m": [[-2, -1, -1], [2, 1, 1]], "longest_side_m": 4,
            "normalized_longest_side_m": size}))
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(importer.subprocess, "run", run)
    result = import_one(model, source, library, "blender")
    assert result["metadata"]["size_m"] == .30
    assert result["metadata"]["license_status"] == "unknown"
    assert result["metadata"]["import_facts"]["longest_side_m"] == 4
    assert import_one(model, source, library, "blender")["status"] == "cached"
    assert len(calls) == 1
    target = library / result["id"] / "prop.json"
    old = json.loads(target.read_text())
    old["importer_version"] = 1
    old["size_m"] = 4  # stale source-size metadata must migrate, not become runtime size
    target.write_text(json.dumps(old))
    assert import_one(model, source, library, "blender")["metadata"]["size_m"] == .30
    assert len(calls) == 2
    assert import_one(model, source, library, "blender", size_m=.6)["metadata"]["size_m"] == .6
    assert len(calls) == 3
    with pytest.raises(ValueError):
        import_one(model, source, library, "blender", size_m=float('nan'))


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("runtime,expected_longest", [({}, .30), ({"size_m": .6}, .6),
                                                      ({"size_m": .6, "scale": 2}, 1.2)])
def test_load_props_runtime_normalization(tmp_path, monkeypatch, runtime, expected_longest, legacy):
    from tools.scene3d.plate import prop_render
    from unittest.mock import MagicMock
    directory = tmp_path / "asset"
    directory.mkdir()
    (directory / "model.glb").write_bytes(b"test-cache")
    source_bbox = [[10, 20, 30], [14, 22, 31]]
    facts = ({"source_bbox_center_m": [12, 21, 30.5], "dimensions_m": [4, 2, 1]} if legacy
             else {"source_bbox_m": source_bbox})
    (directory / "prop.json").write_text(json.dumps({"size_m": 99, "import_facts": facts}))
    root = SimpleNamespace(matrix_world=None)
    mesh = MagicMock()
    mesh.type, mesh.parent = "MESH", None
    scene = SimpleNamespace(objects=[], collection=SimpleNamespace(objects=SimpleNamespace(link=lambda _: None)))

    def import_gltf(**kwargs):
        scene.objects.append(mesh)

    bpy = SimpleNamespace(data=SimpleNamespace(objects=MagicMock()),
                          ops=SimpleNamespace(import_scene=SimpleNamespace(gltf=import_gltf)),
                          context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None)))
    # data.objects must also support set() in the renderer.
    bpy.data.objects.__iter__.return_value = iter([])
    bpy.data.objects.new.return_value = root
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    matrix = MagicMock()
    matrix.Translation.return_value = matrix
    matrix.Scale.return_value = matrix
    matrix.__matmul__.return_value = matrix
    euler = MagicMock()
    euler.return_value.to_matrix.return_value.to_4x4.return_value = matrix
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Matrix=matrix, Euler=euler, Vector=lambda p: np.array(p)))
    cache_bbox = [[-.15, -.075, -.0375], [.15, .075, .0375]]
    if legacy:
        cache_bbox = [[-2, -1, -.5], [2, 1, .5]]
    final = [[-expected_longest / 2, 0, 0], [expected_longest / 2, 0, 0]]
    bounds = iter([cache_bbox, final])
    monkeypatch.setattr(prop_render, "_bounds", lambda _: next(bounds))
    group, = load_props(scene, [{"prop_id": "asset", **runtime}], {"char_pelvis": [0, 0, 0]}, tmp_path)
    assert group["source_bbox_m"] == source_bbox
    assert group["final_bbox_m"] == final
    assert max(np.subtract(group["scaled_bbox_m"][1], group["scaled_bbox_m"][0])) == pytest.approx(expected_longest)
    assert group["normalization_factor"] == pytest.approx(expected_longest / (4 if legacy else .30))
