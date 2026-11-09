"""09.11e 0e: seed-free surface drafts and read-only previews, CPU only."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

import auto_surfaces as auto
import profiles
import top_region as tr

CFG = {"height_bin_m": .02, "min_area_m2": .15}


def discover(height):
    return auto.surfaces_from_grid([0., 0.], height, [0., -2.], config=CFG)


def plate():
    return SimpleNamespace(C=np.array([0., -2., 2.]), rgb=Image.new("RGB", (320, 240), "#909090"),
        Wp=320, Hp=240, project=lambda p: (p[:, 0] * 80 + 20, 220 - p[:, 1] * 80, np.ones(len(p))))


def test_bin_components_area_and_metadata():
    height = np.full((100, 100), np.nan)
    height[:40, :40] = .005  # .64m2; largest AND lowest
    height[50:70, :20] = .405  # .16m2
    height[75:95, :20] = .405  # disconnected, same height
    height[50:69, 30:49] = .805  # .1444m2, rejected
    regs, report = discover(height)
    assert [r.object_id for r in regs] == ["floor_1", "surf_1", "surf_2"]
    assert report["floor_status"] == "ok"
    assert report["surf_count"] == 2 and report["rejected_components"] == 1
    assert regs[1].h0 == .405
    assert regs[1].mask.sum() * tr.CELL ** 2 == pytest.approx(.16)
    np.testing.assert_array_equal(regs[1].object_mask, regs[1].mask)
    assert regs[1].dt.max() > 0 and np.isnan(regs[1].height[~regs[1].mask]).all()


@pytest.mark.parametrize("mode", ["disagree", "area_tie", "height_tie"])
def test_floor_ambiguity_never_invents_choice(mode):
    height = np.full((60, 60), np.nan)
    height[:20, :20] = .005
    height[30:50 if mode == "area_tie" else 60, 30:50] = .005 if mode == "height_tie" else .405
    regs, report = discover(height)
    assert report["floor_status"] == "floor_ambiguous"
    assert report["floor_id"] is None and all(r.object_id.startswith("surf_") for r in regs)
    assert "no choice invented" in report["reason"]


def test_bins_are_disjoint_and_connectivity_is_four_neighbour():
    height = np.full((50, 50), np.nan)
    height[:20, :20] = .019
    height[20:40, 20:40] = .019  # diagonal contact does not merge
    height[:20, 20:40] = .021  # directly adjoining, but different bin
    regs, _ = discover(height)
    assert len(regs) == 3
    assert all(r.mask.sum() == 400 for r in regs)
    assert np.stack([r.mask for r in regs]).sum(0).max() == 1


def test_empty_and_subthreshold_grid_report():
    height = np.full((30, 30), np.nan)
    regs, report = discover(height)
    assert not regs and report["floor_status"] == "no_surfaces"
    height[:10, :10] = 0
    assert discover(height)[1]["surface_count"] == 0


def test_missing_config_no_fallback_and_invalid_thresholds():
    with patch.object(auto, "section", side_effect=KeyError("auto_surfaces")):
        with pytest.raises(KeyError):
            auto.surfaces_from_grid([0, 0], np.zeros((20, 20)), [0, 0])
    for bad in (0., float("nan"), -1., True):
        with pytest.raises(ValueError):
            auto.surfaces_from_grid([0, 0], np.zeros((20, 20)), [0, 0],
                                   config={"height_bin_m": bad, "min_area_m2": .15})


def test_holes_remain_missing_and_real_boundary_includes_hole():
    height = np.zeros((30, 30))
    height[10:20, 10:20] = np.nan
    reg = discover(height)[0][0]
    assert not reg.mask[15, 15] and not reg.object_mask[15, 15]
    edges = auto._edges(reg)
    assert len(edges) == 4 * 30 + 4 * 10
    assert len(edges.shape) == 3 and edges.shape[1:] == (2, 3)


def test_writer_round_trip_ransac_walls_and_no_overwrite(tmp_path):
    room = plate()
    height = np.zeros((30, 30))
    wall = dict(n=np.array([0., -1., 0.]), d=1., area_m2=2., pts=np.array([[0., 1., 1.]]),
                lat=[0., 1.], z=[0., 2.], center=np.array([.5, 1., 1.]))
    with patch.object(tr, "_grid", return_value=(np.zeros(2), height.shape, height)), \
            patch.object(tr, "walls", return_value=[wall]) as ransac:
        path = auto.write_auto_profile("new_room", room, tmp_path / "new", config=CFG)
    ransac.assert_called_once_with(room)
    p = profiles.LocationProfile(path)
    assert p.data["reviewed"] is False and not p.data["anchors"]
    assert p.data["auto_surfaces"]["wall_count"] == 1
    obj, typ = p.find_object("floor_1")
    assert typ == "floor" and obj["area_m2"] == pytest.approx(.36)
    assert len(obj["center_m"]) == len(obj["seed_world"]) == 3
    reg = p.region(obj)
    assert reg.mask.all() and reg.in_mask([obj["seed_world"]]).all()
    assert len(p.walls()) == 1
    assert reg.top_diagnostics["method"] == "auto_height_component"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        auto.write_auto_profile("new_room", room, path.parent, config=CFG)
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        auto.write_auto_profile("new_room", room, tmp_path, config=CFG)


def test_preview_png_dimensions_pixels_and_read_only_profile(tmp_path):
    room = plate()
    height = np.zeros((30, 30))
    height[10:20, 10:20] = np.nan
    with patch.object(tr, "_grid", return_value=(np.zeros(2), height.shape, height)), \
            patch.object(tr, "walls", return_value=[]):
        path = auto.write_auto_profile("new_room", room, tmp_path / "new", config=CFG)
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    output = auto.render_objects_preview(room, path, tmp_path / "preview.png")
    with Image.open(output) as im:
        assert im.size == (320, 240) and im.format == "PNG"
        assert np.any(np.asarray(im) != np.asarray(room.rgb))
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}
    with pytest.raises(ValueError):
        auto.render_objects_preview(room, path, path)


def test_writer_surface_metadata_and_seed_inside_disconnected_surface(tmp_path):
    room = plate()
    height = np.full((60, 60), np.nan)
    height[:30, :30] = .005
    height[40:60, 40:60] = .405
    with patch.object(tr, "_grid", return_value=(np.zeros(2), height.shape, height)), \
            patch.object(tr, "walls", return_value=[]):
        path = auto.write_auto_profile("new_room", room, tmp_path / "new", config=CFG)
    p = profiles.LocationProfile(path)
    obj, typ = p.find_object("surf_1")
    assert typ == "other" and obj["h0"] == .405
    assert obj["center_m"] == pytest.approx([1., 1., .405])
    assert p.region(obj).in_mask([obj["seed_world"]]).all()
    assert obj["soft"] is False
