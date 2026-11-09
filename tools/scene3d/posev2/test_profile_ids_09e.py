"""Explicit instance targets; legacy type/hint/substitutes stay compatible."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import profiles


@pytest.fixture
def profile(tmp_path):
    data = dict(loc_id="test", version=2, reviewed=False, scale={}, anchors=[],
        substitutes={"sofa": ["bed"]}, objects=[
            dict(id="bed_1", name="left", type="bed"), dict(id="bed_2", name="right", type="bed"),
            dict(id="floor_1", name="floor", type="floor"), dict(id="surf_1", name="other", type="other"),
            dict(id="wall_1", name="wall", type="wall")])
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    return profiles.LocationProfile(path)


@pytest.mark.parametrize("target,expected_type", [("bed_2", "bed"), ("surf_1", "other"),
                                                  ("floor_1", "floor"), ("wall_1", "wall")])
def test_exact_id_wins_without_loading_geometry(profile, target, expected_type):
    with patch.object(profile, "region", side_effect=AssertionError("not needed")):
        obj, typ = profile.find_object(target)
    assert obj["id"] == target and typ == expected_type
    assert profile.find_object(target, target)[0] == obj
    assert profile.find_object(target, "different_instance") == (None, None)


def test_types_hints_substitutes_and_unknown_id(profile):
    assert profile.find_object("bed") == (None, None)
    assert profile.find_object("bed", "bed_2")[0]["id"] == "bed_2"
    assert profile.find_object("bed", "floor_1")[0] is None
    assert profile.find_object("sofa", "bed_2")[1] == "bed"
    assert profile.find_object("surf_999") == (None, None)
    assert profile.find_object("left") == (None, None)
    assert profile.find_object("floor")[0]["id"] == "floor_1"
    assert profile.find_object("other")[0]["id"] == "surf_1"


def test_type_nearest_selection_unchanged(profile):
    with patch.object(profile, "region", side_effect=lambda o: SimpleNamespace(
            points3d=lambda: np.array([[0., 0., 0.]]) if o["id"] == "bed_1" else np.array([[5., 0., 0.]]))):
        assert profile.find_object("bed", anchor_xy=[4., 0.])[0]["id"] == "bed_2"
    assert profile.find_object("bed_1", anchor_xy=[4., 0.])[0]["id"] == "bed_1"


def test_explicit_id_can_resolve_legacy_name_without_type(profile):
    profile.data["objects"].append(dict(id="stool_1", name="stool"))
    assert profile.find_object("stool_1")[1] == "stool"
