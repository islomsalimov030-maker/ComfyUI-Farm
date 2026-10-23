"""23.10: служебные ноды farm_auto — кадр из папки по ID, ZIP итогов."""
import zipfile

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
from farm import util_nodes as U  # noqa: E402


def test_find_frame_and_zip(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_OUTPUT", str(tmp_path))
    d = tmp_path / "frames"; d.mkdir()
    Image.new("RGB", (8, 8)).save(d / "SAMH-STD01_ref2v.png")
    Image.new("RGB", (8, 8)).save(d / "SAMH-STD010_ref2v.png")
    assert U.find_frame(str(d), "samh-std01").name == "SAMH-STD01_ref2v.png"
    img = torch.from_numpy(np.zeros((1, 8, 8, 3), dtype="float32"))
    r = U.FarmZip().run("photo", "SAMH-STD01", image_1=img)
    z = tmp_path / "farm_zip" / "photo.zip"
    assert z.is_file() and "SAMH-STD01_ref2v.png" in zipfile.ZipFile(z).namelist()
    assert "/view?filename=photo.zip" in r["ui"]["text"][0]
    pick = U.FarmPickFirst()
    assert pick.check_lazy_status(True, new_2=None) == ["new_2"]
    assert pick.check_lazy_status(True, new_1=None, new_2=None) == ["new_1"]
    assert pick.check_lazy_status(False, from_folder=img) == []
    assert pick.run(True, new_2=img)[0] is img


def test_facepass_panel_and_finalize(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_INPUT", str(tmp_path / "in"))
    monkeypatch.setenv("COMFY_OUTPUT", str(tmp_path / "out"))
    d = tmp_path / "in" / "stacy_fp"; d.mkdir(parents=True)
    for n in ("b.mp4", "a.mov", "a_facepass_stacy.mp4", "note.txt"):
        (d / n).write_bytes(b"x")
    assert [p.name for p in U.list_videos("stacy_fp")] == ["a.mov", "b.mp4"]
    assert U.list_videos(str(d / "b.mp4"))[0].name == "b.mp4"
    kw = {k: d0 for k, _, d0, *_ in U.FP_SPEC}
    r = U.FarmPanelFacepass().run("stacy_fp", 1, True, "p", face_mod_name="farm/nope", **kw)
    assert r[0].endswith("b.mp4") and r[1] == "b_facepass_stacy" and r[2] == "p" and len(r) == 5 + len(U.FP_SPEC)
    assert r[-1]["mode"] is False and r[-1]["face_denoise"] == 0.55 and r[-1]["face_lora"] == 0.75
    assert r[3]["mod"] == "farm/stacy_face" and U.pick_mod("x", ["(none)", "ss340/face_n10"]) == "ss340/face_n10"
    assert len(U.FarmPanelFacepass.RETURN_TYPES) == len(U.FarmPanelFacepass.RETURN_NAMES)
    assert all(isinstance(t, str) for t in U.FarmPanelFacepass.RETURN_TYPES)
    r = U.FarmPanelFacepass().run("x", 0, True, "p", video="stacy_fp/a.mov", **kw)
    assert r[1] == "a_facepass_stacy"
    with pytest.raises(IndexError):
        U.FarmPanelFacepass().run("stacy_fp", 5, True, "p", **kw)
    o = tmp_path / "out" / "stacy_facepass"; v = o / "_vhs"; v.mkdir(parents=True)
    (v / "b_facepass_stacy_00001.mp4").write_bytes(b"v")
    (v / "b_facepass_stacy_00001.png").write_bytes(b"i")
    dst = U.finalize_video([v / "b_facepass_stacy_00001.png", v / "b_facepass_stacy_00001.mp4"], "b_facepass_stacy")
    assert dst.name == "b_facepass_stacy.mp4" and sorted(p.name for p in o.iterdir()) == ["_vhs", "b_facepass_stacy.mp4"]
    assert sorted(p.name for p in v.iterdir()) == ["b_facepass_stacy_00001.mp4"] and dst.read_bytes() == b"v"
    assert "studio" not in U.FP_PROMPT.lower()
