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
    assert pick.check_lazy_status(True) == ["new_1", "new_2", "new_3"]
    assert pick.run(True, new_2=img)[0] is img
