"""0.3.8: зона ожидаемых теней входит в область H3 с мягким краем; тень из рендера мягче и светлее."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/scene3d'), str(ROOT / 'tools/scene3d/plate')]
import pack_cli


def test_shadow_zone_extends_strength_and_softens_shadow(tmp_path):
    sil = np.zeros((200, 300), np.uint8); sil[50:150, 50:100] = 255
    Image.fromarray(sil).save(tmp_path / 'silhouette.png')
    shadow = np.zeros((200, 300), np.uint8); shadow[120:150, 100:220] = 200
    Image.fromarray(shadow).save(tmp_path / 'shadow.png')
    strength = np.where(sil > 0, 255, 0).astype(np.uint8)
    Image.fromarray(strength).save(tmp_path / 'strength.png')
    out = pack_cli.shadow_zone(tmp_path, [tmp_path / 'strength.png'], 8, .7, .6)
    s = np.asarray(Image.open(tmp_path / 'strength.png')) / 255
    room = np.asarray(Image.open(out)) / 255
    assert abs(s[135, 180] - .7) < .02 and s[10, 280] == 0 and s[100, 70] == 1
    assert room[135, 180] < .6 * 200 / 255 + .02 and room[100, 70] == 0


def test_no_shadow_files_no_zone(tmp_path):
    Image.fromarray(np.zeros((20, 20), np.uint8)).save(tmp_path / 'silhouette.png')
    assert pack_cli.shadow_zone(tmp_path, [], 8) is None
