"""WS-06: проверки видео и post.loop/post.color на синтетических клипах (CPU, без моделей).

Синтетика: текстурный фон + «человек» (эллипс) с управляемым движением. Каждый тест ломает одно
свойство и ждёт свой код отказа; чистая петля проходит все проверки.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from farm.contracts import Asset, QAContext  # noqa: E402
from farm.qa import registry  # noqa: E402
from farm.qa.video import _frames as F  # noqa: E402

TH = {"video": {"disabled": ["identity_dip"], "cut": {"scenedetect": False}}}
FPS = 24.0
W, H = 320, 180


def _bg(seed=0, w=W + 40, h=H + 40):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8), (0, 0), 1.0)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(40):   # углы для трекинга фона
        x, y = int(rng.integers(0, w - 10)), int(rng.integers(0, h - 10))
        cv2.rectangle(base, (x, y), (x + 6, y + 6), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return base


def make_clip(path: Path, n=120, motion=lambda t: (0.0, 0.0), cam=lambda t: (0.0, 0.0),
              light=lambda t: 0.0, cut_at=None, amp=1.0) -> Path:
    bg = _bg(0)
    bg2 = _bg(7)
    frames = []
    for i in range(n):
        t = i / n
        src = bg2 if cut_at is not None and i >= cut_at else bg
        cx, cy = cam(t)
        y0, x0 = 20 + int(round(cy)), 20 + int(round(cx))
        img = src[y0:y0 + H, x0:x0 + W].copy()
        dx, dy = motion(t)
        cv2.ellipse(img, (int(W / 2 + dx * amp), int(H / 2 + dy * amp)), (20, 40), 0, 0, 360, (90, 120, 200), -1)
        cv2.circle(img, (int(W / 2 + dx * amp), int(H / 2 - 50 + dy * amp)), 14, (150, 170, 220), -1)
        img = np.clip(img.astype(np.float32) + light(t), 0, 255).astype(np.uint8)
        frames.append(img)
    return F.write(path, np.stack(frames), FPS)


def loop_motion(t):   # покачивание 2 периода за 5 с (живая скорость), кадр n ≡ кадр 0
    return 15 * math.sin(4 * math.pi * t), 3 * math.sin(8 * math.pi * t)


def run(path: Path, loop=True, th=None):
    ctx = QAContext(shot={"shot_id": "s", "loop": int(loop)}, stage="video", profile={}, attempt={},
                    thresholds=th or TH, refs={"loop": loop}, extra={})
    res = registry.run_stage("video", Asset("a", "a", "video", str(path), ""), ctx)
    return {r.name: r for r in res}


def codes(res):
    return {r.fail_code for r in res.values() if not r.passed and r.fail_code}


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("ws06")
    out = {
        "good": make_clip(d / "good.mp4", motion=loop_motion),
        # рывок: движение идёт 0..0.8 периода и обрывается — последний кадр далеко от первого
        "seam": make_clip(d / "seam.mp4", motion=lambda t: loop_motion(t * 0.8)),
        "pan": make_clip(d / "pan.mp4", motion=loop_motion, cam=lambda t: (15 * t, 0.0)),
        "light": make_clip(d / "light.mp4", motion=loop_motion, light=lambda t: 25 * math.sin(math.pi * t)),
        "cut": make_clip(d / "cut.mp4", motion=loop_motion, cut_at=60),
        # заморозка: движение 0..0.3, стоп на 1.5 с посреди клипа, потом движение и возврат
        "freeze": make_clip(d / "freeze.mp4", n=144, motion=lambda t: loop_motion(
            0.0 if 0.35 < t < 0.65 else (t if t <= 0.35 else t - 0.3) / 0.7)),
        "slow": make_clip(d / "slow.mp4", motion=loop_motion, amp=0.08),
    }
    return out


def test_all_video_checks_registered():
    names = {c.name for c in registry.checks("video")}
    assert {"seam", "seam_velocity", "cut", "camera_move", "light_drift", "color_drift", "flicker",
            "freeze", "slowmo", "identity_dip"} <= names


def test_good_loop_passes(clips):
    res = run(clips["good"])
    assert not codes(res), {k: (v.value, v.evidence) for k, v in res.items() if not v.passed}


def test_seam_detected_and_skipped_for_transition(clips):
    assert "SEAM" in codes(run(clips["seam"]))
    res = run(clips["seam"], loop=False)
    assert "seam" not in res and "seam_velocity" not in res


def test_camera_pan_detected(clips):
    res = run(clips["pan"])
    assert "CAMERA_MOVE" in codes(res)
    assert res["camera_move"].evidence["shift_px_src"] > 8


def test_light_drift_detected(clips):
    assert "LIGHT_DRIFT" in codes(run(clips["light"]))


def test_cut_detected(clips):
    res = run(clips["cut"])
    assert "CUT" in codes(res)
    assert 60 in res["cut"].evidence["cuts"]


def test_freeze_detected(clips):
    assert "FREEZE" in codes(run(clips["freeze"]))


def test_slowmo_detected(clips):
    assert "SLOWMO" in codes(run(clips["slow"]))
    assert "SLOWMO" not in codes(run(clips["good"]))


def test_post_loop_trims_duplicate_and_fixes_seam(clips, tmp_path):
    from farm.post import loop as PL
    fr, fps = F.read_full(clips["seam"])
    # дубль first=last: добавить копию кадра 0 в конец — post.loop должен его отрезать
    fr = np.concatenate([fr, fr[:1]])
    out, rep = PL.repair(fr, {})
    assert rep["trim_to"] == len(fr) - 1
    assert rep["ok"] and rep["seam_after"] < rep["seam_before"]
    assert len(out) == rep["trim_to"] - rep["crossfade"]
    p = F.write(tmp_path / "fixed.mp4", out, fps)
    assert "SEAM" not in codes(run(p))


def test_post_loop_fix_interface(clips):
    from farm.post import loop as PL
    ctx = QAContext(shot={"shot_id": "s", "loop": 1}, stage="video", profile={}, attempt={},
                    thresholds=TH, extra={})
    path = PL.fix(Asset("a", "a", "video", str(clips["seam"]), ""), ctx)
    assert Path(path).is_file() and ctx.extra["post_loop"]["ver"].startswith("post.loop")


def test_post_color_removes_light_drift(clips, tmp_path):
    from farm.post import color as PC
    ctx = QAContext(shot={"shot_id": "s", "loop": 1}, stage="video", profile={}, attempt={},
                    thresholds=TH, extra={})
    path = PC.fix(Asset("a", "a", "video", str(clips["light"]), ""), ctx)
    res = run(Path(path))
    assert "LIGHT_DRIFT" not in codes(res)
    assert res["light_drift"].value < 2.0


def test_crossfade_keeps_uint8_and_length():
    from farm.post.loop import crossfade
    fr = np.random.default_rng(0).integers(0, 255, (30, 8, 8, 3), dtype=np.uint8)
    out = crossfade(fr, 6)
    assert out.dtype == np.uint8 and len(out) == 24
    assert np.array_equal(out[6:], fr[6:24])   # тело не тронуто


def test_routing_sends_cut_to_seed_before_seam():
    from farm.config import load_config, load_routing
    routes = list((load_routing(load_config()).get("routes") or {}))
    assert routes.index("CUT") < routes.index("SEAM")
    r = load_routing(load_config())["routes"]
    assert r["SEAM"]["action"] == "post_fix" and r["SEAM"]["fixer"] == "post.loop"
    assert r["COLOR_DRIFT"]["fixer"] == "post.color"


def test_results_are_plain_json(clips):
    """Воркер пишет int(passed) и json(evidence): numpy-типы ломали прогон (identity_dip, np.bool_)."""
    import json
    for name in ("good", "pan", "cut"):
        for r in run(clips[name]).values():
            assert type(r.passed) is bool and isinstance(r.value, float)
            json.dumps(r.evidence)
            assert (r.fail_code is None) == r.passed
