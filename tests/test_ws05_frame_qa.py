"""WS-05: реестр проверок кадра, маршрут, геометрия масштаба, разбор logprobs — без моделей и сети."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from farm import config
from farm.contracts import Asset, CheckResult, QAContext
from farm.qa import registry, vlm
from farm.qa.image import scale


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(registry, "_CHECKS", {})
    monkeypatch.setattr(registry, "_loaded", True)
    return registry


def _ctx(th=None):
    return QAContext(shot={"shot_id": "T"}, stage="frame", profile={}, attempt={}, thresholds=th or {})


def _asset():
    return Asset("a1", "t1", "frame", "x.png", "sha")


def test_run_stage_order_stop_and_errors(clean_registry):
    calls = []

    @registry.register("frame", "cheap", cost=0)
    def _a(asset, ctx):
        calls.append("cheap")
        return CheckResult("cheap", 1.0, False, "hard", "SHARP")

    @registry.register("frame", "boom", cost=1)
    def _b(asset, ctx):
        calls.append("boom")
        raise RuntimeError("модель не загрузилась")

    @registry.register("frame", "costly", cost=5)
    def _c(asset, ctx):
        calls.append("costly")
        return CheckResult("costly", 1.0, True, "soft")

    @registry.register("frame", "silent", cost=2)
    def _d(asset, ctx):
        return None

    res = registry.run_stage("frame", _asset(), _ctx({"frame": {"stop_cost": 3}}))
    assert calls == ["cheap", "boom"]                     # дешёвые первыми, дорогая пропущена после hard
    by = {r.name: r for r in res}
    assert by["boom"].fail_code == "TECH" and by["boom"].severity == "info"   # исключение не роняет воркер
    assert by["costly"].evidence["skipped"] and by["costly"].passed
    assert "silent" not in by                             # None — проверка неприменима
    assert "ms" in by["cheap"].evidence


def test_video_runs_all_and_disabled(clean_registry):
    seen = []
    for name, cost in (("seam", 1), ("cut", 2)):
        @registry.register("video", name, cost=cost)
        def _f(asset, ctx, name=name):
            seen.append(name)
            return CheckResult(name, 0.0, False, "hard", name.upper())
    registry.run_stage("video", _asset(), _ctx())
    assert seen == ["seam", "cut"]                        # видео: без остановки на первом hard (WS-06)
    seen.clear()
    registry.run_stage("video", _asset(), _ctx({"video": {"disabled": ["seam"]}}))
    assert seen == ["cut"]


def test_routing_override_priority():
    r = config.load_routing({})
    keys = list(r["routes"])
    assert r["routes"]["FACE_UNRELIABLE"]["action"] == "needs_human"
    assert keys.index("CUT") < keys.index("SEAM")         # CUT+SEAM → новый seed, не постобработка
    assert keys.index("FACE_ID") < keys.index("FACE_UNRELIABLE")


def test_thresholds_yaml_has_frame_sections():
    th = yaml.safe_load(open(config.PKG_DIR / "qa" / "thresholds_sfw_base.yaml", encoding="utf-8"))
    for k in ("sharp", "face_detect", "face_id", "age", "hands", "scale", "checklist"):
        assert k in th["frame"]
    assert th["frame"]["face_detect"]["detector"] == "mediapipe"     # прод — коммерчески чистый детектор
    assert th["frame"]["face_id"]["backend"] == "auraface"
    assert th["frame"]["age"]["model"] == "none"                     # genderage некоммерческий — не в проде


def test_scale_frac_interpolation():
    # стоя: лодыжки y=1000 (0.04), колени 800 (0.285), таз 600 (0.53), плечи 300 (0.81)
    lv = [(1000, 0.04), (800, 0.285), (600, 0.53), (300, 0.81)]
    assert scale.frac_at(700, lv) == pytest.approx((0.285 + 0.53) / 2)
    assert scale.frac_at(1000, lv) == pytest.approx(0.04)
    assert scale.frac_at(200, lv) > 0.81                  # выше плеч — экстраполяция
    assert scale.frac_at(500, [(600, 0.53)]) is None


def test_scale_body_levels_visibility():
    p = {f"{s}_{n}": (0.0, y, vis) for n, y in (("ankle", 1000), ("knee", 800), ("hip", 600),
                                                 ("shoulder", 300), ("eye", 150)) for s in "lr"
         for vis in [0.9 if n != "eye" else 0.1]}
    lv = scale.body_levels(p)
    assert [f for _, f in lv] == [0.04, 0.285, 0.53, 0.81]   # невидимые глаза отброшены


def test_vlm_p_yes_from_logprobs():
    lp = lambda p: float(np.log(p))  # noqa: E731
    resp = {"logprobs": [{"token": "Yes", "logprob": lp(0.6),
                          "top_logprobs": [{"token": "Yes", "logprob": lp(0.6)}, {"token": " yes", "logprob": lp(0.1)},
                                           {"token": "No", "logprob": lp(0.2)}, {"token": "The", "logprob": lp(0.1)}]}]}
    assert vlm._p_from_logprobs(resp) == pytest.approx(0.7 / 0.9)
    assert vlm._p_from_logprobs({"response": "Yes"}) is None


def test_vlm_unavailable_is_info(monkeypatch):
    from farm.qa.image import checklist
    monkeypatch.setattr(vlm, "DEFAULT_URL", "http://127.0.0.1:9")   # закрытый порт
    import farm.qa.image._common as C
    monkeypatch.setattr(C, "state", lambda asset: {"img": np.zeros((8, 8, 3), np.uint8)})
    ctx = _ctx({"frame": {"checklist": {"enabled": True}}})
    ctx.profile = {"llm": {"url": "http://127.0.0.1:9"}}
    r = checklist.check(_asset(), ctx)
    assert r.severity == "info" and r.passed and "error" in r.evidence


def test_face_yaw_roll_from_points():
    from farm.qa.face import Face
    # анфас, глаза горизонтально
    f = Face(bbox=np.array([0, 0, 100, 100.0]), kps=np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80.0]]), score=1)
    assert abs(f.yaw) < 1 and abs(f.roll) < 1
    # тот же анфас, повёрнутый на 60° в плоскости кадра: yaw не меняется, roll ≈ 60
    c = np.array([50, 60.0])
    rot = np.array([[np.cos(np.pi / 3), -np.sin(np.pi / 3)], [np.sin(np.pi / 3), np.cos(np.pi / 3)]])
    f2 = Face(bbox=f.bbox, kps=(f.kps - c) @ rot.T + c, score=1)
    assert abs(f2.yaw) < 1 and f2.roll == pytest.approx(60, abs=1)
