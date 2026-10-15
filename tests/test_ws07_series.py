"""WS-07: графы серий, варианты промптов (линтер WS-04), раннер против мока ComfyUI — без GPU и без пода."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import ws07_graphs as G  # noqa: E402
import ws07_lib as L  # noqa: E402
import ws07_run as R  # noqa: E402
import ws07_fetch as F  # noqa: E402
from farm.prompts.lint import errors, lint  # noqa: E402
from mock_comfy import MockComfy, tiny_png  # noqa: E402

GOLDEN = REPO / "tests" / "golden" / "prompts"
ARM = {"model": "m.safetensors", "loras": [{"name": "l.safetensors", "w": 1.5}], "knobs": {"pass1_steps": 8}}


@pytest.mark.parametrize("gen", G.GENS)
@pytest.mark.parametrize("ins", [("plate", "sheet"), ("plate", "sheet", "pose"), ("plate", "sheet", "outfit", "pose")])
def test_frame_graph_links_and_output(gen, ins):
    labels = {"plate": 1, "sheet": 2, "outfit": 3, "pose": 4}
    txt = " ".join(f"<Picture {labels[k]}>" for k in ins)
    g = G.frame_graph(gen, {k: f"{k}.png" for k in ins}, txt, 7, ARM)
    assert not G.check_links(g)
    out = G.nid(g, G.OUT_RAW)
    assert g[G.nid(g, "KNOB seed")]["inputs"]["value"] == 7
    assert g[G.nid(g, "KNOB pass1_steps")]["inputs"]["value"] == 8
    assert not any(v["_meta"]["title"].startswith(("faceZ", "faceH3")) for v in g.values())  # проходы лица срезаны
    if gen.startswith(("B", "C")):
        cond = [v for v in g.values() if v["class_type"] in ("MiniMaxH3ReferenceToVideo", "MM1FrameMiniMaxH3ReferenceToImage")]
        assert len(cond) == 1
        refs = sorted(k for k in cond[0]["inputs"] if k.startswith("ref_images."))
        assert refs == [f"ref_images.ref_image_{i}" for i in range(len(ins))]
        # метки перенумерованы подряд под слоты Autogrow
        assert cond[0]["inputs"]["prompt"] == " ".join(f"<Picture {i}>" for i in range(1, len(ins) + 1))
        assert "H3ReferenceEditPrepare" not in G.classes(g)
    if gen == "A_t1":
        assert g[G.nid(g, G.T["prep"])]["inputs"]["quality_profile"] == G.SINGLE_PROFILE
    if gen == "A_p2lo":
        assert g[G.nid(g, "KNOB pass2_steps")]["inputs"]["value"] == 3
    assert g[out]["class_type"] == "SaveImage"


def test_active_configs_exclude_beta4():
    import yaml

    models = yaml.safe_load((REPO / "tools" / "ws07_models.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((REPO / "tools" / "ws07_series.yaml").read_text(encoding="utf-8"))
    assert not any("beta4" in m["file"].lower() for m in models["models"])
    assert not any("beta4" in arm["model"].lower() for arm in cfg["arms"].values())
    for series in cfg["series"].values():
        assert set(series.get("arm", [])) <= cfg["arms"].keys()


def test_renumber_rejects_unconnected_label():
    with pytest.raises(ValueError):
        L.renumber_pictures("<Picture 1> <Picture 4>", [1, 2])


@pytest.mark.parametrize("row", ["SIT-08", "STD-02", "SIT-02", "LIE-04"])
@pytest.mark.parametrize("var", ["fixe", "skeleton", "green", "wide", "idcrop"])
def test_prompt_variants_pass_lint(row, var):
    base = (GOLDEN / f"{row}.frame.txt").read_text(encoding="utf-8")
    t = L.prompt_variant(base, var)
    conn = {"<Picture 1>", "<Picture 2>"} | ({"<Picture 3>"} if "<Picture 3>" in t else set())
    if var == "skeleton" or "<Picture 4>" in base:
        conn.add("<Picture 4>")
    iss = lint(t, "frame", {"connected": conn})
    assert not errors(iss), [(i.code, i.msg) for i in errors(iss)]
    assert "studio" not in t.lower()
    assert not [i for i in iss if i.code == "W_LABEL_UNUSED"]
    if var == "skeleton":
        assert "stick-figure skeleton in <Picture 4>" in t


def test_scale_metrics_and_clip():
    t = [0.4, 0.2, 0.6, 0.9]
    m = L.scale_metrics([0.35, 0.1, 0.65, 0.95], t)
    assert m["scale_ratio"] > 1.2 and not m["clipped"]
    assert L.scale_metrics([0.4, 0.0, 0.6, 0.9], t)["clipped"]
    assert L.scale_metrics(t, t)["scale_err"] == 0


def _kit(tmp: Path) -> Path:
    kit = tmp / "kit"
    for pose in ("Sitting-8", "Standing-2", "Sitting-2", "Lying-4"):
        d = kit / "refs" / pose
        d.mkdir(parents=True)
        for f in ("plate.png", "skeleton.png", "plate_green.png", "plate_wide.png", "pose_photo.jpg", "outfit.png"):
            (d / f).write_bytes(tiny_png())
        for v in ("fixe", "skeleton", "green", "wide", "idcrop"):
            (d / f"prompt_{v}.txt").write_text("<Picture 1> <Picture 2> <Picture 4>", encoding="utf-8")
    for f in ("sheet.jpg", "idcrop.png", "face_ref.png"):
        (kit / "refs" / f).write_bytes(tiny_png())
    tg = {p: {"files": {"plate": "plate.png", "skeleton": "skeleton.png", "plate_green": "plate_green.png",
                        "plate_wide": "plate_wide.png", "pose_photo": "pose_photo.jpg", "outfit": "outfit.png"},
              "has_outfit": False, "has_pose": True, "target_box": [0.4, 0.2, 0.6, 0.9], "wide_crop": [0.14, 0.14, 0.86, 0.86]}
          for p in ("Sitting-8", "Standing-2", "Sitting-2", "Lying-4")}
    (kit / "targets.json").write_text(json.dumps(tg), encoding="utf-8")
    return kit


def test_plan_counts_and_order():
    cfg = R.load_cfg()
    js = R.jobs(cfg, ["s2_scale", "s1_gen"])
    assert len([j for j in js if j["series"] == "s2_scale"]) == 5 * 4 * 2
    assert len({j["id"] for j in js}) == len(js)
    arms = [j["arm"] for j in js]
    assert arms == sorted(arms)               # одна рука подряд
    t, loads = R.estimate(js)
    assert loads == 1 and t > 0


def test_run_resumable_against_mock(tmp_path, monkeypatch):
    kit = _kit(tmp_path)
    monkeypatch.setenv("WS07_KIT", str(kit))
    monkeypatch.setenv("WS07_OUT", str(tmp_path / "out"))
    cfg = R.load_cfg()
    js = [j for j in R.jobs(cfg, ["s2_scale"]) if j["pose"] == "Sitting-8"][:3]
    m = MockComfy().start()
    try:
        from farm.comfy_client import ComfyClient
        c = ComfyClient(m.url, poll_s=0.05)
        assert R.run(c, cfg, js, 0, 0) == 0
        lines = (tmp_path / "out" / "s2_scale" / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(x)["status"] for x in lines] == ["success"] * 3
        files = [Path(f) for x in lines for f in json.loads(x)["files"]]
        assert files and all(f.is_file() for f in files)
        n_sub = sum(m.submissions.values())
        R.run(c, cfg, js, 0, 0)                     # повторный запуск — ничего не отправляет
        assert sum(m.submissions.values()) == n_sub
        body = next(iter(m.bodies.values()))["prompt"]
        assert any(v["_meta"]["title"] == G.OUT_RAW for v in body.values())
        assert any(v["class_type"] == "LoadImage" and v["inputs"]["image"].startswith("ws07/") for v in body.values())
    finally:
        m.stop()


def test_upscale_and_refmod_graphs():
    for meth in ("none", "rtx", "seedvr2_3b", "seedvr2_7b"):
        g = G.upscale_graph("v.mp4", meth, "x")
        assert not G.check_links(g) and "VHS_VideoCombine" in G.classes(g)
    g = G.refmod_extract_graph(["a.png", "b.png"])
    ex = [v for v in g.values() if v["class_type"] == "MiniMaxH3RefModExtract"][0]["inputs"]
    assert ex["extraction_preset"] == "identity_encode" and "refs_image.ref_image_1" in ex


def test_fill_defaults_from_object_info():
    g = {"1": {"class_type": "X", "inputs": {"a": 1}, "_meta": {"title": "x"}}}
    info = {"X": {"input": {"required": {"a": ["INT", {"default": 0}], "b": [["p", "q"], {}],
                                         "c": ["FLOAT", {"default": 0.5}], "d": ["IMAGE"]}}}}
    log = G.fill_defaults(g, info)
    assert g["1"]["inputs"] == {"a": 1, "b": "p", "c": 0.5} and len(log) == 2


ROUND_A = REPO / "tools" / "ws07_round_a.yaml"
ROUND_MODELS = REPO / "tools" / "ws07_round_a_models.yaml"


@pytest.fixture
def round_a(tmp_path, monkeypatch):
    monkeypatch.setenv("WS07_KIT", str(_kit(tmp_path)))
    monkeypatch.setenv("WS07_OUT", str(tmp_path / "out"))
    return R.load_cfg(ROUND_A)


def _round_info(cfg):
    g = G.frame_graph("A", {"plate": "p.png", "sheet": "s.jpg"}, "", 1, cfg["arms"]["base_ref2va"])
    info = {cls: {"input": {"required": {}}} for cls in G.classes(g)}
    info["DiffusionModelLoaderKJ"]["input"]["required"]["model_name"] = [[a["model"] for a in cfg["arms"].values()], {}]
    for node in g.values():
        for key in ("clip_name", "vae_name"):
            if key in node["inputs"]:
                info[node["class_type"]]["input"]["required"][key] = [[node["inputs"][key]], {}]
    return info


def test_round_a_matrix_and_manifest():
    import yaml

    cfg = R.load_cfg(ROUND_A)
    old = R.load_cfg()
    js = R.jobs(cfg, ["round_a"])
    assert cfg["poses"] == old["poses"] and cfg["seeds"] == old["seeds"]
    assert len(js) == len({j["id"] for j in js}) == 24
    assert {j["gen"] for j in js} == {"A"}
    assert {j["variant"] for j in js} == {"fixe"}
    for arm in cfg["arms"]:
        aj = [j for j in js if j["arm"] == arm]
        assert len(aj) == 8
        for pose in cfg["poses"]:
            assert {j["seed"] for j in aj if j["pose"] == pose} == set(cfg["seeds"][pose])
    manifest = yaml.safe_load(ROUND_MODELS.read_text(encoding="utf-8"))
    ms = F.items(manifest, {"round_a"})
    assert len(ms) == 5
    assert {m["file"] for m in ms if m["folder"] == "diffusion_models"} == {a["model"] for a in cfg["arms"].values()}
    assert {m["folder"] for m in ms} == {"diffusion_models", "vae", "text_encoders"}
    assert not any("beta4" in m["file"].lower() or "turbo" in m["file"].lower() for m in ms)
    assert not any(a.get("loras") or a.get("alt_model") for a in cfg["arms"].values())
    new = [m for m in ms if m.get("sha256")]
    assert len(new) == 2 and sum(m["bytes"] for m in new) == 42026673704
    assert all(len(m["sha256"]) == 64 and len(m["revision"]) == 40 for m in new)
    assert {m["sha256"] for m in new} == {
        "488e0d51fad9fd6b277b6ebfbe46b3fd44374ff7baa1e3c9dfce58d3a1e5b33c",
        "cff4adf96f503d4a7f631e5757306192d3bbc6477980da244caba86e2656654e",
    }
    assert not F.items(manifest, {"s1", "s2"})


@pytest.mark.parametrize("arm", ["base_ref2va", "eros_beta5", "pinkcherry_v18"])
def test_round_a_graph_no_lora_and_explicit_sampling(arm):
    cfg = R.load_cfg(ROUND_A)
    a = cfg["arms"][arm]
    g = G.frame_graph("A", {"plate": "p.png", "sheet": "s.jpg"}, "", 17, a)
    assert not G.check_links(g)
    assert not any("lora" in cls.lower() for cls in G.classes(g))
    assert not any(v["_meta"]["title"].startswith(("faceZ", "faceH3")) for v in g.values())
    assert sum(n["class_type"] == "SaveImage" for n in g.values()) == 1
    assert {n["inputs"]["model_name"] for n in g.values() if "model_name" in n["inputs"]} == {a["model"]}
    for t in ("p1", "p2"):
        inp = g[G.nid(g, G.T[t])]["inputs"]
        assert {k: inp[k] for k in a["sampler"]} == a["sampler"]
        assert inp["cfg"] == 1.0 and inp["sampler_name"] == "res_multistep" and inp["scheduler"] == "simple"
    for knob, value in {"pass1_steps": 20, "pass2_total": 20, "pass2_steps": 8, "shift": 12}.items():
        assert g[G.nid(g, f"KNOB {knob}")]["inputs"]["value"] == value
    assert "MiniMaxH3RefModApply" not in G.classes(g)
    assert not any(cls.startswith(("VHS_", "MM1Frame", "SeedVR2")) for cls in G.classes(g))
    math = g[G.nid(g, "gen · старт 2-го прохода = total - steps")]["inputs"]
    assert math["values.a"] == [G.nid(g, "KNOB pass2_total"), 0]
    assert math["values.b"] == [G.nid(g, "KNOB pass2_steps"), 0]


def test_round_a_sampling_changes_invalidate_resume_ids():
    import copy

    cfg = R.load_cfg(ROUND_A)
    other = copy.deepcopy(cfg)
    other["arms"]["base_ref2va"]["knobs"] = {**other["arms"]["base_ref2va"]["knobs"], "pass1_steps": 30}
    old = R.jobs(cfg, ["round_a"])
    new = R.jobs(other, ["round_a"])
    assert {j["id"] for j in old if j["arm"] == "base_ref2va"}.isdisjoint(j["id"] for j in new)
    assert {j["id"] for j in old if j["arm"] != "base_ref2va"} <= {j["id"] for j in new}


def test_round_a_never_substitutes_alt_model():
    cfg = R.load_cfg(ROUND_A)
    a = cfg["arms"]["base_ref2va"]
    a["alt_model"] = "other.safetensors"
    result = R._arm(cfg, "base_ref2va", {a["alt_model"]})
    assert result["model"] == a["model"] and result is not a


def test_round_a_plan_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ws07_run", "plan", "round_a", "--config", str(ROUND_A)])
    assert R.main() == 0
    text = capsys.readouterr().out
    assert "24" in text and "--frame-seconds" in text and "≈" not in text
    monkeypatch.setattr(sys, "argv", ["ws07_run", "plan", "round_a", "--config", str(ROUND_A),
                                     "--arm", "base_ref2va", "--frame-seconds", "120"])
    assert R.main() == 0
    text = capsys.readouterr().out
    assert "  8" in text and "≈" in text


def test_round_a_run_requires_budget(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ws07_run", "run", "round_a", "--config", str(ROUND_A)])
    with pytest.raises(SystemExit) as e:
        R.main()
    assert e.value.code == 2


def test_round_a_check_scoped_to_jobs(round_a, monkeypatch):
    info = _round_info(round_a)
    round_a["arms"]["unused"] = {"model": "missing.safetensors", "loras": [{"name": "missing_lora", "w": 1}]}
    monkeypatch.setattr(R, "object_info", lambda client: info)
    js = R.jobs(round_a, ["round_a"])
    seen = []
    original = R.frame_job_graph

    def graph(cfg, j, *args):
        seen.append((j["arm"], j["pose"]))
        return original(cfg, j, *args)

    monkeypatch.setattr(R, "frame_job_graph", graph)
    assert R.check(None, round_a, js) == 0
    assert len(seen) == len(set(seen)) == 12
    assert "unused" not in {a for a, _ in seen}
    info["DiffusionModelLoaderKJ"]["input"]["required"]["model_name"][0].pop()
    assert R.check(None, round_a, js) == 1


@pytest.mark.parametrize("cls,key", [("CLIPLoader", "clip_name"), ("VAELoader", "vae_name")])
def test_round_a_check_missing_common_file(round_a, monkeypatch, cls, key):
    info = _round_info(round_a)
    info[cls]["input"]["required"][key][0].clear()
    monkeypatch.setattr(R, "object_info", lambda client: info)
    assert R.check(None, round_a, R.jobs(round_a, ["round_a"])) == 1


def test_round_a_run_resume_24_and_stop_on_error(round_a, tmp_path, monkeypatch):
    from farm.comfy_client import ComfyClient

    monkeypatch.setattr(R, "object_info", lambda client: _round_info(round_a))
    legacy = Path(round_a["out"]) / "s2_scale" / "jobs.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    js = R.jobs(round_a, ["round_a"])
    m = MockComfy().start()
    try:
        c = ComfyClient(m.url, poll_s=0.01)
        assert R.run(c, round_a, js, 0, 1) == 0
        journal = Path(round_a["out"]) / "round_a" / "jobs.jsonl"
        recs = [json.loads(x) for x in journal.read_text(encoding="utf-8").splitlines()]
        assert len(recs) == 24 and {r["status"] for r in recs} == {"success"}
        assert all(Path(f).is_file() for r in recs for f in r["files"])
        n = sum(m.submissions.values())
        assert n == 24
        assert R.run(c, round_a, js, 0, 1) == 0 and sum(m.submissions.values()) == n
        assert legacy.read_text(encoding="utf-8") == "legacy"
        for body in m.bodies.values():
            assert not any("lora" in cls.lower() for cls in G.classes(body["prompt"]))
        round_a["out"] = str(tmp_path / "errors")
        interrupted = []
        monkeypatch.setattr(c, "wait", lambda *args: (_ for _ in ()).throw(TimeoutError("test timeout")))
        monkeypatch.setattr(c, "interrupt", lambda pid: interrupted.append(pid))
        assert R.run(c, round_a, js, 0, 1) == 1
        assert sum(m.submissions.values()) == n + 1
        assert len(interrupted) == 1 and interrupted[0]
    finally:
        m.stop()


def test_round_a_run_missing_model_does_not_upload(round_a, monkeypatch):
    monkeypatch.setattr(R, "object_info", lambda client: {})
    monkeypatch.setattr(R, "upload_kit", lambda *args: pytest.fail("не надо загружать kit"))
    assert R.run(None, round_a, R.jobs(round_a, ["round_a"]), 0, 1) == 1


def test_fetch_verify_size_and_lfs_sha(tmp_path):
    import hashlib

    p = tmp_path / "m.safetensors"
    p.write_bytes(b"weights")
    m = {"file": p.name, "bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}
    F.verify(m, p)
    with pytest.raises(ValueError, match="SHA256"):
        F.verify({**m, "sha256": "0" * 64}, p)
    with pytest.raises(ValueError, match="размер"):
        F.verify({**m, "bytes": 8}, p)


@pytest.mark.parametrize("corrupt", [False, True])
def test_fetch_config_staged_and_failure_exit(tmp_path, monkeypatch, corrupt):
    import hashlib
    from types import SimpleNamespace
    import yaml

    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(F, "MODELS", root)
    monkeypatch.setattr(F.shutil, "disk_usage", lambda path: SimpleNamespace(free=1e12))
    m = {"file": "selected.safetensors", "folder": "diffusion_models", "need": "round_a", "src": "hf",
         "license": "test", "bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": [m]}), encoding="utf-8")
    paths = []

    def fetch(model, dst):
        assert model["file"] == m["file"] and dst.parent.name.startswith(".ws07-")
        paths.append(dst)
        dst.write_bytes(b"corrupt" if corrupt else b"weights")

    monkeypatch.setattr(F, "fetch_hf", fetch)
    monkeypatch.setattr(sys, "argv", ["ws07_fetch", "--config", str(config), "--need", "round_a"])
    assert F.main() == int(corrupt)
    final = root / "diffusion_models" / m["file"]
    assert final.exists() == (not corrupt)
    assert len(paths) == 1 and not paths[0].parent.exists()
    if not corrupt:
        monkeypatch.setattr(F, "fetch_hf", lambda *args: pytest.fail("не перекачивать существующие веса"))
        assert F.main() == 0


def test_fetch_missing_local_dependency_blocks_downloads(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(F, "MODELS", tmp_path)
    monkeypatch.setattr(F, "find", lambda name: None)
    monkeypatch.setattr(F.shutil, "disk_usage", lambda path: SimpleNamespace(free=1e12))
    monkeypatch.setattr(F, "fetch_hf", lambda *args: pytest.fail("нет общего кодировщика — не качать"))
    monkeypatch.setattr(sys, "argv", ["ws07_fetch", "--config", str(ROUND_MODELS), "--need", "round_a"])
    assert F.main() == 1
