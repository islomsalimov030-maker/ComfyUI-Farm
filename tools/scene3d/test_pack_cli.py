"""Offline orchestration tests: no model downloads, GPU, H3 or Blender execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

SPEC = importlib.util.spec_from_file_location("scene3d_pack_cli", Path(__file__).with_name("pack_cli.py"))
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def config(tmp_path):
    return cli.config_for({"config": {"farm_root": str(tmp_path), "python": sys.executable}})


def profile_dir(tmp_path):
    directory = tmp_path / "locations" / "room"
    directory.mkdir(parents=True)
    for name in ("plate.json", "plate_mesh.obj", "plate_rgb.png", "depth_loc.png"):
        (directory / name).write_text("fixture", encoding="utf-8")
    cli.save_json(directory / "profile.json", {"loc_id": "room", "objects": [
        {"id": "bed_2", "type": "bed", "h0": .5}, {"id": "wall_1", "type": "wall", "z_m": [0, 2.5]}]})
    return directory


def test_config_defaults_and_fixed_cell(tmp_path):
    cfg = config(tmp_path)
    assert cfg["cell_px"] == 16
    assert cfg["geometry_python"] == "C:/work/ComfyUI-next/venv/Scripts/python.exe"
    assert cfg["comfy"] == "C:/work/ComfyUI-next"
    for cell in (8, 16.5, True):
        with pytest.raises(ValueError, match="cell_px"):
            cli.config_for({"config": {"farm_root": str(tmp_path), "cell_px": cell}})


@pytest.mark.parametrize("value", ["../x", "a/b", "a\\b", "..", "", None, "/tmp"])
def test_location_id_does_not_escape_cache(value):
    with pytest.raises(ValueError):
        cli.location_id(value)


def test_location_cache_without_gpu_or_source(tmp_path, monkeypatch):
    directory = profile_dir(tmp_path)
    monkeypatch.setattr(cli, "run_stage", lambda *a, **k: pytest.fail("cached location must not run GPU"))
    result = cli.do_location({"location_id": "room", "plate": "not-needed.png"}, config(tmp_path))
    assert result["status"] == "ok"
    assert result["location"] == str((directory / "profile.json").resolve())
    assert result["objects"] == [{"id": "bed_2", "type": "bed", "height_m": .5},
                                 {"id": "wall_1", "type": "wall", "height_m": 2.5}]


def test_legacy_d4_flag_is_preserved_without_fake_mask():
    region = SimpleNamespace(flags=["dominant_top_unavailable"], object_mask=None)
    profile = SimpleNamespace(data={"objects": [{"id": "bed_1", "top_grid_npz": "top.npz"}]},
                              region=lambda o: region)
    cli.require_profile_geometry(profile)
    assert region.flags == ["dominant_top_unavailable"]
    assert region.object_mask is None


def test_legacy_geometry_directory_does_not_modify_profile(tmp_path):
    directory = profile_dir(tmp_path)
    original = (directory / "profile.json").read_bytes()
    external = tmp_path / "geometry"
    external.mkdir()
    for name in ("plate.json", "plate_mesh.obj", "plate_rgb.png", "depth_loc.png"):
        (directory / name).replace(external / name)
    cfg = config(tmp_path)
    cfg["location_asset_dirs"] = {"room": str(external)}
    result = cli.do_location({"location_id": "room"}, cfg)
    assert result["status"] == "ok"
    assert result["plate"] == str((external / "plate_rgb.png").resolve())
    assert (directory / "profile.json").read_bytes() == original
    assert not (directory / "plate_mesh.obj").exists()


def test_profile_without_mesh_is_explicit_error(tmp_path):
    directory = profile_dir(tmp_path)
    (directory / "plate_mesh.obj").unlink()
    with pytest.raises(ValueError, match="plate_mesh.obj"):
        cli.do_location({"location_id": "room"}, config(tmp_path))


def test_new_location_runs_existing_moge_then_stops_without_hints(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)
    calls = []
    def run(command, stage):
        calls.append((command, stage))
        out = Path(command[3])
        for name in ("plate.json", "plate_mesh.obj", "plate_rgb.png", "depth_loc.png"):
            (out / name).write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(cli, "run_stage", run)
    with pytest.raises(ValueError, match="location_hints_by_id"):
        cli.do_location({"location_id": "new_room", "plate": str(source)}, config(tmp_path))
    assert len(calls) == 1
    assert str(calls[0][0][1]).endswith("plate_geometry.py")
    assert not (tmp_path / "locations/new_room/profile.json").exists()


def test_contacts_and_support_defaults():
    assert cli.parts("knee.L, shin.L") == ["knee.L", "shin.L"]
    assert cli.contacts("hand.L:wall, hand.R:floor") == [
        {"part": "hand.L", "surface": "wall"}, {"part": "hand.R", "surface": "floor"}]
    assert cli.contacts('[{"part":"hand.L","surface":"wall"}]')[0]["surface"] == "wall"
    assert len(cli.DEFAULT_SUPPORT) == 9
    with pytest.raises(ValueError):
        cli.contacts([{"part": "hand.L"}])


class Profile:
    data = {"loc_id": "room", "objects": []}
    def find_object(self, target, **kw):
        self.selection = (target, kw)
        return {"id": "bed_2"}, target
    def anchor(self, oid, pc):
        return {"object_id": oid, "pose_classes": [pc]}
    def zone_metrics(self, anchor):
        return {"xy": np.array([[1., 2.], [3., 4.]]), "h": np.array([.5, .5])}


def test_seed_reads_profile_zone_and_passes_instance_hint(tmp_path):
    profile = Profile()
    spec = SimpleNamespace(target_object="bed", instance_hint="bed_2", pose_class="kneel")
    assert np.allclose(cli.seed_for(profile, spec, config(tmp_path)), [2, 3, .5])
    assert profile.selection[1]["instance_hint"] == "bed_2"


def test_seed_missing_or_ambiguous_never_guesses(tmp_path):
    profile = Profile()
    spec = SimpleNamespace(target_object="bed", instance_hint=None, pose_class="kneel")
    profile.find_object = lambda *a, **kw: (None, None)
    with pytest.raises(ValueError, match="ambiguous"):
        cli.seed_for(profile, spec, config(tmp_path))
    profile.find_object = lambda *a, **kw: ({"id": "bed_1"}, "bed")
    profile.anchor = lambda *a: None
    with pytest.raises(ValueError, match="no anchor"):
        cli.seed_for(profile, spec, config(tmp_path))


def test_explicit_config_seed_is_not_replaced_by_zone_center(tmp_path):
    cfg = config(tmp_path)
    cfg["location_hints_by_id"] = {"room": {"seeds": {"bed": [5, 6, .8]}}}
    spec = SimpleNamespace(target_object="bed", instance_hint=None, pose_class="kneel")
    assert np.allclose(cli.seed_for(Profile(), spec, cfg), [5, 6, .8])


class FakeProcess:
    pid = 987654  # Only mocked termination calls ever receive this PID.
    stdout = stderr = None
    def __init__(self, returncode=0, output=("", ""), timeouts=0):
        self.returncode, self.output, self.timeouts = returncode, output, timeouts
        self.communications, self.killed = [], False
    def communicate(self, timeout):
        self.communications.append(timeout)
        if self.timeouts:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired("fixture", timeout, stderr=b"gpu timed out")
        return self.output
    def kill(self):
        self.killed = True
    def wait(self, timeout):
        return self.returncode


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_run_stage_timeout_kills_tree_and_reaps(monkeypatch, platform):
    process = FakeProcess(timeouts=1)
    launches, kills = [], []
    monkeypatch.setattr(cli.sys, "platform", platform)
    monkeypatch.setattr(cli.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    import signal
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    def launch(command, **kw):
        launches.append(kw)
        assert kw["cwd"] == str(cli.REPO)
        return process
    monkeypatch.setattr(cli.subprocess, "Popen", launch)
    def terminate(command, **kw):
        kills.append((command, kw))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.subprocess, "run", terminate)
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: kills.append((pid, sig)), raising=False)
    with pytest.raises(cli.StageError) as caught:
        cli.run_stage(["python", "x.py"], "SAM")
    assert caught.value.diagnostic == "gpu timed out"
    assert "600s" in str(caught.value)
    assert process.communications == [600, 5]
    if platform == "win32":
        assert launches[0]["creationflags"] == 512
        assert kills[0][0] == ["taskkill", "/PID", str(process.pid), "/T", "/F"]
        assert kills[0][1]["timeout"] == 10
    else:
        assert launches[0]["start_new_session"] is True
        import signal
        assert kills == [(process.pid, signal.SIGKILL)]


def test_run_stage_failed_tail_and_redaction(monkeypatch):
    monkeypatch.setenv("FARM_AUTH", "fixture-secret-value")
    lines = "\n".join(f"line {i} fixture-secret-value" for i in range(90))
    process = FakeProcess(returncode=3, output=("", lines))
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: process)
    with pytest.raises(cli.StageError) as caught:
        cli.run_stage(["python"], "render")
    assert caught.value.diagnostic.splitlines() == [s.replace("fixture-secret-value", "[redacted]") for s in lines.splitlines()[-30:]]


def test_run_stage_success_returns_completed_process(monkeypatch):
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: FakeProcess(output=("ok", "warning")))
    result = cli.run_stage(["python"], "fixture")
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.stdout == "ok" and result.stderr == "warning" and result.returncode == 0


def test_run_stage_timeout_reap_is_bounded(monkeypatch):
    process = FakeProcess(timeouts=2)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.os, "killpg", lambda *a: None, raising=False)
    with pytest.raises(cli.StageError, match="timeout"):
        cli.run_stage(["python"], "fixture")
    assert process.killed
    assert process.communications == [600, 5]


def test_blender_stdout_traceback_is_failure(monkeypatch):
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: FakeProcess(
        output=("Traceback (most recent call last):\nbad render", "")))
    with pytest.raises(cli.StageError, match="subprocess failed"):
        cli.run_stage(["blender"], "render")


def test_secret_environment_values_are_redacted(monkeypatch):
    monkeypatch.setenv("FARM_AUTH", "login:secret-fixture")
    result = cli.error_result(cli.StageError("failed", "auth=login:secret-fixture https://user:pass@example.org/x"))
    assert "secret-fixture" not in json.dumps(result)
    assert "user:pass" not in json.dumps(result)


def test_prepare_images_exact_background_and_compact_mask(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(cli.HERE / "plate"))
    shape = (768, 1344)
    plate = np.full((*shape, 3), [20, 40, 60], np.uint8)
    pb = np.full((*shape, 3), [120, 120, 120], np.uint8)
    sil = np.zeros(shape, np.uint8)
    sil[50:70, 60:80] = 255
    for name, array in (("plate.png", plate), ("playblast.png", pb), ("silhouette.png", sil)):
        Image.fromarray(array).save(tmp_path / name)
    cli.prepare_images(tmp_path, tmp_path / "plate.png", "grey", None, .75, 16)
    init = np.asarray(Image.open(tmp_path / "init.png"))
    mask = np.asarray(Image.open(tmp_path / "strength.png"))
    assert np.array_equal(init[sil == 0], plate[sil == 0])
    assert np.array_equal(init[sil > 0], pb[sil > 0])
    assert mask[55, 65] == round(.75 * 255)
    assert mask[0, 0] == 0
    assert (tmp_path / "debug.png").is_file()
    cli.prepare_images(tmp_path, tmp_path / "plate.png", "grey", None, 0., 16)
    assert not np.asarray(Image.open(tmp_path / "strength.png")).any()


def test_h3_raster_resize_keeps_geometry_plate_unchanged(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(cli.HERE / "plate"))
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    source = geometry / "plate_rgb.png"
    Image.new("RGB", (1536, 864), (20, 30, 40)).save(source)
    original = source.read_bytes()
    Image.new("RGB", (1536, 864), (120, 120, 120)).save(tmp_path / "playblast.png")
    sil = np.zeros((864, 1536), np.uint8)
    sil[400:450, 700:750] = 255
    Image.fromarray(sil).save(tmp_path / "silhouette.png")
    cli.prepare_images(tmp_path, source, "grey", None, 1., 16)
    for name in ("playblast.png", "init.png", "silhouette.png", "strength.png", "debug.png"):
        assert Image.open(tmp_path / name).size == (1344, 768)
    assert source.read_bytes() == original
    mask = np.asarray(Image.open(tmp_path / "strength.png"))
    silhouette = np.asarray(Image.open(tmp_path / "silhouette.png")) > 0
    y, x = np.argwhere(silhouette)[0]
    assert mask[y, x - 16] == 255
    assert mask[y, x - 48] == 0


def test_color_proxy_uses_existing_palette(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(cli.HERE / "plate"))
    plate = np.full((12, 12, 3), 30, np.uint8)
    sil = np.zeros((12, 12), np.uint8)
    sil[4:8, 4:8] = 255
    labels = np.zeros_like(plate)
    labels[sil > 0] = [0, 255, 0]  # top transport label
    for name, array in (("plate.png", plate), ("playblast.png", plate + 60),
                        ("silhouette.png", sil), ("labels.png", labels)):
        Image.fromarray(array).save(tmp_path / name)
    body = tmp_path / "body.json"
    cli.save_json(body, {"outfit_colors": {"skin": "#C8A08A", "top": "#FFFFFF"}})
    cli.prepare_images(tmp_path, tmp_path / "plate.png", "color", body, 1., 16)
    assert (tmp_path / "proxy.png").is_file()
    assert np.asarray(Image.open(tmp_path / "init.png"))[350, 600].tolist() == [255, 255, 255]
    assert Image.open(tmp_path / "silhouette.png").size == (1344, 768)
    values = np.unique(np.asarray(Image.open(tmp_path / "silhouette.png")))
    assert set(values) == {0, 255}


def test_real_step9_prompt_generator_without_table(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cli.modules(cfg)
    import pose_round
    monkeypatch.setattr(pose_round, "prompts", lambda: pytest.fail("must not read the 17-pose table"))
    cfg["prompt_context_by_location"] = {"room": {"anchor": "the bed reaches her knees"}}
    fit = {"shot_spec": {"pose_class": "kneel", "target_object": "bed", "secondary_contacts": []}}
    text = cli.prompt_for(Profile(), cfg, fit)
    assert "She kneels" in text
    assert "<Video 1>" in text
    assert "\nThe lighting is preserved unchanged throughout.\n" in text
    assert "\nColors and saturation stay unchanged from the reference image.\n" in text
    assert "studio" not in text.lower()
    assert "locked-off" in text
    with pytest.raises(ValueError, match="scale anchor missing"):
        cli.prompt_for(Profile(), config(tmp_path), fit)


def test_main_error_json_replaces_stale_success(tmp_path):
    req, output = tmp_path / "request.json", tmp_path / "result.json"
    cli.save_json(req, {"config": {"farm_root": str(tmp_path)}, "location_id": "../bad"})
    cli.save_json(output, {"status": "ok", "playblast": "stale.png"})
    assert cli.main(["location", "--request", str(req), "--output", str(output), "--worker"]) == 0
    result = cli.read_json(output)
    assert result["status"] == "error"
    assert "playblast" not in result


def test_main_roundtrip_uses_requested_farm_python(tmp_path):
    profile_dir(tmp_path)
    req, output = tmp_path / "request.json", tmp_path / "result.json"
    cli.save_json(req, {"config": config(tmp_path), "location_id": "room"})
    assert cli.main(["location", "--request", str(req), "--output", str(output)]) == 0
    assert cli.read_json(output)["status"] == "ok"
    assert not list(tmp_path.glob(".pack-*.json"))


def test_arbitrary_pose_pipeline_and_content_addressed_cache(tmp_path, monkeypatch):
    directory = profile_dir(tmp_path)
    # The mocked geometry/renderer still exchange the real filesystem artifacts.
    Image.new("RGB", (128, 128), (20, 30, 40)).save(directory / "plate_rgb.png")
    pose = tmp_path / "arbitrary_pose.png"
    Image.new("RGB", (16, 16), "white").save(pose)
    mannequin = tmp_path / "mannequin.blend"
    mannequin.write_bytes(b"fixture")
    cfg = config(tmp_path)
    cfg.update(blender="blender-fixture", mannequin=str(mannequin))
    events = []
    profile = Profile()
    pf = SimpleNamespace(
        Plate=lambda p: SimpleNamespace(dir=p),
        load_ref=lambda p: {"F": np.array([[0, 1, 2]]), "V": np.zeros((3, 3))},
        fit_input_hash=lambda *a: "hash-fixture",
    )
    monkeypatch.setattr(cli, "modules", lambda c: (pf, SimpleNamespace(LocationProfile=lambda p: profile)))
    def place(ref, plate, spec, seed, height, **kw):
        events.append("place")
        assert spec.pose_class == "kneel" and spec.target_object == "bed"
        assert spec.instance_hint == "bed_2"
        assert spec.secondary_contacts == [{"part": "hand.L", "surface": "wall"}]
        assert np.allclose(seed, [2, 3, .5])
        assert ref["path"].endswith("pose.ref.npz")
        assert ref["body_cache_dir"].endswith("body_geometry")
        assert kw["refine_ctx"] == "existing-refine-context"
        return {"rep": {"status": "partial"}}
    def write(out, result, ref, plate, spec, fit_hash):
        events.append("write")
        out.mkdir(parents=True)
        (out / "body.obj").write_text("fixture")
        cli.save_json(out / "fit.json", {"status": "partial", "shot_spec": spec.to_dict()})
    monkeypatch.setitem(sys.modules, "place", SimpleNamespace(place=place, write=write))
    monkeypatch.setitem(sys.modules, "refine", SimpleNamespace(load_ctx=lambda p: "existing-refine-context"))
    monkeypatch.syspath_prepend(str(cli.HERE / "posev2"))
    monkeypatch.syspath_prepend(str(cli.HERE / "plate"))
    monkeypatch.setattr(cli, "prompt_for", lambda *a: "existing-prompt-fixture")
    def run(command, stage, **kw):
        events.append(stage)
        if stage.startswith("pose MoGe"):
            out = Path(command[command.index("--out") + 1])
            (out / "pose.ref.npz").write_bytes(b"mesh-fixture")
        else:
            assert command[0] == "blender-fixture"
            assert "render_plate_scene.py" in str(command)
            out = Path(command[command.index("--out") + 1]) / "pose"
            out.mkdir(parents=True)
            Image.new("RGB", (128, 128), (100, 100, 100)).save(out / "playblast.png")
            sil = np.zeros((128, 128), np.uint8)
            sil[40:90, 40:90] = 255
            Image.fromarray(sil).save(out / "silhouette.png")
    monkeypatch.setattr(cli, "run_stage", run)
    request = {"location": str(directory / "profile.json"), "pose_ref": str(pose), "pose_class": "kneel",
               "target_object": "bed", "instance_hint": "bed_2", "support_parts": "knee.L,shin.L",
               "secondary_contacts": "hand.L:wall", "height_m": 1.62, "strength": 1., "seed": 55}
    result = cli.do_place(request, cfg)
    assert events == ["pose MoGe + SAM 3D Body", "place", "write", "Blender render"]
    assert result["status"] == "partial"
    assert result["h3_prompt"] == "existing-prompt-fixture"
    for key in ("playblast", "init", "strength", "silhouette", "debug"):
        assert Path(result[key]).is_file()
    assert set(result["timings"]) == {"reference_s", "placement_s", "render_s", "maps_s", "total_s"}
    assert len(list((tmp_path / "cache/pose_ref").iterdir())) == 1
    events.clear()
    # Same bytes, different filename: cache must still hit.
    alias = tmp_path / "renamed.png"
    alias.write_bytes(pose.read_bytes())
    cli.do_place(dict(request, pose_ref=str(alias)), cfg)
    assert events == ["place", "write", "Blender render"]


def test_place_worker_dispatches_refine_to_geometry_python(tmp_path, monkeypatch):
    req, output = tmp_path / "request.json", tmp_path / "result.json"
    cfg = config(tmp_path)
    cfg["geometry_python"] = "geometry-python-fixture"
    cli.save_json(req, {"config": cfg})
    def run(command, stage, **kwargs):
        assert command[0] == "geometry-python-fixture"
        assert "--geometry-worker" in command
        cli.save_json(command[command.index("--output") + 1], {"status": "ok"})
    monkeypatch.setattr(cli, "run_stage", run)
    cli.main(["place", "--request", str(req), "--output", str(output), "--worker"])
    assert cli.read_json(output)["status"] == "ok"


def test_main_malformed_request_is_json_error(tmp_path):
    req, output = tmp_path / "request.json", tmp_path / "result.json"
    req.write_text("not json", encoding="utf-8")
    cli.main(["place", "--request", str(req), "--output", str(output)])
    assert cli.read_json(output)["status"] == "error"
