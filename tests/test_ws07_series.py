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
    g = G.refmod_extract_graph(["a.png", "b.png", "c.png"], mod="ss340/face_n3", masks=True, max_tokens=4096)
    ex = [v for v in g.values() if v["class_type"] == "MiniMaxH3RefModExtract"][0]["inputs"]
    assert (ex["subfolder"], ex["name"], ex["budget_policy"]) == ("ss340", "face_n3", "error")
    assert "mask_list" in ex and not G.check_links(g)
    mb = [v for v in g.values() if v["class_type"] == "BatchMasksNode"][0]["inputs"]
    assert [k for k in mb] == ["masks.mask0", "masks.mask1", "masks.mask2"]


def test_frame_refmod_slots():
    arm = {"model": "m.safetensors", "loras": []}
    g = G.frame_graph("A_refmod", {"plate": "p.png", "sheet": "s.png"}, "<Picture 1> <Picture 2>", 1, arm,
                      refmods=[{"mod": "ss340/a", "copies": 2}, {"mod": "ss340/b", "strength": 0.5}], retention=0.7)
    ld = [v for v in g.values() if v["class_type"] == "MiniMaxH3RefModsLoader"][0]["inputs"]
    ap = [v for v in g.values() if v["class_type"] == "MiniMaxH3RefModApply"][0]["inputs"]
    assert (ld["mod_1"], ld["copies_1"], ld["mod_2"], ld["strength_2"], ld["mod_3"]) == \
        ("ss340/a", 2, "ss340/b", 0.5, "(none)")
    assert ap["retention"] == 0.7 and not G.check_links(g)


def test_fill_defaults_from_object_info():
    g = {"1": {"class_type": "X", "inputs": {"a": 1}, "_meta": {"title": "x"}}}
    info = {"X": {"input": {"required": {"a": ["INT", {"default": 0}], "b": [["p", "q"], {}],
                                         "c": ["FLOAT", {"default": 0.5}], "d": ["IMAGE"]}}}}
    log = G.fill_defaults(g, info)
    assert g["1"]["inputs"] == {"a": 1, "b": "p", "c": 0.5} and len(log) == 2
