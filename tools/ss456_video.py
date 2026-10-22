#!/usr/bin/env python3
"""SS-456: видео-петли FullHD по принятым кадрам — движок video_h3_loop@1 (StacyLoop), Саманта
без LoRA Стейси (ARM = ref2va convrot + turbo 8, face_pass off — личность держит ключевой кадр).

    python tools/ss456_video.py run C:/stacy-lora/refmod_inputs/round_456v1.yaml
    python tools/ss456_video.py lint C:/stacy-lora/refmod_inputs/round_456v1.yaml   # только линтер

Промпты — [keyframe completion] по строкам SFW-таблицы «Дом» (Confluence, ss456_meta.json):
фазы 0.6–2.2 с, действия ≤62 % клипа, возврат в старт, отдых ≥2 с; узел StacyPromptRetime сам
подгоняет таймкоды под сетку H3. Ключевой кадр — финал (лицо 1 Мп) раунда round_456f4.
Выход: OUT video (FullHD mp4) → <out>/<round>/<сцена>/…  + jobs.jsonl (время на клип).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import wf_params as WP  # noqa: E402
from ss340_round import Comfy  # noqa: E402

PARAMS = REPO / "workflows/video_h3_loop@1.params.yaml"
FACE_REF = "07_face_front_s001.png"        # рефы лица Саманты (мод characters/samantha)
SEED = 61875092394976
OUT_ROOM = "in <Picture 1>, exactly as in the keyframe"

CAM = ("filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, "
       "with no camera movement, no zoom and no pan")

# сцена → (сек, стартовая поза из кадра, выражение, фазы [текст фаз без «From X to Y»])
SCENES = {
    "C2_bedside": dict(sec=10.0, room="the loft bedroom with the low platform bed",
        open=("she sits barefoot on the edge of the bed, a small vintage dumbbell in her left hand, "
              "polishing it with a white cloth in her right hand"),
        expr="a neutral, relaxed look",
        phases=[
            "she holds the starting pose, polishing the dumbbell and looking down at it",
            "she looks up at the camera and keeps polishing with small quick strokes",
            "she puts the dumbbell and the cloth down on the bed beside her",
            "her right hand slips under her loose hair to the back of her neck and kneads it gently, "
            "while her head turns slowly to the left and to the right",
            "her right hand comes back down and picks up the dumbbell and the cloth, "
            "and she returns to polishing it, looking down at it",
        ]),
    "C3_sofa": dict(sec=8.0, room="the loft living room with the burgundy velvet sofa",
        open=("she sits on the sofa with her right leg tucked up on the seat, her left foot on the floor, "
              "her left forearm resting on her knee, holding a protein bar in her left hand"),
        expr="confidence with a light smirk",
        phases=[
            "she holds the starting pose, looking at the camera with the protein bar in her hand",
            "she raises the protein bar to her mouth and takes a single clean bite",
            "she chews calmly, looking at the camera",
            "she lowers the protein bar back to her knee and stops chewing",
        ]),
    "C4_rack": dict(sec=8.0, room="the home gym corner with the black power rack",
        open="she sits in a deep squat inside the power rack, her hands gripping the side beams",
        expr="a proud smirk",
        phases=[
            "she holds the starting pose, gripping the beams, looking at the camera",
            "she releases the beams and brings both forearms up in a crisp double biceps flex, "
            "her head giving small light nods",
            "she keeps the double biceps flex, smirking at the camera",
            "she lowers her arms and grips the side beams again",
        ]),
    "C5_greenhouse": dict(sec=9.0, room="the indoor greenhouse with wooden shelves of plants",
        open=("she squats by the low shelf, tending a green potted plant with both hands; a spray bottle "
              "stands on the floor beside her"),
        expr="a calm, focused look, her face turned to the plant",
        phases=[
            "she holds the starting pose, turning the pot with both hands and checking the leaves",
            "her right hand picks up the spray bottle from the floor",
            "she sprays the plant twice with quick presses, mist settling on the leaves",
            "she puts the spray bottle back on the floor and returns both hands to the pot",
        ]),
    "L2_mat": dict(sec=10.0, room="the loft bedroom with the exercise mat on the floor",
        open=("she lies on her back on the mat, feet toward the camera, knees bent, feet flat on the mat, "
              "her hands behind her head"),
        expr="a satisfied smile between reps",
        phases=[
            "she holds the starting pose for one breath",
            "she does two crisp sit-ups, shoulder blades lifting off the mat each time",
            "on the next rise she stops halfway, plants both hands on the mat behind her and looks "
            "at the camera, smiling",
            "she holds the raised pose for a moment, still smiling at the camera",
            "she lies back down, laces her hands behind her head and returns to crunches",
        ]),
    "L3_sofa": dict(sec=8.0, room="the loft living room with the burgundy velvet sofa",
        open=("she lies on her side along the sofa, her head propped on her left hand, her right hand "
              "resting on her hip"),
        expr="a relaxed, sleepy look",
        phases=[
            "she holds the starting pose, looking at the camera",
            "a yawn builds on her face, her brows knitting slightly",
            "the back of her right hand comes up to cover her mouth as she yawns fully",
            "her right hand lowers back to her hip and her face returns to the relaxed look",
        ]),
    "S4_bookshelf": dict(sec=9.0, room="the loft living room with the tall dark bookshelves",
        open=("she stands upright at the bookshelf, her arms hanging relaxed at her sides"),
        expr="a thoughtful, curious look",
        phases=[
            "she holds the starting pose, her eyes scanning the book spines",
            "her head turns slowly left and right along the shelves",
            "her right hand rises in a quick, lively reach toward one book and freezes beside it",
            "her right hand lowers back to her side without taking the book",
            "her head resumes scanning the shelves left and right",
        ]),
    "S5_rack": dict(sec=10.0, room="the home gym corner with the black power rack",
        open=("she stands inside the power rack holding a loaded barbell with both hands, the bar at her "
              "chest in the top position of a biceps curl"),
        expr="a slightly tensed focus, breaking into a smile",
        phases=[
            "she holds the starting pose, the bar at her chest",
            "she does two explosive biceps curls, the bar rising briskly to her chest and dropping "
            "to her hips each time",
            "with the bar back at her chest she begins one slow, controlled lowering to her hips, "
            "a smile growing on her face",
            "she curls the bar briskly back to her chest and resumes explosive repetitions",
        ]),
    "S6_greenhouse": dict(sec=9.0, room="the indoor greenhouse with the wooden work desk",
        open=("she stands behind the wooden desk, examining a potted plant under the grow lamp"),
        expr="a soft, tender smile",
        phases=[
            "she holds the starting pose, looking down at the plant",
            "she stops, raises her head and looks at the camera, giving one small nod",
            "her right hand turns the pot a little to the left, then a little to the right",
            "she lowers her eyes back to the plant and keeps examining it",
        ]),
}


def build_prompt(sc: dict, sec: float) -> str:
    """[keyframe completion] петля: 0.00–0.80 удержание старта, фазы ≤2.2 с до ~62 % клипа,
    отдых ≥2 с. Каждое предложение одной строкой — линтер режет секции по строкам."""
    n = len(sc["phases"])
    act = min(sec * 0.62 - 0.8, 2.2 * n)                     # действия до 62 % клипа
    step = act / n
    ts = [0.0, 0.8] + [0.8 + (i + 1) * step for i in range(n)]
    if ts[-1] > sec - 2.0:                                   # отдых ≥ 2 с
        ts[-1] = max(sec - 2.0, ts[-2] + 0.4)
    ph = [f"From {a:.2f} to {b:.2f} seconds, {t}." for a, b, t in zip(ts, ts[1:], sc["phases"])]
    rest = f"From {ts[-1]:.2f} to {sec:.2f} seconds, she stays relaxed in the starting pose, breathing softly and blinking."
    desc = [f"The target video is photorealistic live-action footage with warm indoor light, {CAM}.",
            f"[Shot 1] The shot begins from <Picture 1>, showing <Subject 1> in <Subject 2>: {sc['open']}.",
            *ph, rest,
            f"At {sec:.2f} seconds the shot ends on <Picture 1>: she is back in exactly the starting pose, expression and position, so the last frame matches the first frame.",
            "The video is one continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen.",
            f"Her facial expression is {sc['expr']}; it never goes blank.",
            "Every movement happens at natural real-time speed, crisp and clearly visible, never in slow motion.",
            "Only her head, arms and hands move; the rest of her body and the whole room stay still.",
            "The lighting is preserved unchanged throughout.",
            "Colors and saturation stay unchanged from the reference image."]
    return "\n".join([
        "subject_definitions:",
        f"<Subject 1> is the young woman shown in <Picture 1> and <Picture 2>, with exactly the face, hair, skin tone and body proportions shown there, wearing the same clothes as in <Picture 1>.",
        f"<Subject 2> is {sc['room']}, exactly as shown in <Picture 1>.",
        "<Picture 2> is a close-up identity reference of the face of <Subject 1>.",
        "<Picture 1> is the first frame of [Shot 1] and also its last frame.",
        "",
        "summary:",
        f"[keyframe completion] The target video is one continuous static shot of {sec:.2f} seconds that begins from <Picture 1> and ends on <Picture 1>: <Subject 1> performs a small lively action in <Subject 2> and returns to exactly the starting pose, so the clip loops seamlessly.",
        "",
        "retention_analysis:",
        "<Subject 1> (appears in [Shot 1]): fully_preserved - her face, hair, skin tone, body proportions and clothes are retained.",
        "<Subject 2> (appears in [Shot 1]): fully_preserved - the room, furniture, light and colors are retained.",
        "<Picture 2>: fully_preserved - the facial identity of <Subject 1> is retained in every frame.",
        "",
        "detailed_description:",
        " ".join(desc),
        "",
        "overall_soundscape:",
        "Quiet indoor room tone.",
        "",
        "non_diegetic_music:",
        "N/A",
        ""])


def values(scene: str, sc: dict, kf: str, face: str) -> dict:
    return {"keyframe": kf, "face_ref": face, "prompt": build_prompt(sc, sc["sec"]),
            "face_prompt": "", "seed": SEED, "duration_s": sc["sec"], "loop": True,
            "face_pass": "off", "char_lora": 1.0,
            "arm": {"model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                    "lora": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lint", "run"])
    ap.add_argument("config")
    ap.add_argument("--only", help="только эти сцены, через запятую")
    ap.add_argument("--url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    a = ap.parse_args()
    from farm.prompts.lint import errors, lint
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    p, api = WP.load(str(PARAMS))
    frames = {f.stem: f for f in Path(cfg["frames_dir"]).glob("*.png")}
    face_dir = Path(cfg["face_ref_dir"])
    only = set(a.only.split(",")) if a.only else None
    bad = 0
    for scene, sc in SCENES.items():
        if only and scene not in only:
            continue
        t = build_prompt(sc, sc["sec"])
        errs = [i.msg for i in errors(lint(t, "video", {"mode": "loop", "end_s": sc["sec"]}))]
        print(scene, f"{sc['sec']:.0f} с", "линтер:", errs or "0")
        bad += bool(errs)
    if a.cmd == "lint" or bad:
        return 1 if bad else 0
    cli = Comfy(a.url)
    out = Path(cfg["out"]) / cfg["round"]
    out.mkdir(parents=True, exist_ok=True)
    log = out / "jobs.jsonl"
    done = {json.loads(l)["scene"] for l in log.read_text(encoding="utf-8").splitlines()} if log.exists() else set()
    face_up = cli.upload(face_dir / FACE_REF, f"{cfg['round']}/face")
    for scene, sc in SCENES.items():
        if (only and scene not in only) or scene in done:
            continue
        kf = frames.get(scene) or frames.get(f"{scene}_plate")
        assert kf, f"нет ключевого кадра {scene} в {cfg['frames_dir']}"
        t0 = time.time()
        vals = values(scene, sc, cli.upload(kf, f"{cfg['round']}/kf"), face_up)
        g = WP.apply(p, api, vals)
        for k, v in g.items():
            if v["class_type"] == "VHS_VideoCombine" and v["inputs"].get("filename_prefix", "").endswith("Stacy"):
                v["inputs"]["filename_prefix"] = f"{cfg['round']}/{scene}"
            elif v["class_type"] == "VHS_VideoCombine":
                v["inputs"]["filename_prefix"] = f"{cfg['round']}/{scene}_raw"
            elif v["class_type"] == "SaveImage":
                v["inputs"]["filename_prefix"] = f"{cfg['round']}/{scene}_last"
        h = cli.wait(cli.submit(g), 2400)
        vid = next(f for o in h["outputs"].values() for f in o.get("gifs", []))
        rep = next((t for o in h["outputs"].values() for t in o.get("text", [])), "")
        rec = {"scene": scene, "sec": sc["sec"], "video": f"{vid['subfolder']}/{vid['filename']}",
               "wall_s": round(time.time() - t0), "report": rep[:600]}
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{scene}: {rec['wall_s']} с → {rec['video']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
