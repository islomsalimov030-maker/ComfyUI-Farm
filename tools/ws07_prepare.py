#!/usr/bin/env python3
"""WS-07 офлайн-подготовка (ПК): пакет входов для серий пода → <out>/kit (рефы, скелеты, рамки, промпты, targets.json).

    .venv/Scripts/python tools/ws07_prepare.py [--src C:/stacy-lora] [--out C:/stacy-lora/_claude/ws07]

4 позы серий (бриф): Sitting-8, Standing-2, Sitting-2 и Lying-4 (Lying-7 нет среди SFW-строк листа — WS-03,
решение команды; берём Lying-4 — лежит, есть фото-реф позы). Для каждой позы:
  - плита, лист персонажа, одиночный кроп лица, одежда (если у строки есть наряд), фото-реф позы (если есть);
  - эталон масштаба = одобренный кадр v2 (frames/sfw_final_frames) — плиты и кадры v2 совпадают по раскладке;
    из него MediaPipe Pose → рамка тела (target_box) → скелет нужного размера (Picture 4, серия 2b) и зелёная
    рамка на копии плиты (серия 2c); широкая плита с полями 20 % (серия 2d);
  - промпты кадра по вариантам (база — golden WS-04 `tests/golden/prompts/<ID>.frame.txt`, правила h3-prompt-writer).
Медиа в репозиторий не кладём: kit копируется на под (`scp -r`), см. docs/experiments/WS-07_plan.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import ws07_lib as L  # noqa: E402
from farm.prompts.lint import errors, lint  # noqa: E402

# поза → строка листа и файлы (относительно --src); outfit None — без Picture 3 (как кадры v2: одежда из листа)
POSES = {
    "Sitting-8": {"row": "SIT-08", "plate": "_claude/up2/pl_stool.jpg", "outfit": "wardrobe/default.png",
                  "pose": "_claude/up2/pose_sit8.jpg", "frame_v2": "frames/sfw_final_frames/Sitting-8.png"},
    "Standing-2": {"row": "STD-02", "plate": "_claude/up/pl_window.jpg", "outfit": None,
                   "pose": None, "frame_v2": "frames/sfw_final_frames/Standing-2.png"},
    "Sitting-2": {"row": "SIT-02", "plate": "_claude/up/pl_sofa.jpg", "outfit": None,
                  "pose": None, "frame_v2": "frames/sfw_final_frames/Sitting-2.png"},
    "Lying-4": {"row": "LIE-04", "plate": "_claude/up/pl_bed.jpg", "outfit": None,
                "pose": "_claude/up2/pose_lie4.jpg", "frame_v2": "frames/sfw_final_frames/Lying-4.png"},
}
SHEET = "_claude/up/stacy_sheet_body.jpg"           # лист, на котором делались кадры v2
ID_CROP = "character_sheets/ref_single_0003.png"    # одиночный кроп лица/плеч (серия 2e)
FACE_REF = "h3lab/assets/face_ref.png"              # эталон лица видео (Picture 2 движка видео)
FACE_DIR = "dataset_face_v1_ready"                  # эталон для face_id (центроид AuraFace)
VARIANTS = ["fixe", "skeleton", "green", "wide", "idcrop"]
CLIPS = ["h3lab/newwf/Sitting-7_raw_00001.mp4", "h3lab/newwf/Standing-12_raw_00001.mp4"]


def _save(im, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(path, quality=95) if path.suffix == ".jpg" else im.save(path)
    return path.name


def main() -> int:
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="C:/stacy-lora")
    ap.add_argument("--out", default="C:/stacy-lora/_claude/ws07")
    a = ap.parse_args()
    src, kit = Path(a.src), Path(a.out) / "kit"
    refs = kit / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / SHEET, refs / "sheet.jpg")
    shutil.copy2(src / ID_CROP, refs / "idcrop.png")
    shutil.copy2(src / FACE_REF, refs / "face_ref.png")
    face_dir = kit / "face_ref_dir"
    face_dir.mkdir(exist_ok=True)
    for f in sorted((src / FACE_DIR).glob("*.png")):
        if not (face_dir / f.name).exists():
            shutil.copy2(f, face_dir / f.name)

    fv2 = kit / "frames_v2"                       # серия 6 (VLM) — 26 кадров v2
    fv2.mkdir(exist_ok=True)
    for f in sorted((src / "frames/sfw_final_frames").glob("*.png")):
        if f.stem != "Lying-3" and not (fv2 / f.name).exists():
            shutil.copy2(f, fv2 / f.name)

    golden = REPO / "tests" / "golden" / "prompts"
    targets, report = {}, []
    for pose, c in POSES.items():
        d = refs / pose
        plate = Image.open(src / c["plate"]).convert("RGB")
        w, h = plate.size
        files = {"plate": _save(plate, d / "plate.png")}
        if c["outfit"]:
            files["outfit"] = _save(Image.open(src / c["outfit"]), d / "outfit.png")
        if c["pose"]:
            files["pose_photo"] = _save(Image.open(src / c["pose"]), d / "pose_photo.jpg")
        v2 = L.read_bgr(src / c["frame_v2"])
        p = L.pose33(v2)
        if p is None:
            print(f"!! {pose}: MediaPipe не нашёл позу в кадре v2", file=sys.stderr)
            return 1
        vh, vw = v2.shape[:2]
        box = L.body_box(p, vw / vh)
        files["skeleton"] = _save(L.draw_skeleton(p, (w, h)), d / "skeleton.png")
        files["plate_green"] = _save(L.draw_green_box(plate, box), d / "plate_green.png")
        wide, crop = L.widen_plate(plate, 0.2)
        files["plate_wide"] = _save(wide, d / "plate_wide.png")
        _save(Image.open(src / c["frame_v2"]), d / "frame_v2.png")

        base = (golden / f"{c['row']}.frame.txt").read_text(encoding="utf-8")
        prompts = {}
        for v in VARIANTS:
            t = L.prompt_variant(base, v)
            # скелет занимает слот позы: фото-реф позы в этом варианте не подаём
            labels = {f"<Picture {n}>" for n in (1, 2)} | ({"<Picture 3>"} if c["outfit"] else set())
            if v == "skeleton" or c["pose"]:
                labels.add("<Picture 4>")
            iss = lint(t, "frame", {"connected": labels})
            errs = [(i.code, i.msg) for i in errors(iss)]
            report.append({"pose": pose, "variant": v, "words": len(t.split()), "errors": errs,
                           "warn": sorted({i.code for i in iss if i.severity != "error"})})
            prompts[v] = t
            (d / f"prompt_{v}.txt").write_text(t, encoding="utf-8")
        targets[pose] = {"row": c["row"], "size": [w, h], "target_box": [round(x, 4) for x in box],
                         "frame_v2_size": [vw, vh], "wide_crop": [round(x, 4) for x in crop],
                         "files": files, "has_outfit": bool(c["outfit"]), "has_pose": bool(c["pose"])}
        print(f"{pose:11s} плита {w}x{h}  рамка тела {[round(x, 2) for x in box]}  "
              f"диагональ {L.box_diag(box):.3f}")
    # серия 5 (апскейл): raw 1344x768 без апскейла, первые 5 с (h3lab/newwf, движок video_h3_loop)
    clips = kit / "clips"
    clips.mkdir(exist_ok=True)
    for c in CLIPS:
        dst = clips / Path(c).name.replace("_raw_00001", "")
        if not dst.exists() and (src / c).is_file():
            import subprocess
            subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-i", str(src / c), "-t", "5",
                            "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(dst)], check=True)
    (kit / "targets.json").write_text(json.dumps(targets, ensure_ascii=False, indent=1), encoding="utf-8")
    (kit / "prompts_lint.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [r for r in report if r["errors"]]
    for r in bad:
        print("ЛИНТЕР:", r["pose"], r["variant"], r["errors"], file=sys.stderr)
    print(f"kit: {kit}  промптов {len(report)}, с ошибками линтера {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
