"""3D-A 08.11a п.11а: дамп H3-задач шага 7 по /history сервера (реальный граф, а не реконструкция).
  python tools/scene3d/plate/h3_dump.py V2_Q03_bed_kneel_full_s101,... --out docs/coord/h3_dump_0711f.md
Для каждой задачи: что подано в MiniMaxH3ReferenceToVideo под какими метками (ref_image_i -> <Picture i+1>,
ref_video_i -> <Video i+1>), полный промпт, retention_analysis, seed, steps, сэмплер, разрешение, линтер (stage=frame).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import farm_env  # noqa: E402
import quality_round as qr  # noqa: E402


def trace(g, ref, depth=0):
    """цепочка до LoadImage: 'LoadImage(x) <- ImageScale <- RepeatImageBatch'."""
    nid = ref[0]
    n = g[nid]
    ct = n["class_type"]
    if ct == "LoadImage":
        return f"LoadImage `{n['inputs']['image']}`"
    for k in ("image", "images", "frames"):
        v = n["inputs"].get(k)
        if isinstance(v, list) and depth < 6:
            return f"{trace(g, v, depth + 1)} → {ct}"
    return ct


def lint_text(text, connected):
    from farm.prompts.lint import lint
    iss = lint(text, "frame", {"connected": connected})
    return [f"{i.severity}:{i.code}" for i in iss]


def dump_one(name, pid):
    h = qr.api(f"/history/{pid}")[pid]
    g = h["prompt"][2]
    r2v = next(k for k, n in g.items() if n["class_type"] == "MiniMaxH3ReferenceToVideo")
    ri = g[r2v]["inputs"]
    feeds, connected = [], set()
    for k, v in sorted(ri.items()):
        if k.startswith("ref_images.ref_image_"):
            lab = f"<Picture {int(k.rsplit('_', 1)[1]) + 1}>"
        elif k.startswith("ref_videos.ref_video_"):
            lab = f"<Video {int(k.rsplit('_', 1)[1]) + 1}>"
        else:
            continue
        connected.add(lab)
        feeds.append(f"- {lab} ← `{k}`: {trace(g, v)}")
    text = ri["prompt"]
    ret = text.split("retention_analysis:")[1].split("detailed_description:")[0].strip()
    smp = [(k, n) for k, n in g.items() if n["class_type"] in ("KSamplerAdvanced", "KSampler", "SamplerCustomAdvanced")]
    smp_txt = "; ".join(
        f"`{k}` {n['class_type']} " + ", ".join(f"{a}={n['inputs'][a]}" for a in
                                                 ("noise_seed", "seed", "steps", "cfg", "sampler_name", "scheduler",
                                                  "start_at_step", "denoise") if a in n["inputs"])
        for k, n in smp)
    other = sorted({n["class_type"] for n in g.values()})
    lint = lint_text(text, connected)
    outs = []
    for nid, o in (h.get("outputs") or {}).items():
        for im in o.get("images", []):
            outs.append(f"{im.get('subfolder')}/{im['filename']}")
    return "\n".join([
        f"## {name}", f"prompt_id `{pid}`; разрешение {ri['width']}×{ri['height']}, length {ri['length']}, "
                      f"ref_image_size {ri.get('ref_image_size')}",
        "Подано:", *feeds, f"Сэмплер: {smp_txt}",
        f"Модель/шифт: " + "; ".join(f"`{k}` {n['class_type']} " + json.dumps(
            {a: b for a, b in n["inputs"].items() if not isinstance(b, list)}, ensure_ascii=False)
            for k, n in g.items() if n["class_type"] in ("DiffusionModelLoaderKJ", "MiniMaxH3SigmaShift",
                                                          "MiniMaxH3RefModsLoader")),
        f"Узлы: {', '.join(other)}", f"Выходы: {', '.join(outs)}",
        f"Линтер (stage=frame, connected={sorted(connected)}): {lint or 'чисто'}",
        "retention_analysis:", "```", ret, "```", "Промпт целиком:", "```", text, "```", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = json.loads((farm_env.ROOT / "pose_round" / "h3_runs.json").read_text())
    parts = ["# H3 дамп шага 7 (07.11f) — по /history сервера студии", ""]
    for name in a.names.split(","):
        r = [x for x in runs if x["name"] == name and x["status"] == "success"][-1]
        parts.append(dump_one(name, r["prompt_id"]))
    Path(a.out).write_text("\n".join(parts), encoding="utf-8")
    print("dump ->", a.out)


if __name__ == "__main__":
    main()
