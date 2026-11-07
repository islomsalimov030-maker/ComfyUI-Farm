"""Pose round 07.11c driver: render the pose files of a method into the plates, summarise metrics, build sheets.

  python pose_round.py render --method A            # $FARM_ROOT/pose_round/<M>/<id>.json -> input/pose_<M>_m<N>_s2
  python pose_round.py summary --methods A,B,C      # pose_metrics.json -> $FARM_ROOT/pose_round/summary.json
  python pose_round.py sheet --methods A,B,C --out sheet.jpg [--scores scores.json]
Playblasts: neutral mannequin ($FARM_MANNEQUIN), photo emission 2.5, M3 with the window gobo (spec light).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import farm_env  # noqa: E402

BASE = {1: "scenes_mainroom1_r4.json", 2: "scenes_mainroom2.json", 3: "scenes_mainroom3.json"}
PLATE = "C:/work/scene3d_work/plate/mainroom{}"
ROUND = farm_env.ROOT / "pose_round"
ROWS = set()


def prompts():
    return json.loads((HERE / "pose_round_prompts.json").read_text(encoding="utf-8"))["prompts"]


def loc_dir(method, n):
    return farm_env.COMFY / "input" / f"pose_{method}_m{n}_s2"


def render(method, only=None):
    for n in (1, 2, 3):
        base = json.loads((HERE / BASE[n]).read_text(encoding="utf-8"))
        scenes = []
        for p in prompts():
            if p["loc"] != n or (only and p["id"] not in only):
                continue
            f = ROUND / method / f"{p['id']}.json"
            if not f.is_file():
                continue
            s = {"name": p["id"], "pose_file": str(f), "yaw": "camera", "yaw_add": p.get("yaw_add", 0),
                 "action": p["action"], "anchor": p["anchor"], "contacts": p["contacts"]}
            s["seat" if "seat" in p else "feet"] = p.get("seat") or p.get("feet")
            scenes.append(s)
        if not scenes:
            continue
        base["scenes"] = scenes
        spec = ROUND / method / f"spec_m{n}.json"
        spec.write_text(json.dumps(base, indent=1), encoding="utf-8")
        out = loc_dir(method, n)
        (out / "_plate").mkdir(parents=True, exist_ok=True)
        for fn in ("plate_rgb.png", "plate.json"):
            (out / "_plate" / fn).write_bytes((Path(PLATE.format(n)) / fn).read_bytes())
        cmd = farm_env.bpy_cmd(HERE / "render_plate_scene.py") + [
            "--plate", PLATE.format(n), "--spec", str(spec), "--mannequin", os.environ["FARM_MANNEQUIN"],
            "--out", str(out), "--photo-emission", "2.5", "--samples", "48"]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        for line in r.stdout.splitlines():
            if line.startswith(("[plate]", "[pose]", "[limits]", "[posefile]")) or "Error" in line or "Traceback" in line:
                print(f"[{method} m{n}] {line}", flush=True)
        if r.returncode:
            print(r.stderr[-2000:])


def summary(methods):
    rows = []
    for m in methods:
        for p in prompts():
            d = loc_dir(m, p["loc"]) / p["id"]
            f = d / "pose_metrics.json"
            if not f.is_file():
                rows.append({"method": m, "id": p["id"], "rendered": False})
                continue
            x = json.loads(f.read_text())
            gaps = [c["gap_cm"] for c in x.get("contacts", [])]
            rows.append({"method": m, "id": p["id"], "rendered": True,
                         "violations_before": len(x["violations_before"]), "clamped": len(x["clamp"]),
                         "violations_after": len(x["violations_after"]), "bone_dev_mm": x["bones"]["max_dev_mm"],
                         "contacts": {c["part"]: c["gap_cm"] for c in x.get("contacts", [])},
                         "contacts_ok": sum(abs(g) <= 2 for g in gaps), "contacts_n": len(gaps),
                         "max_abs_gap_cm": max((abs(g) for g in gaps), default=None),
                         "max_penetration_cm": x.get("max_penetration_cm")})
    (ROUND / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in rows:
        if r["rendered"]:
            print(f"{r['method']} {r['id']:24} viol {r['violations_before']}->{r['violations_after']} "
                  f"bones {r['bone_dev_mm']}mm contacts {r['contacts_ok']}/{r['contacts_n']} "
                  f"max|gap| {r['max_abs_gap_cm']} pen {r['max_penetration_cm']}")
        else:
            print(f"{r['method']} {r['id']:24} NOT RENDERED")
    return rows


def sheet(methods, out, scores=None):
    from PIL import Image, ImageDraw
    sc = json.loads(Path(scores).read_text(encoding="utf-8")) if scores else {}
    summ = {(r["method"], r["id"]): r for r in json.loads((ROUND / "summary.json").read_text())} \
        if (ROUND / "summary.json").is_file() else {}
    TW, TH = 360, 420
    ps = [p for p in prompts() if not ROWS or p["id"] in ROWS]
    S = Image.new("RGB", (200 + TW * len(methods), 24 + TH * len(ps)), (18, 18, 18))
    d = ImageDraw.Draw(S)
    for j, m in enumerate(methods):
        d.text((200 + j * TW + 6, 6), f"method {m}", fill=(255, 255, 0))
    for i, p in enumerate(ps):
        y = 24 + i * TH
        d.text((6, y + 6), p["id"], fill=(255, 255, 255))
        for j, m in enumerate(methods):
            x = 200 + j * TW
            f = loc_dir(m, p["loc"]) / p["id"] / "playblast.png"
            if not f.is_file():
                d.text((x + 6, y + 6), "no pose", fill=(255, 80, 80))
                continue
            im = Image.open(f).convert("RGB")
            mk = Image.open(f.parent / "mask.png").convert("L")
            if sum(mk.histogram()[128:]) > mk.width * mk.height / 2:
                mk = mk.point(lambda q: 255 - q)
            bb = mk.point(lambda q: 255 if q > 127 else 0).getbbox() or (0, 0, im.width, im.height)
            pad = int(0.25 * max(bb[2] - bb[0], bb[3] - bb[1]))
            c = im.crop((max(0, bb[0] - pad), max(0, bb[1] - pad), min(im.width, bb[2] + pad), min(im.height, bb[3] + pad)))
            c.thumbnail((TW - 4, TH - 44))
            S.paste(c, (x + 2, y + 2))
            r = summ.get((m, p["id"]), {})
            s = sc.get(f"{m}/{p['id']}")
            txt = (f"prompt {s if s is not None else '?'}/2  contacts {r.get('contacts_ok', '?')}/{r.get('contacts_n', '?')}"
                   f"  |gap|max {r.get('max_abs_gap_cm')}")
            txt2 = f"limits {r.get('violations_before', '?')}->{r.get('violations_after', '?')}  pen {r.get('max_penetration_cm')}  bones {r.get('bone_dev_mm')}mm"
            d.rectangle((x, y + TH - 40, x + TW - 2, y + TH - 2), fill=(0, 0, 0))
            d.text((x + 4, y + TH - 38), txt, fill=(255, 255, 255))
            d.text((x + 4, y + TH - 22), txt2, fill=(200, 200, 200))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    S.save(out, quality=88)
    print(out, S.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["render", "summary", "sheet"])
    ap.add_argument("--method", default="A")
    ap.add_argument("--methods", default="A,B,C")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--scores", default="")
    ap.add_argument("--rows", default="", help="comma list of prompt ids for the sheet")
    a = ap.parse_args()
    ROWS.update(x for x in a.rows.split(",") if x)
    if a.mode == "render":
        render(a.method, set(a.only.split(",")) if a.only else None)
    elif a.mode == "summary":
        summary(a.methods.split(","))
    else:
        sheet(a.methods.split(","), a.out, a.scores or None)


if __name__ == "__main__":
    main()
