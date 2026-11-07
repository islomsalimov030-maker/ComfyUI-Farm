"""Постановка v2 — драйвер раунда (ПК): посадка (pose_fit) → плейбласт (render_plate_scene, body_obj) → сводка.
  python tools/scene3d/posev2/run_v2.py collect --confluence <папка рефов Confluence> --out <папка> 
  python tools/scene3d/posev2/run_v2.py fit --refs <dir с *.ref.npz> [--only id1,id2]
  python tools/scene3d/posev2/run_v2.py render [--only ...]
  python tools/scene3d/posev2/run_v2.py summary
  python tools/scene3d/posev2/run_v2.py sheet --refs-img <папка рефов> --refs <папка npz> --out sheet.jpg
Файлы: $FARM_ROOT/pose_round/V2/<id>/{body.obj, fit.json, overlay.jpg}, spec_m<N>.json;
плейбласты: $FARM_COMFY/input/pose_V2_m<N>_s2/<id>/ — дальше pose_h3.py --picks V2/<id> как для A/B/C.
Контакты: из референса (MoGe-сцена); если в рефе их не нашлось — из таблицы промптов (back→torso, elbow→forearm).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "plate"))
import farm_env  # noqa: E402
import pose_round as pr  # noqa: E402
import pose_fit as pf  # noqa: E402

M = "V2"
PART = {"back": "torso", "elbow.L": "forearm.L", "elbow.R": "forearm.R"}


def table_contacts(p):
    return [f"{PART.get(c['part'], c['part'])}:{'v' if c.get('surface') == 'wall' else 'h'}:{c.get('surface', 's')}"
            for c in p.get("contacts", [])]


def do_fit(refs, only):
    root = pr.ROUND / M
    rows = []
    for p in pr.prompts():
        if only and p["id"] not in only:
            continue
        rf = Path(refs) / f"{p['id']}.ref.npz"
        if not rf.is_file():
            continue
        ref = pf.load_ref(rf)
        plate = pf.Plate(pr.PLATE.format(p["loc"]))
        target = tuple(p.get("seat") or p.get("feet"))
        X, rep, reg = pf.fit(ref, plate, target, 1.62, None, log=lambda m: print(f"[{p['id']}] {m}", flush=True))
        src = "ref"
        if rep["status"] != "ok" and p.get("contacts"):     # таблица — только если реф не посадил, и только если лучше
            X2, rep2, reg2 = pf.fit(ref, plate, target, 1.62, table_contacts(p),
                                    log=lambda m: print(f"[{p['id']}] (таблица) {m}", flush=True))
            score = lambda r: (r["status"] != "ok", r["support_gap_cm"] + r["max_penetration_cm"])  # noqa: E731
            if score(rep2) < score(rep):
                X, rep, reg, src = X2, rep2, reg2, "table"
        rep["contacts_from"] = src
        pf.write_outputs(root / p["id"], X, ref["F"], rep, plate, reg, rep["contacts"].keys())
        rows.append(p["id"])
    print(f"[v2] fit: {len(rows)} поз", flush=True)


def do_render(only):
    root = pr.ROUND / M
    for n in (1, 2, 3):
        base = json.loads((pr.HERE / pr.BASE[n]).read_text(encoding="utf-8"))
        scenes = []
        for p in pr.prompts():
            if p["loc"] != n or (only and p["id"] not in only) or not (root / p["id"] / "body.obj").is_file():
                continue
            s = {"name": p["id"], "body_obj": str(root / p["id"] / "body.obj"), "action": p["action"],
                 "anchor": p["anchor"], "contacts": p["contacts"], "yaw": "camera"}
            s["seat" if "seat" in p else "feet"] = p.get("seat") or p.get("feet")
            scenes.append(s)
        if not scenes:
            continue
        base["scenes"] = scenes
        spec = root / f"spec_m{n}.json"
        spec.write_text(json.dumps(base, indent=1), encoding="utf-8")
        out = pr.loc_dir(M, n)
        (out / "_plate").mkdir(parents=True, exist_ok=True)
        for fn in ("plate_rgb.png", "plate.json"):
            (out / "_plate" / fn).write_bytes((Path(pr.PLATE.format(n)) / fn).read_bytes())
        cmd = farm_env.bpy_cmd(pr.HERE / "render_plate_scene.py") + [
            "--plate", pr.PLATE.format(n), "--spec", str(spec), "--mannequin", os.environ["FARM_MANNEQUIN"],
            "--out", str(out), "--photo-emission", "2.5", "--samples", "48"]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        for line in r.stdout.splitlines():
            if line.startswith(("[plate]",)) or "Error" in line or "Traceback" in line:
                print(f"[V2 m{n}] {line}", flush=True)
        if r.returncode:
            print(r.stderr[-2000:])


def do_summary():
    root = pr.ROUND / M
    rows = []
    for p in pr.prompts():
        f = root / p["id"] / "fit.json"
        if not f.is_file():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"id": p["id"], **{k: r[k] for k in ("status", "support", "support_gap_cm", "max_penetration_cm",
                                                         "pen_verts_over_2cm", "balance_ok", "contacts_from",
                                                         "angle_to_camera_deg", "camera_elevation_deg", "ref_plan")},
                     "contacts": {k: v["gap_cm"] for k, v in r["contacts"].items()},
                     "refine": {k: r["refine"][k] for k in ("max_joint_delta_deg", "n_joints_over_10deg", "before")}
                     if "refine" in r else None})
        x = rows[-1]
        print(f"{x['id']:28} {x['status']:13} опора {x['support']} {x['support_gap_cm']:+.1f} см, "
              f"прон. {x['max_penetration_cm']:.1f} см ({x['pen_verts_over_2cm']} в.), равн. {x['balance_ok']}, "
              f"ракурс {x['angle_to_camera_deg']}, камера {x['camera_elevation_deg']}, конт. {x['contacts_from']}"
              + (f", refine ≤{x['refine']['max_joint_delta_deg']}° (было {x['refine']['before']['status']})"
                 if x["refine"] else ""))
    (root / "summary.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


def do_collect(conf, out):
    """Рефы раунда в одну папку <id>.png: реф Confluence, иначе картинка Z-Image, из которой считалась поза B."""
    import shutil
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    got = {}
    for p in pr.prompts():
        c = Path(conf) / f"{p['id']}.png" if conf else None
        if c is not None and c.is_file():
            shutil.copy(c, out / f"{p['id']}.png")
            got[p["id"]] = "confluence"
            continue
        b = pr.ROUND / "B" / f"{p['id']}.json"
        if b.is_file():
            src = Path(json.loads(b.read_text(encoding="utf-8")).get("source_image", ""))
            if src.is_file():
                shutil.copy(src, out / f"{p['id']}.png")
                got[p["id"]] = "B_zimage"
    print(f"[v2] collect: {len(got)} рефов -> {out}: {got}", flush=True)


def font_ru(size=13):
    """Шрифт с кириллицей (по умолчанию PIL рисует квадраты)."""
    from PIL import ImageFont
    for f in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def do_sheet(refs_img, refs, out):
    """Строка = поза: реф | реф с телом (ref_check) | посадка поверх фото (overlay) | плейбласт; подпись — fit.json."""
    from PIL import Image, ImageDraw
    font = font_ru()
    root = pr.ROUND / M
    ids = [p for p in pr.prompts() if (root / p["id"] / "fit.json").is_file()]
    TW, TH = 330, 300
    S = Image.new("RGB", (4 * TW, len(ids) * (TH + 34)), (18, 18, 18))
    d = ImageDraw.Draw(S)
    for i, p in enumerate(ids):
        y = i * (TH + 34)
        r = json.loads((root / p["id"] / "fit.json").read_text(encoding="utf-8"))
        files = [Path(refs_img) / f"{p['id']}.png", Path(refs) / f"{p['id']}.ref_check.jpg",
                 root / p["id"] / "overlay.jpg", pr.loc_dir(M, p["loc"]) / p["id"] / "playblast.png"]
        for j, f in enumerate(files):
            if f.is_file():
                im = Image.open(f).convert("RGB")
                if j >= 2:                                              # кроп вокруг тела по маске/overlay
                    mk = pr.loc_dir(M, p["loc"]) / p["id"] / "mask.png"
                    if mk.is_file():
                        m = Image.open(mk).convert("L").resize(im.size)
                        if sum(m.histogram()[128:]) > m.width * m.height / 2:
                            m = m.point(lambda q: 255 - q)
                        bb = m.point(lambda q: 255 if q > 127 else 0).getbbox()
                        if bb:
                            pad = int(0.3 * max(bb[2] - bb[0], bb[3] - bb[1]))
                            im = im.crop((max(0, bb[0] - pad), max(0, bb[1] - pad),
                                          min(im.width, bb[2] + pad), min(im.height, bb[3] + pad)))
                im.thumbnail((TW - 6, TH - 6))
                S.paste(im, (j * TW + 3, y + 3))
        txt = (f"{p['id']}  {r['status']}  опора {r['support']} {r['support_gap_cm']:+.1f}см  прон {r['max_penetration_cm']}см"
               f"  равн {r['balance_ok']}  ракурс {r['angle_to_camera_deg']}  камера {r['camera_elevation_deg']}"
               f"  конт {r.get('contacts_from')}")
        d.text((6, y + TH + 4), txt, fill=(255, 255, 255), font=font)
    S.save(out, quality=88)
    print(f"[v2] sheet {out}: {len(ids)} строк", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["collect", "fit", "render", "summary", "sheet"])
    ap.add_argument("--refs")
    ap.add_argument("--refs-img")
    ap.add_argument("--confluence")
    ap.add_argument("--out")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    {"collect": lambda: do_collect(a.confluence, a.out), "fit": lambda: do_fit(a.refs, only),
     "render": lambda: do_render(only), "summary": do_summary,
     "sheet": lambda: do_sheet(a.refs_img, a.refs, a.out)}[a.cmd]()
