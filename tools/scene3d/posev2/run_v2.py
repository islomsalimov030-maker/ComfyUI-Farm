"""Постановка v2 — драйвер раунда (ПК): посадка (pose_fit) → плейбласт (render_plate_scene, body_obj) → сводка.
  python tools/scene3d/posev2/run_v2.py collect --confluence <папка рефов Confluence> --out <папка> 
  python tools/scene3d/posev2/run_v2.py fit --refs <dir с *.ref.npz> [--only id1,id2]
  python tools/scene3d/posev2/run_v2.py render [--only ...]
  python tools/scene3d/posev2/run_v2.py summary
  v2.4 (07.11f): profiles → черновики профилей локаций (height_proxy); place --refs … --comfy C:/work/ComfyUI-next
  (посадка по shot_spec: Р1–Р5, refine внутри); sheet24 --refs-img … --refs … --out poses_v24.jpg (tools/rowsheet.py)
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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "plate"))
import farm_env  # noqa: E402
import pose_round as pr  # noqa: E402
import pose_fit as pf  # noqa: E402

M = "V2"
CHAR_ID = "stacy"
DEFAULT_HEIGHT = 1.62          # рост манекена Стейси (м), пока нет characters/stacy/body.json
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
        seat = "seat" in p
        X, rep, reg = pf.fit(ref, plate, target, 1.62, None, log=lambda m: print(f"[{p['id']}] {m}", flush=True),
                             seat=seat)
        src = "ref"
        if rep["status"] != "ok" and p.get("contacts"):     # таблица — только если реф не посадил, и только если лучше
            X2, rep2, reg2 = pf.fit(ref, plate, target, 1.62, table_contacts(p),
                                    log=lambda m: print(f"[{p['id']}] (таблица) {m}", flush=True), seat=seat)
            score = lambda r: (r["status"] != "ok", r["support_gap_cm"] + r["max_penetration_cm"])  # noqa: E731
            if score(rep2) < score(rep):
                X, rep, reg, src = X2, rep2, reg2, "table"
        rep["contacts_from"] = src
        pf.write_outputs(root / p["id"], X, ref["F"], rep, plate, reg, rep["contacts"].keys())
        rows.append(p["id"])
    print(f"[v2] fit: {len(rows)} поз", flush=True)


def loc_id(n):
    return f"mainroom{n}"


def geometry_asset_hashes(plate):
    import hashlib
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((plate.dir / "object_masks").glob("*.npz"))}


def attach_geometry_assets(plate):
    """Optional authoritative XY object masks, aligned with top_region._grid.

    <plate>/object_masks/<target>.npz: mask(bool), origin(2). Never infer one
    from the existing height-grown top: it has already discarded pillows.
    """
    plate.object_masks = {}
    for path in sorted((plate.dir / "object_masks").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            plate.object_masks[path.stem] = {"mask": data["mask"].astype(bool), "origin": data["origin"]}
    return plate


def table_seed(p, plate):
    """точка из таблицы: seat — поднятая точка сиденья (Plate.seat_point), feet — точка пола."""
    return plate.seat_point(*p["seat"])[0] if "seat" in p else plate.pixel_point(*p["feet"])


def do_profiles():
    """Черновые профили локаций текущего прогона (reviewed=false) из height_proxy + правил зоны."""
    import profiles as prof
    import shot_spec as ss
    import top_region as tr
    for n in sorted({p["loc"] for p in pr.prompts()}):
        plate = attach_geometry_assets(pf.Plate(pr.PLATE.format(n)))
        seeds = {}
        for p in pr.prompts():
            if p["loc"] != n:
                continue
            spec = ss.from_table(p)
            if spec.target_object != "floor" and "seat" in p:
                seeds.setdefault(spec.target_object, table_seed(p, plate))
        f = prof.write_draft(loc_id(n), plate, seeds, DEFAULT_HEIGHT, root=farm_env.ROOT)
        lp = prof.LocationProfile(f)
        regs = [lp.region(o) for o in lp.data["objects"] if "top_grid_npz" in o]
        tr.debug_overlay(plate, regs, f.parent / "overlay.jpg")
        print(f"[v2.4] profile {loc_id(n)}: " + "; ".join(
            f"{o['id']} h0 {o['h0']} {o['summary'].get('width_m')}×{o['summary'].get('depth_m')} м, "
            f"спинка {o['summary'].get('backrest_pts')}, {o['flags']}" for o in lp.data["objects"] if "summary" in o)
            + f"; якоря {[(a['id'], a['pose_classes']) for a in lp.data['anchors']]}", flush=True)


def do_place(refs, only, comfy, iters):
    import place
    import profiles as prof
    import shot_spec as ss
    ctx = None
    if comfy:
        import refine
        ctx = refine.load_ctx(comfy, iters=iters)
    body = prof.load_body(CHAR_ID, farm_env.ROOT)
    height = float(body["height_m"]) if body else DEFAULT_HEIGHT
    plates, profs = {}, {}
    root = pr.ROUND / M
    for p in pr.prompts():
        if only and p["id"] not in only:
            continue
        rf = Path(refs) / f"{p['id']}.ref.npz"
        if not rf.is_file():
            continue
        spec = ss.from_table(p)
        ref = pf.load_ref(rf)
        ref["path"] = str(rf)
        ref["body_cache_dir"] = str(farm_env.ROOT / "characters" / CHAR_ID / "geometry_cache")
        anatomy = os.environ.get("SCENE3D_BODY_ANATOMY")
        if anatomy:
            ref["body_model_path"] = anatomy
            ref["body_cache_dir"] = str(farm_env.ROOT / "characters" / CHAR_ID / "geometry_cache")
        if p["loc"] not in plates:
            plates[p["loc"]] = attach_geometry_assets(pf.Plate(pr.PLATE.format(p["loc"])))
        plate = plates[p["loc"]]
        if p["loc"] not in profs:
            profs[p["loc"]] = prof.LocationProfile.load(loc_id(p["loc"]), farm_env.ROOT)
        seed = table_seed(p, plate)
        res = place.place_pose(ref, plate, spec, seed, height, profile=profs[p["loc"]],
                               body_src="profile" if body else "proxy",
                               dims_override=body["body_dims"] if body else None, refine_ctx=ctx,
                               log=lambda m: print(m, flush=True))
        z = res["rep"].get("zone", {})
        fh = pf.fit_input_hash(rf, plate.dir, {"spec": spec.to_dict(), "height": height, "zone": z.get("source"),
                                               "cell": z.get("cell"),
                                               "placement_config_sha256": __import__("hashlib").sha256(
                                                   __import__("placement_config").CONFIG_PATH.read_bytes()).hexdigest(),
                                               "anatomy_sha256": __import__("hashlib").sha256(Path(anatomy).read_bytes()).hexdigest()
                                                   if anatomy else None,
                                               "object_mask_sha256": geometry_asset_hashes(plate),
                                               "profile_sha256": __import__("hashlib").sha256(profs[p["loc"]].path.read_bytes()).hexdigest()
                                                   if profs[p["loc"]] is not None else None,
                                               "top_definition": res["rep"].get("zone_diagnostics", {}).get("top_definition")})
        place.write(root / p["id"], res, ref, plate, spec, fh)
        r = res["rep"]
        print(f"[v2.5] {p['id']}: {r['status']} target {spec.target_object} опора {r.get('support_object')}", flush=True)


def do_diagnostics(refs, only, out):
    """E, read-only geometry of saved placements. Never fit/refine/render or modify fit.json."""
    import place
    import profiles as prof
    import shot_spec as ss
    from body import regions
    results = []
    plates, profs = {}, {}
    for p in pr.prompts():
        if only and p["id"] not in only:
            continue
        d = pr.ROUND / M / p["id"]
        rf = Path(refs) / (p["id"] + ".ref.npz")
        required = [rf, d / "body.obj", d / "label_joints.npz", d / "fit_state.npz", d / "fit.json"]
        missing = [str(f) for f in required if not f.is_file()]
        if missing:
            results.append({"id": p["id"], "blocked": "missing saved geometry", "missing": missing})
            continue
        ref = pf.load_ref(rf)
        X, faces = pf.load_obj_mesh(d / "body.obj")
        if len(X) != len(ref["V"]) or not np.array_equal(faces, ref["F"]):
            results.append({"id": p["id"], "blocked": "saved body/ref topology mismatch"})
            continue
        with np.load(d / "label_joints.npz", allow_pickle=False) as data:
            KP = data["kp"]
        report = json.loads((d / "fit.json").read_text(encoding="utf-8"))
        spec = ss.ShotSpec.load(d / "shot_spec.json") if (d / "shot_spec.json").is_file() else ss.from_table(p)
        if p["loc"] not in plates:
            plates[p["loc"]] = pf.Plate(pr.PLATE.format(p["loc"]))
            profs[p["loc"]] = prof.LocationProfile.load(loc_id(p["loc"]), farm_env.ROOT)
        plate, profile = plates[p["loc"]], profs[p["loc"]]
        if profile is None:
            results.append({"id": p["id"], "blocked": "missing location profile"})
            continue
        treg, freg, _, obj = place.resolve_regions(plate, spec, table_seed(p, plate), DEFAULT_HEIGHT, profile)
        if treg is None:
            results.append({"id": p["id"], "blocked": "missing target region"})
            continue
        treg = place.tr.pose_region(treg, spec.pose_class)
        treg.wall_planes = profile.walls()
        treg.diagnostic_surfaces = [{"id": o["id"], "soft": bool(o.get("soft", False)),
                                     "region": profile.region(o)} for o in profile.data["objects"]
                                    if "top_grid_npz" in o]
        groups = []
        with np.load(d / "fit_state.npz", allow_pickle=False) as state:
            for desc in state["groups"]:
                name, surf, role = str(desc).split(":")
                groups.append((name, state["g_" + name], surf, role))
        B = {"reg": regions(ref["V"], ref["kp"])}
        if spec.pose_class == "kneel":
            oriented = place.orient_body(place.prepare_body(ref, spec, report.get("height_m", DEFAULT_HEIGHT)), spec)
            selected, _, _ = place.build_groups(oriented, spec, treg, oriented["dims"]["H"], False)
            selection = place.zone_diagnostics(oriented, spec, treg, selected)
        else:
            selection = report.get("zone_diagnostics", {})
        S = place.build_surfaces(plate, treg, freg)
        measured = place.geometry_diagnostics(X, KP, B, spec, treg, groups, S, plate, bool(obj.get("soft")))
        results.append({"id": p["id"], "target_object": spec.target_object,
                        "saved_status": report["status"], "source": "saved placement, NOT a step9d rerun",
                        "fit_hash": report.get("fit_hash"),
                        "selection": selection, **measured})
    text = json.dumps(results, ensure_ascii=False, indent=1)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return results


def do_render(only):
    root = pr.ROUND / M
    for n in (1, 2, 3):
        base = json.loads((pr.HERE / pr.BASE[n]).read_text(encoding="utf-8"))
        scenes = []
        for p in pr.prompts():
            if p["loc"] != n:
                continue
            if not (root / p["id"] / "body.obj").is_file():          # no_target и т.п.: убрать старый плейбласт
                import shutil
                shutil.rmtree(pr.loc_dir(M, n) / p["id"], ignore_errors=True)
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
        render_ids = [s["name"] for s in scenes if not only or s["name"] in only]
        if not render_ids:
            continue
        out = pr.loc_dir(M, n)
        (out / "_plate").mkdir(parents=True, exist_ok=True)
        for fn in ("plate_rgb.png", "plate.json"):
            (out / "_plate" / fn).write_bytes((Path(pr.PLATE.format(n)) / fn).read_bytes())
        cmd = farm_env.bpy_cmd(pr.HERE / "render_plate_scene.py") + [
            "--plate", pr.PLATE.format(n), "--spec", str(spec), "--mannequin", os.environ["FARM_MANNEQUIN"],
            "--out", str(out), "--only", ",".join(render_ids), "--label-pass", "--photo-emission", "2.5", "--samples", "48"]
        if farm_env.BLENDER:
            cmd[1:1] = ["--python-exit-code", "1"]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        for line in r.stdout.splitlines():
            if line.startswith(("[plate]",)) or "Error" in line or "Traceback" in line:
                print(f"[V2 m{n}] {line}", flush=True)
        if r.returncode or "Traceback (most recent call last)" in r.stdout + r.stderr:
            raise RuntimeError(f"render m{n} failed ({r.returncode}):\n{(r.stdout + r.stderr)[-4000:]}")
        for scene in scenes:
            if scene["name"] not in render_ids:
                continue
            if not (out / scene["name"] / "labels.png").is_file():
                raise FileNotFoundError(out / scene["name"] / "labels.png")


def line25(pid, r):
    """строка v2.5 (08.11a): статус, опора, ориентация, след/кадр, проникновение, флаги."""
    ss, o, pen = r.get("shot_spec") or {}, r.get("orient") or {}, r.get("penetration") or {}
    y = o.get("yaw") or {}
    fp, fr = r.get("footprint") or {}, r.get("frame") or {}
    rf = r.get("refine") or {}
    z = r.get("zone", {})
    failure = (r.get("validation") or {}).get("first_failure")
    diagnostic = (f"{failure['check']} +{failure['excess']} {failure['unit']}" if failure else "—")
    return (f"{pid[:3]} {r['status']} {ss.get('pose_class')}→{ss.get('target_object')} опора {r.get('support')} "
            f"{r.get('support_gap_cm')}см втор {[(c['part'], c['surface']) for c in r.get('secondary', []) if c['role'] in ('wall', 'back')]} "
            f"стена {r.get('wall_gap_cm')} спинка {r.get('back_gap_cm')} | yaw реф {y.get('ref_yaw_deg')}→{y.get('yaw_deg')} "
            f"окно {y.get('window_deg')} сторона {o.get('side_ref')}→{o.get('side_final')} наклон {o.get('tilt_ref_deg')}→"
            f"{o.get('tilt_final_deg')} крен {o.get('roll_ref_deg')}→{o.get('roll_final_deg')} | след {fp.get('n_out')} "
            f"lie+{o.get('lie_yaw_add_deg')} кадр {fr.get('ok')} попыток {len(r.get('attempts', []))} сдвиг {z.get('shift_m')} | "
            f"прон p95 {pen.get('p95_cm')} max {pen.get('max_cm')} (soft {pen.get('soft_p95_cm')}/hard {pen.get('hard_p95_cm')}) | "
            f"refine {rf.get('groups_deg')} | провал {diagnostic} | высота {r.get('height_adjust')} "
            f"откаты {r.get('refine_root_clamps')} | {r.get('flags')} | зона {r.get('zone_diagnostics')}")


def do_summary():
    """v2.5 (08.11a п.10): shot_spec, опора/secondary, ориентация, след, кадр, p95 проникновения, статусы и флаги."""
    root = pr.ROUND / M
    rows = []
    for p in pr.prompts():
        f = root / p["id"] / "fit.json"
        if not f.is_file():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        keep = ("status", "shot_spec", "body_dims", "body_source", "support", "support_info", "secondary",
                "support_object", "support_parts_on", "support_gap_cm", "back_gap_cm", "wall_gap_cm", "penetration",
                "footprint", "frame", "frame_tries", "orient", "angle_to_camera_deg", "stage", "attempts", "flags",
                "target_info", "validation", "height_adjust", "refine_root_clamps", "zone_diagnostics",
                "back_contact_gap_m")
        row = {"id": p["id"], **{k: r.get(k) for k in keep},
               "zone": {k: r.get("zone", {}).get(k) for k in ("source", "object_id", "anchor_id", "via", "rule",
                                                                 "band_m", "cell", "attempt", "shift_m", "sides",
                                                                 "side_yaw_deg", "wall", "area_m2")},
               "refine": {k: (r.get("refine") or {}).get(k) for k in ("groups_deg", "limits_ok")}}
        rows.append(row)
        print(line25(p["id"], r))
    (root / "summary.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


def do_sheet24(refs_img, refs, out):
    """Лист v2.4 через tools/rowsheet.py: реф | реф с телом | посадка поверх фото | плейбласт."""
    root = pr.ROUND / M
    spec = []
    for p in pr.prompts():
        f = root / p["id"] / "fit.json"
        if not f.is_file():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        ss = r.get("shot_spec") or {}
        o, pen = r.get("orient") or {}, r.get("penetration") or {}
        lab = (f"{p['id']}  {r['status']}  {ss.get('pose_class')} → {ss.get('target_object')}  опора "
               f"{r.get('support_object')} {r.get('support_gap_cm')} см  прон p95 {pen.get('p95_cm')} max "
               f"{pen.get('max_cm')} см  наклон {o.get('tilt_ref_deg')}→{o.get('tilt_final_deg')}  сторона "
               f"{o.get('side_ref')}→{o.get('side_final')}  след {(r.get('footprint') or {}).get('n_out')}  кадр "
               f"{(r.get('frame') or {}).get('ok')}  {','.join(r.get('flags', []))}")
        pb = pr.loc_dir(M, p["loc"]) / p["id"] / "playblast.png"
        spec.append({"label": lab, "cells": [
            [str(Path(refs_img) / f"{p['id']}.png"), "реф"], [str(Path(refs) / f"{p['id']}.ref_check.jpg"), "реф + тело"],
            [str(root / p["id"] / "overlay.jpg") if (root / p["id"] / "overlay.jpg").is_file() else None, "посадка"],
            [str(pb) if pb.is_file() else None, "плейбласт"]]})
    sp = root / "sheet24_spec.json"
    sp.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    subprocess.run([sys.executable, str(HERE.parents[2] / "tools" / "rowsheet.py"), str(sp), "--out", out], check=True)


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
    ap.add_argument("cmd", choices=["collect", "fit", "render", "summary", "sheet", "profiles", "place", "sheet24", "diagnostics"])
    ap.add_argument("--comfy", default="", help="ComfyUI с SAM 3D Body (refine внутри place); пусто — без refine")
    from placement_config import section
    ap.add_argument("--iters", type=int, default=section("refine")["iters"])
    ap.add_argument("--refs")
    ap.add_argument("--refs-img")
    ap.add_argument("--confluence")
    ap.add_argument("--out")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    {"collect": lambda: do_collect(a.confluence, a.out), "fit": lambda: do_fit(a.refs, only),
     "render": lambda: do_render(only), "summary": do_summary,
     "sheet": lambda: do_sheet(a.refs_img, a.refs, a.out), "profiles": do_profiles,
     "place": lambda: do_place(a.refs, only, a.comfy, a.iters),
     "diagnostics": lambda: do_diagnostics(a.refs, only, a.out),
     "sheet24": lambda: do_sheet24(a.refs_img, a.refs, a.out)}[a.cmd]()
