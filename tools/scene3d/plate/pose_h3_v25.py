"""3D-A 08.11a п.11–12: H3 по плейбластам v2.5 — варианты A, B, C1 (denoise .55/.70/.85), C2 (DifferentialDiffusion),
метрики по ПОЛНЫМ кадрам H3 и лист poses_v25_h3.jpg. Сервер студии (FARM_REMOTE=1), seed 101, план full (как шаг 7).

  python pose_h3_v25.py prep    --ids Q03,Q04,...   # кропы плана full, промпты A/B + линтер (prompts_v25.json)
  python pose_h3_v25.py run     --ids ...           # задача 1 на шот: A + B + C1 ×3 (одна задача, один seed)
  python pose_h3_v25.py metrics --ids ...           # PCK/PCK_lr, bg_ssim, obj_shift_px, grey_left (+ DWPose)
  python pose_h3_v25.py run_c2  --ids ...           # задача 2: C2, denoise — лучший C1 по bg_ssim при PCK_lr ≥ 0.6
  python pose_h3_v25.py sheet   --out poses_v25_h3.jpg

A  = рецепт шага 7 (L_C2: <Picture 1> плита-кроп, <Picture 2> лицо 0003.png, <Video 1> плейбласт), промпт table_round.
B  = A + предложение позы из shot_spec сразу после первого предложения [Shot 1] + манекен/комната (п.11б);
     «One single photorealistic static image.» → «One single photograph.» (A не проходит линтер; решение пользователя).
C1 = стартовый латент H3LatentFromImage(плейбласт) с denoise < 1, промпт B, SamplerCustomAdvanced (штатные ноды).
C2 = C1 + DifferentialDiffusion и SetLatentNoiseMask с картой силы: 1.0 на силуэте манекена и тени, гауссово падение
     (σ = 8 % высоты кадра) до 0.40.
"""
import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "posev2"))
sys.path.insert(0, str(REPO))
import build_ab as ab  # noqa: E402
import farm_env  # noqa: E402
import plan_crop  # noqa: E402
import pose_round as pr  # noqa: E402
import quality_round as qr  # noqa: E402
import remote_sync  # noqa: E402
import table_server as ts  # noqa: E402

SEED = 101
C1_DENOISE = (0.55, 0.70, 0.85)
C2_FLOOR = 0.40
C2_SIGMA = 0.08
PCK_LR_MIN = 0.6
PCK_TORSO = 0.25            # порог PCK: доля длины корпуса (шея → середина бёдер цели)
BG_DILATE = 0.06            # × высоты кадра
GREY_S, GREY_V = 0.08, 0.10
STATE = farm_env.ROOT / "pose_round" / "h3_v25"
OUT_SUB = "plate3d_v25"
POSE = {
    "kneel": "She kneels upright on top of the {obj}: both knees and shins rest on its surface, her thighs and torso "
             "are vertical.",
    "lie": "She lies on top of the {obj}; her whole body, legs and feet included, rests on its surface and nothing "
           "hangs over the edge.",
    "recline": "She reclines on the {obj}: her hips rest on its surface and her back and shoulders lean against the "
               "{backrest}, her head higher than her hips.",
    "sit_edge": "She sits on the edge of the {obj}: her hips rest on its surface and her feet stand on the floor.",
    "sit_tucked": "She sits on top of the {obj} with her legs folded beside her, resting on its surface.",
    "sit_floor": "She sits on the floor.",
    "stand": "She stands on the floor.",
    "crouch": "She crouches on the floor.",
    "all_fours": "She is on all fours on the floor, her hands and knees resting on it.",
}
MANNEQUIN = "Her body pose and placement exactly match the grey mannequin in {pb}, and the mannequin is fully replaced by her."
ROOM = "The room, the {obj} and all furniture stay exactly as in {plate}, with the same shape, height, size and position."
PB_LABEL, PLATE_LABEL = "<Video 1>", "<Picture 1>"     # реальные метки: плейбласт — ref_video_0, плита — ref_image_0


def prompts_by_id():
    return {p["id"]: p for p in pr.prompts()}


def full_id(short):
    return next(k for k in prompts_by_id() if k.startswith(short))


def loc_of(pid):
    return f"pv25_{pid.split('_')[0]}"


def fit_of(pid):
    return json.loads((pr.ROUND / "V2" / pid / "fit.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ промпты
def prompt_b(text_a, fit):
    ss = fit["shot_spec"]
    ti = fit.get("target_info") or {}
    obj = ti.get("name") or ss["target_object"]
    pose = POSE[ss["pose_class"]].format(obj=obj, backrest=ti.get("backrest_name") or "backrest")
    if ss["pose_class"] == "stand" and any(c["surface"] == "wall" for c in ss.get("secondary_contacts", [])):
        pose = pose[:-1] + ", her hands resting on the wall."
    add = " ".join([pose, MANNEQUIN.format(pb=PB_LABEL), ROOM.format(obj=obj, plate=PLATE_LABEL)])
    t = text_a.replace("One single photorealistic static image.", "One single photograph.")
    m = re.search(r"\[Shot 1\] (.*?\.)(\s)", t, re.S)          # первое предложение после [Shot 1]
    if not m:
        raise SystemExit("нет [Shot 1] в промпте A")
    return t[:m.end(1)] + " " + add + t[m.end(1):]


def lint(text):
    from farm.prompts.lint import lint as _lint
    iss = _lint(text, "frame", {"connected": {"<Picture 1>", "<Picture 2>", "<Video 1>"}})
    return [f"{i.severity}:{i.code}" for i in iss]


def scene_of(pid):
    p = prompts_by_id()[pid]
    spec = json.loads((pr.ROUND / "V2" / f"spec_m{p['loc']}.json").read_text(encoding="utf-8"))
    return spec, next(s for s in spec["scenes"] if s["name"] == pid)


def do_prep(ids):
    """кроп плана full (plan_crop, как шаг 7) в input/pv25_<id>_s2/<id>/ + input/pv25_<id>_rgb.png; промпты A/B."""
    inp = farm_env.COMFY / "input"
    STATE.mkdir(parents=True, exist_ok=True)
    out = {}
    for pid in ids:
        p = prompts_by_id()[pid]
        src = inp / f"pose_V2_m{p['loc']}_s2"
        loc = loc_of(pid)
        dst = inp / f"{loc}_s2" / pid
        shutil.rmtree(dst, ignore_errors=True)
        info = plan_crop.crop_scene(src / pid, src / "_plate" / "plate_rgb.png", dst, "full")
        shutil.copy(dst / "plate_rgb.png", inp / f"{loc}_rgb.png")
        spec, scene = scene_of(pid)
        g = ts.graph(loc, spec, scene, SEED)
        a = g["r2v"]["inputs"]["prompt"]
        b = prompt_b(a, fit_of(pid))
        out[pid] = {"plan": info, "A": a, "B": b, "lint_A": lint(a), "lint_B": lint(b)}
        print(f"[v25] prep {pid}: rect {info['rect_render']} zoom {info['zoom']}; линтер A {out[pid]['lint_A']} "
              f"B {out[pid]['lint_B']}", flush=True)
    f = STATE / "prompts_v25.json"
    old = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    old.update(out)
    f.write_text(json.dumps(old, indent=1, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ графы
def sampler_branch(g, tag, cond, latent_from, image, denoise, prefix, model="shift", mask=None):
    g[f"{tag}_lat"] = ab.node("H3LatentFromImage", h3_latent=[latent_from, 1], vae=["vae", 0], image=[image, 0],
                              model=[model, 0], steps=25, scheduler="simple", denoise=denoise)
    lat = [f"{tag}_lat", 0]
    if mask is not None:
        g[f"{tag}_nm"] = ab.node("SetLatentNoiseMask", samples=lat, mask=[mask, 0])
        lat = [f"{tag}_nm", 0]
    g[f"{tag}_noise"] = ab.node("RandomNoise", noise_seed=SEED)
    g[f"{tag}_guider"] = ab.node("BasicGuider", model=[model, 0], conditioning=[cond, 0])
    g[f"{tag}_sampler"] = ab.node("KSamplerSelect", sampler_name="euler")
    g[f"{tag}_smp"] = ab.node("SamplerCustomAdvanced", noise=[f"{tag}_noise", 0], guider=[f"{tag}_guider", 0],
                              sampler=[f"{tag}_sampler", 0], sigmas=[f"{tag}_lat", 1], latent_image=lat)
    g[f"{tag}_dec"] = ab.node("H3ImageDecode", samples=[f"{tag}_smp", 0], vae=["vae", 0])
    g[f"{tag}_sel"] = ab.node("H3ImageFrameSelector", frames=[f"{tag}_dec", 0], recommended_index=[f"{tag}_dec", 3],
                              **{k: v for k, v in g["sel"]["inputs"].items() if k not in ("frames", "recommended_index")})
    g[f"{tag}_save"] = ab.node("SaveImage", images=[f"{tag}_sel", 0], filename_prefix=prefix)


def base_graph(pid):
    pp = json.loads((STATE / "prompts_v25.json").read_text(encoding="utf-8"))[pid]
    if any(x.startswith("error") for x in pp["lint_B"]):
        raise SystemExit(f"{pid}: линтер B {pp['lint_B']} — B/C не запускаем")
    spec, scene = scene_of(pid)
    g = ts.graph(loc_of(pid), spec, scene, SEED)
    for k in ("sam", "sam_grow", "sam_img", "sam_save"):
        g.pop(k, None)
    g = farm_env.prepare(g)
    assert g["r2v"]["inputs"]["prompt"] == pp["A"]
    g["r2vB"] = copy.deepcopy(g["r2v"])
    g["r2vB"]["inputs"]["prompt"] = pp["B"]
    return g, pp


def graph_job1(pid):
    g, _ = base_graph(pid)
    pre = f"{OUT_SUB}/{pid}"
    g["save"]["inputs"]["filename_prefix"] = f"{pre}/A_s{SEED}"
    g["negB"] = ab.node("ConditioningZeroOut", conditioning=["r2vB", 0])
    g["ksB"] = copy.deepcopy(g["ks"])
    g["ksB"]["inputs"].update(positive=["r2vB", 0], negative=["negB", 0], latent_image=["r2vB", 1])
    g["decB"] = ab.node("H3ImageDecode", samples=["ksB", 0], vae=["vae", 0])
    g["selB"] = copy.deepcopy(g["sel"])
    g["selB"]["inputs"].update(frames=["decB", 0], recommended_index=["decB", 3])
    g["saveB"] = ab.node("SaveImage", images=["selB", 0], filename_prefix=f"{pre}/B_s{SEED}")
    for d in C1_DENOISE:
        t = f"C1_{round(d * 100)}"
        sampler_branch(g, t, "r2vB", "r2vB", "clay_scale", d, f"{pre}/{t}_s{SEED}")
    for k, n in g.items():
        n["_meta"] = {"title": f"3DA v25 · {k}"}
    return g


def strength_map(pid, size):
    """карта силы C2: 1.0 на силуэте манекена ∪ тень (shadow > 0.05), гауссово падение σ = 8 % высоты до 0.40."""
    from scipy import ndimage
    d = farm_env.COMFY / "input" / f"{loc_of(pid)}_s2" / pid
    m = silhouette(d / "mask.png", size)
    sh = np.asarray(Image.open(d / "shadow.png").convert("L").resize(size, Image.BILINEAR), np.float32) / 255
    reg = m | (sh > 0.05)
    dist = ndimage.distance_transform_edt(~reg)
    s = C2_SIGMA * size[1]
    st = C2_FLOOR + (1 - C2_FLOOR) * np.exp(-dist ** 2 / (2 * s * s))
    f = d / "c2_strength.png"
    Image.fromarray((st * 255).round().astype(np.uint8)).save(f)
    return f


def graph_c2(pid, denoise):
    g, _ = base_graph(pid)
    pre = f"{OUT_SUB}/{pid}"
    loc = loc_of(pid)
    g["c2_map_img"] = ab.node("LoadImage", image=f"{loc}_s2/{pid}/c2_strength.png")
    g["c2_map_scale"] = ab.node("ImageScale", image=["c2_map_img", 0], upscale_method="bilinear",
                                width=ts.SIZE[0], height=ts.SIZE[1], crop="disabled")
    g["c2_mask"] = ab.node("ImageToMask", image=["c2_map_scale", 0], channel="red")
    g["dd"] = ab.node("DifferentialDiffusion", model=["shift", 0], strength=1.0)
    sampler_branch(g, "C2", "r2vB", "r2vB", "clay_scale", denoise, f"{pre}/C2_s{SEED}", model="dd", mask="c2_mask")
    for k in ("ks", "neg", "dec", "sel", "save", "r2v"):        # A в этой задаче не нужен
        g.pop(k, None)
    g["save"] = g.pop("C2_save")                                # qr.wait ждёт узел save
    for k, n in g.items():
        n["_meta"] = {"title": f"3DA v25 · {k}"}
    return g


# ------------------------------------------------------------------------------------------------ сервер
def queue_free(max_wait=1800):
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_wait:
        q = qr.api("/queue")
        if not q["queue_running"] and not q["queue_pending"]:
            return True
        print("[v25] очередь сервера занята — жду", flush=True)
        time.sleep(30)
    return False


def submit(name, g, uploads):
    runs_f = STATE / "runs.json"
    runs = json.loads(runs_f.read_text()) if runs_f.is_file() else []
    if any(r["name"] == name and r["status"] == "success" for r in runs):
        print(f"[v25] skip {name}", flush=True)
        return "success"
    pend = next((r for r in runs if r["name"] == name and r["status"] == "submitted"), None)
    if pend is None:
        inp = farm_env.COMFY / "input"
        for f in uploads:
            rel = f.relative_to(inp)
            remote_sync.upload(f, str(rel.parent).replace("\\", "/") if str(rel.parent) != "." else "")
        if not queue_free():
            raise SystemExit("очередь сервера занята > 30 мин")
        (STATE / f"{name}.api.json").write_text(json.dumps(g, indent=1), encoding="utf-8")
        pend = {"name": name, "status": "submitted", "prompt_id": ab.post(g), "t0": time.time()}
        runs.append(pend)
        runs_f.write_text(json.dumps(runs, indent=1))
    status, _ = qr.wait(pend["prompt_id"], 1800)
    elapsed = round(time.time() - pend["t0"], 1)
    if status == "success":
        hist = qr.api("/history/" + pend["prompt_id"])[pend["prompt_id"]]
        stamps = {k: v.get("timestamp") for k, v in hist.get("status", {}).get("messages", [])}
        if stamps.get("execution_start") and stamps.get("execution_success"):
            elapsed = round((stamps["execution_success"] - stamps["execution_start"]) / 1000, 1)
    pend.update(status=status, seconds=elapsed)
    runs_f.write_text(json.dumps(runs, indent=1))
    print(f"[v25] {name}: {status} {pend['seconds']} с", flush=True)
    if status != "success":
        raise SystemExit(f"stop: {name} {status}")
    return status


def uploads_of(pid, extra=()):
    inp = farm_env.COMFY / "input"
    loc = loc_of(pid)
    return [inp / f"{loc}_rgb.png", inp / f"{loc}_s2" / pid / "playblast.png", *extra]


def do_run(ids):
    for pid in ids:
        submit(f"{pid}_job1", graph_job1(pid), uploads_of(pid))


def do_run_c2(ids):
    met = json.loads((STATE / "metrics.json").read_text(encoding="utf-8"))
    for pid in ids:
        m = met[pid]
        ok = [(m[f"C1_{round(d * 100)}"]["bg_ssim"], d) for d in C1_DENOISE
              if (m.get(f"C1_{round(d * 100)}") or {}).get("pck_lr", 0) >= PCK_LR_MIN]
        d = max(ok)[1] if ok else 0.85
        size = Image.open(farm_env.COMFY / "input" / f"{loc_of(pid)}_s2" / pid / "playblast.png").size
        f = strength_map(pid, size)
        print(f"[v25] C2 {pid}: denoise {d} ({'лучший C1 по bg_ssim' if ok else 'нет C1 с PCK_lr ≥ 0.6'})", flush=True)
        submit(f"{pid}_c2", graph_c2(pid, d), uploads_of(pid, [f]))
        cf = STATE / "c2_denoise.json"
        cd = json.loads(cf.read_text()) if cf.is_file() else {}
        cd[pid] = d
        cf.write_text(json.dumps(cd, indent=1))


# ------------------------------------------------------------------------------------------------ метрики
def silhouette(path, size):
    m = np.asarray(Image.open(path).convert("L").resize(size, Image.NEAREST)) > 127
    return ~m if m.mean() > 0.5 else m


def latest(d, prefix):
    fs = sorted(Path(d).glob(f"{prefix}_*.png"))
    return fs[-1] if fs else None


def outputs(pid):
    d = farm_env.COMFY / "output" / OUT_SUB / pid
    out = {"A": latest(d, f"A_s{SEED}"), "B": latest(d, f"B_s{SEED}")}
    for dd in C1_DENOISE:
        out[f"C1_{round(dd * 100)}"] = latest(d, f"C1_{round(dd * 100)}_s{SEED}")
    out["C2"] = latest(d, f"C2_s{SEED}")
    return {k: v for k, v in out.items() if v is not None}


def ssim_masked(a, b, m):
    from scipy.ndimage import gaussian_filter
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = gaussian_filter(a, 1.5), gaussian_filter(b, 1.5)
    saa = gaussian_filter(a * a, 1.5) - mu_a ** 2
    sbb = gaussian_filter(b * b, 1.5) - mu_b ** 2
    sab = gaussian_filter(a * b, 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(s[m].mean())


def object_top_edge(pid, plan, size):
    """верхний край target_object в кадре плана: проекция маски верха из профиля камерой плиты → для каждого столбца
    верхняя строка. -> (cols, rows) в пикселях кадра size | None для floor."""
    import pose_fit as pf
    import profiles as prof
    fit = fit_of(pid)
    if fit["shot_spec"]["target_object"] == "floor":
        return None
    p = prompts_by_id()[pid]
    plate = pf.Plate(pr.PLATE.format(p["loc"]))
    lp = prof.LocationProfile.load(f"mainroom{p['loc']}", farm_env.ROOT)
    obj, _ = lp.find_object(fit["shot_spec"]["target_object"])
    P = lp.region(obj).points3d()
    u, v, z = plate.project(P)
    x0, y0, x1, y1 = plan["rect_render"]
    sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)
    U, Vv = (u - x0) * sx, (v - y0) * sy
    ok = (U >= 0) & (U < size[0]) & (Vv >= 0) & (Vv < size[1]) & (z > 0)
    cols = np.floor(U[ok]).astype(int)
    rows = Vv[ok]
    top = {}
    for c, r in zip(cols, rows):
        top[c] = min(top.get(c, 1e9), r)
    cs = np.array(sorted(top))
    return cs, np.array([top[c] for c in cs])


def edge_rows(gray, cols, rows, win):
    """для каждого столбца — строка сильнейшего вертикального градиента в окне ±win вокруг rows."""
    from scipy.ndimage import sobel
    gy = np.abs(sobel(gray.astype(np.float64), axis=0))
    out = []
    for c, r in zip(cols, rows):
        a, b = int(max(0, r - win)), int(min(gray.shape[0], r + win + 1))
        out.append(a + int(np.argmax(gy[a:b, c])))
    return np.array(out, float)


SWAP = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]   # как pose_metrics.SWAP (OpenPose-18 L↔R)


def pck(target, det, torso, swap=False):
    if det is None:
        return 0.0
    d = np.asarray(det)[SWAP] if swap else np.asarray(det)
    vis = [i for i in range(14) if target[i, 2] > 0]
    hits = sum(1 for i in vis if d[i, 2] > 0.3 and np.linalg.norm(d[i, :2] - target[i, :2]) < PCK_TORSO * torso)
    return hits / len(vis) if vis else 0.0


def do_metrics(ids):
    pp = json.loads((STATE / "prompts_v25.json").read_text(encoding="utf-8"))
    inp = farm_env.COMFY / "input"
    files, jobs = [], {}
    for pid in ids:
        d = inp / f"{loc_of(pid)}_s2" / pid
        outs = outputs(pid)
        jobs[pid] = (d, outs)
        files += [str(d / "playblast.png")] + [str(f) for f in outs.values()]
    lst = STATE / "dw_list.txt"
    lst.write_text("\n".join(files), encoding="utf-8")
    det_f = STATE / "dw_det.json"
    subprocess.run([farm_env.PY_POSE, "-I", "-u", str(HERE / "dwpose_detect.py"), "--list", str(lst), "--out", str(det_f),
                    "--weights", farm_env.DWPOSE], check=True)
    det = json.loads(det_f.read_text(encoding="utf-8"))
    mf = STATE / "metrics.json"
    res = json.loads(mf.read_text(encoding="utf-8")) if mf.is_file() else {}
    for pid in ids:
        d, outs = jobs[pid]
        plan = pp[pid]["plan"]
        pb = Image.open(d / "playblast.png").convert("RGB")
        p25 = json.loads((inp / f"pose_V2_m{prompts_by_id()[pid]['loc']}_s2" / pid / "pose25.json").read_text())
        x0, y0, x1, y1 = plan["rect_render"]
        row = {"playblast_conf": None}
        dpb = det.get(str(d / "playblast.png"), {}).get("people", [])
        for name, f in outs.items():
            im = Image.open(f).convert("RGB")
            W, H = im.size
            sx, sy = W / (x1 - x0), H / (y1 - y0)
            T = np.array([[(k[0] - x0) * sx, (k[1] - y0) * sy, k[2]] for k in p25["keypoints_body25"][:18]], float)
            torso = float(np.linalg.norm(T[1, :2] - (T[8, :2] + T[11, :2]) / 2)) if T[1, 2] and T[8, 2] and T[11, 2] \
                else 0.3 * H
            people = det.get(str(f), {}).get("people", [])
            cen = T[T[:, 2] > 0, :2].mean(0)
            best = min(people, key=lambda q: np.linalg.norm((np.array(q["box"][:2]) + q["box"][2:]) / 2 - cen),
                       default=None)
            b18 = best["body18"] if best else None
            p_dir, p_sw = pck(T, b18, torso), pck(T, b18, torso, swap=True)
            sil = silhouette(d / "mask.png", (W, H))
            from scipy import ndimage
            dil = ndimage.binary_dilation(sil, iterations=max(1, int(BG_DILATE * H)))
            s512 = 512 / max(W, H)
            sz = (round(W * s512), round(H * s512))
            g1 = np.asarray(im.convert("L").resize(sz, Image.BILINEAR))
            plate = Image.open(d / "plate_rgb.png").convert("RGB").resize((W, H), Image.LANCZOS)
            g0 = np.asarray(plate.convert("L").resize(sz, Image.BILINEAR))
            bgm = ~np.asarray(Image.fromarray(dil.astype(np.uint8) * 255).resize(sz, Image.NEAREST)).astype(bool)
            bg = ssim_masked(g0, g1, bgm)
            edge = object_top_edge(pid, plan, (W, H))
            shift = None
            if edge is not None:
                cols, rows = edge
                keep = ~dil[np.clip(rows.astype(int), 0, H - 1), cols]
                if keep.sum() >= 10:
                    win = int(0.08 * H)
                    e0 = edge_rows(np.asarray(plate.convert("L")), cols[keep], rows[keep], win)
                    e1 = edge_rows(np.asarray(im.convert("L")), cols[keep], rows[keep], win)
                    shift = round(float(np.mean(np.abs(e1 - e0))), 1)
            hsv = np.asarray(im.convert("HSV"), np.float32) / 255
            vpb = np.asarray(pb.resize((W, H), Image.BILINEAR).convert("HSV"), np.float32)[..., 2] / 255
            grey = (hsv[..., 1] < GREY_S) & (np.abs(hsv[..., 2] - vpb) <= GREY_V * np.maximum(vpb, 1e-3))
            a1 = np.asarray(im, np.float32)
            a0 = np.asarray(pb.resize((W, H), Image.BILINEAR), np.float32)
            same = np.abs(a1 - a0).max(-1) <= 0.08 * 255                 # доп.: пиксель силуэта = плейбласт
            row[name] = {"file": str(f), "pck": round(p_dir, 2), "pck_lr": round(max(p_dir, p_sw), 2),
                         "pb_same": round(float(same[sil].mean()), 3) if sil.any() else None,
                         "lr_swapped": bool(p_sw > p_dir), "people": len(people), "bg_ssim": round(bg, 3),
                         "obj_shift_px": shift, "grey_left": round(float(grey[sil].mean()), 3) if sil.any() else None}
        if dpb:
            q = dpb[0]
            row["playblast_conf"] = round(float(np.mean([k[2] for k in q["body18"][:14]])), 2)
        row["playblast_people"] = len(dpb)
        res[pid] = row
        print(f"[v25] metrics {pid}: " + "; ".join(
            f"{k} {v['pck_lr']}/{v['bg_ssim']}/{v['obj_shift_px']}/{v['grey_left']}"
            for k, v in row.items() if isinstance(v, dict)) + f"; conf pb {row['playblast_conf']}", flush=True)
    mf.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def do_sheet(ids, out):
    met = json.loads((STATE / "metrics.json").read_text(encoding="utf-8"))
    spec = []
    names = ["A", "B", "C1_55", "C1_70", "C1_85", "C2"]
    for pid in ids:
        fit = fit_of(pid)
        d = farm_env.COMFY / "input" / f"{loc_of(pid)}_s2" / pid
        m = met.get(pid, {})
        cells = [[str(Path(farm_env.ROOT).parent / "refs_v2" / f"{pid}.png"), "реф"],
                 [str(d / "playblast.png"), f"плейбласт (conf {m.get('playblast_conf')})"]]
        for n in names:
            v = m.get(n)
            cells.append([v["file"] if v else None,
                          f"{n} {v['pck_lr']} / {v['bg_ssim']} / {v['obj_shift_px']} / {v['grey_left']}" if v else n])
        spec.append({"label": f"{pid}  {fit['status']}  {fit['shot_spec']['pose_class']}→"
                              f"{fit['shot_spec']['target_object']}   подписи: PCK_lr / bg_ssim / obj_shift_px / grey_left",
                     "cells": cells})
    sp = STATE / "sheet_spec.json"
    sp.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    subprocess.run([sys.executable, str(REPO / "tools" / "rowsheet.py"), str(sp), "--out", out], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "run", "metrics", "run_c2", "sheet"])
    ap.add_argument("--ids", default="Q03,Q04,Q08,T03,T01,Q09,T05,Q11,Q12,T02")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ids = [full_id(x) for x in a.ids.split(",") if x]
    {"prep": lambda: do_prep(ids), "run": lambda: do_run(ids), "metrics": lambda: do_metrics(ids),
     "run_c2": lambda: do_run_c2(ids), "sheet": lambda: do_sheet(ids, a.out)}[a.cmd]()
