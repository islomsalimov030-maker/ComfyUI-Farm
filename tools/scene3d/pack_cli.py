"""File-based bridge for 3DA nodes; no ComfyUI server or H3 execution.

  python tools/scene3d/pack_cli.py location|place --request request.json --output result.json

config: farm_root, comfy, python (farm venv), geometry_python (Comfy venv),
blender, mannequin (.blend), cell_px=16. Optional, explicit preparation data:
location_hints_by_id[id] = {seeds: {object_type: [world_x, world_y, world_z]},
                          footprints: {object_type: {bbox: [[x0,y0],[x1,y1]]}
                                       or {polygon: [[x,y], ...]}}}
prompt_context_by_location[id] = {room, action, anchor}; anchor is the textual
scale anchor consumed by the existing step-9 generator, not a placement rule.
Existing profile.prompt_context is also accepted. No table or secret file is read.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
# Isolated (-I) worker processes omit the script directory from sys.path.
sys.path.insert(0, str(HERE))
import threeda_blender  # bpy wheel in a child process, else Blender binary
REPO = HERE.parents[1]
TIMEOUT = 600
# Input defaults only; support selection/geometry belongs exclusively to posev2.
DEFAULT_SUPPORT = {
    "stand": ["foot.L", "foot.R"], "crouch": ["foot.L", "foot.R"],
    "kneel": ["knee.L", "knee.R", "shin.L", "shin.R"],
    "sit_edge": ["pelvis", "thigh.L", "thigh.R"],
    "sit_tucked": ["pelvis", "thigh.L", "thigh.R"], "sit_floor": ["pelvis"],
    "recline": ["pelvis"], "lie": ["pelvis", "torso", "thigh.L", "thigh.R"],
    "all_fours": ["knee.L", "knee.R"],
    "auto": ["reference"],
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def plain(value):
    """0.3.1: numpy scalars/arrays (np.bool is named «bool» in numpy 2) are written as plain JSON values.
    A new pose put a numpy bool into the scene report and the whole preparation fell back (13.11)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


NONFINITE = 1e9  # 0.3.15: «нет опоры/не найдено» (inf) в JSON — большое конечное число того же знака, NaN → null


def finite(value):
    """0.3.15: inf/NaN из проверок геометрии (нет опоры → зазор inf) не должны ронять подготовку сцены:
    до 0.3.15 save_json падал «Out of range float values are not JSON compliant: inf», сцена не экспортировалась,
    а 3D-окно оставалось со старой позой и комнатой."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return finite(value.tolist())
        if isinstance(value, np.generic):
            value = value.item()
    except ImportError:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else math.copysign(NONFINITE, value)
    return value


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(finite(data), ensure_ascii=False, indent=2, allow_nan=False, default=plain), encoding="utf-8")
    tmp.replace(path)


def required_file(value, label):
    if not value or not Path(value).is_file():
        raise ValueError(f"{label}: required file is missing")
    return Path(value).resolve()


def config_for(request):
    supplied = request.get("config", {})
    if not isinstance(supplied, dict):
        raise ValueError("config must be an object")
    config = dict(supplied)
    defaults = {
        "farm_root": os.environ.get("FARM_ROOT", ""),
        "comfy": os.environ.get("THREEDA_COMFY_ROOT", str(REPO.parent.parent)),
        "python": sys.executable,
        "geometry_python": sys.executable,
        "blender": os.environ.get("FARM_BLENDER", ""),
        "mannequin": os.environ.get("FARM_MANNEQUIN", ""), "cell_px": 16,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    if not config["farm_root"]:
        raise ValueError("config.farm_root or FARM_ROOT is required")
    if isinstance(config["cell_px"], bool) or config["cell_px"] != 16:
        raise ValueError("09c requires config.cell_px=16")
    config["farm_root"] = str(Path(config["farm_root"]).resolve())
    return config


def worker_python(config, executable):
    # Server dependencies may live in user-site/PYTHONPATH supplied by its
    # launcher. -I discards those paths even with the same sys.executable.
    # Workers remain separate OS processes; explicit PC environments stay isolated.
    return [executable] if config.get('server_mode') else [executable, '-I']


def stderr_tail(value, lines=30):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value or "")
    # Subprocess diagnostics can contain credentials in third-party exceptions.
    for key, secret in os.environ.items():
        if len(secret) >= 4 and re.search(r"TOKEN|PASSWORD|PASS$|SECRET|AUTH|API_KEY", key, re.I):
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", text)
    return text if lines is None else "\n".join(text.splitlines()[-lines:])


class StageError(RuntimeError):
    def __init__(self, stage, diagnostic):
        super().__init__(stage)
        self.diagnostic = stderr_tail(diagnostic)


def run_stage(command, stage, env=None):
    import signal

    command = [str(x) for x in command]
    def log_output(stdout, stderr):
        directory = (env if env is not None else os.environ).get('THREEDA_STAGE_LOG_DIR')
        if directory:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            name = re.sub(r'[^a-z0-9_-]+', '_', stage.lower())
            for kind, value in [('stdout', stdout), ('stderr', stderr)]:
                (root/(name+'.'+kind+'.txt')).write_text(stderr_tail(value, lines=None), encoding='utf-8')
    windows = sys.platform == "win32"
    process_options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if windows
                       else {"start_new_session": True})
    try:
        proc = subprocess.Popen(command, cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                **process_options)
    except OSError as exc:
        raise StageError(f"{stage}: could not start subprocess", str(exc)) from exc
    try:
        stdout, stderr = proc.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        diagnostic = exc.stderr or exc.stdout or ""
        try:
            if windows:
                # Kill the worker AND its geometry/Blender descendants, not just
                # the direct child as subprocess.run(timeout=...) would do.
                terminated = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
                if terminated.returncode:
                    proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, subprocess.TimeoutExpired):
            # Still reap the direct child if the tree already exited or taskkill
            # itself failed; pipe collection below is bounded independently.
            proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=5)
            diagnostic = stderr or stdout or diagnostic
        except subprocess.TimeoutExpired:
            proc.kill()
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        log_output(locals().get('stdout', ''), locals().get('stderr', diagnostic))
        raise StageError(f"{stage}: timeout after {TIMEOUT}s", diagnostic) from exc
    stdout, stderr = stdout or "", stderr or ""
    log_output(stdout, stderr)
    if proc.returncode or "Traceback (most recent call last)" in stdout + stderr:
        # Blender often sends Python tracebacks to stdout, not stderr.
        raise StageError(f"{stage}: subprocess failed ({proc.returncode})", stderr or stdout)
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def modules(config):
    # Set paths before importing farm_env (imported indirectly by step-9 helpers).
    os.environ["FARM_ROOT"] = config["farm_root"]
    os.environ["FARM_COMFY"] = config["comfy"]
    for directory in (REPO, HERE, HERE / "plate", HERE / "posev2"):
        sys.path.insert(0, str(directory))
    import pose_fit
    import profiles
    return pose_fit, profiles


def location_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("location_id must be a safe single directory name")
    return value


def vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label}: expected {size} finite numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f"{label}: expected {size} finite numbers")
    return list(value)


def plate_assets(directory):
    directory = Path(directory)
    for name in ("plate.json", "plate_mesh.obj", "plate_rgb.png", "depth_loc.png"):
        required_file(directory / name, name)
    return directory


def hints_for(config, loc_id):
    return config.get("location_hints_by_id", {}).get(loc_id, {})


def assets_for(config, loc_id, profile_directory):
    return plate_assets(config.get("location_asset_dirs", {}).get(loc_id, profile_directory))


def require_profile_geometry(profile):
    """Ask existing profile/D4 code to validate its assets; never manufacture a footprint."""
    for obj in profile.data["objects"]:
        if "top_grid_npz" in obj:
            # Missing D4 footprint is an existing diagnostic, not permission to
            # invent a mask or abort other placement checks. Missing assets raise.
            profile.region(obj)
        if "pts_npz" in obj:
            required_file(profile.dir / obj["pts_npz"], f"{obj['id']} wall points")


def keep_photo(request, geometry_dir):
    """0.2.0 «б»: фото комнаты в полном разрешении для рендера до H3 в МП из пульта.
    Пишется один раз; геометрия и кэш локации не меняются."""
    photo = request.get("plate")
    target = Path(geometry_dir) / "plate_photo.png"
    if photo and Path(photo).is_file() and not target.exists():
        from PIL import Image
        try:
            with Image.open(photo) as image:
                image.convert("RGB").save(target)
        except OSError:
            pass  # read-only preset location: render falls back to plate_rgb.png


PLATE_KEYS = ("R_world_from_cv", "cam_height_m", "inv_norm", "fov_x_deg", "size")
LOCATION_FILES = ("plate.json", "plate_mesh.obj", "plate_rgb.png", "depth_loc.png")


def location_cache_problem(directory, loc_id):
    """0.3.0: why a cached location cannot be used (None = usable). Broken/partial JSON is never trusted."""
    directory = Path(directory)
    for name in LOCATION_FILES:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            return name + " missing or empty"
    try:
        plate = read_json(directory / "plate.json")
        if not isinstance(plate, dict) or any(key not in plate for key in PLATE_KEYS):
            return "plate.json incomplete"
    except (OSError, ValueError) as exc:
        return "plate.json unreadable: " + str(exc)[:120]
    profile = directory / "profile.json"
    if profile.exists():
        try:
            data = read_json(profile)
            if not isinstance(data, dict) or data.get("loc_id") != loc_id or not isinstance(data.get("objects"), list):
                return "profile.json incomplete"
        except (OSError, ValueError) as exc:
            return "profile.json unreadable: " + str(exc)[:120]
    return None


def quarantine(directory):
    """Move a broken AUTO location aside (never delete) so it is prepared again from the photo."""
    directory = Path(directory)
    if directory.exists():
        target = directory.with_name(directory.name + ".broken-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        directory.rename(target)
        return str(target)
    return None


def copy_atomic(source, target):
    import shutil
    tmp = Path(target).with_name(Path(target).name + "." + uuid.uuid4().hex + ".tmp")
    shutil.copy2(source, tmp)
    tmp.replace(target)


def do_location(request, config):
    started = time.monotonic()
    timings = {'moge_s': 0., 'auto_profile_s': 0.}
    loc_id = location_id(request.get("location_id"))
    directory = Path(config["farm_root"]) / "locations" / loc_id
    profile_path = directory / "profile.json"
    # 0.3.0: an AUTO location (made from the user's photo) with a broken or partial cache is
    # prepared again instead of failing; preset locations keep their read-only data.
    if loc_id.startswith("auto") and directory.exists() and loc_id not in config.get("location_asset_dirs", {}):
        problem = location_cache_problem(directory, loc_id)
        if problem is None and not profile_path.is_file() and "seeds" not in hints_for(config, loc_id):
            # Geometry is fine but the auto profile never finished: drop only its partial files.
            if any(directory.glob("*_top.npz")) or any(directory.glob("*_pts.npz")):
                problem = "auto profile incomplete"
        if problem is not None:
            timings['broken_cache'] = problem
            timings['quarantined'] = quarantine(directory)
    timings['profile_cache_hit'] = profile_path.is_file()
    if profile_path.is_file():
        geometry_dir = assets_for(config, loc_id, directory)
        data = read_json(profile_path)
        if data.get("loc_id") != loc_id:
            raise ValueError("cached profile loc_id mismatch")
    else:
        stage = time.monotonic()
        photo = required_file(request.get("plate"), "plate")
        directory.mkdir(parents=True, exist_ok=True)
        if not all((directory / n).is_file() for n in ("plate.json", "plate_mesh.obj", "plate_rgb.png", "depth_loc.png")):
            core = ["--comfy", config["comfy"]] if config.get('server_mode') else []
            run_stage([config["geometry_python"], HERE / "plate/plate_geometry.py", photo, directory,
                       "--size", "1344x768", *core], "plate MoGe")
            timings['moge_s'] = round(time.monotonic() - stage, 3)
        geometry_dir = plate_assets(directory)
        hints = hints_for(config, loc_id)
        if "seeds" not in hints:
            stage = time.monotonic()
            pf, profiles = modules(config)
            import auto_surfaces
            import tempfile
            plate = pf.Plate(directory)
            with tempfile.TemporaryDirectory(prefix="3da_auto_") as td:
                generated = auto_surfaces.write_auto_profile(loc_id, plate, Path(td))
                assets = sorted(Path(td).iterdir(), key=lambda a: a.name == "profile.json")
                for asset in assets:
                    if (directory / asset.name).exists():
                        raise ValueError("auto profile refuses overwrite: " + asset.name)
                # profile.json last and atomically: its presence marks a finished location.
                for asset in assets:
                    copy_atomic(asset, directory / asset.name)
            data = read_json(profile_path)
            preview = directory / "objects_preview.png"
            auto_surfaces.render_objects_preview(plate, profile_path, preview)
            timings['auto_profile_s'] = round(time.monotonic() - stage, 3)
            timings['total_s'] = round(time.monotonic() - started, 3)
            keep_photo(request, geometry_dir)
            return {"status": "ok", "location": str(profile_path.resolve()),
                    "plate": str((geometry_dir / "plate_rgb.png").resolve()),
                    "objects": data["objects"], "objects_preview": str(preview.resolve()), 'timings': timings}
        seeds = {name: vector(point, 3, f"seed {name}") for name, point in hints["seeds"].items()}
        pf, profiles = modules(config)
        plate = pf.Plate(directory)
        import numpy as np
        plate.object_masks = {}
        for asset in sorted((directory / "object_masks").glob("*.npz")):
            with np.load(asset, allow_pickle=False) as data:
                plate.object_masks[asset.stem] = {"mask": data["mask"].astype(bool), "origin": data["origin"]}
        profiles.write_draft(loc_id, plate, seeds, 1.62, root=config["farm_root"])
        data = read_json(profile_path)
        for obj in data["objects"]:
            hint = hints.get("footprints", {}).get(obj["id"], hints.get("footprints", {}).get(obj["type"], {}))
            if "bbox" in hint:
                bbox = hint["bbox"]
                if len(bbox) != 2:
                    raise ValueError("footprint bbox requires two XY points")
                obj["footprint_bbox"] = [vector(p, 2, "footprint bbox") for p in bbox]
            if "polygon" in hint:
                if len(hint["polygon"]) < 3:
                    raise ValueError("footprint polygon requires at least three XY points")
                obj["footprint_polygon"] = [vector(p, 2, "footprint polygon") for p in hint["polygon"]]
        save_json(profile_path, data)
    if any("top_grid_npz" in obj for obj in data["objects"]):
        _, profiles = modules(config)
        require_profile_geometry(profiles.LocationProfile(profile_path))
    pf, _ = modules(config)
    import auto_surfaces
    # Preview lives outside the immutable prepared location profile.
    preview_dir = Path(config.get("cache_dir", str(Path(config["farm_root"]) / "cache"))) / "objects_preview" / loc_id
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview = preview_dir / "objects_preview.png"
    auto_surfaces.render_objects_preview(pf.Plate(geometry_dir), profile_path, preview)
    timings['total_s'] = round(time.monotonic() - started, 3)
    keep_photo(request, geometry_dir)
    return {"status": "ok", "location": str(profile_path.resolve()),
            "plate": str((geometry_dir / "plate_rgb.png").resolve()), "objects_preview": str(preview.resolve()),
            'timings': timings,
            "objects": [{"id": o["id"], "type": o.get("type", o.get("name")),
                         "height_m": o.get("h0", o.get("z_m", [None, None])[-1])} for o in data["objects"]]}


def parts(value):
    if not value:
        return []
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    raise ValueError("support_parts must be a list or comma-separated string")


def contacts(value):
    if not value:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = [dict(zip(("part", "surface"), item.strip().split(":"))) for item in value.split(",")]
    if not isinstance(value, list) or any(not isinstance(c, dict) or not isinstance(c.get("part"), str)
                                         or not isinstance(c.get("surface"), str) for c in value):
        raise ValueError("secondary_contacts requires [{part,surface}] or part:surface,...")
    return value


def seed_for(profile, spec, config):
    """Read an explicit world anchor or the existing profile's stored zone, never infer furniture."""
    import numpy as np
    hints = hints_for(config, profile.data["loc_id"])
    seed = hints.get("seeds", {}).get(spec.instance_hint) or hints.get("seeds", {}).get(spec.target_object)
    obj, _ = profile.find_object(spec.target_object, instance_hint=spec.instance_hint,
                                 anchor_xy=seed[:2] if seed else None, pose_class=spec.pose_class)
    if obj is None:
        raise ValueError("target instance is absent or ambiguous; prepare profile/config spatial hints")
    if obj.get("seed_world") is not None and seed is None:
        return np.asarray(vector(obj["seed_world"], 3, "auto-surface seed_world"), float)
    if seed is not None:
        return np.asarray(vector(seed, 3, "location seed"), float)
    anchor = profile.anchor(obj["id"], spec.pose_class)
    if anchor is None:
        raise ValueError("profile has no anchor for this pose class; prepare explicit location seed")
    if "seed_world" in anchor:
        return np.asarray(vector(anchor["seed_world"], 3, "profile seed_world"), float)
    zone = profile.zone_metrics(anchor)
    xy, h = np.asarray(zone["xy"]), np.asarray(zone["h"])
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all() or not np.isfinite(h).all() or h.size == 0:
        raise ValueError("profile anchor zone is missing finite spatial hints")
    return np.r_[xy.mean(axis=0), h.mean()]


def prompt_for(profile, config, fit):
    """Keep step-9/httpx imports in the farm venv, never the geometry venv."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="3da-prompt-") as temp:
        request_path, output_path = Path(temp) / "request.json", Path(temp) / "result.json"
        save_json(request_path, {"config": config, "profile_data": profile.data, "fit": fit})
        run_stage([*worker_python(config, config["python"]), Path(__file__).resolve(), "place", "--request", request_path,
                   "--output", output_path, "--prompt-worker"], "step9 prompt")
        result = read_json(required_file(output_path, "step9 prompt result"))
        if result.get("status") == "error":
            raise StageError(result.get("error", "step9 prompt failed"), result.get("stderr_tail", ""))
        return result["h3_prompt"]


def prompt_direct(profile, config, fit):
    import prompt_h3
    context = dict(profile.data.get("prompt_context", {}))
    context.update(config.get("prompt_context_by_location", {}).get(profile.data["loc_id"], {}))
    if not context.get("anchor") and profile.data.get("auto_surfaces"):
        # A fresh room has no human-authored furniture/anatomy scale caption.
        # Use the actual guide's geometry rather than inventing a table/bed.
        context["anchor"] = ("the reconstructed surfaces and body in <Video 1> define the projected scale; "
                             "preserve their exact relative size and contact positions")
    if not context.get("anchor"):
        raise ValueError("prompt scale anchor missing: prepare profile.prompt_context.anchor or "
                         "config.prompt_context_by_location[id].anchor")
    scene = {"action": context.get("action", "holding the body pose shown in <Video 1>"), "anchor": context["anchor"]}
    text = prompt_h3.prompt_d(prompt_h3.table_prompt({"room": context.get("room", "the room in <Picture 1>")}, scene), fit)
    additions = []
    pose_text = str(fit.get("pose_text", "")).strip()
    if pose_text:
        if "\n" in pose_text or any(tag in pose_text for tag in ("subject_definitions:", "summary:", "overall_soundscape:")):
            raise ValueError("pose_text must be one pose sentence, not prompt sections")
        additions.append(pose_text.rstrip(".!?") + ".")
    for prop in fit.get("props_prompt", []):
        name = str(prop.get("description_en") or prop["name"]).strip()
        if any(c in name for c in "\r\n"):
            raise ValueError("prop description must be one line")
        additions.append(f"The {name} floats motionless in the air; preserve its exact shape, color and material as in the init image.")
    if additions:
        marker = "\noverall_soundscape:"
        text = text.replace(marker, "\n" + " ".join(additions) + marker, 1)
    prompt_h3.require_lint(text, video=True)
    return text


def prepare_images(directory, plate_path, proxy_mode, body_json, strength, cell_px, size=(1344, 768), margin=1.):
    import numpy as np
    from PIL import Image
    import pose_proxy_v26 as proxy
    # size — растр H3 (0.2.0 «б», МП из пульта); cell_px=16 = клетка латента H3 в любом растре.
    with Image.open(plate_path) as image:
        plate = np.asarray(image.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    with Image.open(directory / "playblast.png") as image:
        rendered = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        playblast = np.asarray(rendered)
    rendered.save(directory / "playblast.png")
    with Image.open(directory / "silhouette.png") as image:
        sil = np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST)) >= 128
    Image.fromarray(sil.astype(np.uint8) * 255).save(directory / "silhouette.png")
    Image.fromarray(plate).save(directory / "plate.png")
    if not sil.any():
        raise ValueError("renderer produced empty silhouette")
    figure = playblast
    if proxy_mode == "color":
        colors = proxy.outfit_colors(required_file(body_json, "body_json for color proxy"))
        with Image.open(directory / "labels.png") as image:
            labels = proxy.decode_labels(np.asarray(image.convert("RGB").resize(size, Image.Resampling.NEAREST)), sil)
        figure = proxy.proxy_image(playblast, plate, sil, labels, colors)
        Image.fromarray(figure).save(directory / "proxy.png")
    # 0.3.14: playblast без падающих теней Cycles (фигура поверх чистого фото) — для режима «Тени от H3»:
    # иначе H3 копировал тени рендера из видео-референса в обоих режимах.
    Image.fromarray(np.where(sil[..., None], playblast, plate)).save(directory / "playblast_ns.png")
    # Exact plate outside the figure, matching ImageCompositeMasked in step 9.
    Image.fromarray(np.where(sil[..., None], figure, plate)).save(directory / "init.png")
    # 0.3.7: запас вокруг тела (волосы, тени H3) — клетка × margin; margin растёт с растром, чтобы в 2 Мп
    # полоса не становилась относительно тоньше, чем в 1 Мп. margin=1 — прежние 1+2 клетки.
    band = max(int(cell_px), int(round(cell_px * float(margin))))
    sm = proxy.strength_map(sil, strength, band) if strength else np.zeros(sil.shape)
    Image.fromarray(np.rint(sm * 255).astype(np.uint8)).save(directory / "strength.png")
    overlay = plate.astype(float) * (1 - .45 * sm[..., None])
    overlay += np.array([255, 70, 30]) * (.45 * sm[..., None])
    from scipy.ndimage import binary_erosion
    overlay[sil & ~binary_erosion(sil)] = [0, 255, 180]
    Image.fromarray(np.rint(overlay).astype(np.uint8)).save(directory / "debug.png")


def shadow_zone(directory, strength_files, band, strength=.7, opacity=.6):
    """0.3.8: зона теней в области H3. Тени тела и предметов из рендера (Cycles) — мягче (размытие ~ полосы) и
    светлее (opacity) → room_shadow.png (подсказка для режима «тени от рендера»). Там, где тень ожидается,
    H3 получает силу strength с тем же мягким краем, что у тела, — тень не обрезается краем полосы."""
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    import pose_proxy_v26 as proxy
    directory = Path(directory)
    with Image.open(directory / "silhouette.png") as image:
        sil = np.asarray(image.convert("L")) > 127
    size = (sil.shape[1], sil.shape[0])
    layers = []
    for name in ("shadow.png", "prop_shadow.png"):
        if (directory / name).is_file():
            with Image.open(directory / name) as image:
                layers.append(np.asarray(image.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255)
    if not layers:
        return None
    shadow = gaussian_filter(np.max(layers, axis=0), sigma=max(1., band / 2)) * float(opacity)
    shadow[sil] = 0
    zone = shadow > .03
    if not zone.any():
        return None
    Image.fromarray(np.rint(shadow.clip(0, 1) * 255).astype("uint8")).save(directory / "room_shadow.png")
    extra = proxy.strength_map(zone, float(strength), band)
    # 0.4.0 (тест «чистые тени»): зона тени и полоса тела/предмета без неё — для переноса только затемнения H3.
    Image.fromarray(np.rint(np.clip(extra / max(float(strength), 1e-6), 0, 1) * 255).astype("uint8")).save(directory / "shadow_zone.png")
    first = Path(strength_files[-1]) if strength_files else None
    if first is not None and first.is_file():
        with Image.open(first) as image:
            image.convert("L").save(directory / "band.png")
    for path in dict.fromkeys(Path(f) for f in strength_files):
        with Image.open(path) as image:
            current = np.asarray(image.convert("L"), dtype=np.float32) / 255
        Image.fromarray(np.rint(np.maximum(current, extra) * 255).astype("uint8")).save(path)
    return str(directory / "room_shadow.png")


def prop_halo(strength, body, prop_sils, band):
    """0.3.12: запас вокруг предмета — с силой тела (как полоса тела), а не prop_strength: до 0.3.12 предмет лежал
    в полосе тела, а его собственная полоса (0.6 и ниже) почти не добавляла H3 свободы — «запас не работает».
    Сам предмет сохраняет prop_strength (форма модели), полоса вокруг него — полная."""
    import numpy as np
    import pose_proxy_v26 as proxy
    peak = float(np.max(body)) if float(np.max(body)) > 0 else 1.
    out = np.asarray(strength, np.float32)
    for sil in prop_sils:
        sil = np.asarray(sil, bool)
        if sil.any():
            out = np.maximum(out, np.where(sil, 0., proxy.strength_map(sil, peak, int(band)))).astype(np.float32)
    return out


def compose_props(directory, overlay_dir, size=(1344, 768), band=16):
    import numpy as np
    from PIL import Image
    import prop_render
    desc = read_json(overlay_dir / "props_render.json")
    shadow = None
    if desc.get("shadow") and Path(desc["shadow"]).is_file():
        # 0.3.0 «C»: cast shadow of the props (room light) darkens room/body under them in every guide.
        shadow = np.asarray(Image.open(desc["shadow"]).convert("L").resize(size, Image.Resampling.BILINEAR),
                            dtype=np.float32)[..., None] / 255.

    def shaded(image):
        if shadow is None:
            return image.convert("RGBA")
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) * (1. - shadow)
        return Image.fromarray(np.rint(rgb).clip(0, 255).astype("uint8")).convert("RGBA")
    # Props retain their source appearance in both guides. A grey prop video
    # otherwise contradicts material refs at every guided frame. Body proxy
    # remains unchanged; clay is still saved as a diagnostic render.
    for target, source in (("init.png", "textured_rgba"), ("playblast.png", "textured_rgba")):
        clean = Image.open(directory / target)
        base = shaded(clean)
        overlay = Image.open(desc[source]).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
        # 0.3.6/0.4.0: «предмет рисует H3». playblast (видео-референс и превью рендера до H3) — как у тела: с тенью
        # предмета из рендера; init — без тени Cycles (тень по правилу «Тени» добавит 3DA Place Pose).
        joint = base if target == "playblast.png" else clean.convert("RGBA")
        Image.alpha_composite(joint, overlay).convert("RGB").save(directory / ("h3_" + target))
        Image.alpha_composite(base, overlay).convert("RGB").save(directory / target)
    if (directory / "playblast_ns.png").is_file():
        overlay = Image.open(desc["textured_rgba"]).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
        Image.alpha_composite(Image.open(directory / "playblast_ns.png").convert("RGBA"), overlay).convert("RGB").save(
            directory / "h3_playblast_ns.png")
    body = np.asarray(Image.open(directory / "strength.png").convert("L"), dtype=float) / 255
    props = [(Image.open(p["silhouette"]).convert("L").resize(size, Image.Resampling.NEAREST),
              p["prop_strength"]) for p in desc["props"]]
    # 0.4.0: тот же запас (body_margin), что вокруг тела, — и вокруг предмета.
    strength = prop_render.combine_strength(body, props, dilation_px=int(band), gaussian_px=2 * int(band))
    strength = prop_halo(strength, body, [np.asarray(mask) > 127 for mask, _ in props], int(band))
    Image.fromarray(np.rint(strength * 255).astype("uint8")).save(directory / "strength.png")
    from prop_composite import repair_mask
    # Work from the actual quantized masks passed to Remote.
    quantized = np.rint(strength*255).astype('uint8')
    body_sil=np.asarray(Image.open(directory/'silhouette.png').convert('L'))>127
    prop_sil=np.logical_or.reduce([np.asarray(mask)>127 for mask,_ in props])
    repair = repair_mask(body,quantized,body_sil,prop_sil)
    source = shaded(Image.open(directory/'plate.png'))
    layer = Image.open(desc['textured_rgba']).convert('RGBA').resize(size,Image.Resampling.LANCZOS)
    Image.alpha_composite(source,layer).convert('RGB').save(directory/'prop_repair.png')
    if shadow is not None:
        Image.fromarray(np.rint(shadow[..., 0] * 255).astype("uint8")).save(directory / "prop_shadow.png")
    Image.fromarray(repair.astype('uint8')*255).save(directory/'prop_repair_mask.png')
    plate = np.asarray(Image.open(directory / "plate.png"), dtype=float)
    debug = plate * (1 - .45 * strength[..., None]) + np.array([255,70,30]) * (.45 * strength[..., None])
    Image.fromarray(np.rint(debug).astype("uint8")).save(directory / "debug.png")


def do_place(request, config):
    t0 = time.monotonic()
    timings = {}
    profile_path = required_file(request.get("location"), "location profile.json")
    profile_data = read_json(profile_path)
    plate_dir = assets_for(config, location_id(profile_data.get("loc_id")), profile_path.parent)
    pose = required_file(request.get("pose_ref"), "pose_ref")
    pc = request.get("pose_class")
    if pc not in DEFAULT_SUPPORT:
        raise ValueError("pose_class must be one of " + ", ".join(DEFAULT_SUPPORT))
    if not isinstance(request.get("target_object"), str) or not request["target_object"].strip():
        raise ValueError("target_object is required")
    height = float(request.get("height_m", 1.62))
    strength = float(request.get("strength", 1.0))
    if not math.isfinite(height) or height <= 0 or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("invalid height_m or strength")
    mode = request.get("proxy", "grey")
    if mode not in ("grey", "color"):
        raise ValueError("proxy must be grey or color")
    seed_value = request.get("seed", 101)
    if isinstance(seed_value, bool) or not isinstance(seed_value, int) or seed_value < 0:
        raise ValueError("seed must be a nonnegative integer")
    mannequin = required_file(config["mannequin"], "config.mannequin / FARM_MANNEQUIN")
    body_path = request.get("body_json")
    body = read_json(required_file(body_path, "body_json")) if body_path else None
    if body is not None:
        if not isinstance(body.get("body_dims"), dict):
            raise ValueError("body_json requires body_dims")
        for key in ("H", "L_thigh", "L_torso"):
            value = body["body_dims"].get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"body_json.body_dims.{key} must be finite and positive")
    if mode == "color" and not body:
        raise ValueError("color proxy requires body_json with outfit_colors")
    pf, profiles = modules(config)
    import numpy as np
    import place
    import refine
    from shot_spec import ShotSpec
    profile = profiles.LocationProfile(profile_path)
    require_profile_geometry(profile)
    spec = ShotSpec(id="pose", pose_class=pc, target_object=request["target_object"].strip(),
                    support_parts=parts(request.get("support_parts")) or DEFAULT_SUPPORT[pc],
                    instance_hint=request.get("instance_hint") or None,
                    secondary_contacts=contacts(request.get("secondary_contacts")))
    explicit = next((o for o in profile.data["objects"] if o["id"] == spec.target_object), None)
    if explicit is not None:
        if spec.instance_hint and spec.instance_hint != explicit["id"]:
            raise ValueError("target ID conflicts with instance_hint")
        spec.instance_hint = explicit["id"]
        spec.target_object = explicit.get("type", explicit.get("name", "other"))
    if pc != 'auto' and spec.target_object == 'floor' and profile.data.get("auto_surfaces", {}).get("floor_status") not in (None, "ok"):
        raise ValueError("auto profile floor is ambiguous; explicit floor preparation required before placement")
    seed = np.zeros(3) if pc == 'auto' else seed_for(profile, spec, config)
    # Validate required prompt metadata before spending GPU time.
    prompt_for(profile, config, {"shot_spec": spec.to_dict()})
    cache = Path(config.get("cache_dir", str(Path(config["farm_root"]) / "cache"))) / "pose_ref" / hashlib.sha256(pose.read_bytes()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    rf = cache / "pose.ref.npz"
    stage = time.monotonic()
    timings['reference_cache_hit'] = rf.is_file()
    if not rf.is_file():
        canonical = cache / "pose.png"
        from PIL import Image
        with Image.open(pose) as image:
            if image.format != "PNG":
                raise ValueError("pose_ref must be a PNG")
            image.convert("RGB").save(canonical)
        run_stage([config["geometry_python"], HERE / "posev2/ref_scene.py", "--comfy", config["comfy"],
                   "--out", cache, canonical], "pose MoGe + SAM 3D Body")
    required_file(rf, "SAM body mesh (no person / incomplete reference cache)")
    # 0.3.20: своё телосложение персонажа из body.json; с фото позы берутся только поза и кисти.
    import body_shape
    character_shape = body_shape.body_shape(body)
    timings["body_shape"] = "profile" if character_shape else "pose_photo"
    if character_shape:
        shaped = cache / ("shape_v1_" + body_shape.shape_hash(*character_shape)) / "pose.ref.npz"
        if not shaped.is_file():
            run_stage([config["geometry_python"], HERE / "posev2/body_shape.py", "--comfy", config["comfy"],
                       "--ref", rf, "--body", body_path, "--out", shaped], "character body shape (MHR)")
        rf = required_file(shaped, "character-shaped body mesh")
    ref = pf.load_ref(rf)
    if ref["F"].ndim != 2 or ref["F"].shape[1] != 3 or not len(ref["F"]) or not np.isfinite(ref["V"]).all():
        raise ValueError("SAM reference body mesh missing/invalid")
    ref.update(path=str(rf), body_cache_dir=str(Path(rf).parent / "body_geometry"), image_path=str(pose), vertical_python=config['python'])
    timings["reference_s"] = round(time.monotonic() - stage, 3)
    if (cache / 'pose.ref.json').is_file():
        timings['reference_stages'] = read_json(cache / 'pose.ref.json').get('timings', {})
    job = Path(config["farm_root"]) / "pack" / "jobs" / uuid.uuid4().hex
    job.mkdir(parents=True)
    plate = pf.Plate(plate_dir)
    stage = time.monotonic()
    import manual_pose
    key = manual_pose.placement_key(rf, profile, config, spec, height, seed, {"body": body, "random_seed": seed_value})
    cache_dir = Path(config.get("cache_dir", str(Path(config["farm_root"]) / "cache"))) / "placement" / key
    base = manual_pose.load_placement(cache_dir, key)
    cache_hit = base is not None
    if base is None:
        np.random.seed(seed_value % (2 ** 32))
        if pc == 'auto':
            import reference_place
            result = reference_place.place(ref, plate, profile, spec, height)
        else:
            refine_ctx = refine.load_ctx(config["comfy"])
            result = place.place(ref, plate, spec, seed, height, profile=profile,
                             body_src="profile" if body else "proxy", dims_override=body["body_dims"] if body else None,
                             refine_ctx=refine_ctx, trace_dir=job/'trace')
        manual_pose.save_placement(cache_dir, result, ref, plate, spec, key)
        base = manual_pose.load_placement(cache_dir, key)
        if base is None:
            raise ValueError(f"placement has no body geometry (status={result['status']})")
    values = {k: request.get(k, 0.) for k in ("body_dx", "body_dy", "body_dz", "body_dyaw", "body_dpitch", "body_droll")}
    if request.get('body_transform'):
        from scene_edit import body_delta
        values['body_matrix'] = body_delta(request['body_transform'], key)
    if 'body_matrix' in values or any(float(v) != 0 for v in values.values()):
        if pc == 'auto':
            import reference_place
            X, KP, fit = reference_place.revalidate(base, plate, profile, spec, values)
        else:
            X, KP, fit = manual_pose.revalidate(base, ref, plate, profile, spec, seed, height, values)
    else:
        X, KP, fit = base["X"], base["KP"], base["report"]
    # Advisory frame coverage of the projected 3D bbox, not fraction of vertices.
    import itertools
    import prop_render
    corners = np.array(list(itertools.product(*zip(X.min(axis=0), X.max(axis=0)))))
    u, v, z = plate.project(corners)
    fraction = prop_render.projected_bbox_fraction(np.column_stack((u / plate.Wp, v / plate.Hp, z)))
    placement_frame=dict(fit.get('frame',{}))
    fit['frame'] = {**placement_frame,'body': {'inside_fraction': fraction, 'out_of_frame': fraction < .95,
                                            'method':'advisory projected 3D AABB; not mesh coverage'}}
    flags = set(fit.get('flags', []))
    flags.discard('out_of_frame')
    flags.discard('body_out_of_frame')
    if fraction < .95:
        flags.add('body_out_of_frame')
    if not placement_frame.get('ok',True):
        flags.add('out_of_frame')
    fit['flags'] = sorted(flags)
    manual_pose.write_transformed(job / "fit", X, KP, ref["F"], fit)
    timings["placement_cache_hit"] = cache_hit
    timings["placement_s"] = round(time.monotonic() - stage, 3)
    fit = read_json(job / "fit/fit.json")
    fit["pose_text"] = request.get("pose_text", "")
    props_value = request.get("props", "[]")
    props_value = json.loads(props_value) if isinstance(props_value, str) else props_value
    if not isinstance(props_value, list):
        raise ValueError("props must be a JSON list")
    fit["props_prompt"] = []
    library = Path(config["farm_root"]) / "props"
    for prop in props_value:
        pid = location_id(prop.get("prop_id"))
        meta = read_json(required_file(library / pid / "prop.json", "prop metadata"))
        fit["props_prompt"].append(meta)
    save_json(job / "fit/fit.json", fit)
    required_file(job / "fit/body.obj", f"placement body.obj (status={fit.get('status')})")
    h3_prompt = prompt_for(profile, config, fit)
    body_h3_prompt = prompt_for(profile,config,{**fit,'props_prompt':[]}) if props_value else h3_prompt
    scene = {"name": "pose", "body_obj": str(job / "fit/body.obj"),
             "label_joints": str(job / "fit/label_joints.npz")}
    save_json(job / "render_spec.json", {"scenes": [scene]})
    stage = time.monotonic()
    env = os.environ.copy()
    env.setdefault("SCENE3D_BLENDER_SITE", str(Path(config["farm_root"]) / "runtime/blender_pkgs"))
    if config.get("blender_pkgs"):
        env["FARM_BPY_PKGS"] = config["blender_pkgs"]
    import raster
    size = raster.raster_for(request.get("megapixels", 0))
    timings["raster"] = list(size)
    run_stage(threeda_blender.command(config, HERE / "plate/render_plate_scene.py", ["--plate", plate_dir, "--spec", job / "render_spec.json",
               "--mannequin", mannequin, "--out", job / "render", "--label-pass", "--photo-emission", "2.5",
               "--samples", "48", "--raster", f"{size[0]}x{size[1]}"]), "Blender render", env=env)
    timings["render_s"] = round(time.monotonic() - stage, 3)
    directory = job / "render/pose"
    stage = time.monotonic()
    # Фото комнаты в полном разрешении (с 0.2.0 сохраняется при подготовке локации); старые локации — plate_rgb.
    source_plate = plate_dir / "plate_photo.png" if (plate_dir / "plate_photo.png").is_file() else plate_dir / "plate_rgb.png"
    # 0.3.9: настройки из «3D-контроля» (запрос) важнее config.yaml.
    def setting(name, default):
        value = request.get(name, config.get(name, default))
        return float(default if value is None else value)
    margin = setting("strength_margin", 2.0) * size[0] / 1344
    timings["strength_margin"] = round(margin, 3)
    prepare_images(directory, source_plate, mode, body_path, strength, config["cell_px"], size, margin)
    if props_value:
        from shutil import copy2
        body_dir=directory/'body_guides'
        body_dir.mkdir(exist_ok=True)
        for name in ('playblast.png','init.png','strength.png'):
            copy2(directory/name,body_dir/name)
    else:
        body_dir=directory
    body_guides={key:str(body_dir/(key+'.png')) for key in ('playblast','init','strength')}
    body_guides['plate']=str(directory/'plate.png')
    if (directory/'playblast_ns.png').is_file():
        body_guides['playblast_ns']=str(directory/'playblast_ns.png')
    # 0.3.7: тень тела из рендера (Cycles, свет комнаты) — для режима «тени от рендера».
    if (directory/'shadow.png').is_file():
        body_guides['body_shadow']=str(directory/'shadow.png')
    body_guides.update(prompt=body_h3_prompt,render_prompt=h3_prompt)
    render_props_value = []
    if props_value:
        # Input offsets use the same camera-horizontal location axes as body_*.
        render_props_value = []
        basis = manual_pose.location_basis(plate)
        for prop in props_value:
            prop = dict(prop)
            if 'absolute_matrix_world' not in prop:
                offset = basis @ np.asarray([prop.get(k, 0.) for k in "xyz"], dtype=float)
                prop.update(dict(zip("xyz", offset.tolist())))
            render_props_value.append(prop)
        # 0.3.4: stick по смещению вне кадра не пропадает из 3D-окна и рендера — придвигаем к персонажу.
        from prop_validation import keep_props_in_frame
        render_props_value, frame_notes = keep_props_in_frame(render_props_value, KP, plate.project, (plate.Wp, plate.Hp))
        fit["prop_frame_notes"] = frame_notes
        props_path = job / "props.json"
        save_json(props_path, render_props_value)
        overlay_dir = job / "props_render"
        run_stage(threeda_blender.command(config, HERE / "plate/prop_render.py", ["--plate", plate_dir,
                   "--body", job / "fit/body.obj", "--joints", job / "fit/label_joints.npz",
                   "--props", props_path, "--library", library, "--out", overlay_dir,
                   "--raster", f"{size[0]}x{size[1]}", "--spec", job / "render_spec.json",
                   "--photo-emission", "2.5", "--samples", "32"]), "Blender props", env=env)
        compose_props(directory, overlay_dir, size, max(int(config["cell_px"]), int(round(config["cell_px"] * margin))))
        prop_report = read_json(overlay_dir / "props_render.json")
        # 0.2.0 «е»: проникновение и опора предметов — те же правила, что у 3D-окна.
        import prop_checks
        for item in prop_report["props"]:
            points_path = item.get("points_world")
            if points_path and Path(points_path).is_file():
                prop_flags, prop_info = prop_checks.check(plate, np.load(points_path), X,
                    item.get("surface_intersection_pairs", {}).get("room", 0), depth_fn=pf.penetration)
                item["flags"].update(prop_flags)
                item["checks"] = prop_info
        from PIL import Image
        layer=Image.open(prop_report['textured_rgba']).convert('RGBA').resize(size,Image.Resampling.LANCZOS)
        layer.convert('RGB').save(directory/'prop_layer_rgb.png')
        layer.getchannel('A').save(directory/'prop_layer_alpha.png')
        body_guides.update(prop_rgb=str(directory/'prop_layer_rgb.png'),prop_alpha=str(directory/'prop_layer_alpha.png'))
        # 0.3.6: карты для режима «предмет рисует H3»: тело + предмет в одной сцене, сила с областью предмета,
        # промпт с предметом; вклейка и тень Cycles в этом режиме не применяются.
        body_guides['h3_props']=dict(playblast=str(directory/'h3_playblast.png'),init=str(directory/'h3_init.png'),
                                     strength=str(directory/'strength.png'),prompt=h3_prompt)
        if (directory/'h3_playblast_ns.png').is_file():
            body_guides['h3_props']['playblast_ns']=str(directory/'h3_playblast_ns.png')
        if (directory/'prop_shadow.png').is_file():
            body_guides['prop_shadow']=str(directory/'prop_shadow.png')
        fit["props"] = prop_report["props"]
        fit["flags"] = sorted(set(fit.get("flags", []) + [flag for prop in prop_report["props"]
                                    for flag, active in prop["flags"].items() if active]))
        from prop_validation import include_props
        include_props(fit)
        save_json(job / "fit/fit.json", fit)
    # 0.3.8: зона ожидаемых теней (тело + предметы) входит в область H3 с мягким краем.
    zone = shadow_zone(directory, [body_guides['strength'], directory / 'strength.png'] +
                       ([body_guides['h3_props']['strength']] if body_guides.get('h3_props') else []),
                       max(int(config["cell_px"]), int(round(config["cell_px"] * margin))),
                       setting("shadow_strength", .7), setting("render_shadow_opacity", .6))
    if zone:
        body_guides['room_shadow'] = zone
        if (directory / 'shadow_zone.png').is_file() and (directory / 'band.png').is_file():
            body_guides['shadow_zone'] = str(directory / 'shadow_zone.png')
            body_guides['band'] = str(directory / 'band.png')
    # Keep geometry diagnostics visible even in a saved preview, without painting
    # them into init/playblast (which are H3 references).
    from PIL import Image, ImageDraw, ImageFont
    preview = Image.open(directory / 'debug.png').convert('RGB')
    lines = [f"body out_of_frame={fraction < .95} bbox_inside={fraction:.4f}"]
    for p in fit.get('props', []):
        lines.append(f"prop {p['prop_id']}: source bbox {p.get('bbox_source_m')} m")
        lines.append(f"prop final bbox {p.get('bbox_final_m')} m; dims {p.get('final_dimensions_m')}")
        lines.append(f"prop out_of_frame={p['flags'].get('prop_out_of_frame')} inside={p.get('frame_inside_fraction')}")
    surface = fit.get('surface_match', {})
    if surface:
        lines.append(f"support={surface.get('support_surface_id')} soft={surface.get('soft_surface_ids')}")
    canvas = Image.new('RGB', (preview.width, preview.height + 24 * len(lines) + 10), '#141414')
    canvas.paste(preview)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, line in enumerate(lines):
        draw.text((8, preview.height + 5 + i * 24), line, fill='white', font=font)
    canvas.save(directory / 'debug.png')
    timings["maps_s"] = round(time.monotonic() - stage, 3)
    timings["total_s"] = round(time.monotonic() - t0, 3)
    from prop_validation import blocking_reasons, frame_warnings
    from reasons_ru import explain
    codes=blocking_reasons(fit)
    # Русские причины для GeometryRejected/статуса; коды — для тестов и логов. Кадр — предупреждение.
    reasons=(explain(fit) or codes) if codes else []
    fit['reasons_ru']=reasons
    fit['warnings']=frame_warnings(fit)+list(fit.get('prop_frame_notes') or [])
    if timings.get("body_shape") == "pose_photo":
        fit['warnings'].append('телосложение персонажа не снято — тело взято с фото позы («3DA · Персонаж: телосложение»)')
    save_json(job/'fit/fit.json',fit)
    body_guides['validation']=dict(ok=not codes,reasons=reasons,codes=codes,warnings=fit['warnings'])
    save_json(job/'body_guides.json',body_guides)
    return dict(status=fit["status"], **{key: str((directory / filename).resolve()) for key, filename in
                {"playblast": "playblast.png", "init": "init.png", "strength": "strength.png",
                 "silhouette": "silhouette.png", "debug": "debug.png"}.items()},
                prop_repair=str(directory/'prop_repair.png') if props_value else None,
                prop_repair_mask=str(directory/'prop_repair_mask.png') if props_value else None,
                body_guides=body_guides,
                scene_setup=dict(placement_key=key, body_obj=str(cache_dir/'body.obj'),
                                 body_cache=str(cache_dir/'manual_base.npz'),
                                 joints=str(cache_dir/'label_joints.npz'), location_id=profile.data['loc_id'],
                                 plate_dir=str(plate_dir), props=render_props_value),
                h3_prompt=h3_prompt, report=fit, timings=timings)


def do_body_profile(request, config):
    """0.3.20: телосложение персонажа по его фото (SAM 3D Body, медиана MHR) -> body.json (по флагу write)."""
    modules(config)
    import body_shape
    body_path = required_file(request.get("body_json"), "body_json")
    keys = sorted((k for k in request if re.fullmatch(r"photo_\d+", k)), key=lambda k: int(k[6:]))
    photos = [required_file(p, "photo") for p in (request.get("photos") or [request[k] for k in keys])]
    if not photos:
        raise ValueError("body profile: нужно хотя бы одно фото персонажа")
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in photos]
    work = Path(config.get("cache_dir", str(Path(config["farm_root"]) / "cache"))) / "body_profile" / \
        hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()[:16]
    work.mkdir(parents=True, exist_ok=True)
    # Имя в кэше — хэш содержимого: порядок и имена загрузок не влияют на результат.
    names = [h[:12] for h in hashes]
    todo = []
    for p, name in zip(photos, names):
        if not (work / (name + ".ref.npz")).is_file():
            shutil.copyfile(p, work / (name + ".png"))
            todo.append(work / (name + ".png"))
    if todo:
        run_stage([config["geometry_python"], HERE / "posev2/ref_scene.py", "--comfy", config["comfy"],
                   "--out", work, *todo], "body profile MoGe + SAM 3D Body")
    import numpy as np
    refs = [(f"фото {i + 1}", dict(np.load(work / (name + ".ref.npz"), allow_pickle=True)))
            for i, name in enumerate(names) if (work / (name + ".ref.npz")).is_file()]
    prof = body_shape.aggregate(refs)
    missing = len(photos) - prof["photos"]
    backup = body_shape.write_profile(body_path, prof, hashes) if request.get("write") else None
    sh = body_shape.shape_hash(prof["shape_params"], prof["scale_params"])
    text = (f"телосложение {sh}: фото с человеком {prof['photos']} из {len(photos)}"
            + (f" (без человека: {missing})" if missing else "")
            + "; разброс к медиане " + ", ".join(f"{k} {v}" for k, v in prof["spread"].items())
            + ("; записано в body.json, копия " + backup.name if backup else "; не записано (write выключен)"))
    return {"status": "ok" if not missing else "partial", "text": text, "shape_hash": sh,
            "written": bool(backup), "spread": prof["spread"]}


def error_result(exc):
    return {"status": "error", "error": stderr_tail(str(exc)),
            "stderr_tail": stderr_tail(getattr(exc, "diagnostic", str(exc)))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("location", "place", "body_profile"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--geometry-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prompt-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        request = read_json(args.request)
        config = config_for(request)
        if args.prompt_worker:
            from types import SimpleNamespace
            modules(config)
            text = prompt_direct(SimpleNamespace(data=request["profile_data"]), config, request["fit"])
            result = {"status": "ok", "h3_prompt": text}
        elif args.geometry_worker or (args.worker and args.command == "location"):
            result = {"location": do_location, "place": do_place,
                      "body_profile": do_body_profile}[args.command](request, config)
        else:
            # Node -> farm Python -> geometry Python for place+refine. Never import
            # torch/SAM into the farm or calling ComfyUI environment.
            executable = config["geometry_python"] if args.worker else config["python"]
            worker_flag = "--geometry-worker" if args.worker else "--worker"
            worker_out = Path(args.output).resolve().with_name(".pack-" + uuid.uuid4().hex + ".json")
            try:
                run_stage([*worker_python(config, executable), Path(__file__).resolve(), args.command,
                           "--request", Path(args.request).resolve(), "--output", worker_out, worker_flag], "farm worker")
                result = read_json(required_file(worker_out, "worker result"))
            finally:
                worker_out.unlink(missing_ok=True)
    except (Exception, SystemExit) as exc:
        result = error_result(exc)
    save_json(args.output, result)
    # Node errors are a normal JSON outcome, not an exception that aborts the graph.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
