"""Seed-free, CPU-only draft surfaces for a NEW location (09.11e, item 0e).

No furniture semantics, soft materials, scale anchor or backrests are inferred.
Existing profiles are read-only: write_auto_profile requires an empty directory.
Numeric tolerances are mandatory placement.yaml values, not code fallbacks.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from placement_config import section
import top_region as tr


def _config(config):
    cfg = section("auto_surfaces") if config is None else config
    for key in ("height_bin_m", "min_area_m2"):
        if isinstance(cfg[key], bool) or not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"auto_surfaces.{key} must be a positive finite number")
    return cfg


def surfaces_from_grid(origin, height, cam_xy, *, config=None):
    """Return (TopRegion list, diagnostics) from the existing height_proxy grid.

    Four-connected components in anchored, disjoint height bins. Missing cells
    and holes remain missing: no fabricated bridges, floors or object footprints.
    The floor must be BOTH uniquely largest and uniquely lowest. If fixed-bin
    boundaries fragment one plane, compare observed coplanar unions using the
    same height tolerance; never fill gaps or join different height bands.
    """
    cfg = _config(config)
    height = np.asarray(height, float)
    origin = np.asarray(origin, float)
    cam_xy = np.asarray(cam_xy, float)
    if height.ndim != 2 or origin.shape != (2,) or cam_xy.shape != (2,):
        raise ValueError("expected 2D height grid and 2D origin/cam_xy")
    if not np.isfinite(origin).all() or not np.isfinite(cam_xy).all():
        raise ValueError("origin/cam_xy must be finite")
    valid = np.isfinite(height)
    bins = np.zeros(height.shape, dtype=np.int64)
    bins[valid] = np.floor(height[valid] / cfg["height_bin_m"]).astype(np.int64)
    regions, counts, heights = [], [], []
    rejected = 0
    for bin_id in np.unique(bins[valid]):
        labels, n = ndimage.label(valid & (bins == bin_id))  # 4-neighbours
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        for label in range(1, n + 1):
            if sizes[label] * tr.CELL ** 2 < cfg["min_area_m2"]:
                rejected += 1
                continue
            mask = labels == label
            ij = np.argwhere(mask)
            h0 = float(np.median(height[mask]))
            center_xy = origin + (ij.mean(0) + .5) * tr.CELL
            xy = origin + (ij + .5) * tr.CELL
            seed = ij[np.argmin(np.linalg.norm(xy - center_xy, axis=1))]
            dt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * tr.CELL
            regions.append(tr.TopRegion("other", "auto_height_proxy", origin.copy(), height.shape,
                np.where(mask, height, np.nan), mask, dt, h0, tuple(int(x) for x in seed), cam_xy.copy(),
                object_mask=mask.copy(), top_diagnostics={"method": "auto_height_component",
                    "height_bin": int(bin_id), "height_bin_m": cfg["height_bin_m"],
                    "footprint_scope": "observed surface component, not a semantic object"}))
            counts.append(int(sizes[label]))
            heights.append(h0)
    largest = np.flatnonzero(np.asarray(counts) == max(counts)) if counts else np.array([], int)
    lowest = np.flatnonzero(np.asarray(heights) == min(heights)) if heights else np.array([], int)
    floor = int(largest[0]) if len(largest) == len(lowest) == 1 and largest[0] == lowest[0] else None
    component_count = len(regions)
    floor_components = 1 if floor is not None else 0
    if floor is None and regions:
        # Absolute bin boundaries (especially zero) can split a noisy floor.
        # Cluster retained components by a bounded median-height range, not
        # transitive chaining, so stairs cannot grow into one floor plane.
        bands = []
        for index in np.argsort(heights, kind='stable'):
            if not bands or heights[index] - heights[bands[-1][0]] > cfg['height_bin_m']:
                bands.append([])
            bands[-1].append(int(index))
        areas = [sum(counts[i] for i in band) for band in bands]
        dominant = np.flatnonzero(np.asarray(areas) == max(areas))
        if len(dominant) == 1 and dominant[0] == 0 and len(bands[0]) > 1:
            members = bands[0]
            seed_region = regions[max(members, key=lambda i: counts[i])]
            mask = np.logical_or.reduce([regions[i].mask for i in members])
            dt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * tr.CELL
            merged = tr.TopRegion('floor', 'auto_height_proxy', origin.copy(), height.shape,
                np.where(mask, height, np.nan), mask, dt, float(np.median(height[mask])),
                seed_region.seed_cell, cam_xy.copy(), object_mask=mask.copy(),
                top_diagnostics={'method': 'observed_coplanar_floor_union',
                    'component_count': len(members), 'height_bin_m': cfg['height_bin_m'],
                    'component_medians_m': [heights[i] for i in members],
                    'footprint_scope': 'union of observed cells only; holes and occlusions remain missing'})
            regions = [merged] + [r for i, r in enumerate(regions) if i not in members]
            counts = [int(r.mask.sum()) for r in regions]
            heights = [r.h0 for r in regions]
            largest = np.flatnonzero(np.asarray(counts) == max(counts))
            lowest = np.flatnonzero(np.asarray(heights) == min(heights))
            floor, floor_components = 0, len(members)
    k = 0
    for i, reg in enumerate(regions):
        if i == floor:
            reg.object_id, reg.target_object = "floor_1", "floor"
        else:
            k += 1
            reg.object_id = f"surf_{k}"
    report = {"method": "height_proxy_connected_bins", "cell_m": tr.CELL,
              "height_bin_m": cfg["height_bin_m"], "min_area_m2": cfg["min_area_m2"],
              "surface_count": len(regions), "surf_count": k, "rejected_components": rejected,
              "observed_component_count": component_count, "floor_component_count": floor_components,
              "floor_status": "no_surfaces" if not regions else "ok" if floor is not None else "floor_ambiguous",
              "floor_id": None if floor is None else "floor_1",
              "largest_candidates": [regions[i].object_id for i in largest],
              "lowest_candidates": [regions[i].object_id for i in lowest]}
    if regions and floor is None:
        report["reason"] = "floor requires the same uniquely largest AND uniquely lowest observed plane; no choice invented"
    return regions, report


def discover_surfaces(plate, *, config=None):
    """Use the existing height_proxy sampling/rasterization, without spatial seeds."""
    cfg = _config(config)
    try:
        origin, _, height = tr._grid(plate)
    except ValueError as exc:
        # _grid cannot reduce an empty horizontal sample cloud.
        samples = tr._horizontal_samples(plate)
        if len(samples):
            raise
        raise ValueError("auto_surfaces: no horizontal height_proxy samples") from exc
    return surfaces_from_grid(origin, height, plate.C[:2], config=cfg)


def write_auto_profile(loc_id, plate, directory, *, config=None):
    """Write NEW profile.json + surface/wall NPZs; return Path to profile.json.

    directory is the profile directory itself, NOT a FARM_ROOT. No existing
    profile/assets are overwritten. scale remains explicitly unverified.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"auto_surfaces requires a new/empty profile directory: {directory}")
    regions, report = discover_surfaces(plate, config=config)
    walls = tr.walls(plate)  # existing vertical-plane RANSAC, unchanged
    directory.mkdir(parents=True, exist_ok=True)
    objects = []
    for reg in regions:
        oid = reg.object_id
        tr.save_npz(reg, directory / f"{oid}_top.npz")
        tr.mask_png(reg, directory / f"{oid}_mask.png")
        points = reg.points3d()
        center = points.mean(0)
        seed = np.r_[reg.center(*reg.seed_cell), reg.height[reg.seed_cell]]
        objects.append({"id": oid, "name": reg.target_object, "type": reg.target_object,
                        "soft": False, "backrest_name": None, "backrest_npz": None,
                        "h0": reg.h0, "area_m2": float(reg.mask.sum()) * tr.CELL ** 2,
                        "center_m": center.tolist(), "seed_world": seed.tolist(),
                        "top_grid_npz": f"{oid}_top.npz", "mask_png": f"{oid}_mask.png",
                        "flags": ["unclassified_surface"] if oid != "floor_1" else []})
    for k, wall in enumerate(walls, 1):
        oid = f"wall_{k}"
        np.savez_compressed(directory / f"{oid}_pts.npz", P=wall["pts"])
        objects.append({"id": oid, "name": "wall", "type": "wall", "soft": False,
                        "backrest_name": None, "plane": {"n": wall["n"].tolist(), "d": wall["d"]},
                        "area_m2": wall["area_m2"], "lat_m": wall["lat"], "z_m": wall["z"],
                        "center_m": np.asarray(wall["center"]).tolist(), "pts_npz": f"{oid}_pts.npz",
                        "flags": [] if wall["z"][1] >= section("zone")["wall_min_top"] else ["low_plane"]})
    report["wall_count"] = len(walls)
    profile = {"loc_id": loc_id, "version": 2, "reviewed": False,
               "scale": {"factor": 1.0, "seat_heights_m": [], "scale_suspect": True,
                         "source": "inherited plate metric geometry; not independently verified"},
               "objects": objects, "anchors": [], "substitutes": {}, "auto_surfaces": report}
    path = directory / "profile.json"
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def _edges(reg):
    """World XYZ line segments along the actual cell union, including hole edges."""
    mask = reg.mask
    padded = np.pad(mask, 1)
    edges = []
    # Offset endpoints and neighbour index for each of the four cell sides.
    for di, dj, a, b in ((-1, 0, (0, 0), (0, 1)), (1, 0, (1, 0), (1, 1)),
                         (0, -1, (0, 0), (1, 0)), (0, 1, (0, 1), (1, 1))):
        neighbour = padded[1 + di:1 + di + mask.shape[0], 1 + dj:1 + dj + mask.shape[1]]
        ij = np.argwhere(mask & ~neighbour)
        z = reg.height[ij[:, 0], ij[:, 1]]
        p = np.column_stack([reg.origin + (ij + a) * tr.CELL, z])
        q = np.column_stack([reg.origin + (ij + b) * tr.CELL, z])
        edges.append(np.stack([p, q], axis=1))
    return np.concatenate(edges)


def render_objects_preview(plate, profile_path, output_path):
    """PNG plate overlay: exact surface contours + RANSAC wall rectangles + IDs.

    Reads either an auto or legacy profile without modifying it. One neutral
    high-contrast ink; identity is in direct labels, never in cycling colours.
    Returns Path. The caller loads this PNG as ComfyUI IMAGE.
    """
    profile_path, output_path = Path(profile_path), Path(output_path)
    if output_path.resolve() == profile_path.resolve():
        raise ValueError("objects_preview must not overwrite the profile")
    if plate.rgb is None:
        raise ValueError("objects_preview requires plate.rgb")
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    im = plate.rgb.convert("RGB").resize((plate.Wp, plate.Hp))
    draw = ImageDraw.Draw(im)
    occupied = []
    for obj in data["objects"]:
        if "top_grid_npz" in obj:
            reg = tr.load_npz(profile_path.parent / obj["top_grid_npz"])
            segments = _edges(reg)
        elif obj.get("type") == "wall":
            n = np.asarray(obj["plane"]["n"], float)
            lat = np.cross([0., 0., 1.], n)
            base = -obj["plane"]["d"] * n
            corners = np.array([base + t * lat + [0, 0, z] for t, z in
                ((obj["lat_m"][0], obj["z_m"][0]), (obj["lat_m"][1], obj["z_m"][0]),
                 (obj["lat_m"][1], obj["z_m"][1]), (obj["lat_m"][0], obj["z_m"][1]))])
            segments = np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)
        else:
            continue
        if not len(segments):
            continue
        u, v, z = plate.project(segments.reshape(-1, 3))
        projected = np.column_stack([u, v]).reshape(-1, 2, 2)
        keep = np.isfinite(projected).all(axis=(1, 2)) & (z.reshape(-1, 2) > 0).all(axis=1)
        # Discard entirely off-screen segments before asking Pillow to clip.
        keep &= (projected[:, :, 0].max(1) >= 0) & (projected[:, :, 0].min(1) < plate.Wp)
        keep &= (projected[:, :, 1].max(1) >= 0) & (projected[:, :, 1].min(1) < plate.Hp)
        projected = projected[keep]
        if not len(projected):
            continue
        for segment in projected:
            pts = [tuple(p) for p in segment]
            draw.line(pts, fill="black", width=4)
            draw.line(pts, fill="white", width=2)
        # Use visible contour, not centroid (which may project into an empty hole).
        position = projected.reshape(-1, 2).mean(0)
        label = obj["id"]
        box = draw.textbbox((0, 0), label)
        w, h = box[2] - box[0] + 8, box[3] - box[1] + 8
        x = int(np.clip(position[0], 0, max(0, plate.Wp - w)))
        y = int(np.clip(position[1], 0, max(0, plate.Hp - h)))
        for candidate_y in range(y, max(y + 1, plate.Hp - h + 1), h + 2):
            rect = (x, candidate_y, x + w, candidate_y + h)
            if not any(rect[0] < b[2] and rect[2] > b[0] and rect[1] < b[3] and rect[3] > b[1] for b in occupied):
                y = candidate_y
                break
        occupied.append((x, y, x + w, y + h))
        draw.rectangle(occupied[-1], fill="black", outline="white", width=1)
        draw.text((x + 4, y + 4 - box[1]), label, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path, format="PNG")
    return output_path
