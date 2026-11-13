"""Prop overlay renderer; no changes to the existing body renderer.

Blender CLI (trusted script, never run from the asset directory):
  blender -b --factory-startup --disable-autoexec --python prop_render.py --
    --plate DIR --body body.obj --joints label_joints.npz --props props.json
    --library farm/props --out DIR

Python API inside existing bpy scene:
  render_props(scene, props, anchors_world, library_dir, out_dir,
               body_objects=None, room_objects=None) -> paths and diagnostics.
Axes are location-world XYZ, Euler XYZ=(roll,pitch,yaw); world anchor is pelvis.
CPU helpers are importable without bpy. Collisions use world-space BVH surface intersections.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import sys

try:
    from ..import_props_blender import normalization_transform
except ImportError:
    # Direct Blender CLI: only add the trusted repository script directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from import_props_blender import normalization_transform


def projected_bbox_fraction(coords):
    """CPU helper: fraction of projected bbox inside normalized frame [0,1]^2.

    coords is an iterable of (x, y, positive_camera_depth), e.g. Blender's
    world_to_camera_view output or (u/width, v/height, depth). This measures the
    projected axis-aligned rectangle, NOT triangle coverage or occlusion.
    Any corner on/behind the camera or a zero-area bbox returns 0 (conservative;
    near-plane clipping is not attempted). Malformed/nonfinite input is rejected.
    Use fraction < .95 for the advisory out_of_frame flag (body or prop).
    """
    points = [tuple(float(v) for v in p) for p in coords]
    if not points or any(len(p) != 3 for p in points):
        raise ValueError("expected nonempty projected points (N,3)")
    if not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("projected points must be finite")
    if any(p[2] <= 0 for p in points):
        return 0.0
    x0, x1 = min(p[0] for p in points), max(p[0] for p in points)
    y0, y1 = min(p[1] for p in points), max(p[1] for p in points)
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return 0.0
    # Divide the axes separately to avoid unnecessary area under/overflow.
    ratio_x = max(0.0, min(x1, 1.0) - max(x0, 0.0)) / width
    ratio_y = max(0.0, min(y1, 1.0) - max(y0, 0.0)) / height
    return min(1.0, ratio_x * ratio_y)


def anchors_from_mhr70(kp):
    import numpy as np
    k = np.asarray(kp, dtype=float)
    if k.ndim != 2 or k.shape[0] < 70 or k.shape[1] != 3 or not np.isfinite(k).all():
        raise ValueError("expected finite MHR70 world joints (>=70,3)")
    pelvis = (k[9] + k[10]) / 2
    neck = k[69]
    return {"world": pelvis.tolist(), "char_pelvis": pelvis.tolist(),
            "char_head": ((k[3] + k[4]) / 2).tolist(),
            "char_chest": (pelvis * .35 + neck * .65).tolist(),
            "char_hand_l": k[62].tolist(), "char_hand_r": k[41].tolist()}


def prop_transform(prop, anchors_world):
    """Return XYZ translation, XYZ Euler radians, scale; never manipulate bones."""
    anchor = prop.get("anchor", "world")
    anchor = "char_pelvis" if anchor == "world" else anchor
    if anchor not in anchors_world:
        raise ValueError(f"missing anchor {anchor}")
    xyz = [float(anchors_world[anchor][i]) + float(prop.get(axis, 0)) for i, axis in enumerate("xyz")]
    rotation = [math.radians(float(prop.get(axis, 0))) for axis in ("roll", "pitch", "yaw")]
    scale = float(prop.get("scale", 1))
    if not all(math.isfinite(v) for v in xyz + rotation + [scale]) or scale <= 0:
        raise ValueError("prop transform must be finite, scale > 0")
    return xyz, rotation, scale


def combine_strength(body_strength, prop_masks, *, dilation_px=16, gaussian_px=32):
    """max(body, existing compact pose_proxy_v26 strength_map); float32 [0,1].

    body_strength already has the existing body edge treatment: never blur it twice.
    prop_masks is iterable of (PIL mask or HxW array [0,1]/uint8, strength).
    Pixel units refer to the supplied full-resolution image.
    """
    import numpy as np
    from PIL import Image
    try:
        from .pose_proxy_v26 import strength_map
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pose_proxy_v26 import strength_map
    if gaussian_px != 2 * dilation_px:
        raise ValueError("existing compact strength edge requires gaussian_px == 2*dilation_px")
    result = np.asarray(body_strength, dtype=np.float32).copy()
    if result.ndim != 2 or not np.isfinite(result).all() or result.min(initial=0) < 0 or result.max(initial=0) > 1:
        raise ValueError("body strength must be finite HxW [0,1]")
    for mask, strength in prop_masks:
        strength = float(strength)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("prop strength must be [0,1]")
        if isinstance(mask, Image.Image):
            image = mask.convert("L")
        else:
            a = np.asarray(mask)
            a = a if a.dtype == np.uint8 else np.clip(a * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(a).convert("L")
        if image.size != (result.shape[1], result.shape[0]):
            raise ValueError("prop silhouette and body strength sizes differ")
        silhouette = np.asarray(image) >= 128
        if strength > 0 and silhouette.any():
            result = np.maximum(result, strength_map(silhouette, strength, cell_px=dilation_px)).astype(np.float32)
    return result


def aabb_overlap(a, b):
    return all(a[0][i] <= b[1][i] and b[0][i] <= a[1][i] for i in range(3))


def _bounds(objects):
    from mathutils import Vector
    pts = [o.matrix_world @ Vector(c) for o in objects for c in o.bound_box]
    return [[min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]]


def _world_bvh(objects):
    from mathutils.bvhtree import BVHTree
    vertices,faces=[],[]
    for ob in objects:
        offset=len(vertices)
        vertices.extend(ob.matrix_world@v.co for v in ob.data.vertices)
        faces.extend(tuple(offset+i for i in p.vertices) for p in ob.data.polygons)
    return BVHTree.FromPolygons(vertices,faces) if faces else None


def load_props(scene, props, anchors_world, library_dir):
    import bpy
    from mathutils import Euler, Matrix, Vector
    library = Path(library_dir).resolve(strict=True)
    groups = []
    for index, prop in enumerate(props):
        ident = prop["prop_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
            raise ValueError("invalid prop_id")
        directory = (library / ident).resolve(strict=True)
        if not directory.is_relative_to(library):
            raise ValueError("prop path escapes library")
        metadata = json.loads((directory / "prop.json").read_text(encoding="utf-8"))
        path = (directory / metadata.get("model", "model.glb")).resolve(strict=True)
        if not path.is_relative_to(directory) or path.suffix.lower() != ".glb":
            raise ValueError("only local normalized GLB cache is accepted")
        strength = float(prop.get("prop_strength", metadata.get("prop_strength", .35)))
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("prop_strength must be [0,1]")
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(path))
        imported = [o for o in scene.objects if o not in before]
        meshes = [o for o in imported if o.type == "MESH"]
        if not meshes:
            raise ValueError("normalized prop has no mesh")
        root = bpy.data.objects.new(f"prop_{index}_{ident}", None)
        scene.collection.objects.link(root)
        for obj in imported:
            obj.animation_data_clear()
            if obj.parent not in imported:
                obj.parent = root
            obj.pass_index = index + 10
        bpy.context.view_layer.update()
        cache_bbox = _bounds(meshes)
        # Runtime size is independent of source units and of the cache's target size.
        # Missing Prop size means .30 m, including for legacy metre-sized caches.
        size = float(prop.get("size_m", .30))
        xyz, rotation, scale = prop_transform(prop, anchors_world)
        centre, factor, scaled_bbox = normalization_transform(cache_bbox, size_m=size, scale=scale)
        if 'absolute_matrix_world' in prop:
            from prop_gizmo import validate_matrix
            root.matrix_world = Matrix(validate_matrix(prop['absolute_matrix_world']).tolist())
            # Absolute transform acts on the original GLB. Numeric size/anchor
            # remain serializable for legacy mode but do not act a second time.
            factor = None
        else:
            root.matrix_world = (Matrix.Translation(Vector(xyz)) @ Euler(rotation, "XYZ").to_matrix().to_4x4()
                                 @ Matrix.Scale(factor, 4) @ Matrix.Translation(-Vector(centre)))
        bpy.context.view_layer.update()
        facts = metadata.get("import_facts", {})
        source_bbox = facts.get("source_bbox_m")
        if source_bbox is None and "dimensions_m" in facts and "source_bbox_center_m" in facts:
            # Importer v1 facts retain the original metres even though cache was not resized.
            centre_source, dims = facts["source_bbox_center_m"], facts["dimensions_m"]
            source_bbox = [[centre_source[i] + sign * dims[i] / 2 for i in range(3)] for sign in (-1, 1)]
        final_bbox = _bounds(meshes)
        vertex_probe=[]
        for mesh in meshes:
            verts=mesh.data.vertices
            step=max(1,(len(verts)+511)//512)
            vertex_probe.extend([list(mesh.matrix_world@v.co) for v in list(verts)[::step]])
        groups.append({"id": ident, "index": index, "meshes": meshes, "root": root,
                       "strength": strength, "bbox_world": final_bbox,
                       "vertex_probe_world":vertex_probe,"vertex_probe_complete":all(len(m.data.vertices)<=512 for m in meshes),
                       "absolute_matrix_world":prop.get('absolute_matrix_world'),
                       "size_m": size, "scale": scale, "normalization_factor": factor,
                       "source_bbox_m": source_bbox, "cache_bbox_m": cache_bbox,
                       "scaled_bbox_m": scaled_bbox, "final_bbox_m": final_bbox,
                       "source_dimensions_m": ([source_bbox[1][i] - source_bbox[0][i] for i in range(3)]
                                               if source_bbox is not None else None),
                       "final_dimensions_m": [final_bbox[1][i] - final_bbox[0][i] for i in range(3)]})
    return groups


def _material(name, kind, color=(.7, .7, .7, 1)):
    import bpy
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    shader = nt.nodes.new(kind)
    if "Color" in shader.inputs:
        shader.inputs["Color"].default_value = color
    nt.links.new(shader.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    return mat


@contextmanager
def _materials(scene, replacements):
    saved = []
    for ob, material in replacements.items():
        if not ob.material_slots:
            saved.append((ob, None))
            ob.data.materials.append(material)
        else:
            values = [(slot.link, slot.material) for slot in ob.material_slots]
            saved.append((ob, values))
            for slot in ob.material_slots:
                slot.link, slot.material = "OBJECT", material
    try:
        yield
    finally:
        for ob, values in saved:
            if values is None:
                ob.data.materials.clear()
            else:
                for slot, (link, material) in zip(ob.material_slots, values):
                    slot.material, slot.link = material, link


def sun_vector(camera_matrix, azimuth_deg, elevation_deg):
    """Direction TO the key light, as render_plate_scene.render_playblast: azimuth 0 = from the camera,
    +90 = from the right of the frame, 180 = from behind the subject. CPU helper (no bpy)."""
    import numpy as np
    m = np.asarray(camera_matrix, dtype=float)
    fwd = -m[:3, 2]
    fwd = np.array([fwd[0], fwd[1], 0.])
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    right = np.array([fwd[1], -fwd[0], 0.])
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    v = (-fwd * math.cos(az) + right * math.sin(az)) * math.cos(el) + np.array([0., 0., math.sin(el)])
    return v / np.linalg.norm(v)


def room_light(scene, room, photo, light, photo_emission=2.5):
    """0.3.0 «C»: the prop is lit by the SAME light as the body playblast — sun from the resolved room light
    (plate.json / spec light, else the brightest area of the photo), world ambient in the light colour and the photo
    re-emitted from a copy of the room mesh (image-based bounce light). Returns (resolved light, created objects)."""
    import bpy
    from mathutils import Matrix, Vector
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import render_plate_scene as renderer
    L = renderer.resolve_light(light, str(photo))
    cam = scene.camera
    to_light = Vector(sun_vector([list(r) for r in cam.matrix_world], L["azimuth_deg"], L["elevation_deg"]).tolist())
    data = bpy.data.lights.new("prop_sun", "SUN")
    data.energy, data.color, data.angle = L["strength"], L["color"][:3], math.radians(L["softness_deg"])
    sun = bpy.data.objects.new("prop_sun", data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = to_light.to_track_quat("Z", "Y").to_euler()
    created = [sun]
    world = scene.world or bpy.data.worlds.new("prop_world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    amb = L.get("ambient_color", L["color"])
    bg.inputs["Color"].default_value = (amb[0], amb[1], amb[2], 1)
    bg.inputs["Strength"].default_value = L["ambient"] * (0.3 if photo_emission > 0 else 1.0)
    if photo_emission > 0 and room is not None:
        em = bpy.data.materials.new("prop_plate_emit")
        em.use_nodes = True
        nt = em.node_tree
        nt.nodes.clear()
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(photo))
        tex.extension = "EXTEND"
        emi = nt.nodes.new("ShaderNodeEmission")
        emi.inputs["Strength"].default_value = photo_emission
        nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Window"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
        nt.links.new(emi.outputs["Emission"], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
        emitter = room.copy()
        emitter.data = room.data.copy()
        scene.collection.objects.link(emitter)
        emitter.visible_camera = emitter.visible_shadow = False
        c = cam.matrix_world.translation
        emitter.matrix_world = Matrix.Translation(c) @ Matrix.Scale(1.01, 4) @ Matrix.Translation(-c) @ room.matrix_world
        emitter.data.materials.clear()
        emitter.data.materials.append(em)
        created.append(emitter)
    L["photo_emission"] = photo_emission
    return L, created


def shadow_alpha(rgba, threshold=.06, opacity=.85):
    """Shadow-catcher alpha -> cast-shadow opacity: drop catcher noise, scale. CPU helper (numpy)."""
    import numpy as np
    a = np.asarray(rgba, dtype=np.float32)[..., 3] / 255.
    return np.where(a < threshold, 0., a) * opacity


def render_props(scene, props, anchors_world, library_dir, out_dir, *, body_objects=None, room_objects=None,
                 light=None, samples=32):
    """Keep props in scene for optional unified depth render; restore original materials.

    Overlay RGBA is depth occluded by *all* existing mesh geometry (body and room),
    not a 2D stamp. Per-prop alpha is occluded also by other props. No contact solver.
    EEVEE only, 3+N bounded passes. No promise of <=15s on unbounded high-poly assets.
    0.3.0: light=(resolved room light, emitter objects) → the textured pass is Cycles in the room light
    (own shadows of the prop, body shadows on it) and one more pass renders the prop's CAST shadow on the
    room and the body (both shadow catchers, they cast nothing themselves) -> props_shadow.png.
    """
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    import numpy as np
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lit, emitters = (light if light is not None else (None, []))
    old_meshes = [o for o in scene.objects if o.type == "MESH" and not o.hide_render and o not in emitters]
    body_objects = list(body_objects or [])
    room_objects = list(room_objects or [o for o in old_meshes if o not in body_objects])
    groups = load_props(scene, props, anchors_world, library_dir)
    prop_meshes = [o for g in groups for o in g["meshes"]]
    holdout = _material("prop_occluder", "ShaderNodeHoldout")
    clay = _material("prop_clay", "ShaderNodeBsdfDiffuse")
    white = _material("prop_mask", "ShaderNodeEmission", (1, 1, 1, 1))
    saved = (scene.render.engine, scene.render.film_transparent, scene.render.filepath,
             scene.render.image_settings.file_format, scene.render.image_settings.color_mode,
             scene.render.image_settings.color_depth, bpy.context.view_layer.material_override,
             scene.view_settings.view_transform)
    def render(path, replacements, raw=False):
        scene.render.filepath = str(path)
        scene.view_settings.view_transform = "Raw" if raw else saved[-1]
        with _materials(scene, replacements):
            bpy.ops.render.render(write_still=True)
        return str(path)
    room_bvh,body_bvh=_world_bvh(room_objects),_world_bvh(body_objects)
    result = {"props": [], "diagnostic_method": "world BVH surface intersections; AABB recorded separately; projected bbox area ratio",
              "diagnostic_limits": "Complete closed-volume containment without surface crossing is not resolved; "
                                   "frame fraction measures projected bbox, not visible triangle coverage; "
                                   "any corner on/behind camera conservatively returns zero. Diagnostic only."}
    try:
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        bpy.context.view_layer.material_override = None
        occluders = {o: holdout for o in old_meshes}
        if lit is not None:
            # Cycles in the room light: own (self) shadows; the body/room holdouts still block the sun.
            renderer = sys.modules.get("render_plate_scene")
            scene.render.engine = "CYCLES"
            device = renderer.use_gpu(scene) if renderer else "CPU"
            scene.cycles.samples = int(samples)
            scene.cycles.use_denoising = True
            scene.cycles.denoiser = "OPTIX" if device == "OPTIX" else "OPENIMAGEDENOISE"
            for o in room_objects:
                o.visible_shadow = False  # noisy photo relief must not shade the sun (as the body playblast)
        result["textured_rgba"] = render(out / "props_textured_rgba.png", occluders)
        if lit is not None:
            # Cast shadow of the props only: room + body are catchers that cast nothing; props are unseen by the camera.
            catcher = _material("prop_catcher", "ShaderNodeBsdfDiffuse", (.6, .6, .6, 1))
            state = [(o, o.is_shadow_catcher, o.visible_shadow) for o in old_meshes]
            seen = [(o, o.visible_camera) for o in prop_meshes]
            try:
                for o in old_meshes:
                    o.is_shadow_catcher, o.visible_shadow = True, False
                for o in prop_meshes:
                    o.visible_camera = False
                shadow_png = render(out / "props_shadow_rgba.png", {o: catcher for o in old_meshes})
            finally:
                for o, catch, cast in state:
                    o.is_shadow_catcher, o.visible_shadow = catch, cast
                for o, value in seen:
                    o.visible_camera = value
            image = bpy.data.images.load(shadow_png, check_existing=False)
            px = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(px)
            w, h = image.size
            bpy.data.images.remove(image)
            rgba = (px.reshape(h, w, 4)[::-1] * 255)
            from PIL import Image as PILImage
            PILImage.fromarray(np.rint(shadow_alpha(rgba) * 255).clip(0, 255).astype(np.uint8), "L").save(out / "props_shadow.png")
            result["shadow"] = str(out / "props_shadow.png")
            result["light"] = {k: lit[k] for k in ("azimuth_deg", "elevation_deg", "strength", "color", "ambient",
                                                   "softness_deg", "photo_emission") if k in lit}
            scene.render.engine = "BLENDER_EEVEE"
        result["clay_rgba"] = render(out / "props_clay_rgba.png", {**occluders, **{o: clay for o in prop_meshes}})
        for g in groups:
            visible = {o: holdout for o in prop_meshes}
            visible.update({o: white for o in g["meshes"]})
            rgba = render(out / f"prop_{g['index']:03}_rgba.png", {**occluders, **visible}, raw=True)
            mask = out / f"prop_{g['index']:03}_silhouette.png"
            image = bpy.data.images.load(rgba, check_existing=False)
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
            pixels = pixels.reshape(-1, 4)
            alpha = pixels[:, 3].copy()
            pixels[:, :3] = alpha[:, None]
            pixels[:, 3] = 1
            image.pixels.foreach_set(pixels.ravel())
            image.filepath_raw, image.file_format = str(mask), "PNG"
            image.save()
            bpy.data.images.remove(image)
            prop_bvh=_world_bvh(g['meshes'])
            intersections={name:len(prop_bvh.overlap(tree)) if tree is not None else 0
                           for name,tree in [('room',room_bvh),('body',body_bvh)]}
            aabb={name:any(aabb_overlap(g['bbox_world'],_bounds([o])) for o in objects)
                  for name,objects in [('room',room_objects),('body',body_objects)]}
            flags = {"prop_hits_room": intersections['room']>0, "prop_hits_body": intersections['body']>0}
            # 0.2.0 «е»: world points (= posev2 world, OBJ imported Y-forward/Z-up) for the server
            # penetration/support checks in pack_cli (prop_checks.py).
            points = np.array([tuple(o.matrix_world @ v.co) for o in g["meshes"] for v in o.data.vertices], dtype=np.float32)
            if len(points) > 6000:
                points = points[np.linspace(0, len(points) - 1, 6000).astype(int)]
            points_path = out / f"prop_{g['index']:03}_points.npy"
            np.save(points_path, points)
            coords = [world_to_camera_view(scene, scene.camera, o.matrix_world @ Vector(c))
                      for o in g["meshes"] for c in o.bound_box]
            fraction = projected_bbox_fraction(coords)
            flags["out_of_frame"] = flags["prop_out_of_frame"] = fraction < .95
            diagnostics = {key: g[key] for key in ("vertex_probe_world","vertex_probe_complete","absolute_matrix_world","size_m", "scale", "normalization_factor", "source_bbox_m",
                                                  "source_dimensions_m", "cache_bbox_m", "scaled_bbox_m",
                                                  "final_bbox_m", "final_dimensions_m")}
            result["props"].append({"prop_id": g["id"], "index": g["index"], "prop_strength": g["strength"],
                                     "surface_intersection_pairs":intersections,"aabb_overlap":aabb,
                                     "silhouette": str(mask), "bbox_world": g["bbox_world"],
                                     **diagnostics, "bbox_source_m": g["source_bbox_m"],
                                     "bbox_final_m": g["final_bbox_m"],
                                     "frame_inside_fraction": fraction, "flags": flags,
                                     "points_world": str(points_path)})
    finally:
        (scene.render.engine, scene.render.film_transparent, scene.render.filepath,
         scene.render.image_settings.file_format, scene.render.image_settings.color_mode,
         scene.render.image_settings.color_depth, bpy.context.view_layer.material_override,
         scene.view_settings.view_transform) = saved
    (out / "props_render.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    import bpy
    import numpy as np
    # Only trusted repository paths are added. Never asset source paths.
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import render_plate_scene as renderer
    ap = argparse.ArgumentParser()
    for key in ("plate", "body", "joints", "props", "library", "out"):
        ap.add_argument("--" + key, required=True, type=Path)
    ap.add_argument("--raster", default="")
    ap.add_argument("--spec", type=Path, default=None, help="0.3.0: render_spec.json (scene light), as the body render")
    ap.add_argument("--photo-emission", type=float, default=2.5)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--neutral-light", action="store_true", help="pre-0.3.0 neutral EEVEE light, no shadows")
    ns = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    info = json.loads((ns.plate / "plate.json").read_text(encoding="utf-8"))
    if ns.raster:
        info["size"] = [int(v) for v in ns.raster.lower().split("x")]
    renderer.setup_plate_camera(scene, info)
    plate_file = next((ns.plate / n for n in ("plate_mesh.obj", "plate.obj", "mesh.obj") if (ns.plate / n).exists()), None)
    if plate_file is None:
        raise FileNotFoundError("plate.obj/mesh.obj missing")
    room = renderer.import_plate(str(plate_file))
    before = set(scene.objects)
    bpy.ops.wm.obj_import(filepath=str(ns.body), forward_axis="Y", up_axis="Z")
    body = [o for o in scene.objects if o not in before and o.type == "MESH"]
    with np.load(ns.joints, allow_pickle=False) as data:
        anchors = anchors_from_mhr70(data["kp"])
    props = json.loads(ns.props.read_text(encoding="utf-8"))
    light = None
    if ns.neutral_light:
        # Pre-0.3.0 neutral illumination (diagnostics/regression only).
        world = scene.world or bpy.data.worlds.new("prop_world")
        scene.world = world
        world.use_nodes = True
        world.node_tree.nodes.get("Background").inputs["Color"].default_value = (.35, .35, .35, 1)
        world.node_tree.nodes.get("Background").inputs["Strength"].default_value = .35
        bpy.ops.object.light_add(type="AREA", location=scene.camera.location)
        lamp = bpy.context.object
        lamp.rotation_euler = scene.camera.rotation_euler
        lamp.data.energy, lamp.data.shape, lamp.data.size = 75, "DISK", 5
        # Highlight rolloff preserves surface shape even with absent source textures.
        scene.view_settings.view_transform = "AgX"
    else:
        spec = json.loads(ns.spec.read_text(encoding="utf-8")) if ns.spec else {}
        scene_spec = (spec.get("scenes") or [{}])[0]
        photo = ns.plate / "plate_rgb.png"
        light = room_light(scene, room, photo, scene_spec.get("light", spec.get("light", info.get("light"))),
                           ns.photo_emission)
        scene.view_settings.view_transform = "Standard"  # as the body playblast
    result = render_props(scene, props, anchors, ns.library, ns.out, body_objects=body, room_objects=[room],
                          light=light, samples=ns.samples)
    if light is not None:
        for o in light[1]:
            bpy.data.objects.remove(o, do_unlink=True)
    # Existing implementation provides unified room/body/prop depth, same plate inverse normalization.
    lo, hi = info["inv_norm"]["lo"], info["inv_norm"]["hi"]
    renderer.render_pass(scene, str(ns.out / "depth_with_props.png"), engine="BLENDER_EEVEE",
                         override=renderer.depth_material(1 / hi, 1 / lo), bw16=True)
    result["depth"] = str(ns.out / "depth_with_props.png")
    (ns.out / "props_render.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PROP_RENDER_OK", json.dumps(result))


if __name__ == "__main__":
    main()
