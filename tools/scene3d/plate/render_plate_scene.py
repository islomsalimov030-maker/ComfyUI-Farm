"""Scene3D plate: put the mannequin INTO a location photo (camera and geometry from plate_geometry.py).

blender -b --python tools/scene3d/plate/render_plate_scene.py -- --plate C:/work/scene3d_work/plate/mainroom1
    --spec tools/scene3d/plate/scenes_mainroom1.json --mannequin C:/work/scene3d_work/stacy_mannequin_162.blend
    --out C:/work/scene3d_work/plate_maps

Per scene, in <out>/<name>/ (same layout as render_h3_maps.py, so draw_h3_maps.py works unchanged):
  clay.png       mannequin (clay) over the photo mesh, from the photo camera
  mask.png       the mannequin, white; parts hidden behind furniture of the photo are black
  depth_loc.png  copy of the photo's depth map (the location as it is in the photo)
  depth_char.png 16-bit inverse depth of the whole render (mannequin + props + photo mesh), normalised exactly like
                 depth_loc (plate.json inv_norm): build_ab pastes it inside the mask = real body depth (round 4)
  pose25.json    BODY-25 keypoints in pixels + placement info
  scene.blend    for manual fixes
Placement is given in PHOTO pixels (fractions 0..1 of width/height):
  stand: "feet" = the floor point between the feet; sit: "seat" = a point on the seat surface.
The pixel is ray-cast into the photo mesh; the mannequin is put there at real scale (the mesh is metric).
  lie: "seat" = where the pelvis lies, "head" = a second pixel on the same surface the head points to.
seat: if the pixel lands below 0.25 m (floor between the legs of a bench), the pixel walks up its column to the
first surface 0.25-1.0 m high facing up (round 4, P2 bench).
yaw: "camera" = faces the camera, or degrees (0 = faces -Y), plus optional "yaw_add" (and the pose's own yaw_add).
props: [{"file": path (.fbx .obj .glb .gltf .blend) or "builtin:phone", "size_m": largest side,
         "hand": "L"|"R" (at the palm) or "at": [u, v] (stands on the surface under that pixel),
         "rot_deg": [x, y, z], "offset_m": [x, y, z], "decimate": 0..1, "desc": text for the prompt}]
  Props get pass_index 1: they go into mask.png and depth_char.png with the mannequin.
  builtin:phone, builtin:book, builtin:cup - simple boxes / cylinder of real size.
playblast (05.11): Cycles (OptiX/CUDA if present, else CPU), 64 samples + denoise, transparent film; the photo mesh is a
  shadow catcher, mannequin and props are light-grey clay. Light = spec "light" (scene overrides location):
  {"azimuth_deg": 0 from the camera, +90 from the right of the frame, -90 left, 180 from behind the subject,
   "elevation_deg", "strength", "color": [r, g, b], "ambient", "softness_deg"}, or {"key_dir": left|right|back|front|top,
   "temp": warm|neutral|cool} (plate card), else the brightest area of the photo. Out: playblast_rgba.png and
  playblast.png = photo + clay + its cast/contact shadows (alpha_composite) in the photo camera.
  07.11: light "photo_emission": k (default 1.5, 0 = off) - the photo mesh also EMITS the photo itself (projected from
  the camera, Window coords), so the clay gets the room's real bounce light, colour and window rim light (image-based
  lighting from the plate); world ambient is then scaled by 0.3. Also writes shadow.png (L, cast/contact shadow
  opacity outside the figure) for the composite branch.
"""
import argparse
import json
import math
import os
import shutil
import sys

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from poses import POSES, apply_pose, place  # noqa: E402
from render_h3_maps import (BODY25_NAMES, char_objects,  # noqa: E402
                            depth_material, mask_material, keypoints_body25, render_pass, to_px, visibility)


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--mannequin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--samples", type=int, default=64, help="Cycles samples of the playblast pass")
    ap.add_argument("--no-playblast", action="store_true")
    ap.add_argument("--photo-emission", type=float, default=None,
                    help="override light.photo_emission (0 = the 05.11 playblast, 1.5 = photo-lit clay)")
    return ap.parse_args(argv)


KEY_DIR = {"front": (0, 35), "right": (90, 30), "left": (-90, 30), "back": (180, 25), "top": (0, 75)}
TEMP = {"warm": (1.0, 0.82, 0.62), "neutral": (1.0, 1.0, 1.0), "cool": (0.8, 0.88, 1.0)}


def resolve_light(light, photo_path):
    """spec light -> azimuth/elevation/strength/color/ambient/softness. See the module docstring."""
    light = dict(light or {})
    if "azimuth_deg" not in light and light.get("key_dir") in KEY_DIR:
        light["azimuth_deg"], light["elevation_deg"] = KEY_DIR[light["key_dir"]]
    if "color" not in light and light.get("temp") in TEMP:
        light["color"] = TEMP[light["temp"]]
    if "azimuth_deg" not in light:
        # no data: the brightest column of the photo (a window) gives the side, low sun
        from PIL import Image, ImageFilter
        im = Image.open(photo_path).convert("L").resize((64, 36)).filter(ImageFilter.GaussianBlur(3))
        px = list(im.getdata())
        u = (px.index(max(px)) % 64 + 0.5) / 64
        light["azimuth_deg"], light["elevation_deg"] = max(-90.0, min(90.0, (u - 0.5) * 180)), 30
        print(f"[plate] light from the photo: brightest u {u:.2f} -> azimuth {light['azimuth_deg']:.0f}")
    light.setdefault("elevation_deg", 30)
    light.setdefault("strength", 3.0)
    light.setdefault("color", TEMP["neutral"])
    light.setdefault("ambient", 0.3)
    light.setdefault("softness_deg", 8)
    return light


def use_gpu(sc):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except Exception:  # noqa: BLE001 - no such backend in this build
            continue
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            sc.cycles.device = "GPU"
            return kind
    sc.cycles.device = "CPU"
    return "CPU"


def render_playblast(a, sc, cam, plate, light, out, feet=None):
    """Clay mannequin + props lit like the photo, photo mesh = shadow catcher, over the photo.
    feet = (x, y, z) of the floor contact: a flat 0.8 m disc catcher there gives a clean contact shadow
    (the MoGe relief under the feet is noisy and catches almost none)."""
    photo = os.path.join(a.plate, "plate_rgb.png")
    L = resolve_light(light, photo)
    if a.photo_emission is not None:
        L["photo_emission"] = a.photo_emission
    sc.render.engine = "CYCLES"
    dev = use_gpu(sc)
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPTIX" if dev == "OPTIX" else "OPENIMAGEDENOISE"
    sc.render.film_transparent = True
    plate.is_shadow_catcher = True
    # the depth relief of the photo (back walls, shower) must not shade the sun itself: it only catches shadows
    plate.visible_shadow = False
    disc = None
    if feet is not None:
        bpy.ops.mesh.primitive_circle_add(radius=L.get("contact_radius_m", 0.8), vertices=48, fill_type="NGON",
                                          location=(feet[0], feet[1], feet[2] + 0.003))
        disc = bpy.context.active_object
        disc.is_shadow_catcher = True
        disc.visible_shadow = False
    clay = bpy.data.materials.new("clay_light")
    clay.use_nodes = True
    bsdf = clay.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1)
    bsdf.inputs["Roughness"].default_value = 0.6
    k_emit = float(L.get("photo_emission", 1.5))
    saved = {}
    if k_emit > 0:
        # per-object clay (a view-layer override would also replace the plate's emission)
        for o in sc.objects:
            if o.type == "MESH" and o is not plate and o is not disc:
                saved[o.name] = [sl.material for sl in o.material_slots]
                if not o.material_slots:
                    o.data.materials.append(clay)
                for sl in o.material_slots:
                    sl.link = "OBJECT"
                    sl.material = clay
        em = bpy.data.materials.new("plate_emit")
        em.use_nodes = True
        nt = em.node_tree
        nt.nodes.clear()
        tc = nt.nodes.new("ShaderNodeTexCoord")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(photo)
        tex.extension = "EXTEND"
        emi = nt.nodes.new("ShaderNodeEmission")
        emi.inputs["Strength"].default_value = k_emit
        outn = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(tc.outputs["Window"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
        nt.links.new(emi.outputs["Emission"], outn.inputs["Surface"])
        saved["__plate__"] = [sl.material for sl in plate.material_slots]
        if not plate.material_slots:
            plate.data.materials.append(em)
        for sl in plate.material_slots:
            sl.link = "OBJECT"
            sl.material = em
    else:
        bpy.context.view_layer.material_override = clay
    # sun: azimuth 0 = from the camera, +90 = from the right of the frame, 180 = from behind the subject
    fwd = cam.matrix_world.to_3x3() @ Vector((0, 0, -1))
    fwd = Vector((fwd.x, fwd.y, 0)).normalized()
    right = Vector((fwd.y, -fwd.x, 0))
    az, el = math.radians(L["azimuth_deg"]), math.radians(L["elevation_deg"])
    to_light = (-fwd * math.cos(az) + right * math.sin(az)) * math.cos(el) + Vector((0, 0, math.sin(el)))
    sd = bpy.data.lights.new("sun", "SUN")
    sd.energy = L["strength"]
    sd.color = L["color"][:3]
    sd.angle = math.radians(L["softness_deg"])
    sun = bpy.data.objects.new("sun", sd)
    sc.collection.objects.link(sun)
    sun.rotation_euler = to_light.to_track_quat("Z", "Y").to_euler()   # the lamp shines along its -Z
    world = sc.world or bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    amb = L.get("ambient_color", L["color"])
    bg.inputs["Color"].default_value = (amb[0], amb[1], amb[2], 1)
    bg.inputs["Strength"].default_value = L["ambient"] * (0.3 if k_emit > 0 else 1.0)
    s = sc.render.image_settings
    s.file_format, s.color_mode, s.color_depth = "PNG", "RGBA", "8"
    sc.view_settings.view_transform = "Standard"
    sc.render.filepath = f"{out}/playblast_rgba.png"
    bpy.ops.render.render(write_still=True)
    bpy.context.view_layer.material_override = None
    for name, mats in saved.items():
        o = plate if name == "__plate__" else sc.objects.get(name)
        if o is None:
            continue
        for sl, mat in zip(o.material_slots, mats):
            sl.material = mat
            sl.link = "DATA"
    plate.is_shadow_catcher = False
    plate.visible_shadow = True
    if disc is not None:
        bpy.data.objects.remove(disc, do_unlink=True)
    sc.render.film_transparent = False
    import numpy as np
    from PIL import Image, ImageFilter
    base = Image.open(photo).convert("RGBA")
    layer = Image.open(f"{out}/playblast_rgba.png").convert("RGBA").resize(base.size, Image.LANCZOS)
    # shadow alpha outside the figure: drop the catcher noise (< 0.08), fade with the distance from the figure
    # (a long low-sun shadow across the room reads as a stain), scale by shadow_opacity
    char = Image.open(f"{out}/mask.png").convert("L").resize(base.size, Image.NEAREST)
    m = np.asarray(char) > 127
    ys = np.nonzero(m.any(axis=1))[0]
    fig_h = int(ys[-1] - ys[0]) if len(ys) else base.height // 3
    near = char.point(lambda p: 255 if p > 127 else 0).filter(ImageFilter.MaxFilter(9))
    radius = float(max(4.0, fig_h * float(L.get("shadow_reach", 0.25))))
    prox = np.asarray(near.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    prox = np.clip(prox / max(prox.max(), 1e-6) * 4.0, 0, 1)
    rgba = np.asarray(layer).astype(np.float32)
    al = rgba[..., 3] / 255.0
    sh = np.where(al < 0.08, 0, al) * prox * L.get("shadow_opacity", 0.8)
    rgba[..., 3] = np.where(m, al, sh) * 255
    Image.fromarray((np.where(m, 0, sh) * 255).clip(0, 255).astype(np.uint8), "L").save(f"{out}/shadow.png")
    layer = Image.fromarray(rgba.clip(0, 255).astype(np.uint8), "RGBA")
    Image.alpha_composite(base, layer).convert("RGB").save(f"{out}/playblast.png")
    print(f"[plate] playblast ({dev}, {a.samples} spp): sun az {L['azimuth_deg']:.0f} el {L['elevation_deg']:.0f} "
          f"strength {L['strength']}, ambient {L['ambient']}, photo emission {k_emit}")


def setup_plate_camera(sc, info):
    W, H = info["size"]
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = info["sensor_width_mm"]
    cd.lens = info["lens_mm"]
    k = info["intrinsics_norm"]
    cd.shift_x = 0.5 - k["cx"]
    cd.shift_y = (k["cy"] - 0.5) * H / W
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    Rw = Matrix(info["R_world_from_cv"])  # rows = world X, Y, Z axes in camera (OpenCV) coords: world = Rw @ cv
    # Blender camera axes: X right = cv x, Y up = -cv y, Z back = -cv z
    cols = [Rw @ Vector((1, 0, 0)), Rw @ Vector((0, -1, 0)), Rw @ Vector((0, 0, -1))]
    m = Matrix((
        (cols[0].x, cols[1].x, cols[2].x),
        (cols[0].y, cols[1].y, cols[2].y),
        (cols[0].z, cols[1].z, cols[2].z)))
    cam.matrix_world = Matrix.Translation((0, 0, info["cam_height_m"])) @ m.to_4x4()
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.resolution_percentage = 100
    return cam


def pixel_ray(info, cam, u, v):
    """World ray through photo pixel (u, v) given as fractions of width/height."""
    k = info["intrinsics_norm"]
    W, H = info["size"]
    # fx is normalised by W and fy by H, as u and v are: both terms come out in camera units
    d_cv = Vector(((u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], 1.0))
    R = Matrix(info["R_world_from_cv"])
    return cam.matrix_world.translation.copy(), (R @ d_cv).normalized()


def import_plate(path):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    ob = [o for o in bpy.data.objects if o not in before][0]
    ob.name = "plate"
    ob.color = (0.55, 0.55, 0.55, 1)
    ob.pass_index = 0
    return ob


def ray_hit(sc, origin, direction, skip, normal=False):
    dg = bpy.context.evaluated_depsgraph_get()
    o = origin.copy()
    for _ in range(20):
        hit, loc, n, _i, ob, _m = sc.ray_cast(dg, o, direction, distance=100.0)
        if not hit:
            return (None, None) if normal else None
        if ob.name in skip:
            o = loc + direction * 1e-3
            continue
        return (loc, n) if normal else loc
    return (None, None) if normal else None


def seat_hit(sc, info, cam, u, v, skip):
    """Seat pixel -> surface. Below 0.25 m = floor under the seat: walk up the column to a seat top."""
    org, d = pixel_ray(info, cam, u, v)
    hit, _n = ray_hit(sc, org, d, skip, normal=True)
    if hit is None or hit.z >= 0.25:
        return hit, v
    fallback = None
    for i in range(1, 160):
        vv = v - i * 0.002
        org, d = pixel_ray(info, cam, u, vv)
        h, n = ray_hit(sc, org, d, skip, normal=True)
        if h is None or not 0.25 <= h.z <= 1.0:
            continue
        if n.z > 0.6:
            print(f"[plate] seat pixel {u:.3f},{v:.3f} hit z {hit.z:.2f}; moved up to v {vv:.3f} (z {h.z:.2f})")
            return h, vv
        fallback = fallback or (h, vv)
    if fallback:
        print(f"[plate] seat: no upward surface, using first hit above 0.25 m at v {fallback[1]:.3f}")
        return fallback
    raise SystemExit(f"seat pixel {u},{v}: floor (z {hit.z:.2f}) and nothing 0.25-1.0 m above it in the column")


def import_prop(pr):
    """Import one prop, return the empty that carries it (origin = bbox centre, scaled to size_m)."""
    before = set(bpy.data.objects)
    f = pr["file"]
    if f in ("builtin:phone", "builtin:book"):
        bpy.ops.mesh.primitive_cube_add(size=1)
        o = bpy.context.active_object
        o.scale = (0.075, 0.008, 0.15) if f == "builtin:phone" else (0.15, 0.025, 0.22)
        bpy.ops.object.transform_apply(scale=True)
    elif f == "builtin:cup":
        bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.095, vertices=24)
    else:
        ext = os.path.splitext(f)[1].lower()
        if ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=f)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=f, forward_axis="NEGATIVE_Z", up_axis="Y")
        elif ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=f)
        elif ext == ".blend":
            with bpy.data.libraries.load(f) as (src, dst):
                dst.objects = src.objects
            for o in dst.objects:
                if o is not None:
                    bpy.context.scene.collection.objects.link(o)
        else:
            raise SystemExit(f"prop {f}: unsupported format {ext}")
    meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"prop {f}: no mesh imported")
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    emp = bpy.data.objects.new("prop", None)
    bpy.context.scene.collection.objects.link(emp)
    emp.location = (lo + hi) / 2
    bpy.context.view_layer.update()
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = emp
        o.matrix_world = mw
        o.pass_index = 1
        o.color = (0.75, 0.6, 0.45, 1)
        if pr.get("decimate"):
            m = o.modifiers.new("dec", "DECIMATE")
            m.ratio = pr["decimate"]
    k = pr.get("size_m", 0.15) / max(hi - lo)
    emp.scale = (k, k, k)
    emp.rotation_euler = [math.radians(a) for a in pr.get("rot_deg", (0, 0, 0))]
    return emp, meshes


def place_props(sc, info, cam, scene, rig, skip):
    objs = []
    for i, pr in enumerate(scene.get("props", [])):
        emp, meshes = import_prop(pr)
        off = Vector(pr.get("offset_m", (0, 0, 0)))
        if pr.get("hand"):
            pb = rig.pose.bones[f"finger3-1.{pr['hand']}"]
            emp.location = rig.matrix_world @ pb.head + off
        else:
            org, d = pixel_ray(info, cam, *pr["at"])
            hit = ray_hit(sc, org, d, skip | {m.name for m in objs})
            if hit is None:
                raise SystemExit(f"prop {i}: pixel {pr['at']} hits nothing")
            emp.location = hit + off
            bpy.context.view_layer.update()
            low = min((m.matrix_world @ Vector(c)).z for m in meshes for c in m.bound_box)
            emp.location.z += hit.z - low
        bpy.context.view_layer.update()
        objs += meshes
        print(f"[plate] prop {i} {os.path.basename(pr['file'])}: at {tuple(round(c, 2) for c in emp.location)}")
    return objs


def run_scene(a, info, scene, light=None):
    bpy.ops.wm.open_mainfile(filepath=a.mannequin)
    sc = bpy.context.scene
    out = os.path.join(a.out, scene["name"])
    os.makedirs(out, exist_ok=True)
    plate = import_plate(os.path.join(a.plate, "plate_mesh.obj"))
    cam = setup_plate_camera(sc, info)
    rig, body, chars = char_objects()
    names = {o.name for o in chars}
    for o in chars:
        o.pass_index = 1
    bpy.context.view_layer.update()

    # target point from the photo pixel
    key = "seat" if scene.get("seat") else "feet"
    u, v = scene[key]
    if key == "seat" and POSES[scene["pose"]]["contact"] == "seat":   # lie may be on the floor (a mat)
        hit, v_used = seat_hit(sc, info, cam, u, v, names)
    else:
        org, dirn = pixel_ray(info, cam, u, v)
        hit, v_used = ray_hit(sc, org, dirn, names), v
    if hit is None:
        raise SystemExit(f"{scene['name']}: pixel {u},{v} hits nothing in the plate mesh")
    yaw = scene.get("yaw", "camera")
    if yaw == "camera":
        to_cam = cam.matrix_world.translation - hit
        yaw = math.degrees(math.atan2(to_cam.x, -to_cam.y))  # yaw 0 faces -Y
    yaw += scene.get("yaw_add", 0) + POSES[scene["pose"]].get("yaw_add", 0)
    contact = apply_pose(rig, scene["pose"])
    seat_z = hit.z if key == "seat" else 0.0
    if key == "seat" and contact not in ("seat", "lie"):
        raise SystemExit(f"{scene['name']}: seat point given for a standing pose")
    head_dir = None
    if contact == "lie":
        if not scene.get("head"):
            raise SystemExit(f"{scene['name']}: lie pose needs a 'head' pixel")
        o2, d2 = pixel_ray(info, cam, *scene["head"])
        h2 = ray_hit(sc, o2, d2, names)
        if h2 is None:
            raise SystemExit(f"{scene['name']}: head pixel hits nothing")
        head_dir = ((h2 - hit).x, (h2 - hit).y)
    placed = place(rig, body, hit.x, hit.y, yaw, contact, seat_z, head_dir)
    props = place_props(sc, info, cam, scene, rig, names)
    names |= {o.name for o in props}

    # clay pass (plate grey, mannequin light)
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "OBJECT"
    sc.world = sc.world or bpy.data.worlds.new("w")
    sc.world.color = (0, 0, 0)
    render_pass(sc, f"{out}/clay.png", engine="BLENDER_WORKBENCH")
    # mask: mannequin white, the plate mesh occludes it
    sc.eevee.taa_render_samples = 8
    # mask_material(): object index > 0.5 -> white; the photo mesh AND its holes stay black
    # (char_mask_material() inverted that and made the mesh holes white, found 04.11 round 4)
    render_pass(sc, f"{out}/mask.png", engine="BLENDER_EEVEE", override=mask_material())
    shutil.copy(os.path.join(a.plate, "depth_loc.png"), f"{out}/depth_loc.png")
    # real depth of the body: the same inverse-depth normalisation as the photo depth (map 1/z lo..hi -> 0..1)
    lo, hi = info["inv_norm"]["lo"], info["inv_norm"]["hi"]
    sc.eevee.taa_render_samples = 1
    render_pass(sc, f"{out}/depth_char.png", engine="BLENDER_EEVEE", override=depth_material(1.0 / hi, 1.0 / lo),
                bw16=True)
    if not a.no_playblast:
        render_playblast(a, sc, cam, plate, light, out, feet=(hit.x, hit.y, placed["feet_z"]))

    W, H = info["size"]
    size = (W, H)
    pts, top_z = keypoints_body25(body, rig)
    vis = visibility(sc, cam, pts, names)
    kp = []
    for k in BODY25_NAMES:
        p = pts.get(k)
        if p is None:
            kp.append([0.0, 0.0, 0])
            continue
        px, z = to_px(sc, cam, p, size)
        inside = 0 <= px[0] < W and 0 <= px[1] < H and z > 0
        kp.append(px + [1 if (vis.get(k) and inside) else 0])
    head_xy = (pts["l_ear"] + pts["r_ear"]) / 2
    head_top, _ = to_px(sc, cam, Vector((head_xy.x, head_xy.y, top_z)), size)
    feet = (pts["l_ankle"] + pts["r_ankle"]) / 2
    feet_px, _ = to_px(sc, cam, Vector((feet.x, feet.y, placed["feet_z"])), size)
    res = {"scene": scene, "size": [W, H], "plate": a.plate, "hit_world": list(hit), "yaw_deg": yaw,
           "pixel_used": [u, v_used], "contact": contact,
           "placed": placed, "keypoints_body25": kp, "keypoint_names": BODY25_NAMES,
           "head_top_px": head_top, "feet_px": feet_px,
           "cam_to_char_m": (cam.matrix_world.translation - hit).length}
    with open(f"{out}/pose25.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    bpy.ops.wm.save_as_mainfile(filepath=f"{out}/scene.blend")
    print(f"[plate] {scene['name']}: hit {tuple(round(c, 2) for c in hit)}, yaw {yaw:.0f}, "
          f"dist {res['cam_to_char_m']:.2f} m, head-feet px {round(feet_px[1] - head_top[1])}, "
          f"visible {sum(p[2] for p in kp)}/{len(kp)}")


def main():
    a = args()
    with open(os.path.join(a.plate, "plate.json"), encoding="utf-8") as f:
        info = json.load(f)
    with open(a.spec, encoding="utf-8") as f:
        spec = json.load(f)
    for scene in spec["scenes"]:
        if a.only and scene["name"] not in a.only.split(","):
            continue
        run_scene(a, info, scene, scene.get("light", spec.get("light")))


if __name__ == "__main__":
    main()
