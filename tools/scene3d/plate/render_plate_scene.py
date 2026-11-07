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
if os.environ.get("FARM_BPY_PKGS"):   # Blender's own Python has no Pillow: pip --target dir (appended: bundled numpy wins)
    sys.path.append(os.environ["FARM_BPY_PKGS"])
from poses import POSES, apply_pose, place, reach, rotate  # noqa: E402
import pose_tools  # noqa: E402
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


def add_window_gobo(sc, cam, plate, disc, sun, to_light, photo, L):
    """07.11 (M3): sun patches through the window on the FIGURE, like the patches on the wall of the photo.
    The exact window is out of frame, so the pattern cannot be traced; it is CLONED: the wall band of the photo
    (rows 'gobo_rows' of the frame, default top 45 %) -> local luminance ratio (albedo cancels) -> transmittance of a
    blocker plane facing the sun 'gobo_dist_m' in front of the figure, scaled to the wall band's size in metres (so the
    panes and bars keep their size and slant). Light linking: the gobo-shaded sun lights only the figure; an unshaded
    copy lights the catchers (plate/disc), so the photo's own patches are not darkened twice."""
    import numpy as np
    from PIL import Image, ImageFilter
    figure = [o for o in sc.objects if o.type == "MESH" and o not in (plate, disc) and o.visible_camera]
    pts = [o.matrix_world @ Vector(c) for o in figure for c in o.bound_box]
    centre = sum(pts, Vector()) / len(pts)
    r0, r1 = L.get("gobo_rows", [0.0, 0.45])
    g = Image.open(photo).convert("L")
    W, H = g.size
    band = g.crop((0, int(r0 * H), W, int(r1 * H)))
    fine = np.asarray(band.filter(ImageFilter.GaussianBlur(2)), np.float32) + 1.0
    broad = np.asarray(band.filter(ImageFilter.GaussianBlur(max(band.size) / 6)), np.float32) + 1.0
    ratio = fine / broad
    lo, hi = np.percentile(ratio, L.get("gobo_lo_pct", 20)), np.percentile(ratio, L.get("gobo_hi_pct", 80))
    t = np.clip((ratio - lo) / max(hi - lo, 1e-6), 0, 1)
    # size of the band on the wall: rays through its corners onto the photo mesh
    dg = bpy.context.evaluated_depsgraph_get()
    info = json.load(open(os.path.join(os.path.dirname(photo), "plate.json")))
    corners = []
    for (u, v) in ((0.02, r0 + 0.02), (0.98, r0 + 0.02), (0.02, r1 - 0.02)):
        org, d = pixel_ray(info, cam, u, v)
        hit = sc.ray_cast(dg, org, d, distance=50.0)
        corners.append(hit[1] if hit[0] else None)
    if None in corners:
        print("[plate] gobo: wall band not on the mesh, skipped")
        return []
    w_m = (corners[1] - corners[0]).length / 0.96
    h_m = (corners[2] - corners[0]).length / max(r1 - r0 - 0.04, 1e-3)
    u_ax = to_light.cross(Vector((0, 0, 1)))
    if u_ax.length < 1e-3:
        u_ax = Vector((1, 0, 0))
    u_ax.normalize()
    v_ax = u_ax.cross(to_light).normalized()
    origin = centre + to_light * float(L.get("gobo_dist_m", 1.0))
    rig = next((o for o in sc.objects if o.type == "ARMATURE"), None)
    if L.get("gobo_face_lit", True) and rig is not None and "head" in rig.pose.bones:
        # slide the pattern so the nearest bright pane lands on her face (identity needs a lit face)
        head = rig.matrix_world @ rig.pose.bones["head"].head
        rel = head - origin
        hu, hv = rel.dot(u_ax) / w_m + 0.5, rel.dot(v_ax) / h_m + 0.5          # texture coords of the face
        rows, cols = t.shape
        ys, xs = np.nonzero(t > 0.75)
        if len(xs):
            fu, fv = xs / cols, 1.0 - ys / rows                                   # t row 0 = top of the band
            k = int(np.argmin((fu - hu) ** 2 * w_m ** 2 + (fv - hv) ** 2 * h_m ** 2))
            origin = origin + u_ax * (hu - fu[k]) * w_m + v_ax * (hv - fv[k]) * h_m
    img = bpy.data.images.new("gobo", t.shape[1], t.shape[0], alpha=False)
    rgba = np.repeat(np.flipud(t)[..., None], 4, axis=2)
    rgba[..., 3] = 1
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=origin)
    gobo = bpy.context.active_object
    gobo.name = "gobo"
    gobo.scale = (w_m, h_m, 1)
    gobo.rotation_euler = Matrix((u_ax, v_ax, to_light)).transposed().to_euler()
    m = bpy.data.materials.new("gobo")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "REPEAT"
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    dark = nt.nodes.new("ShaderNodeBsdfDiffuse")
    dark.inputs["Color"].default_value = (0, 0, 0, 1)
    mix = nt.nodes.new("ShaderNodeMixShader")
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(dark.outputs["BSDF"], mix.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], outn.inputs["Surface"])
    gobo.data.materials.append(m)
    gobo.visible_camera = gobo.visible_diffuse = gobo.visible_glossy = gobo.visible_transmission = False
    fig_col = bpy.data.collections.new("gobo_receivers")
    for o in figure:
        fig_col.objects.link(o)
    cat_col = bpy.data.collections.new("gobo_catchers")
    for o in (plate, disc):
        if o is not None:
            cat_col.objects.link(o)
    sun.light_linking.receiver_collection = fig_col
    sun2 = sun.copy()
    sc.collection.objects.link(sun2)
    sun2.light_linking.receiver_collection = cat_col
    blk = bpy.data.collections.new("gobo_blockers_plain")
    for o in figure:
        blk.objects.link(o)
    sun2.light_linking.blocker_collection = blk
    if L.get("gobo_key_strength"):
        sun.data = sun.data.copy()
        sun.data.energy = float(L["gobo_key_strength"])
    print(f"[plate] gobo: wall band {w_m:.2f} x {h_m:.2f} m, open {float((t > 0.5).mean()):.2f}")
    Image.fromarray((t * 255).astype(np.uint8)).save(os.path.join(os.path.dirname(sc.render.filepath) or ".",
                                                                   "gobo_debug.png"))
    return [gobo, sun2]


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
        # 07.11: drop the override left by the previous pass (depth emission -> flat grey "clay" on 07.11 run 1)
        bpy.context.view_layer.material_override = None
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
        # 07.11 fix: an EMITTING shadow catcher kills its own matte (its emission enters the shadow ratio -> no cast
        # shadow) and floods the clay. The catcher stays clay (as in _s) and invisible to bounce rays; a copy of the
        # plate pushed 1 % away from the camera (same Window projection) emits the photo, unseen by camera/shadows.
        saved["__plate__"] = [sl.material for sl in plate.material_slots]
        if not plate.material_slots:
            plate.data.materials.append(clay)
        for sl in plate.material_slots:
            sl.link = "OBJECT"
            sl.material = clay
        plate.visible_diffuse = plate.visible_glossy = plate.visible_transmission = False
        emitter = plate.copy()
        sc.collection.objects.link(emitter)
        emitter.is_shadow_catcher = False
        emitter.visible_camera = emitter.visible_shadow = False
        c = cam.matrix_world.translation
        emitter.matrix_world = (Matrix.Translation(c) @ Matrix.Scale(1.01, 4) @ Matrix.Translation(-c)
                                @ plate.matrix_world)
        for sl in emitter.material_slots:
            sl.link = "OBJECT"
            sl.material = em
        saved["__emitter__"] = emitter
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
    gobo_objs = add_window_gobo(sc, cam, plate, disc, sun, to_light, photo, L) if L.get("gobo") else []
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
    for o in gobo_objs:
        bpy.data.objects.remove(o, do_unlink=True)
    emitter = saved.pop("__emitter__", None)
    if emitter is not None:
        bpy.data.objects.remove(emitter, do_unlink=True)
        plate.visible_diffuse = plate.visible_glossy = plate.visible_transmission = True
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
    # shadow.png: also drop the 2 px anti-aliased rim of the figure (07.11: it drew a dark outline of the mannequin
    # around the generated person in the composite branch)
    rim = np.asarray(char.point(lambda p: 255 if p > 127 else 0).filter(ImageFilter.MaxFilter(5))) > 127
    Image.fromarray((np.where(rim, 0, sh) * 255).clip(0, 255).astype(np.uint8), "L").save(f"{out}/shadow.png")
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
    if scene.get("body_obj"):            # 07.11e Постановка v2: меш тела уже посажен pose_fit.py (мир плиты)
        return _run_body_obj(a, sc, info, cam, plate, light, out, scene, chars)

    # 07.11c pose round: external pose file (methods A/B/C) -> registered as a pose named after the scene
    spec = None
    if scene.get("pose_file"):
        with open(scene["pose_file"], encoding="utf-8") as f:
            spec = json.load(f)
        scene["pose"] = "file:" + scene["name"]
        POSES[scene["pose"]] = {"contact": spec.get("contact", "solve"), "yaw_add": spec.get("yaw_add", 0), "steps": []}
    # target point from the photo pixel
    key = "seat" if scene.get("seat") else "feet"
    u, v = scene[key]
    if key == "seat" and POSES[scene["pose"]]["contact"] in ("seat", "solve"):   # lie may be on the floor (a mat)
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
    if spec is not None:
        contact = pose_tools.apply_pose_file(rig, spec, reach, rotate)
    else:
        contact = apply_pose(rig, scene["pose"])
    # joint limits (07.11c): measure, warn, clamp (scene "no_clamp": true keeps the pose as is, report only)
    ang0 = pose_tools.measure(rig)
    viol0 = pose_tools.violations(ang0)
    clamp_log = [] if scene.get("no_clamp") else pose_tools.clamp(rig)
    # the PROMPT contacts (same for every method) drive the solver and the metric; method A's own list is kept in its file
    contacts = scene.get("contacts") or (spec or {}).get("contacts") or []
    if contact == "solve":
        rig.rotation_euler = (0, 0, math.radians(yaw))
        rig.location = (hit.x, hit.y, hit.z)
        bpy.context.view_layer.update()
        low = min(p.z for p in pose_tools._all_points(body, step=3))
        rig.location.z += hit.z - low
        bpy.context.view_layer.update()
        solved = pose_tools.solve_contacts(rig, body, plate, contacts, cam.matrix_world.translation,
                                              reach_fn=reach, rotate_fn=rotate) if contacts else {}
        feet_z = min(p.z for p in pose_tools._all_points(body, step=3))
        _write_pose_metrics(out, rig, body, plate, cam, ang0, viol0, clamp_log, contacts, solved)
        placed = {"feet_z": round(feet_z, 3)}
        props = place_props(sc, info, cam, scene, rig, names)
        names |= {o.name for o in props}
        return _finish_scene(a, sc, info, cam, plate, light, out, scene, rig, body, names, hit, yaw, u, v_used,
                             contact, placed)
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

    if contacts:
        solved = {"contacts": pose_tools.contact_gaps(body, pose_tools.plate_bvh(plate), contacts,
                                                      cam.matrix_world.translation)}
    else:
        solved = {}
    _write_pose_metrics(out, rig, body, plate, cam, ang0, viol0, clamp_log, contacts, solved)
    return _finish_scene(a, sc, info, cam, plate, light, out, scene, rig, body, names, hit, yaw, u, v_used,
                         contact, placed)


def _run_body_obj(a, sc, info, cam, plate, light, out, scene, chars):
    for o in chars:                       # манекен не рендерится; его материалы/свет сцены остаются
        o.hide_render = True
        o.hide_viewport = True
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=scene["body_obj"], forward_axis="Y", up_axis="Z")
    body = [o for o in bpy.data.objects if o not in before][0]
    body.name = "body_v2"
    body.pass_index = 1
    body.color = chars[0].color if chars else (0.8, 0.8, 0.8, 1)
    for poly in body.data.polygons:
        poly.use_smooth = True
    bpy.context.view_layer.update()
    zs = [(body.matrix_world @ v.co) for v in body.data.vertices]
    low = min(p.z for p in zs)
    cx = sum(p.x for p in zs) / len(zs)
    cy = sum(p.y for p in zs) / len(zs)
    hit = Vector((cx, cy, low))
    placed = {"feet_z": round(low, 3)}
    fit_json = os.path.join(os.path.dirname(scene["body_obj"]), "fit.json")
    if os.path.isfile(fit_json):
        shutil.copy(fit_json, f"{out}/pose_fit.json")
    return _finish_scene(a, sc, info, cam, plate, light, out, scene, None, body, {body.name}, hit, 0.0, 0, 0,
                         "body_obj", placed)


def _write_pose_metrics(out, rig, body, plate, cam, ang0, viol0, clamp_log, contacts, solved):
    ang1 = pose_tools.measure(rig)
    res = {"angles_before": ang0, "violations_before": {k: list(v) for k, v in viol0.items()},
           "clamp": clamp_log, "angles": ang1,
           "violations_after": {k: list(v) for k, v in pose_tools.violations(ang1).items()},
           "bones": pose_tools.bone_lengths(rig), "contacts_req": contacts, **solved}
    with open(f"{out}/pose_metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    gaps = ", ".join(f"{c['part']} {c['gap_cm']:+.1f}" for c in solved.get("contacts", []))
    print(f"[pose] limits: {len(viol0)} violations, {len(clamp_log)} clamped; bones dev "
          f"{res['bones']['max_dev_mm']} mm; contacts [{gaps}] pen {solved.get('max_penetration_cm', '-')} cm")


def _finish_scene(a, sc, info, cam, plate, light, out, scene, rig, body, names, hit, yaw, u, v_used, contact, placed):
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
    if rig is None:                       # Постановка v2: ключевые точки BODY-25 посчитал pose_fit.py (fit.json)
        res = {"scene": scene, "size": [W, H], "plate": a.plate, "hit_world": list(hit), "contact": contact,
               "placed": placed, "cam_to_char_m": (cam.matrix_world.translation - hit).length}
        fj = f"{out}/pose_fit.json"
        if os.path.isfile(fj):
            with open(fj, encoding="utf-8") as f:
                fit = json.load(f)
            kx, ky = W / fit["px_size"][0], H / fit["px_size"][1]
            res["keypoints_body25"] = [[x * kx, y * ky, v] for x, y, v in fit["keypoints_body25_px"]]
            res["keypoint_names"] = BODY25_NAMES
        with open(f"{out}/pose25.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        bpy.ops.wm.save_as_mainfile(filepath=f"{out}/scene.blend")
        print(f"[plate] {scene['name']}: body_obj, low z {placed['feet_z']}, dist {res['cam_to_char_m']:.2f} m")
        return
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
