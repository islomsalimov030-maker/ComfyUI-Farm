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
from render_h3_maps import (BODY25_NAMES, char_mask_material, char_objects,  # noqa: E402
                            depth_material, keypoints_body25, render_pass, to_px, visibility)


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--mannequin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    return ap.parse_args(argv)


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
    if f == "builtin:phone":
        bpy.ops.mesh.primitive_cube_add(size=1)
        o = bpy.context.active_object
        o.scale = (0.075, 0.008, 0.15)
        bpy.ops.object.transform_apply(scale=True)
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


def run_scene(a, info, scene):
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
    render_pass(sc, f"{out}/mask.png", engine="BLENDER_EEVEE", override=char_mask_material())
    shutil.copy(os.path.join(a.plate, "depth_loc.png"), f"{out}/depth_loc.png")
    # real depth of the body: the same inverse-depth normalisation as the photo depth (map 1/z lo..hi -> 0..1)
    lo, hi = info["inv_norm"]["lo"], info["inv_norm"]["hi"]
    sc.eevee.taa_render_samples = 1
    render_pass(sc, f"{out}/depth_char.png", engine="BLENDER_EEVEE", override=depth_material(1.0 / hi, 1.0 / lo),
                bw16=True)

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
        run_scene(a, info, scene)


if __name__ == "__main__":
    main()
