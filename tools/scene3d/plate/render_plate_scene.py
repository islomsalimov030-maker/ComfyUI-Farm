"""Scene3D plate: put the mannequin INTO a location photo (camera and geometry from plate_geometry.py).

blender -b --python tools/scene3d/plate/render_plate_scene.py -- --plate C:/work/scene3d_work/plate/mainroom1
    --spec tools/scene3d/plate/scenes_mainroom1.json --mannequin C:/work/scene3d_work/stacy_mannequin_162.blend
    --out C:/work/scene3d_work/plate_maps

Per scene, in <out>/<name>/ (same layout as render_h3_maps.py, so draw_h3_maps.py works unchanged):
  clay.png       mannequin (clay) over the photo mesh, from the photo camera
  mask.png       the mannequin, white; parts hidden behind furniture of the photo are black
  depth_loc.png  copy of the photo's depth map (the location as it is in the photo)
  pose25.json    BODY-25 keypoints in pixels + placement info
  scene.blend    for manual fixes
Placement is given in PHOTO pixels (fractions 0..1 of width/height):
  stand: "feet" = the floor point between the feet; sit: "seat" = a point on the seat surface.
The pixel is ray-cast into the photo mesh; the mannequin is put there at real scale (the mesh is metric).
yaw: "camera" = faces the camera, or degrees (0 = faces -Y), plus optional "yaw_add".
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
from poses import apply_pose, place  # noqa: E402
from render_h3_maps import (BODY25_NAMES, char_mask_material, char_objects,  # noqa: E402
                            keypoints_body25, render_pass, to_px, visibility)


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


def ray_hit(sc, origin, direction, skip):
    dg = bpy.context.evaluated_depsgraph_get()
    o = origin.copy()
    for _ in range(20):
        hit, loc, _n, _i, ob, _m = sc.ray_cast(dg, o, direction, distance=100.0)
        if not hit:
            return None
        if ob.name in skip:
            o = loc + direction * 1e-3
            continue
        return loc
    return None


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
    org, dirn = pixel_ray(info, cam, u, v)
    hit = ray_hit(sc, org, dirn, names)
    if hit is None:
        raise SystemExit(f"{scene['name']}: pixel {u},{v} hits nothing in the plate mesh")
    yaw = scene.get("yaw", "camera")
    if yaw == "camera":
        to_cam = cam.matrix_world.translation - hit
        yaw = math.degrees(math.atan2(to_cam.x, -to_cam.y))  # yaw 0 faces -Y
    yaw += scene.get("yaw_add", 0)
    contact = apply_pose(rig, scene["pose"])
    seat_z = hit.z if key == "seat" else 0.0
    if key == "seat" and contact != "seat":
        raise SystemExit(f"{scene['name']}: seat point given for a standing pose")
    placed = place(rig, body, hit.x, hit.y, yaw, contact, seat_z)

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
