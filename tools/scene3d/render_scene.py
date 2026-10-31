"""Scene3D: mannequin + room + pose + camera -> control maps for one or more scenes of a spec.
blender -b --python tools/scene3d/render_scene.py -- --spec tools/scene3d/scenes_pilot.json
    --mannequin C:/work/scene3d_work/stacy_mannequin.blend --out C:/work/scene3d_work/scenes [--only S1_sofa_sit]
Per scene, in <out>/<name>/: clay.png (Workbench, object colours), depth.png (16-bit, near = white, inverse depth),
mask.png (character visible pixels), pose.json (COCO-18 keypoints in pixels, occluded by furniture -> v=0,
plus head-top/feet points for scale checks), scene.blend. Draw the OpenPose image with draw_pose.py.
"""
import argparse
import json
import math
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poses import apply_pose, place  # noqa: E402
from rooms import ROOMS  # noqa: E402

# COCO-18 order; each point = mean of vertex indices or of a joint helper group (MPFB openpose rig definitions)
KEYPOINTS = [
    ("nose", [297, 7063]), ("neck", [1528, 3780]),
    ("r_shoulder", "joint-r-shoulder"), ("r_elbow", "joint-r-elbow"), ("r_wrist", "joint-r-hand"),
    ("l_shoulder", "joint-l-shoulder"), ("l_elbow", "joint-l-elbow"), ("l_wrist", "joint-l-hand"),
    ("r_hip", "joint-r-upper-leg"), ("r_knee", [4609, 16278]), ("r_ankle", "joint-r-ankle"),
    ("l_hip", "joint-l-upper-leg"), ("l_knee", [11227, 17607]), ("l_ankle", "joint-l-ankle"),
    ("r_eye", [14706, 14733]), ("l_eye", [14634, 14661]), ("r_ear", [5507, 5598]), ("l_ear", [12106, 12195]),
]
CHAR_COLORS = {"body": (0.85, 0.68, 0.58, 1), "hair": (0.85, 0.72, 0.45, 1), "clothes": (0.55, 0.55, 0.58, 1)}


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--mannequin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def char_objects():
    rig = bpy.data.objects["stacy_rig"]
    return rig, bpy.data.objects["stacy_body"], [o for o in bpy.data.objects if o.parent == rig or o == rig]


def keypoints(body):
    """World positions from the deformed base mesh with helper masks off (vertex order = base mesh)."""
    masks = [m for m in body.modifiers if m.type == "MASK"]
    for m in masks:
        m.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    mw = ev.matrix_world
    out = {}
    for name, src in KEYPOINTS:
        if isinstance(src, str):
            gi = body.vertex_groups[src].index
            idx = [v.index for v in me.vertices if any(g.group == gi for g in v.groups)]
        else:
            idx = src
        out[name] = sum((mw @ me.vertices[i].co for i in idx), Vector()) / len(idx)
    zs = [(mw @ v.co).z for v in me.vertices]
    ev.to_mesh_clear()
    for m in masks:
        m.show_viewport = True
    return out, max(zs)


def first_hit(sc, origin, direction, max_dist, skip):
    """Nearest hit, passing through boolean cutters and objects in `skip`."""
    dg = bpy.context.evaluated_depsgraph_get()
    travelled = 0.0
    o = origin.copy()
    while travelled < max_dist:
        hit, loc, _n, _i, ob, _m = sc.ray_cast(dg, o, direction, distance=max_dist - travelled)
        if not hit:
            return None, None
        d = (loc - origin).length
        if ob.get("cutter") or ob.name in skip:
            o = loc + direction * 1e-3
            travelled = d + 1e-3
            continue
        return d, ob
    return None, None


def visibility(sc, cam, pts, char_names, head_top):
    """v=0 when furniture is between camera and point, or the face point looks away from the camera."""
    c = cam.matrix_world.translation
    vis = {}
    for k, p in pts.items():
        d = (p - c).length
        hit, _ob = first_hit(sc, c, (p - c).normalized(), d, char_names)
        vis[k] = hit is None or hit > d - 0.02
    head = (pts["l_ear"] + pts["r_ear"]) / 2
    to_cam = (c - head).normalized()
    if (pts["nose"] - head).normalized().dot(to_cam) < -0.1:
        vis["nose"] = vis["l_eye"] = vis["r_eye"] = False
    for side in ("l", "r"):
        ear = pts[f"{side}_ear"]
        if (ear - head).normalized().dot(to_cam) < -0.35:
            vis[f"{side}_ear"] = False
    return vis


def depth_range(sc, cam, skip=()):
    """Near/far view depth from a coarse ray grid over the frame."""
    fr = [cam.matrix_world @ v for v in cam.data.view_frame(scene=sc)]
    c = cam.matrix_world.translation
    fwd = (cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
    ds = []
    for i in range(33):
        for j in range(19):
            u, v = i / 32, j / 18
            top = fr[3].lerp(fr[0], u)
            bot = fr[2].lerp(fr[1], u)
            p = top.lerp(bot, v)
            d, _ = first_hit(sc, c, (p - c).normalized(), 100.0, skip)
            if d:
                ds.append(d * (p - c).normalized().dot(fwd))
    ds.sort()
    return max(0.3, ds[0] * 0.95), ds[-1] * 1.02


def emission_material(name, build):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    nt.links.new(build(nt), em.inputs["Color"])
    return mat


def depth_material(near, far):
    def build(nt):
        cam = nt.nodes.new("ShaderNodeCameraData")
        inv = nt.nodes.new("ShaderNodeMath")
        inv.operation = "DIVIDE"
        inv.inputs[0].default_value = 1.0
        nt.links.new(cam.outputs["View Z Depth"], inv.inputs[1])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.clamp = True
        mr.inputs["From Min"].default_value = 1.0 / far
        mr.inputs["From Max"].default_value = 1.0 / near
        nt.links.new(inv.outputs[0], mr.inputs["Value"])
        return mr.outputs["Result"]
    return emission_material("depth", build)


def mask_material():
    def build(nt):
        info = nt.nodes.new("ShaderNodeObjectInfo")
        gt = nt.nodes.new("ShaderNodeMath")
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = 0.5
        nt.links.new(info.outputs["Object Index"], gt.inputs[0])
        return gt.outputs[0]
    return emission_material("mask", build)


def render_pass(sc, path, *, engine, override=None, bw16=False):
    sc.render.engine = engine
    bpy.context.view_layer.material_override = override
    s = sc.render.image_settings
    s.file_format = "PNG"
    s.color_mode = "BW" if bw16 else "RGB"
    s.color_depth = "16" if bw16 else "8"
    sc.view_settings.view_transform = "Raw" if override else "Standard"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def setup_camera(sc, spec, size):
    cd = bpy.data.cameras.new("cam")
    cd.lens = spec["lens"]
    cd.sensor_width = 36
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    cam.location = Vector(spec["loc"])
    cam.rotation_euler = (Vector(spec["look"]) - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.resolution_percentage = 100
    return cam


def to_px(sc, cam, p, size):
    v = world_to_camera_view(sc, cam, p)
    return [round(v.x * size[0], 1), round((1 - v.y) * size[1], 1)], v.z


def run_scene(a, spec, scene, size):
    bpy.ops.wm.open_mainfile(filepath=a.mannequin)
    sc = bpy.context.scene
    out = os.path.join(a.out, scene["name"])
    os.makedirs(out, exist_ok=True)
    anchors = ROOMS[scene["room"]]()
    rig, body, chars = char_objects()
    contact = apply_pose(rig, scene["pose"])
    seat_z = anchors[scene["seat"]] if scene.get("seat") else 0.0
    placed = place(rig, body, scene["at"][0], scene["at"][1], scene.get("yaw", 0), contact, seat_z)
    for o in chars:
        o.pass_index = 1
        kind = "hair" if o.name.startswith("hair") else "clothes" if o.name.startswith("clothes") else "body"
        o.color = CHAR_COLORS[kind]
    cam = setup_camera(sc, scene["cam"], size)
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "OBJECT"
    sc.display.shading.show_cavity = True
    sc.display.shading.show_shadows = True
    sc.eevee.taa_render_samples = 8
    sc.world = sc.world or bpy.data.worlds.new("w")
    sc.world.color = (0, 0, 0)
    render_pass(sc, f"{out}/clay.png", engine="BLENDER_WORKBENCH")
    near, far = depth_range(sc, cam)
    render_pass(sc, f"{out}/depth.png", engine="BLENDER_EEVEE", override=depth_material(near, far), bw16=True)
    render_pass(sc, f"{out}/mask.png", engine="BLENDER_EEVEE", override=mask_material())
    bpy.context.view_layer.material_override = None
    pts, top_z = keypoints(body)
    names = {o.name for o in chars}
    vis = visibility(sc, cam, pts, names, top_z)
    kp = {}
    for k, p in pts.items():
        px, z = to_px(sc, cam, p, size)
        inside = 0 <= px[0] < size[0] and 0 <= px[1] < size[1] and z > 0
        kp[k] = px + [1 if (vis[k] and inside) else 0]
    head_xy = (pts["l_ear"] + pts["r_ear"]) / 2
    head_top, _ = to_px(sc, cam, Vector((head_xy.x, head_xy.y, top_z)), size)
    feet = (pts["l_ankle"] + pts["r_ankle"]) / 2
    feet_px, _ = to_px(sc, cam, Vector((feet.x, feet.y, placed["feet_z"])), size)
    info = {"scene": scene, "size": size, "anchors": anchors, "placed": placed, "depth_near_far": [near, far],
            "keypoints_coco18": [kp[k] for k, _ in KEYPOINTS], "keypoint_names": [k for k, _ in KEYPOINTS],
            "head_top_px": head_top, "feet_px": feet_px}
    with open(f"{out}/pose.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    bpy.ops.wm.save_as_mainfile(filepath=f"{out}/scene.blend")
    print(f"[scene3d] {scene['name']}: near/far {near:.2f}/{far:.2f} m, placed {placed}, "
          f"visible kp {sum(v[2] for v in kp.values())}/18")


def main():
    a = args()
    with open(a.spec, encoding="utf-8") as f:
        spec = json.load(f)
    size = tuple(spec["size"])
    for scene in spec["scenes"]:
        if a.only and scene["name"] not in a.only.split(","):
            continue
        run_scene(a, spec, scene, size)


main()
