"""Scene3D 3D-A: control maps for the MiniMax-H3 Fun ControlNet (depth of location + OpenPose with hands).

blender -b --python tools/scene3d/render_h3_maps.py -- --spec tools/scene3d/scenes_pilot.json
    --mannequin C:/work/scene3d_work/stacy_mannequin.blend --out C:/work/scene3d_work/gate

Per scene, in <out>/<name>/:
  depth_loc.png   - depth of the LOCATION only, the character cut out by its mask (near = white, inverse depth, 16 bit)
  depth_loc8.png  - the same in 8 bit RGB (what the H3 control video loader gets)
  pose25.png      - OpenPose BODY-25 (COCO-18 + 5 fingers x 3 joints per hand) drawn on black
  pose25.json     - BODY-25 keypoints in pixels (for measuring)
  control/        - a control video as a sequence of identical frames: <n>_00001.png ... (17n+5 frames)
  control.mp4     - the same sequence as an mp4 at 24 fps (for LoadVideo paths)
  head_top_px / feet_px - scale references

Why the character is cut out of the depth map: the H3 Fun ControlNet gets the pose from a separate
OpenPose map, so a depth map that also carries the body fights itself - the model then has to guess
whether the body silhouette in the depth plane is geometry or a person. Depth of the location alone
gives the room geometry; the pose map gives the body.

Hands: the MPFB default rig has finger1-1..finger5-3 per side (5 fingers, 3 joints), so BODY-25
(OpenPose 25 = COCO-18 + 5x3 + 5x2 hand points per side) is computed from the rig itself - no SMPL,
no non-commercial pose assets.
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

# ---------------------------------------------------------------- BODY-25 layout
# Indices 0..17 are the COCO-18 body points (same order as the pilot's pose.json).
# 18..22 left hand (thumb..pinky, 3 joints each: base/proximal/distal),
# 23..27 right hand, 28..29 neck base / mid-hip (OpenPose extras).
BODY25_NAMES = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
    "l_thumb1", "l_thumb2", "l_thumb3", "l_index1", "l_index2", "l_index3",
    "l_middle1", "l_middle2", "l_middle3", "l_ring1", "l_ring2", "l_ring3",
    "l_pinky1", "l_pinky2", "l_pinky3",
    "r_thumb1", "r_thumb2", "r_thumb3", "r_index1", "r_index2", "r_index3",
    "r_middle1", "r_middle2", "r_middle3", "r_ring1", "r_ring2", "r_ring3",
    "r_pinky1", "r_pinky2", "r_pinky3",
]
# MPFB finger bone order -> OpenPose hand order. OpenPose: thumb, index, middle, ring, pinky.
FINGER_BONES = ["finger1", "finger2", "finger3", "finger4", "finger5"]
FINGER_SUFFIX = ["1", "2", "3"]

# ---------------------------------------------------------------- colours (OpenPose BODY25 palette)
# Each limb gets its own colour; the hand points reuse the wrist colour of their side.
BODY25_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
]
BODY25_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
]


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--mannequin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--frames", type=int, default=0, help="control video frames; 0 = still only (5)")
    ap.add_argument("--fps", type=int, default=24)
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


# ---------------------------------------------------------------- scene helpers (mirrors render_scene.py)
def char_objects():
    rig = bpy.data.objects["stacy_rig"]
    body = bpy.data.objects["stacy_body"]
    return rig, body, [o for o in bpy.data.objects if o.parent == rig or o == rig]


def rig_bone_world(rig, bone_name):
    """World position of a pose bone head."""
    pb = rig.pose.bones.get(bone_name)
    if pb is None:
        return None
    return rig.matrix_world @ pb.head


def keypoints_body25(body, rig):
    """BODY-25 keypoints in world space from the deformed base mesh + rig helpers."""
    masks = [m for m in body.modifiers if m.type == "MASK"]
    for m in masks:
        m.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    mw = ev.matrix_world

    def mean_group(group_name):
        gi = body.vertex_groups[group_name].index
        idx = [v.index for v in me.vertices if any(g.group == gi for g in v.groups)]
        if not idx:
            return None
        return sum((mw @ me.vertices[i].co for i in idx), Vector()) / len(idx)

    def mean_verts(idxs):
        return sum((mw @ me.vertices[i].co for i in idxs), Vector()) / len(idxs)

    out = {}
    # COCO-18 (same source points as the pilot, so the numbers stay comparable)
    src = {
        "nose": mean_verts([297, 7063]), "neck": mean_verts([1528, 3780]),
        "r_knee": mean_verts([4609, 16278]), "l_knee": mean_verts([11227, 17607]),
        "r_eye": mean_verts([14706, 14733]), "l_eye": mean_verts([14634, 14661]),
        "r_ear": mean_verts([5507, 5598]), "l_ear": mean_verts([12106, 12195]),
    }
    for side, pref in (("l", "joint-l"), ("r", "joint-r")):
        src[f"{side}_shoulder"] = mean_group(f"{pref}-shoulder")
        src[f"{side}_elbow"] = mean_group(f"{pref}-elbow")
        src[f"{side}_wrist"] = mean_group(f"{pref}-hand")
        src[f"{side}_hip"] = mean_group(f"joint-{side}-upper-leg")
        src[f"{side}_ankle"] = mean_group(f"{pref}-ankle")
    # finger joints from the rig (head of fingerN-1 is the joint base)
    for side, tag in (("l", "L"), ("r", "R")):
        for fi, fb in enumerate(FINGER_BONES):
            for sj in FINGER_SUFFIX:
                p = rig_bone_world(rig, f"{fb}-{sj}.{tag}")
                out[f"{side}_{['thumb', 'index', 'middle', 'ring', 'pinky'][fi]}{sj[-1]}"] = p
    zs = [(mw @ v.co).z for v in me.vertices]
    ev.to_mesh_clear()
    for m in masks:
        m.show_viewport = True
    out.update(src)
    return out, max(zs)


def first_hit(sc, origin, direction, max_dist, skip):
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


def visibility(sc, cam, pts, char_names):
    c = cam.matrix_world.translation
    vis = {}
    for k, p in pts.items():
        if p is None:
            vis[k] = False
            continue
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
    # a finger joint is hidden when the wrist is hidden
    for side in ("l", "r"):
        if not vis.get(f"{side}_wrist"):
            for fi in range(5):
                for sj in range(1, 4):
                    vis[f"{side}_{['thumb', 'index', 'middle', 'ring', 'pinky'][fi]}{sj}"] = False
    return vis


def depth_range(sc, cam, skip=()):
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


def char_mask_material():
    """White on the character, black elsewhere - the inverse of mask_material()."""
    def build(nt):
        info = nt.nodes.new("ShaderNodeObjectInfo")
        lt = nt.nodes.new("ShaderNodeMath")
        lt.operation = "LESS_THAN"
        lt.inputs[1].default_value = 0.5
        nt.links.new(info.outputs["Object Index"], lt.inputs[0])
        return lt.outputs[0]
    return emission_material("char_mask", build)


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


# ---------------------------------------------------------------- pose drawing (numpy/PIL, outside Blender)
def draw_pose25(kp, size, out_png):
    """OpenPose BODY-25 style skeleton on black."""
    import numpy as np
    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("RGB", (w, h), "black")
    over = Image.new("RGB", (w, h), "black")
    d = ImageDraw.Draw(over)
    stick = max(2, round(4 * h / 512))

    def limb(a, b, color):
        if not (kp[a][2] and kp[b][2]):
            return
        (x1, y1), (x2, y2) = kp[a][:2], kp[b][:2]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1:
            return
        ang = math.atan2(y2 - y1, x2 - x1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        pts = []
        for t in range(0, 360, 15):
            r = math.radians(t)
            ex, ey = length / 2 * math.cos(r), stick * math.sin(r)
            pts.append((mx + ex * math.cos(ang) - ey * math.sin(ang), my + ex * math.sin(ang) + ey * math.cos(ang)))
        d.polygon(pts, fill=tuple(int(c * 0.6) for c in color))

    for i, (a, b) in enumerate(BODY25_LIMBS):
        limb(a, b, BODY25_COLORS[i])
    # head: nose -> eyes -> ears (OpenPose draws these thin)
    head_color = (170, 255, 0)
    limb(0, 14, head_color)
    limb(0, 15, head_color)
    limb(14, 16, head_color)
    limb(15, 17, head_color)
    # hands: wrist -> finger1 -> finger2 -> finger3, per finger, per side
    for side, base in (("l", 18), ("r", 32)):
        wrist = 7 if side == "l" else 4
        for fi in range(5):
            p1 = base + fi * 3
            limb(wrist, p1, (0, 170, 255))
            limb(p1, p1 + 1, (0, 170, 255))
            limb(p1 + 1, p1 + 2, (0, 170, 255))
    img = Image.fromarray(np.maximum(np.asarray(img), np.asarray(over)))
    d = ImageDraw.Draw(img)
    for i, (x, y, v) in enumerate(kp):
        if not v:
            continue
        col = BODY25_COLORS[i] if i < len(BODY25_COLORS) else ((0, 170, 255) if i < 32 else (255, 0, 170))
        r = stick if i < 18 else max(1, stick // 2)
        d.ellipse((x - r, y - r, x + r, y + r), fill=col)
    img.save(out_png)


def depth_to_8bit(png16, out_png):
    import numpy as np
    from PIL import Image
    dep = np.asarray(Image.open(png16)).astype(np.float32)
    if dep.ndim == 3:
        dep = dep[..., 0]
    dep8 = Image.fromarray(np.clip(dep / max(dep.max(), 1e-6) * 255, 0, 255).astype(np.uint8)).convert("RGB")
    dep8.save(out_png)


def build_control_frames(pose_png, depth_png, out_dir, frames, fps):
    """A control video is a sequence of IDENTICAL frames (the body does not move in the scene)."""
    import shutil
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    pose = Image.open(pose_png).convert("RGB")
    depth = Image.open(depth_png).convert("RGB")
    for i in range(1, frames + 1):
        pose.save(os.path.join(out_dir, f"pose_{i:05d}.png"))
        depth.save(os.path.join(out_dir, f"depth_{i:05d}.png"))
    # ffmpeg mp4 (H.264, yuv420p) for the LoadVideo path
    mp4 = os.path.join(os.path.dirname(out_dir.rstrip("/\\")), "control.mp4")
    if frames > 0:
        src = os.path.join(out_dir, "pose_%05d.png")
        os.system(f'ffmpeg -y -loglevel error -framerate {fps} -i "{src}" -c:v libx264 -pix_fmt yuv420p '
                  f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "{mp4}"')
    return mp4


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
    cam = setup_camera(sc, scene["cam"], size)
    sc.eevee.taa_render_samples = 8
    names = {o.name for o in chars}

    # --- depth of the LOCATION only: hide the character, render depth
    # The near/far range is measured with the character VISIBLE (same as the pilot's depth.png) so the
    # depth values stay comparable between the pilot and the gate; only the rendered pass hides it.
    # The clay pass comes first: it forces a depsgraph update, which sc.ray_cast needs to see the room.
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "OBJECT"
    sc.world = sc.world or bpy.data.worlds.new("w")
    sc.world.color = (0, 0, 0)
    render_pass(sc, f"{out}/clay.png", engine="BLENDER_WORKBENCH")
    near, far = depth_range(sc, cam)
    for o in chars:
        o.hide_render = True
    render_pass(sc, f"{out}/depth_loc.png", engine="BLENDER_EEVEE", override=depth_material(near, far), bw16=True)
    for o in chars:
        o.hide_render = False
    # the character mask (for reference / inpaint masks)
    render_pass(sc, f"{out}/mask.png", engine="BLENDER_EEVEE", override=char_mask_material())

    # --- BODY-25 keypoints
    pts, top_z = keypoints_body25(body, rig)
    vis = visibility(sc, cam, pts, names)
    kp = {}
    for k in BODY25_NAMES:
        p = pts.get(k)
        if p is None:
            kp[k] = [0.0, 0.0, 0]
            continue
        px, z = to_px(sc, cam, p, size)
        inside = 0 <= px[0] < size[0] and 0 <= px[1] < size[1] and z > 0
        kp[k] = px + [1 if (vis.get(k) and inside) else 0]
    kp_list = [kp[k] for k in BODY25_NAMES]
    head_xy = (pts["l_ear"] + pts["r_ear"]) / 2
    head_top, _ = to_px(sc, cam, Vector((head_xy.x, head_xy.y, top_z)), size)
    feet = (pts["l_ankle"] + pts["r_ankle"]) / 2
    feet_px, _ = to_px(sc, cam, Vector((feet.x, feet.y, placed["feet_z"])), size)

    info = {"scene": scene, "size": size, "anchors": anchors, "placed": placed,
            "depth_near_far": [near, far], "keypoints_body25": kp_list,
            "keypoint_names": BODY25_NAMES, "head_top_px": head_top, "feet_px": feet_px}
    with open(f"{out}/pose25.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    bpy.ops.wm.save_as_mainfile(filepath=f"{out}/scene.blend")

    # --- draw pose + depth 8 bit (numpy/PIL, not available inside Blender's build in some setups)
    try:
        draw_pose25(kp_list, size, f"{out}/pose25.png")
        depth_to_8bit(f"{out}/depth_loc.png", f"{out}/depth_loc8.png")
        frames = a.frames if a.frames else 5
        build_control_frames(f"{out}/pose25.png", f"{out}/depth_loc8.png", f"{out}/control", frames, a.fps)
    except ImportError as e:
        print(f"[scene3d] WARN no numpy/PIL inside Blender ({e}); draw the maps outside")
    nvis = sum(v[2] for v in kp_list)
    print(f"[scene3d] {scene['name']}: near/far {near:.2f}/{far:.2f} m, visible BODY-25 {nvis}/{len(BODY25_NAMES)}, "
          f"head-feet px {round(feet_px[1] - head_top[1])}")


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
