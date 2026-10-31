"""Scene3D poses for the MPFB default rig, as rotations about armature axes through the bone head.
Rest pose: A-pose, character faces -Y, her left side is +X. Steps run in order (parents first).
Axis cheat sheet: thigh forward = X -deg, knee bend = X +deg (after the thigh), arm down = Y +deg (L) / -deg (R),
forearm forward = X -deg, head turn to her left = Z +deg, spine lean back = X +deg.
Contact: 'floor' puts the lowest body point on the floor; 'seat' puts the underside of the hips/thighs on z.
"""
import math

import bpy
from mathutils import Matrix, Vector

AX = {"X": Vector((1, 0, 0)), "Y": Vector((0, 1, 0)), "Z": Vector((0, 0, 1))}


def _arms_down(deg=40, fwd=0, elbow=12):
    return [("upperarm01.L", "Y", deg), ("upperarm01.R", "Y", -deg),
            ("upperarm01.L", "X", -fwd), ("upperarm01.R", "X", -fwd),
            ("lowerarm01.L", "X", -elbow), ("lowerarm01.R", "X", -elbow)]


def _sit(thigh=88, knee=88, spread=6):
    return [("upperleg01.L", "X", -thigh), ("upperleg01.R", "X", -thigh),
            ("upperleg01.L", "Z", -spread), ("upperleg01.R", "Z", spread),
            ("lowerleg01.L", "X", knee), ("lowerleg01.R", "X", knee)]


POSES = {
    "stand_relaxed": {"contact": "floor", "steps": _arms_down(40, 4, 14) + [
        ("upperleg01.R", "X", -4), ("lowerleg01.R", "X", 8), ("head", "X", 4)]},
    "stand_hand_on_hip": {"contact": "floor", "steps": _arms_down(40, 4, 10) + [
        ("upperarm01.L", "Y", -18), ("lowerarm01.L", "Z", -75), ("lowerarm01.L", "X", 20),
        ("upperleg01.L", "Z", -4), ("head", "Z", 8)]},
    # hands on the thighs: grid search (wrist -> mid-thigh), the rest forearm already points forward
    "sit_upright": {"contact": "seat", "steps": _sit() + _arms_down(45, 10, 10)},
    "sit_lean_back": {"contact": "seat", "steps": _sit(80, 75, 8) + [("spine05", "X", 10)] + _arms_down(45, 5, 20)},
    "stand_look_away": {"contact": "floor", "steps": _arms_down(40, 6, 18) + [
        ("head", "Z", 25), ("upperleg01.L", "X", -5), ("lowerleg01.L", "X", 10)]},
}


def rotate(rig, bone, axis, deg):
    pb = rig.pose.bones[bone]
    m = pb.matrix.copy()
    head = m.translation.copy()
    r = Matrix.Rotation(math.radians(deg), 4, AX[axis])
    pb.matrix = Matrix.Translation(head) @ r @ Matrix.Translation(-head) @ m
    bpy.context.view_layer.update()


def apply_pose(rig, name):
    for bone, axis, deg in POSES[name]["steps"]:
        rotate(rig, bone, axis, deg)
    return POSES[name]["contact"]


def _world_verts(obj, groups=None, min_w=0.3):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    idx = None
    if groups:
        idx = {obj.vertex_groups[g].index for g in groups if g in obj.vertex_groups}
    pts = []
    for v in me.vertices:
        if idx is not None and not any(g.group in idx and g.weight >= min_w for g in v.groups):
            continue
        pts.append(ev.matrix_world @ v.co)
    ev.to_mesh_clear()
    return pts


def place(rig, body, x, y, yaw, contact, seat_z=0.0):
    """Yaw about the rig origin, move to (x, y), then drop to the floor or the seat."""
    rig.rotation_euler = (0, 0, math.radians(yaw))
    rig.location.x, rig.location.y = x, y
    bpy.context.view_layer.update()
    if contact == "seat":
        groups = ["pelvis.L", "pelvis.R", "upperleg01.L", "upperleg01.R", "upperleg02.L", "upperleg02.R"]
        for _ in range(15):
            low = min(p.z for p in _world_verts(body, groups))
            rig.location.z += seat_z - low
            bpy.context.view_layer.update()
            if min(p.z for p in _world_verts(body)) > -0.01:
                break
            # feet under the floor: lift both thighs a little (knees up, feet up and forward)
            rotate(rig, "upperleg01.L", "X", -2)
            rotate(rig, "upperleg01.R", "X", -2)
    else:
        low = min(p.z for p in _world_verts(body))
        rig.location.z -= low
    bpy.context.view_layer.update()
    feet = min(p.z for p in _world_verts(body))
    return {"feet_z": round(feet, 3)}
