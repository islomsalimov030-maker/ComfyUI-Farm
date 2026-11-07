"""Scene3D poses for the MPFB default rig, as rotations about armature axes through the bone head.
Rest pose: A-pose, character faces -Y, her left side is +X. Steps run in order (parents first).
Axis cheat sheet: thigh forward = X -deg, knee bend = X +deg (after the thigh), arm down = Y +deg (L) / -deg (R),
forearm forward = X -deg, head turn to her left = Z +deg, spine lean back = X +deg.
Contact: 'floor' puts the lowest body point on the floor; 'seat' puts the underside of the hips/thighs on z;
'lie' lays the body on its back (head along a given direction) with the back on z.
A step ("@reach", side, anchor_bone, offset) moves that arm by coordinate descent until the middle-finger base
(finger3-1) reaches anchor_bone head + offset (armature space, offset x is mirrored for side "R").
A pose may carry "yaw_add" (deg, added to the scene yaw), e.g. 180 for her back to the camera.
"""
import math

import bpy
from mathutils import Matrix, Vector

AX = {"X": Vector((1, 0, 0)), "Y": Vector((0, 1, 0)), "Z": Vector((0, 0, 1))}


def _arms_down(deg=40, fwd=0, elbow=12):
    return [("upperarm01.L", "Y", deg), ("upperarm01.R", "Y", -deg),
            ("upperarm01.L", "X", -fwd), ("upperarm01.R", "X", -fwd),
            ("lowerarm01.L", "X", -elbow), ("lowerarm01.R", "X", -elbow)]


def HANG(side):
    """Arm hanging beside the thigh with a small gap (knuckles 11 cm out from the hip joint, 9 cm below)."""
    return ("@reach", side, f"upperleg01.{side}", (0.11, 0.0, -0.09))


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
    "sit_hands_knees": {"contact": "seat", "steps": _sit(88, 88, 2) + [
        ("@reach", "L", "lowerleg01.L", (0.0, 0.03, 0.13)),
        ("@reach", "R", "lowerleg01.R", (0.0, 0.03, 0.13))]},
    "sit_upright": {"contact": "seat", "steps": _sit() + _arms_down(45, 10, 10)},
    "sit_lean_back": {"contact": "seat", "steps": _sit(80, 75, 8) + [("spine05", "X", 10)] + _arms_down(45, 5, 20)},
    # round 4: hard poses. Arms by @reach (targets in armature space, offset x mirrored for R).
    # NB the up-vector of spine/head tilts FORWARD with X +deg (probe 04.11), the old sheet line above is for limbs.
    "stand_back": {"contact": "floor", "yaw_add": 180, "steps": [
        HANG("L"), HANG("R"), ("upperleg01.R", "X", -4), ("lowerleg01.R", "X", 8), ("head", "Z", 20)]},
    "crouch": {"contact": "floor", "steps": [
        ("upperleg01.L", "X", -100), ("upperleg01.R", "X", -100),
        ("upperleg01.L", "Z", -12), ("upperleg01.R", "Z", 12),
        ("lowerleg01.L", "X", 128), ("lowerleg01.R", "X", 128),
        ("foot.L", "X", -28), ("foot.R", "X", -28),
        ("spine05", "X", 32), ("head", "X", -22),
        ("@reach", "L", "lowerleg01.L", (0.02, -0.10, -0.04)), ("@reach", "R", "lowerleg01.R", (0.02, -0.10, -0.04))]},
    "lie_back": {"contact": "lie", "steps": [
        HANG("L"), HANG("R"),
        ("upperleg01.L", "X", -25), ("upperleg01.R", "X", -10),
        ("lowerleg01.L", "X", 45), ("lowerleg01.R", "X", 15), ("head", "X", 18)]},
    # right hand to the right cheek, left arm hanging clear of the body (so the silhouette shows one hand)
    "stand_hand_face": {"contact": "floor", "steps": [
        HANG("L"), ("@reach", "R", "head", (0.075, -0.08, -0.03)), ("head", "Z", -6)]},
    # left hand on the hip, right arm hangs with a visible gap (round 3: both hands came out)
    "stand_one_hand_hip": {"contact": "floor", "steps": [
        ("@reach", "L", "upperleg01.L", (0.07, -0.06, 0.13)), HANG("R"),
        ("upperleg01.L", "Z", -4), ("head", "Z", 8)]},
    # right hand in front of the chest (a phone), left arm hanging
    "stand_phone": {"contact": "floor", "steps": [
        HANG("L"), ("@reach", "R", "upperarm01.R", (0.08, -0.25, -0.14)), ("head", "X", 12)]},
    "stand_look_away": {"contact": "floor", "steps": _arms_down(40, 6, 18) + [
        ("head", "Z", 25), ("upperleg01.L", "X", -5), ("lowerleg01.L", "X", 10)]},
    # SFW table 05.11 (docs/tables/stacy_sfw_test.md): poses with props / harder turns
    # right hand holds a mug at the chest (builtin:cup at the palm), left arm hanging, weight on the left leg
    "stand_cup": {"contact": "floor", "steps": [
        # offset x is mirrored for R: negative = towards the midline (mug in front of the chest, not out to the side)
        HANG("L"), ("@reach", "R", "upperarm01.R", (-0.10, -0.20, -0.20)),
        ("upperleg01.R", "X", -4), ("lowerleg01.R", "X", 8), ("head", "X", 6)]},
    # crouch with both hands at the right shoe (tying the lace), head down
    "crouch_shoelace": {"contact": "floor", "steps": [
        ("upperleg01.L", "X", -100), ("upperleg01.R", "X", -100),
        ("upperleg01.L", "Z", -12), ("upperleg01.R", "Z", 12),
        ("lowerleg01.L", "X", 128), ("lowerleg01.R", "X", 128),
        ("foot.L", "X", -28), ("foot.R", "X", -28),
        ("spine05", "X", 38), ("head", "X", 18),
        ("@reach", "R", "foot.R", (0.03, -0.10, 0.04)), ("@reach", "L", "foot.R", (-0.06, -0.10, 0.05))]},
    # on her back, both hands hold a book above the chest (builtin:book at the right palm), face visible
    "lie_read": {"contact": "lie", "steps": [
        ("@reach", "R", "upperarm01.R", (0.12, -0.32, -0.08)), ("@reach", "L", "upperarm01.L", (0.12, -0.32, -0.08)),
        ("upperleg01.L", "X", -25), ("upperleg01.R", "X", -10),
        ("lowerleg01.L", "X", 45), ("lowerleg01.R", "X", 15), ("head", "X", 25)]},
    # back to the camera, head and shoulders turned over the left shoulder towards the camera
    "stand_back_look": {"contact": "floor", "yaw_add": 180, "steps": [
        HANG("L"), HANG("R"), ("upperleg01.R", "X", -4), ("lowerleg01.R", "X", 8),
        ("spine03", "Z", 12), ("spine05", "Z", 15), ("neck01", "Z", 20), ("head", "Z", 35)]},
}


def rotate(rig, bone, axis, deg):
    pb = rig.pose.bones.get(bone)
    if pb is None:
        print(f"[poses] WARNING: no bone {bone} in the rig, step skipped")
        return
    m = pb.matrix.copy()
    head = m.translation.copy()
    r = Matrix.Rotation(math.radians(deg), 4, AX[axis])
    pb.matrix = Matrix.Translation(head) @ r @ Matrix.Translation(-head) @ m
    bpy.context.view_layer.update()


# reach: (bone, axis, min, max) relative to the start, degrees
REACH_DOF = [("upperarm01", "X", -130, 50), ("upperarm01", "Y", -110, 110), ("upperarm01", "Z", -90, 90),
             ("lowerarm01", "X", -160, 10), ("lowerarm01", "Z", -90, 90)]
# starts for the descent (deg per dof; Y is "arm down" for L, mirrored for R)
REACH_STARTS = [(0, 0, 0, 0, 0), (0, 45, 0, 0, 0), (-30, 45, 0, -60, 0), (-60, 30, 20, -110, 0), (-20, 60, 0, -130, 0)]


def reach(rig, side, anchor, offset):
    """Coordinate descent on 4 arm angles so the finger3-1 head lands on anchor head + offset (armature space).
    Penalties keep the elbow below the shoulder and the hand outside the head."""
    sx = 1 if side == "L" else -1
    pbs = rig.pose.bones
    eff, elbow, shoulder = f"finger3-1.{side}", f"lowerarm01.{side}", f"upperarm01.{side}"
    if eff not in pbs or anchor not in pbs:
        print(f"[poses] WARNING: reach skipped, no {eff} or {anchor}")
        return
    target = pbs[anchor].head + Vector((offset[0] * sx, offset[1], offset[2]))
    head_c = pbs["head"].head + Vector((0, -0.02, 0.03)) if "head" in pbs else None

    def cost():
        c = (pbs[eff].head - target).length
        c += 0.5 * max(0.0, pbs[elbow].head.z - pbs[shoulder].head.z)
        if head_c is not None:
            c += 2.0 * max(0.0, 0.085 - (pbs[eff].head - head_c).length)
        return c

    names = [f"{b}.{side}" for b, *_ in REACH_DOF]
    base = [pbs[n].matrix_basis.copy() for n in names]
    # every update would re-skin the body (~0.1 s): switch the mesh modifiers off while searching
    mods = [m for o in bpy.data.objects if o.type == "MESH" for m in o.modifiers if m.show_viewport]
    for m in mods:
        m.show_viewport = False

    def restore(ms):
        for n, m in zip(names, ms):
            pbs[n].matrix_basis = m
        bpy.context.view_layer.update()

    best_all = None
    for start in REACH_STARTS:
        restore(base)
        acc = [0.0] * len(REACH_DOF)
        for i, d0 in enumerate(start):
            d = d0 * (sx if REACH_DOF[i][1] == "Y" else 1)
            if d:
                rotate(rig, names[i], REACH_DOF[i][1], d)
                acc[i] = d
        best = cost()
        for step in (16, 8, 4, 2, 1):
            improved = True
            while improved:
                improved = False
                for i, (_b, axis, lo, hi) in enumerate(REACH_DOF):
                    for sgn in (1, -1):
                        d = step * sgn
                        if not lo <= acc[i] + d <= hi:
                            continue
                        saved = [pbs[n].matrix_basis.copy() for n in names]
                        rotate(rig, names[i], axis, d)
                        c = cost()
                        if c < best - 1e-5:
                            best, acc[i], improved = c, acc[i] + d, True
                            break
                        restore(saved)
        if best_all is None or best < best_all[0]:
            best_all = (best, [pbs[n].matrix_basis.copy() for n in names], list(acc))
        if best < 0.01:
            break
    restore(best_all[1])
    for m in mods:
        m.show_viewport = True
    bpy.context.view_layer.update()
    miss = (pbs[eff].head - target).length
    print(f"[poses] reach {side}->{anchor}{tuple(offset)}: miss {miss * 100:.1f} cm (cost {best_all[0] * 100:.1f}), "
          f"angles {best_all[2]}")


def apply_pose(rig, name):
    for st in POSES[name]["steps"]:
        if st[0] == "@reach":
            reach(rig, *st[1:])
        else:
            rotate(rig, *st)
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
    if not groups and obj.parent is not None:
        # rigid parts parented to the same rig (07.11: the neutral mannequin's egg head) count for contacts too
        for o in obj.parent.children:
            if o is not obj and o.type == "MESH" and o.parent_type == "BONE":
                pts += [o.matrix_world @ v.co for v in o.data.vertices]
    return pts


def _mean(pts):
    return sum(pts, Vector()) / len(pts)


def place(rig, body, x, y, yaw, contact, seat_z=0.0, head_dir=None):
    """Yaw about the rig origin, move to (x, y), then drop to the floor or the seat.
    lie: on her back, head towards head_dir (world XY vector), the pelvis over (x, y), the back on seat_z."""
    if contact == "lie":
        dx, dy = head_dir
        # Rx(-90): faces up, head along +Y; then Rz(phi) turns +Y to (dx, dy)
        rig.rotation_euler = (math.radians(-90), 0, math.atan2(-dx, dy))
        rig.location = (0, 0, 0)
        bpy.context.view_layer.update()
        pel = _mean(_world_verts(body, ["pelvis.L", "pelvis.R"]))
        rig.location.x, rig.location.y = x - pel.x, y - pel.y
        bpy.context.view_layer.update()
        low = min(p.z for p in _world_verts(body))
        rig.location.z += seat_z - low - 0.02          # sinks 2 cm into the cushion
        bpy.context.view_layer.update()
        return {"feet_z": round(min(p.z for p in _world_verts(body)), 3)}
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
