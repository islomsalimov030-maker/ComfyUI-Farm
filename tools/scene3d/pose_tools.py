"""Pose round 07.11c (bpy): joint limits + clamp, bone lengths, external pose files, contact solver, MHR70 retarget.

Joint angles are measured GEOMETRICALLY from pose-bone heads in armature space (independent of how the pose was made:
rotation steps, @reach, LLM JSON or a retarget). Limits = typical active ranges of motion of an adult (AAOS / clinical
goniometry tables, rounded); see docs/coord/3DA_POSE_LIMITS.md. A violation is reported and, with clamp=True, the
responsible bones are slerped towards their rest rotation (binary search) until the joint is inside its range.

Pose file (JSON), any of:
  {"contact": "solve", "steps": [[bone, "X|Y|Z", deg], ["@reach", "L", anchor_bone, [x, y, z]], ...],   # method A
   "contacts": [{"part": "pelvis|foot.L|foot.R|hand.L|hand.R|knee.L|knee.R|elbow.L|elbow.R|back|head",
                 "surface": "floor|bed|bench|wall|mat|..."}], "yaw_add": deg}
  {"contact": "solve", "rotations": {bone: [w, x, y, z]}, "contacts": [...]}          # methods B/C (retarget)
"""
import json
import math

import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

# name: (min, max) degrees; see measure()
LIMITS = {
    "elbow_flex.L": (0, 150), "elbow_flex.R": (0, 150),
    "knee_flex.L": (0, 150), "knee_flex.R": (0, 150),
    "knee_hyperext.L": (0, 10), "knee_hyperext.R": (0, 10),
    "hip_flex.L": (-30, 135), "hip_flex.R": (-30, 135),
    "hip_abd.L": (-30, 50), "hip_abd.R": (-30, 50),
    "shoulder_elev.L": (0, 180), "shoulder_elev.R": (0, 180),
    "shoulder_ext.L": (0, 60), "shoulder_ext.R": (0, 60),
    "spine_bend": (0, 80), "spine_twist": (0, 50),
    "neck_bend": (0, 70), "head_turn": (0, 80),
}
# which bones a clamp scales back
CLAMP_BONES = {
    "elbow": ["lowerarm01"], "knee": ["lowerleg01"], "hip": ["upperleg01"], "shoulder": ["upperarm01"],
    "spine": ["spine04", "spine03", "spine02", "spine01"], "neck": ["neck01", "neck02", "neck03"],
    "head": ["head", "neck01", "neck02", "neck03"],
}


def _h(rig, b):
    return rig.pose.bones[b].head.copy()


def _ang(a, b):
    if a.length < 1e-6 or b.length < 1e-6:
        return 0.0
    return math.degrees(a.angle(b))


def measure(rig):
    """Joint angles (deg) from bone heads, armature space."""
    pb = rig.pose.bones
    hipL, hipR = _h(rig, "upperleg01.L"), _h(rig, "upperleg01.R")
    pel = (hipL + hipR) / 2
    left = (hipL - hipR).normalized()
    up_p = (_h(rig, "spine03") - pel)
    up_p = (up_p - left * up_p.dot(left)).normalized()
    fwd_p = left.cross(up_p)
    chest = _h(rig, "spine01")
    neck = _h(rig, "neck01")
    shL, shR = _h(rig, "upperarm01.L"), _h(rig, "upperarm01.R")
    left_c = (shL - shR).normalized()
    up_c = (neck - _h(rig, "spine04"))
    up_c = (up_c - left_c * up_c.dot(left_c)).normalized()
    fwd_c = left_c.cross(up_c)
    out = {}
    for s, sg in (("L", 1), ("R", -1)):
        sh, el, wr = _h(rig, f"upperarm01.{s}"), _h(rig, f"lowerarm01.{s}"), _h(rig, f"wrist.{s}")
        hip, kn, an = _h(rig, f"upperleg01.{s}"), _h(rig, f"lowerleg01.{s}"), _h(rig, f"foot.{s}")
        out[f"elbow_flex.{s}"] = _ang(el - sh, wr - el)
        th, shn = kn - hip, an - kn
        k = _ang(th, shn)
        side = th.cross(shn).dot(left)          # > 0: the knee bends backwards (anatomical)
        out[f"knee_flex.{s}"] = k if side >= -1e-4 or k < 3 else 0.0
        out[f"knee_hyperext.{s}"] = k if (side < -1e-4 and k >= 3) else 0.0
        t = th.normalized()
        out[f"hip_flex.{s}"] = math.degrees(math.atan2(t.dot(fwd_p), -t.dot(up_p)))
        out[f"hip_abd.{s}"] = math.degrees(math.asin(max(-1.0, min(1.0, t.dot(left * sg)))))   # out of the sagittal plane
        a = (el - sh).normalized()
        out[f"shoulder_elev.{s}"] = _ang(a, -up_c)
        back = -a.dot(fwd_c)
        out[f"shoulder_ext.{s}"] = math.degrees(math.atan2(back, max(1e-6, -a.dot(up_c)))) if back > 0 and a.dot(up_c) < 0 else 0.0
    out["spine_bend"] = _ang(up_p, up_c)
    lp = left - up_c * left.dot(up_c)
    lc = left_c - up_c * left_c.dot(up_c)
    out["spine_twist"] = _ang(lp, lc)
    head_up = (pb["head"].tail - pb["head"].head)
    out["neck_bend"] = _ang(up_c, head_up)
    hx = pb["head"].matrix.to_3x3() @ Vector((1, 0, 0))
    hx = hx - up_c * hx.dot(up_c)
    out["head_turn"] = _ang(hx, lc)
    return {k: round(v, 1) for k, v in out.items()}


def violations(ang):
    v = {}
    for k, (lo, hi) in LIMITS.items():
        x = ang.get(k)
        if x is not None and not lo - 0.5 <= x <= hi + 0.5:
            v[k] = (x, lo, hi)
    return v


def _bones_for(key, rig):
    joint, side = (key.split(".") + [None])[:2]
    base = joint.split("_")[0]
    names = CLAMP_BONES[base]
    return [f"{n}.{side}" if side else n for n in names if (f"{n}.{side}" if side else n) in rig.pose.bones]


def clamp(rig, max_iter=12):
    """Scale the responsible bones back towards rest until every joint is inside LIMITS. Returns the log."""
    log = []
    for _ in range(3):
        bad = violations(measure(rig))
        if not bad:
            break
        for key, (x, lo, hi) in bad.items():
            bones = _bones_for(key, rig)
            q0 = {b: rig.pose.bones[b].matrix_basis.to_quaternion() for b in bones}
            loc = {b: rig.pose.bones[b].matrix_basis.to_translation() for b in bones}
            lo_t, hi_t = 0.0, 1.0
            for _ in range(max_iter):
                t = (lo_t + hi_t) / 2
                for b in bones:
                    q = Quaternion().slerp(q0[b], t)
                    rig.pose.bones[b].matrix_basis = Matrix.Translation(loc[b]) @ q.to_matrix().to_4x4()
                bpy.context.view_layer.update()
                if key in violations(measure(rig)):
                    hi_t = t
                else:
                    lo_t = t
            for b in bones:
                q = Quaternion().slerp(q0[b], lo_t)
                rig.pose.bones[b].matrix_basis = Matrix.Translation(loc[b]) @ q.to_matrix().to_4x4()
            bpy.context.view_layer.update()
            log.append({"joint": key, "was": x, "range": [lo, hi], "bones": bones, "kept": round(lo_t, 3),
                        "now": measure(rig).get(key)})
            print(f"[limits] WARNING {key} {x} outside {lo}..{hi} -> clamped ({', '.join(bones)} x{lo_t:.2f})")
    return log


def bone_lengths(rig):
    """Rest length vs posed head-tail distance per deforming bone; max deviation in mm (rotation-only poses: 0)."""
    dev = 0.0
    lengths = {}
    for pb in rig.pose.bones:
        if not pb.bone.use_deform:
            continue
        rest = pb.bone.length
        posed = (pb.tail - pb.head).length
        dev = max(dev, abs(posed - rest))
        if not pb.name.startswith(("finger", "toe", "metacarpal", "oculi", "eye", "jaw", "tongue", "special")):
            lengths[pb.name] = round(rest * 1000, 1)
    return {"max_dev_mm": round(dev * 1000, 3), "lengths_mm": lengths}


# ---------------------------------------------------------------------------------------------- pose files
def apply_pose_file(rig, spec, reach_fn, rotate_fn):
    if "rotations" in spec:
        order = [b.name for b in rig.data.bones]            # parents come before children in Blender's list
        for name in order:
            q = spec["rotations"].get(name)
            if q is not None:
                pb = rig.pose.bones[name]
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = Quaternion(q)
        bpy.context.view_layer.update()
    for st in spec.get("steps", []):
        if st[0] == "@reach":
            reach_fn(rig, st[1], st[2], tuple(st[3]))
        elif st[0] in rig.pose.bones and st[1] in ("X", "Y", "Z"):
            rotate_fn(rig, st[0], st[1], float(st[2]))
        else:
            print(f"[posefile] WARNING bad step {st}")
    return spec.get("contact", "solve")


# ---------------------------------------------------------------------------------------------- contacts
PARTS = {
    "pelvis": ["pelvis.L", "pelvis.R"], "back": ["spine04", "spine03", "spine02"], "head": ["neck03"],
    "foot.L": ["foot.L"], "foot.R": ["foot.R"], "hand.L": ["wrist.L"], "hand.R": ["wrist.R"],
    "knee.L": ["lowerleg01.L"], "knee.R": ["lowerleg01.R"], "elbow.L": ["lowerarm01.L"], "elbow.R": ["lowerarm01.R"],
    "shin.L": ["lowerleg01.L", "lowerleg02.L"], "shin.R": ["lowerleg01.R", "lowerleg02.R"],
    "thigh.L": ["upperleg01.L", "upperleg02.L"], "thigh.R": ["upperleg01.R", "upperleg02.R"],
}


def _part_points(body, groups, step=1, min_w=0.5):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    idx = {body.vertex_groups[g].index for g in groups if g in body.vertex_groups}
    pts = [ev.matrix_world @ v.co for i, v in enumerate(me.vertices)
           if i % step == 0 and any(g.group in idx and g.weight >= min_w for g in v.groups)]
    ev.to_mesh_clear()
    return pts


def _all_points(body, step=5):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    pts = [ev.matrix_world @ v.co for i, v in enumerate(me.vertices) if i % step == 0]
    ev.to_mesh_clear()
    return pts


def plate_bvh(plate):
    dg = bpy.context.evaluated_depsgraph_get()
    return BVHTree.FromObject(plate, dg)


def _signed(bvh, p, cam_pos):
    q, n, _i, _d = bvh.find_nearest(p)
    if q is None:
        return None, None
    if n.dot(cam_pos - q) < 0:          # MoGe mesh normals: orient towards the camera (visible side = free space)
        n = -n
    return (p - q).dot(n), n


def contact_gaps(body, bvh, contacts, cam_pos):
    """Per contact part: min signed distance (m) of its vertices to the plate surface (>0 gap, <0 penetration)."""
    res = []
    for c in contacts:
        groups = PARTS.get(c["part"])
        if not groups:
            continue
        best = None
        for p in _part_points(body, groups, step=3):
            d, n = _signed(bvh, p, cam_pos)
            if d is not None and (best is None or d < best[0]):
                best = (d, n)
        if best:
            res.append({"part": c["part"], "surface": c.get("surface"), "gap_cm": round(best[0] * 100, 1),
                        "normal": [round(x, 2) for x in best[1]]})
    return res


# Support priority (07.11d): the body rests on its heaviest support; other contacts are reached by the limbs.
# Lower number = stronger support. A contact may override with "role": "support" | "secondary".
PRIORITY = {"pelvis": 0, "back": 0, "thigh.L": 1, "thigh.R": 1, "knee.L": 1, "knee.R": 1, "shin.L": 1, "shin.R": 1,
            "foot.L": 2, "foot.R": 2, "elbow.L": 3, "elbow.R": 3, "hand.L": 4, "hand.R": 4, "head": 5}
LIMB_FIX_MAX = 0.35      # m; a secondary contact further than this contradicts the pose -> reported, not forced
LEG_DOF = [("upperleg01", "X", -45, 45), ("upperleg01", "Y", -25, 25), ("upperleg01", "Z", -25, 25),
           ("lowerleg01", "X", -70, 70)]


def _tiers(contacts):
    """(support, secondary). Explicit roles win; otherwise the strongest PRIORITY class present is the support."""
    if any(c.get("role") for c in contacts):
        sup = [c for c in contacts if c.get("role") == "support"]
        sec = [c for c in contacts if c.get("role") != "support"]
        if sup:
            return sup, sec
    best = min(PRIORITY.get(c["part"], 9) for c in contacts)
    return ([c for c in contacts if PRIORITY.get(c["part"], 9) == best],
            [c for c in contacts if PRIORITY.get(c["part"], 9) != best])


def _settle_support(rig, body, bvh, sup, cam_pos, iters):
    """Translate the rig along the mean support normal until the deepest support part touches (min gap = 0)."""
    for _ in range(iters):
        gaps = contact_gaps(body, bvh, sup, cam_pos)
        if not gaps:
            return
        n = Vector()
        for gp in gaps:
            n += Vector(gp["normal"])
        n.normalize()
        d = min(gp["gap_cm"] for gp in gaps) / 100
        if abs(d) < 0.002:
            return
        rig.location -= n * d
        bpy.context.view_layer.update()


def leg_reach(rig, side, target, rotate_fn):
    """Coordinate descent on hip X/Y/Z + knee X so the ankle (foot.<side> head) reaches target (armature space)."""
    pbs = rig.pose.bones
    eff = f"foot.{side}"
    names = [f"{b}.{side}" for b, *_ in LEG_DOF]
    if eff not in pbs or any(n not in pbs for n in names):
        return None
    mods = [m for o in bpy.data.objects if o.type == "MESH" for m in o.modifiers if m.show_viewport]
    for m in mods:
        m.show_viewport = False
    acc = [0.0] * len(LEG_DOF)
    best = (pbs[eff].head - target).length
    for step in (8, 4, 2, 1):
        improved = True
        while improved:
            improved = False
            for i, (_b, axis, lo, hi) in enumerate(LEG_DOF):
                for sgn in (1, -1):
                    d = step * sgn
                    if not lo <= acc[i] + d <= hi:
                        continue
                    saved = [pbs[n].matrix_basis.copy() for n in names]
                    rotate_fn(rig, names[i], axis, d)
                    c = (pbs[eff].head - target).length
                    if c < best - 1e-5:
                        best, acc[i], improved = c, acc[i] + d, True
                        break
                    for n, mb in zip(names, saved):
                        pbs[n].matrix_basis = mb
                    bpy.context.view_layer.update()
    for m in mods:
        m.show_viewport = True
    bpy.context.view_layer.update()
    return best


def _fix_secondary(rig, body, bvh, sec, cam_pos, reach_fn, rotate_fn):
    """Move each hand/foot secondary contact onto its surface with limb IK (other parts are reported only)."""
    log = []
    inv = rig.matrix_world.inverted()
    for c in sec:
        part = c["part"]
        if part.split(".")[0] not in ("hand", "foot"):
            continue
        g = contact_gaps(body, bvh, [c], cam_pos)
        if not g:
            continue
        gap = g[0]["gap_cm"] / 100
        if abs(gap) <= 0.02 or abs(gap) > LIMB_FIX_MAX:
            log.append({"part": part, "gap_cm": round(gap * 100, 1), "fixed": False})
            continue
        side = part.split(".")[1]
        shift = -Vector(g[0]["normal"]) * gap                       # world
        pbs = rig.pose.bones
        if part.startswith("foot"):
            tgt_w = rig.matrix_world @ pbs[f"foot.{side}"].head + shift
            miss = leg_reach(rig, side, inv @ tgt_w, rotate_fn)
        else:
            eff = pbs[f"finger3-1.{side}"]
            tgt = inv @ (rig.matrix_world @ eff.head + shift)
            anchor = f"upperarm01.{side}"
            off = tgt - pbs[anchor].head
            sx = 1 if side == "L" else -1
            reach_fn(rig, side, anchor, (off.x * sx, off.y, off.z))
            miss = (pbs[f"finger3-1.{side}"].head - tgt).length
        log.append({"part": part, "gap_cm": round(gap * 100, 1), "fixed": True,
                    "miss_cm": None if miss is None else round(miss * 100, 1)})
    return log


def solve_contacts(rig, body, plate, contacts, cam_pos, iters=10, reach_fn=None, rotate_fn=None):
    """07.11d: (1) the support tier (highest PRIORITY present) is settled exactly on its surface by translating the
    rig (deepest support part gap = 0, no averaging with other contacts); (2) secondary hand/foot contacts are moved
    onto their surfaces by limb IK (reach_fn/rotate_fn from poses.py), then joint limits are re-clamped;
    (3) everything is re-measured. Without reach_fn/rotate_fn only step (1) runs."""
    bvh = plate_bvh(plate)
    sup, sec = _tiers(contacts)
    _settle_support(rig, body, bvh, sup, cam_pos, iters)
    limb = []
    if sec and reach_fn and rotate_fn:
        limb = _fix_secondary(rig, body, bvh, sec, cam_pos, reach_fn, rotate_fn)
        if any(x.get("fixed") for x in limb):
            clamp(rig)
            _settle_support(rig, body, bvh, sup, cam_pos, iters)
    gaps = contact_gaps(body, bvh, contacts, cam_pos)
    pen = 0.0
    for p in _all_points(body, step=7):
        d, n = _signed(bvh, p, cam_pos)
        # only supporting surfaces (floor, bed, bench tops: normal up); a body behind furniture is not "inside" it
        if d is not None and d < pen and abs(d) < 0.3 and n.z > 0.6:
            pen = d
    return {"contacts": gaps, "max_penetration_cm": round(pen * 100, 1),
            "support": [c["part"] for c in sup], "limb_fix": limb}


# ---------------------------------------------------------------------------------------------- MHR70 retarget
K = {"nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4, "l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8,
     "l_hip": 9, "r_hip": 10, "l_kn": 11, "r_kn": 12, "l_an": 13, "r_an": 14, "l_btoe": 15, "l_stoe": 16,
     "l_heel": 17, "r_btoe": 18, "r_stoe": 19, "r_heel": 20, "r_wr": 41, "l_wr": 62, "r_mid": 32, "l_mid": 53,
     "neck": 69}


def mhr_to_armature(kp):
    """MHR70 camera coords (x right, y down, z forward) -> armature frame (her left +X, up +Z, facing -Y).
    Yaw is removed (the scene sets it); pitch/roll of the body w.r.t. gravity are kept (lying, all fours)."""
    P = {k: Vector(kp[i]) for k, i in K.items()}
    up = Vector((0, -1, 0))
    mid_hip = (P["l_hip"] + P["r_hip"]) / 2
    left = P["l_hip"] - P["r_hip"]
    spine = P["neck"] - mid_hip
    fwd = left.cross(spine)                              # body forward
    f_h = fwd - up * fwd.dot(up)
    s_h = spine - up * spine.dot(up)
    if f_h.length >= 0.5 * fwd.length:                   # upright-ish or lying on the side: facing is horizontal
        y_ax = -f_h.normalized()
    else:                                                # lying on back/front: use the spine direction (head = +Y)
        y_ax = s_h.normalized()
    x_ax = y_ax.cross(up).normalized()
    R = Matrix((x_ax, y_ax, up))                         # rows: armature axes in camera coords
    return {k: R @ (v - mid_hip) for k, v in P.items()}


AIM = [  # (bone, aim joint name in the rig, target from, target to)
    ("spine05", "neck01", "mid_hip", "neck"),
    ("neck01", "head", "neck", "mid_ear"),
    ("upperarm01.L", "lowerarm01.L", "l_sh", "l_el"), ("lowerarm01.L", "wrist.L", "l_el", "l_wr"),
    ("wrist.L", "finger3-1.L", "l_wr", "l_mid"),
    ("upperarm01.R", "lowerarm01.R", "r_sh", "r_el"), ("lowerarm01.R", "wrist.R", "r_el", "r_wr"),
    ("wrist.R", "finger3-1.R", "r_wr", "r_mid"),
    ("upperleg01.L", "lowerleg01.L", "l_hip", "l_kn"), ("lowerleg01.L", "foot.L", "l_kn", "l_an"),
    ("foot.L", "@toe.L", "l_an", "l_toe"),
    ("upperleg01.R", "lowerleg01.R", "r_hip", "r_kn"), ("lowerleg01.R", "foot.R", "r_kn", "r_an"),
    ("foot.R", "@toe.R", "r_an", "r_toe"),
]


def retarget(rig, kp):
    """Direction retarget: every bone keeps OUR length; it is turned (minimal rotation about its head) so the next
    joint lies along the MHR bone direction. Root carries the body pitch/roll (from the hip/spine frame)."""
    T = mhr_to_armature(kp)
    T["mid_hip"] = (T["l_hip"] + T["r_hip"]) / 2
    T["mid_ear"] = (T["l_ear"] + T["r_ear"]) / 2
    T["l_toe"] = (T["l_btoe"] + T["l_stoe"]) / 2
    T["r_toe"] = (T["r_btoe"] + T["r_stoe"]) / 2
    pbs = rig.pose.bones
    # 1) pelvis frame -> root rotation (hip line and spine direction)
    hipL, hipR = pbs["upperleg01.L"].head, pbs["upperleg01.R"].head
    cur_x = (hipL - hipR).normalized()
    cur_z = (pbs["neck01"].head - (hipL + hipR) / 2)
    cur_z = (cur_z - cur_x * cur_z.dot(cur_x)).normalized()
    tgt_x = (T["l_hip"] - T["r_hip"]).normalized()
    tgt_z = (T["neck"] - T["mid_hip"])
    tgt_z = (tgt_z - tgt_x * tgt_z.dot(tgt_x)).normalized()
    Mc = Matrix((cur_x, cur_z.cross(cur_x), cur_z)).transposed()
    Mt = Matrix((tgt_x, tgt_z.cross(tgt_x), tgt_z)).transposed()
    rot = (Mt @ Mc.inverted()).to_4x4()
    root = pbs["root"]
    head = root.head.copy()
    root.matrix = Matrix.Translation(head) @ rot @ Matrix.Translation(-head) @ root.matrix
    bpy.context.view_layer.update()
    for bone, aim, a, b in AIM:
        if bone not in pbs:
            continue
        pb = pbs[bone]
        if aim.startswith("@toe"):
            end = pb.tail.copy()
        elif aim in pbs:
            end = pbs[aim].head.copy()
        else:
            continue
        v = end - pb.head
        d = T[b] - T[a]
        if v.length < 1e-5 or d.length < 1e-5:
            continue
        q = v.rotation_difference(d)
        h = pb.head.copy()
        pb.matrix = Matrix.Translation(h) @ q.to_matrix().to_4x4() @ Matrix.Translation(-h) @ pb.matrix
        bpy.context.view_layer.update()
    return {n: list(pbs[n].matrix_basis.to_quaternion()) for n in [b.name for b in rig.data.bones]
            if pbs[n].matrix_basis.to_quaternion().angle > 1e-4}


# ---------------------------------------------------------------------------------------------- 14.11 MPFB body
# The render body is Stacy's own MPFB mannequin (fingers, toes, face), posed onto the placed MHR body: the MHR mesh
# and its MHR70 joints stay the source of truth for placement/contacts; the mannequin copies the joint directions
# (its own bone lengths), is moved onto the MHR torso and settled to the same contact height above the room mesh.
MPFB_KEEP_HELPERS = ("helper-l-eye", "helper-r-eye")   # eyeballs stay visible (the face keeps its eyes)
FINGER_BASE = {"R": 21, "L": 42}                        # MHR70 hand blocks: per finger tip, DIP, PIP, MCP; thumb..pinky


def prepare_mannequin(rig, body):
    """Hair/clothes/other proxies off, clothes 'Delete.*' masks dropped, helpers hidden except the eyeballs."""
    removed = []
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o is not body and (o.parent == rig or o.find_armature() == rig):
            removed.append(o.name)
            bpy.data.objects.remove(o, do_unlink=True)
    for m in list(body.modifiers):
        if m.type == "MASK" and (m.vertex_group or "").startswith("Delete."):
            removed.append("mask:" + m.name)
            body.modifiers.remove(m)
    src = {body.vertex_groups[n].index for n in ("body",) + MPFB_KEEP_HELPERS if n in body.vertex_groups}
    if src:
        vis = body.vertex_groups.get("visible_body") or body.vertex_groups.new(name="visible_body")
        vis.add([v.index for v in body.data.vertices if any(g.group in src and g.weight > 0 for g in v.groups)],
                1.0, "REPLACE")
        for m in body.modifiers:
            if m.type == "MASK" and m.vertex_group == "body":
                m.vertex_group = "visible_body"
    return removed


def _kabsch(cur, tgt):
    """Rotation (3x3) taking the centred point set cur onto tgt (least squares, no reflection)."""
    import numpy as np
    A = np.array([tuple(p) for p in cur], float)
    B = np.array([tuple(p) for p in tgt], float)
    A -= A.mean(0)
    B -= B.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Matrix([list(r) for r in R])


def _turn(pb, M):
    h = pb.head.copy()
    pb.matrix = Matrix.Translation(h) @ M.to_4x4() @ Matrix.Translation(-h) @ pb.matrix
    bpy.context.view_layer.update()


def _aim_bone(pbs, bone, end, d):
    if bone not in pbs or d.length < 1e-5:
        return
    pb = pbs[bone]
    v = end(pbs) - pb.head
    if v.length > 1e-5:
        _turn(pb, v.rotation_difference(d).to_matrix())


def mhr_targets(kp, to_arm):
    """MHR70 (world, Z up) -> armature-space points incl. finger chains (mcp, pip, dip, tip)."""
    T = {k: to_arm @ Vector(tuple(kp[i])) for k, i in K.items()}
    T["mid_hip"] = (T["l_hip"] + T["r_hip"]) / 2
    T["mid_ear"] = (T["l_ear"] + T["r_ear"]) / 2
    for side, base in FINGER_BASE.items():
        for f in range(5):
            tip, dip, pip, mcp = (to_arm @ Vector(tuple(kp[base + 4 * f + j])) for j in range(4))
            T.update({f"{side}{f + 1}_0": mcp, f"{side}{f + 1}_1": pip, f"{side}{f + 1}_2": dip, f"{side}{f + 1}_3": tip})
    return T


def retarget_world(rig, kp):
    """Pose the MPFB rig onto MHR70 joints given in WORLD space (rig object transform kept).

    Root frame from hips + spine, then bone aims (spine, neck, limbs, finger chains) and rigid frames for the head
    (eyes), hands (MCP row) and feet (toe tips). Bone lengths stay the mannequin's."""
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    T = mhr_targets(kp, rig.matrix_world.inverted())
    pbs = rig.pose.bones
    hipL, hipR = pbs["upperleg01.L"].head, pbs["upperleg01.R"].head
    cur_x = (hipL - hipR).normalized()
    cur_z = pbs["neck01"].head - (hipL + hipR) / 2
    cur_z = (cur_z - cur_x * cur_z.dot(cur_x)).normalized()
    tgt_x = (T["l_hip"] - T["r_hip"]).normalized()
    tgt_z = T["neck"] - T["mid_hip"]
    tgt_z = (tgt_z - tgt_x * tgt_z.dot(tgt_x)).normalized()
    Mc = Matrix((cur_x, cur_z.cross(cur_x), cur_z)).transposed()
    Mt = Matrix((tgt_x, tgt_z.cross(tgt_x), tgt_z)).transposed()
    _turn(pbs["root"], Mt @ Mc.inverted())
    head = lambda n: (lambda p: p[n].head.copy())                     # noqa: E731
    tail = lambda n: (lambda p: p[n].tail.copy())                     # noqa: E731
    # torso: shoulders + neck relative to the hips (bend AND twist), split between the lower and the upper spine
    chest = lambda: [pbs["neck01"].head.copy(), pbs["upperarm01.L"].head.copy(), pbs["upperarm01.R"].head.copy(),
                     pbs["spine05"].head.copy()]                      # noqa: E731
    goal = [T["neck"], T["l_sh"], T["r_sh"], T["mid_hip"]]
    R = _kabsch(chest(), goal).to_quaternion()
    _turn(pbs["spine05"], Quaternion().slerp(R, 0.5).to_matrix())
    if "spine02" in pbs:
        _turn(pbs["spine02"], _kabsch(chest(), goal))
    _aim_bone(pbs, "neck01", head("head"), T["mid_ear"] - T["neck"])
    if all(n in pbs for n in ("head", "eye.L", "eye.R")):              # head: eyes + neck base, rigid
        _turn(pbs["head"], _kabsch([pbs["eye.L"].head, pbs["eye.R"].head, pbs["neck01"].head],
                                   [T["l_eye"], T["r_eye"], T["neck"]]))
    for s, side in (("L", "l"), ("R", "r")):
        if f"clavicle.{s}" in pbs:                                     # shoulder joint onto the MHR shoulder
            base = T["neck"] + (pbs[f"clavicle.{s}"].head - pbs["neck01"].head)
            _aim_bone(pbs, f"clavicle.{s}", head(f"upperarm01.{s}"), T[f"{side}_sh"] - base)
        _aim_bone(pbs, f"upperarm01.{s}", head(f"lowerarm01.{s}"), T[f"{side}_el"] - T[f"{side}_sh"])
        _aim_bone(pbs, f"lowerarm01.{s}", head(f"wrist.{s}"), T[f"{side}_wr"] - T[f"{side}_el"])
        hand = [f"finger{f}-1.{s}" for f in (2, 3, 5)]
        if all(n in pbs for n in hand):                                # hand: wrist + index/middle/pinky MCP
            _turn(pbs[f"wrist.{s}"], _kabsch([pbs[f"wrist.{s}"].head] + [pbs[n].head for n in hand],
                                             [T[f"{side}_wr"], T[f"{s}2_0"], T[f"{s}3_0"], T[f"{s}5_0"]]))
        for f in range(1, 6):
            for j in (1, 2, 3):
                bone = f"finger{f}-{j}.{s}"
                nxt = head(f"finger{f}-{j + 1}.{s}") if j < 3 else tail(bone)
                _aim_bone(pbs, bone, nxt, T[f"{s}{f}_{j}"] - T[f"{s}{f}_{j - 1}"])
        _aim_bone(pbs, f"upperleg01.{s}", head(f"lowerleg01.{s}"), T[f"{side}_kn"] - T[f"{side}_hip"])
        _aim_bone(pbs, f"lowerleg01.{s}", head(f"foot.{s}"), T[f"{side}_an"] - T[f"{side}_kn"])
        if all(n in pbs for n in (f"foot.{s}", f"toe1-2.{s}", f"toe5-3.{s}")):   # foot: ankle + toe tips, rigid
            _turn(pbs[f"foot.{s}"], _kabsch([pbs[f"foot.{s}"].head, pbs[f"toe1-2.{s}"].tail, pbs[f"toe5-3.{s}"].tail],
                                            [T[f"{side}_an"], T[f"{side}_btoe"], T[f"{side}_stoe"]]))
    return T


def mpfb_joints_world(rig):
    """MPFB joints matching MHR70 names used for the rigid fit (world)."""
    p, mw = rig.pose.bones, rig.matrix_world
    J = {"l_hip": p["upperleg01.L"].head, "r_hip": p["upperleg01.R"].head, "l_sh": p["upperarm01.L"].head,
         "r_sh": p["upperarm01.R"].head, "neck": p["neck01"].head, "l_kn": p["lowerleg01.L"].head,
         "r_kn": p["lowerleg01.R"].head, "l_el": p["lowerarm01.L"].head, "r_el": p["lowerarm01.R"].head,
         "l_wr": p["wrist.L"].head, "r_wr": p["wrist.R"].head, "l_an": p["foot.L"].head, "r_an": p["foot.R"].head}
    return {k: mw @ v for k, v in J.items()}


def _height_above(bvh, pts, lift=0.5):
    """Vertical clearance of each point above the room mesh (negative = below the surface); None = nothing below."""
    out = []
    down = Vector((0, 0, -1))
    for p in pts:
        hit = bvh.ray_cast(p + Vector((0, 0, lift)), down, 5.0)
        out.append(None if hit[0] is None else hit[3] - lift)
    return out


def _low_clearance(bvh, pts, q=0.02):
    vals = sorted(v for v in _height_above(bvh, pts) if v is not None)
    return vals[int(q * (len(vals) - 1))] if vals else None


def contact_shift(bvh, V, mesh, report, band=0.02, sample=4000):
    """Vertical shift that gives the mannequin the MHR body's clearance in the MHR contact zone (m; 0 = no data)."""
    import numpy as np
    from mathutils.kdtree import KDTree
    V = V[::max(1, len(V) // sample)]
    cm = np.array([np.nan if c is None else c for c in _height_above(bvh, [Vector(tuple(v)) for v in V])])
    if not np.isfinite(cm).any():
        return 0.
    low = np.nanquantile(cm, 0.02)
    touch = np.where(cm <= low + band)[0]
    tree = KDTree(len(mesh.vertices))
    for v in mesh.vertices:
        tree.insert(v.co, v.index)
    tree.balance()
    near = sorted({tree.find(Vector(tuple(V[i])))[1] for i in touch})
    cp = np.array([np.nan if c is None else c for c in _height_above(bvh, [mesh.vertices[i].co for i in near])])
    if not np.isfinite(cp).any():
        return 0.
    dz = float(np.nanmedian(cm[touch]) - np.nanmedian(cp))
    report.update(contact_dz_m=round(dz, 4), clearance_mhr_m=round(float(low), 4), contact_points=int(len(touch)))
    return dz


def pose_mannequin_to_mhr(rig, body, kp, mhr_verts=None, plate=None, subsurf=1, scale_limits=(0.85, 1.15)):
    """Pose + place the MPFB mannequin on the MHR placement; return a static world mesh object (body_v2) and a report.

    kp: MHR70 joints (world, 70x3). mhr_verts: placed MHR vertices (world) for the contact height; plate: room mesh."""
    import numpy as np
    report = {"model": "mpfb", "removed": prepare_mannequin(rig, body)}
    hidden = [(o, o.hide_viewport) for o in (rig, body)]
    for o, _ in hidden:
        o.hide_viewport = False
    bpy.context.view_layer.update()
    kp = np.asarray(kp, float)
    # body size: MHR and the mannequin are both Stacy (162 cm); a residual size difference is taken from the legs+torso
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    Jr = mpfb_joints_world(rig)
    p = rig.pose.bones
    mw = rig.matrix_world

    def chain(P, ank):
        return sum(((P[a] - P[b]).length for a, b in (("l_hip", "l_kn"), ("r_hip", "r_kn"))), 0.) + \
               ((P["l_hip"] + P["r_hip"]) / 2 - P["neck"]).length + ank
    mh = {k: Vector(tuple(kp[i])) for k, i in K.items()}
    ank_m = (mh["l_kn"] - mh["l_an"]).length + (mh["r_kn"] - mh["r_an"]).length
    ank_r = ((mw @ p["lowerleg01.L"].head) - (mw @ p["foot.L"].head)).length + \
            ((mw @ p["lowerleg01.R"].head) - (mw @ p["foot.R"].head)).length
    scale = chain(mh, ank_m) / max(chain(Jr, ank_r), 1e-6)
    report["scale_raw"] = round(scale, 4)
    scale = min(max(scale, scale_limits[0]), scale_limits[1])
    rig.scale = rig.scale * scale
    bpy.context.view_layer.update()
    report["scale"] = round(scale, 4)
    retarget_world(rig, kp)
    # rigid move onto the MHR torso (hips, shoulders, neck): translation of the centroids
    J = mpfb_joints_world(rig)
    keys = ("l_hip", "r_hip", "l_sh", "r_sh", "neck")
    shift = sum((mh[k] - J[k] for k in keys), Vector()) / len(keys)
    rig.location = rig.location + shift
    bpy.context.view_layer.update()
    if subsurf:
        sub = body.modifiers.get("render_smooth") or body.modifiers.new("render_smooth", "SUBSURF")
        sub.levels = sub.render_levels = int(subsurf)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    mesh = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
    mesh.transform(ev.matrix_world)
    # contact: where the MHR body touches the room mesh (its lowest clearance band), the mannequin's nearest
    # surface gets the same clearance -> feet on the floor, seat on the seat, side on the bed (vertical shift only)
    if plate is not None and mhr_verts is not None:
        dz = contact_shift(plate_bvh(plate), np.asarray(mhr_verts, float), mesh, report)
        if dz:
            mesh.transform(Matrix.Translation((0, 0, dz)))
            rig.location.z += dz
    obj = bpy.data.objects.new("body_v2", mesh)
    bpy.context.scene.collection.objects.link(obj)
    for poly in mesh.polygons:
        poly.use_smooth = True
    for o, value in hidden:
        o.hide_viewport = value
    J = mpfb_joints_world(rig)
    report["joint_err_cm"] = {k: round((J[k] - mh[k]).length * 100, 1) for k in J}
    report["vertices"] = len(mesh.vertices)
    bpy.context.view_layer.update()
    return obj, report


def write_obj(obj, path):
    """World-space triangulated OBJ (same convention as fit/body.obj: Blender axes, forward Y / up Z on import)."""
    me = obj.data
    me.calc_loop_triangles()
    mw = obj.matrix_world
    with open(path, "w", encoding="utf-8") as f:
        for v in me.vertices:
            co = mw @ v.co
            f.write(f"v {co.x:.6f} {co.y:.6f} {co.z:.6f}\n")
        for t in me.loop_triangles:
            f.write("f %d %d %d\n" % tuple(i + 1 for i in t.vertices))
