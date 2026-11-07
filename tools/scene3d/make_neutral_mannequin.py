"""Neutral mannequin for 3D-A (07.11, user's decision): Stacy's body shape and height only - no face features, no
hair, no clothes, no anatomical detail; the MPFB rig (joints from MakeHuman joint groups) is kept unchanged.

  blender -b --python tools/scene3d/make_neutral_mannequin.py -- --src stacy_mannequin_162.blend --out stacy_neutral_162.blend
  (or the bpy wheel: python tools/scene3d/make_neutral_mannequin.py -- ...)

Steps: delete clothes/hair/proxy-eye objects; drop the clothes "Delete.*" mask (helpers stay hidden); smooth the
face (eyes, lids, nose, lips) into a plain egg shape with a weighted Smooth modifier (applied after the Armature, so
it follows the head in any pose); same for nipples/navel if the mesh has them; neutral light-grey material.
Prints height, rest-pose joint check (left/right symmetry, joint heights) for the record.
"""
import argparse
import sys

import bpy
from mathutils import Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--face-iter", type=int, default=20)
    ap.add_argument("--face-voxel", type=float, default=0.03, help="voxel size of the head proxy (closes slits)")
    return ap.parse_args(argv)


FACE_BONES = ("jaw", "eye", "lid", "lip", "tongue", "brow", "cheek", "nose", "special", "levator", "oris", "temporalis",
              "risorius", "orbicularis")


def main():
    a = args()
    bpy.ops.wm.open_mainfile(filepath=a.src)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    body = bpy.data.objects.get("stacy_body") or max((o for o in bpy.data.objects if o.type == "MESH"),
                                                     key=lambda o: len(o.data.vertices))
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o is not body:
            print("[neutral] delete", o.name)
            bpy.data.objects.remove(o, do_unlink=True)
    for m in list(body.modifiers):
        if m.type == "MASK" and m.vertex_group and m.vertex_group.startswith("Delete."):
            body.modifiers.remove(m)
    from mathutils import Matrix
    for pbone in rig.pose.bones:                    # neutral file = rest pose (no leftover pose from earlier rounds)
        pbone.matrix_basis = Matrix()
    rig.location, rig.rotation_euler = (0, 0, 0), (0, 0, 0)
    bpy.context.view_layer.update()
    # MPFB macro shape keys define Stacy's body: bake their mix into the mesh first (edits below then stick)
    if body.data.shape_keys:
        bpy.context.view_layer.objects.active = body
        with bpy.context.temp_override(object=body, active_object=body, selected_objects=[body]):
            bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
        print("[neutral] shape keys baked")
    # vertex indices must stay intact (render_h3_maps keypoints use fixed MPFB indices and the joint-* helper groups),
    # so nothing is deleted: helpers stay behind the "Hide helpers" mask, the face/skull goes behind a second mask
    import bmesh
    for m in list(body.modifiers):
        if m.type == "MASK" and m.vertex_group and m.vertex_group.startswith("Delete."):
            body.modifiers.remove(m)
    # head -> plain egg (artist-mannequin style): the MPFB face/skull vertices go, a smooth ellipsoid of the same
    # size is parented to the head bone (rigid, follows every pose); the neck opening is capped
    vg_names = {g.index: g.name for g in body.vertex_groups}
    face_idx = {i for i, n in vg_names.items() if any(k in n.lower() for k in FACE_BONES)}
    head_idx = {i for i, n in vg_names.items() if n == "head"}
    head_set = face_idx | head_idx
    pb = rig.pose.bones
    mw = body.matrix_world
    cut_vg = body.vertex_groups.new(name="neutral_head_hidden")
    cut, pts = [], []
    body_gi = body.vertex_groups["body"].index if "body" in body.vertex_groups else None
    for v in body.data.vertices:
        in_body = body_gi is None or any(g.group == body_gi and g.weight > 0 for g in v.groups)
        if in_body and sum(g.weight for g in v.groups if g.group in head_set) >= 0.5:
            cut.append(v.index)
            pts.append(mw @ v.co)
    cut_vg.add(cut, 1.0, "REPLACE")
    hide = body.modifiers.new("Hide head", "MASK")
    hide.vertex_group = "neutral_head_hidden"
    hide.invert_vertex_group = True
    # the mask must act before the armature-independent smooth modifiers but its order vs Armature does not matter
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    cap = {"faces": []}
    size = hi - lo
    centre = (lo + hi) / 2
    # ears widen the box: the skull is ~80 % of it; the egg goes a bit lower to cover the cut under the jaw
    rx, ry, rz = size.x * 0.40, size.y * 0.50, size.z * 0.55
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, radius=1.0,
                                         location=(centre.x, centre.y, centre.z - size.z * 0.03))
    egg = bpy.context.active_object
    egg.name = "stacy_head_egg"
    egg.scale = (rx, ry, rz)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth()
    hb = rig.data.bones["head"]
    egg.parent = rig
    egg.parent_type = "BONE"
    egg.parent_bone = "head"
    # bone parenting puts the child at the bone TAIL: compensate so the egg stays where it was built (rest pose)
    egg.matrix_parent_inverse = (rig.matrix_world @ hb.matrix_local @ Matrix.Translation((0, hb.length, 0))).inverted()
    egg.pass_index = body.pass_index
    print(f"[neutral] head -> egg {rx * 2:.3f} x {ry * 2:.3f} x {rz * 2:.3f} m, removed {len(cut)} verts, "
          f"cap faces {len(cap['faces'])}")
    face = body.vertex_groups.new(name="neutral_face")   # kept for compatibility (empty)
    n_face = 0
    # no anatomical detail: smooth nipples and the groin (spheres around rig landmarks, weight falls off to the rim)
    detail = body.vertex_groups.new(name="neutral_detail")
    spots = []
    for side in ("L", "R"):
        if f"breast.{side}" in pb:
            spots.append((rig.matrix_world @ pb[f"breast.{side}"].tail, 0.045))
    if "upperleg01.L" in pb and "upperleg01.R" in pb:
        hips = (rig.matrix_world @ pb["upperleg01.L"].head + rig.matrix_world @ pb["upperleg01.R"].head) / 2
        spots.append((hips + Vector((0, -0.06, -0.05)), 0.09))
    n_det = 0
    for v in body.data.vertices:
        p = mw @ v.co
        w = max((1.0 - (p - c).length / r for c, r in spots), default=0.0)
        if w > 0:
            detail.add([v.index], min(1.0, w * 1.6), "REPLACE")
            n_det += 1
    sm2 = body.modifiers.new("neutral_detail_smooth", "SMOOTH")
    sm2.vertex_group = "neutral_detail"
    sm2.factor = 1.0
    sm2.iterations = 25
    print(f"[neutral] detail-smoothed verts {n_det}")
    mat = bpy.data.materials.new("neutral_clay")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1)
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
    for ob in (body, egg):
        ob.data.materials.clear()
        ob.data.materials.append(mat)
    # rest-pose record
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    zs = [(ev.matrix_world @ v.co).z for v in ev.data.vertices]
    print(f"[neutral] face verts {n_face}, height {max(zs) - min(zs):.3f} m")
    for l, r in (("upperarm01.L", "upperarm01.R"), ("lowerarm01.L", "lowerarm01.R"), ("upperleg01.L", "upperleg01.R"),
                 ("lowerleg01.L", "lowerleg01.R"), ("foot.L", "foot.R"), ("wrist.L", "wrist.R")):
        if l in pb and r in pb:
            pl, pr = rig.matrix_world @ pb[l].head, rig.matrix_world @ pb[r].head
            print(f"[neutral] {l[:-2]:12s} z {pl.z:.3f}/{pr.z:.3f}  |x| {abs(pl.x):.3f}/{abs(pr.x):.3f}  "
                  f"asym {(Vector((-pl.x, pl.y, pl.z)) - pr).length * 100:.1f} cm")
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print("[neutral] ->", a.out)


if __name__ == "__main__":
    main()
