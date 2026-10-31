"""Scene3D: Stacy body mannequin (MPFB2 / MakeHuman, CC0 assets) with the default rig, height 168 cm.
blender -b --python tools/scene3d/build_stacy.py -- --out C:/work/scene3d_work/stacy_mannequin.blend
    [--assets C:/work/scene3d_work/mh_assets] [--hair long01] [--clothes female_sportsuit01] [--preview DIR]
Shape is approximate (from sheet_body_sfw): slim waist, wide hips, full glutes, medium-full bust, athletic.
Hair and clothes come from the MakeHuman system asset pack (CC0, files.makehumancommunity.org); no face detail:
the mannequin is only for depth/pose/clay maps.
"""
import argparse
import math
import os
import sys

import bpy

from bl_ext.blender_org.mpfb.services.humanservice import HumanService
from bl_ext.blender_org.mpfb.services.locationservice import LocationService
from bl_ext.blender_org.mpfb.services.targetservice import TargetService

HEIGHT_M = 1.68
MACRO = {"gender": 0.0, "age": 0.5, "muscle": 0.55, "weight": 0.45, "proportions": 0.85,
         "height": 0.55, "cupsize": 0.7, "firmness": 0.8,
         "race": {"asian": 0.0, "caucasian": 1.0, "african": 0.0}}
# relative path under mpfb/data/targets -> weight
SHAPE = {"buttocks/buttocks-volume-incr": 0.4,
         "hip/hip-scale-depth-incr": 0.1,
         "torso/measure-waist-circ-decr": 0.45,
         "torso/measure-hips-circ-incr": 0.15,
         "stomach/stomach-pregnant-decr": 0.4}


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--assets", default="C:/work/scene3d_work/mh_assets")
    ap.add_argument("--hair", default="long01")
    ap.add_argument("--hair-cut", type=float, default=1.22, help="world z (m) below which hair is removed, 0 = keep")
    ap.add_argument("--clothes", default="female_sportsuit01")
    ap.add_argument("--preview", default="")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def body_height(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    zs = [(ev.matrix_world @ v.co).z for v in ev.data.vertices]
    return min(zs), max(zs)


def add_asset(basemesh, assets, kind, name):
    if not name:
        return None
    path = os.path.join(assets, kind, name, f"{name}.mhclo")
    ob = HumanService.add_mhclo_asset(path, basemesh, asset_type="Hair" if kind == "hair" else "Clothes",
                                      subdiv_levels=0, material_type="MAKESKIN")
    ob.name = f"{kind}_{name}"
    return ob


def trim_below(ob, z_cut):
    """Delete asset vertices below world z (rest pose): long01 -> hair just below the shoulders."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    mw = ob.matrix_world
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if (mw @ v.co).z < z_cut], context="VERTS")
    bm.to_mesh(ob.data)
    bm.free()


def inflate(ob, offset):
    """Push clothes out along normals so the body does not poke through (bust is larger than the asset fit)."""
    mod = ob.modifiers.new("inflate", "DISPLACE")
    mod.strength = offset
    mod.mid_level = 0.0
    mod.direction = "NORMAL"


def preview(out_dir, tag):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "SINGLE"
    sc.display.shading.single_color = (0.75, 0.62, 0.55)
    sc.render.resolution_x, sc.render.resolution_y = 768, 1024
    cam_data = bpy.data.cameras.new("prev")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 1.95
    cam = bpy.data.objects.new("prev", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    for name, loc, rot in [("front", (0, -4, 0.86), (90, 0, 0)), ("side", (4, 0, 0.86), (90, 0, 90)),
                           ("back", (0, 4, 0.86), (90, 0, 180))]:
        cam.location = loc
        cam.rotation_euler = [math.radians(a) for a in rot]
        sc.render.filepath = f"{out_dir}/{tag}_{name}.png"
        bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)


def main():
    a = args()
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    basemesh = HumanService.create_human(macro_detail_dict=MACRO)
    root = LocationService.get_mpfb_data("targets")
    for rel, w in SHAPE.items():
        TargetService.load_target(basemesh, f"{root}/{rel}.target.gz", weight=w)
    rig = HumanService.add_builtin_rig(basemesh, "default")
    hair = add_asset(basemesh, a.assets, "hair", a.hair)
    clothes = add_asset(basemesh, a.assets, "clothes", a.clothes)
    z0, z1 = body_height(basemesh)
    k = HEIGHT_M / (z1 - z0)
    rig.scale = (k, k, k)
    bpy.context.view_layer.update()
    z0, z1 = body_height(basemesh)
    rig.location.z -= z0
    bpy.context.view_layer.update()
    if hair and a.hair_cut:
        trim_below(hair, a.hair_cut)
    if clothes:
        inflate(clothes, 0.01)
    basemesh.name = "stacy_body"
    rig.name = "stacy_rig"
    z0, z1 = body_height(basemesh)
    print(f"[scene3d] body height {100 * (z1 - z0):.1f} cm, rig scale {k:.3f}, "
          f"objects: {[o.name for o in bpy.data.objects]}")
    if a.preview:
        preview(a.preview, f"stacy_{a.hair}_{a.clothes or 'none'}")
    bpy.ops.wm.save_as_mainfile(filepath=a.out)


main()
