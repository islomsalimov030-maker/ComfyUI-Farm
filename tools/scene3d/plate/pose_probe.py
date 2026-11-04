"""Scene3D: look at a pose without the photo - front + side clay views of the mannequin and joint positions.

Runs with the bpy wheel (pod: /workspace/venvs/bpy/bin/python) or blender -b --python ... -- <args>:
  python tools/scene3d/plate/pose_probe.py --mannequin M.blend --poses crouch,stand_hand_face --out DIR
  python tools/scene3d/plate/pose_probe.py --make-mannequin SCENE.blend --mannequin OUT.blend   (strip a scene.blend)
Writes DIR/probe_<pose>.png (front | side | top-down), prints joints in metres (rig placed at the origin, facing -Y).
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from poses import POSES, apply_pose, place  # noqa: E402


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--mannequin", required=True)
    ap.add_argument("--make-mannequin", default="")
    ap.add_argument("--poses", default="")
    ap.add_argument("--out", default=".")
    ap.add_argument("--chain", action="store_true", help="print the arm bone chain")
    return ap.parse_args(argv)


def make_mannequin(src, dst):
    bpy.ops.wm.open_mainfile(filepath=src)
    for o in list(bpy.data.objects):
        if o.name.startswith("plate") or o.type == "CAMERA" or o.name.startswith("prop"):
            bpy.data.objects.remove(o, do_unlink=True)
    rig = bpy.data.objects["stacy_rig"]
    rig.location = (0, 0, 0)
    rig.rotation_euler = (0, 0, 0)
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix()
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=dst)
    print("[probe] mannequin ->", dst)


def cam(sc, loc, look, lens=35):
    cd = bpy.data.cameras.new("pc")
    cd.lens = lens
    c = bpy.data.objects.new("pc", cd)
    sc.collection.objects.link(c)
    c.location = Vector(loc)
    c.rotation_euler = (Vector(look) - c.location).to_track_quat("-Z", "Y").to_euler()
    return c


def joints(rig):
    pb = rig.pose.bones
    mw = rig.matrix_world
    out = {}
    for k, b in (("head", "head"), ("sh_L", "upperarm01.L"), ("sh_R", "upperarm01.R"), ("el_L", "lowerarm01.L"),
                 ("el_R", "lowerarm01.R"), ("hand_L", "finger3-1.L"), ("hand_R", "finger3-1.R"),
                 ("hip_L", "upperleg01.L"), ("hip_R", "upperleg01.R"), ("knee_L", "lowerleg01.L"),
                 ("knee_R", "lowerleg01.R"), ("foot_L", "foot.L"), ("foot_R", "foot.R")):
        if b in pb:
            out[k] = tuple(round(c, 2) for c in (mw @ pb[b].head))
    return out


def main():
    a = args()
    if a.make_mannequin:
        make_mannequin(a.make_mannequin, a.mannequin)
        return
    os.makedirs(a.out, exist_ok=True)
    for name in a.poses.split(","):
        bpy.ops.wm.open_mainfile(filepath=a.mannequin)
        sc = bpy.context.scene
        rig, body = bpy.data.objects["stacy_rig"], bpy.data.objects["stacy_body"]
        if a.chain:
            b = rig.pose.bones["finger3-1.R"]
            ch = []
            while b:
                ch.append(b.name)
                b = b.parent
            print("[probe] chain finger3-1.R -> root:", " < ".join(ch))
            a.chain = False
        contact = apply_pose(rig, name)
        seat = {"seat": 0.45, "lie": 0.40}.get(contact, 0.0)
        place(rig, body, 0, 0, POSES[name].get("yaw_add", 0) * 0 , contact, seat, head_dir=(1, 0))
        bpy.context.view_layer.update()
        print(f"[probe] {name} ({contact}):", joints(rig))
        sc.render.engine = "BLENDER_WORKBENCH"
        sc.display.shading.light = "STUDIO"
        sc.render.resolution_x = sc.render.resolution_y = 400
        sc.render.image_settings.file_format = "PNG"
        views = {"front": ((0, -4.5, 1.0), (0, 0, 0.8)), "side": ((4.5, 0, 1.0), (0, 0, 0.8)),
                 "top": ((0.01, -0.01, 5.0), (0, 0, 0))}
        paths = []
        for v, (loc, look) in views.items():
            sc.camera = cam(sc, loc, look)
            p = os.path.join(a.out, f"_p_{name}_{v}.png")
            sc.render.filepath = p
            bpy.ops.render.render(write_still=True)
            paths.append(p)
        try:
            from PIL import Image
            ims = [Image.open(p) for p in paths]
            sh = Image.new("RGB", (1200, 400), "white")
            for i, im in enumerate(ims):
                sh.paste(im.convert("RGB"), (i * 400, 0))
            sh.save(os.path.join(a.out, f"probe_{name}.png"))
            for p in paths:
                os.remove(p)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
