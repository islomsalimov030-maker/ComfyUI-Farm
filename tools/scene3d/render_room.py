"""Scene3D: build a room blockout from rooms.py and render a clay preview from a camera.
blender -b --python tools/scene3d/render_room.py -- --room mainroom1 --out C:/work/scene3d_work/room.png
    [--cam 1.9,-1.3,1.55] [--look 2.6,5.0,0.9] [--lens 18] [--size 1344x768] [--blend PATH]
"""
import argparse
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rooms import ROOMS  # noqa: E402


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="mainroom1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="1.9,-1.3,1.55")
    ap.add_argument("--look", default="2.6,5.0,0.9")
    ap.add_argument("--lens", type=float, default=18.0)
    ap.add_argument("--size", default="1344x768")
    ap.add_argument("--blend", default="")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def vec(s):
    return Vector([float(x) for x in s.split(",")])


def main():
    a = args()
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    anchors = ROOMS[a.room]()
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("cam")
    cd.lens = a.lens
    cd.sensor_width = 36
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    cam.location = vec(a.cam)
    cam.rotation_euler = (vec(a.look) - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    w, h = (int(x) for x in a.size.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "OBJECT"
    sc.display.shading.show_cavity = True
    sc.render.filepath = a.out
    bpy.ops.render.render(write_still=True)
    print(f"[scene3d] room {a.room} anchors {anchors}")
    if a.blend:
        bpy.ops.wm.save_as_mainfile(filepath=a.blend)


main()
