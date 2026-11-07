"""Pose round, methods B/C: MHR70 keypoints (sam3d_body_kp.py) -> MPFB rig rotations (pose_tools.retarget) -> pose file.

  blender -b <mannequin.blend> --python tools/scene3d/pose_retarget.py -- --method B --kp-dir <dir> --out <dir>
Bone lengths stay ours (directions only); the root carries the body pitch/roll; yaw comes from the scene.
"""
import json
import os
import sys

import bpy
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("FARM_BPY_PKGS"):
    sys.path.append(os.environ["FARM_BPY_PKGS"])
import pose_tools  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    a = dict(zip(argv[::2], argv[1::2]))
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    os.makedirs(a["--out"], exist_ok=True)
    for f in sorted(os.listdir(a["--kp-dir"])):
        if not f.endswith(".kp.json"):
            continue
        with open(os.path.join(a["--kp-dir"], f), encoding="utf-8") as fh:
            kp = json.load(fh)
        for pb in rig.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        rot = pose_tools.retarget(rig, kp["kp3d"])
        pid = f[:-len(".kp.json")]
        spec = {"contact": "solve", "method": a["--method"], "source_image": kp["image"], "rotations": rot,
                "angles": pose_tools.measure(rig)}
        with open(os.path.join(a["--out"], pid + ".json"), "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=1)
        print(f"[retarget] {a['--method']} {pid}: {len(rot)} bones", flush=True)


main()
