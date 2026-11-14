"""14.11: Stacy's MPFB mannequin posed onto MHR70 joints (pose_tools.pose_mannequin_to_mhr).

Self-consistency: a posed copy of the mannequin gives MHR70-like joints; the reset mannequin must reproduce them.
Needs bpy (Blender 5 wheel) and the mannequin .blend (env THREEDA_MANNEQUIN or C:/work/scene3d_work); skipped otherwise.
"""
import math
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/scene3d'))
BLEND = Path(os.environ.get('THREEDA_MANNEQUIN', 'C:/work/scene3d_work/stacy_mannequin_162.blend'))
try:
    import bpy  # noqa: F401
    HAVE_BPY = True
except Exception:  # noqa: BLE001
    HAVE_BPY = False


def mhr70_from_rig(rig):
    """MHR70 joints (world) read off the MPFB rig: body joints, toe tips, finger chains (tip, DIP, PIP, MCP)."""
    import numpy as np
    import pose_tools
    p, mw = rig.pose.bones, rig.matrix_world
    h = lambda n: mw @ p[n].head  # noqa: E731
    t = lambda n: mw @ p[n].tail  # noqa: E731
    kp = np.zeros((70, 3))
    K = pose_tools.K
    eye_mid = (h('eye.L') + h('eye.R')) / 2
    put = {'nose': eye_mid + (eye_mid - h('neck03')) * 0.0, 'l_eye': h('eye.L'), 'r_eye': h('eye.R'),
           'l_ear': h('eye.L') * 1.0, 'r_ear': h('eye.R') * 1.0,
           'l_sh': h('upperarm01.L'), 'r_sh': h('upperarm01.R'), 'l_el': h('lowerarm01.L'), 'r_el': h('lowerarm01.R'),
           'l_hip': h('upperleg01.L'), 'r_hip': h('upperleg01.R'), 'l_kn': h('lowerleg01.L'), 'r_kn': h('lowerleg01.R'),
           'l_an': h('foot.L'), 'r_an': h('foot.R'), 'l_btoe': t('toe1-2.L'), 'r_btoe': t('toe1-2.R'),
           'l_stoe': t('toe5-3.L'), 'r_stoe': t('toe5-3.R'), 'l_heel': h('foot.L'), 'r_heel': h('foot.R'),
           'neck': h('neck01'), 'l_wr': h('wrist.L'), 'r_wr': h('wrist.R')}
    for k, v in put.items():
        kp[K[k]] = tuple(v)
    for side, base in pose_tools.FINGER_BASE.items():
        for f in range(5):
            chain = [t(f'finger{f + 1}-3.{side}'), h(f'finger{f + 1}-3.{side}'), h(f'finger{f + 1}-2.{side}'),
                     h(f'finger{f + 1}-1.{side}')]
            for j, v in enumerate(chain):
                kp[base + 4 * f + j] = tuple(v)
    return kp


@unittest.skipUnless(HAVE_BPY and BLEND.is_file(), 'bpy or the MPFB mannequin is not available')
class MpfbBodyTests(unittest.TestCase):
    def test_retarget_reproduces_a_known_pose(self):
        import bpy
        from mathutils import Euler
        import pose_tools
        bpy.ops.wm.open_mainfile(filepath=str(BLEND))
        rig, body = bpy.data.objects['stacy_rig'], bpy.data.objects['stacy_body']
        pose = {'upperarm01.L': (40, 0, 60), 'lowerarm01.L': (0, 0, 70), 'upperleg01.R': (-60, 0, 10),
                'lowerleg01.R': (80, 0, 0), 'spine03': (15, 20, 0), 'neck02': (0, 0, 25), 'finger2-2.R': (50, 0, 0)}
        for pb in rig.pose.bones:
            pb.rotation_mode = 'XYZ'
        for name, deg in pose.items():
            rig.pose.bones[name].rotation_euler = Euler([math.radians(a) for a in deg])
        rig.rotation_euler = (0, 0, math.radians(35))
        rig.location = (1.0, 2.0, 0.3)
        bpy.context.view_layer.update()
        kp = mhr70_from_rig(rig)
        for pb in rig.pose.bones:
            pb.rotation_mode = 'QUATERNION'
            pb.matrix_basis.identity()
        rig.rotation_euler = (0, 0, 0)
        rig.location = (0, 0, 0)
        bpy.context.view_layer.update()
        obj, rep = pose_tools.pose_mannequin_to_mhr(rig, body, kp, subsurf=0)
        self.assertEqual(rep['model'], 'mpfb')
        self.assertIn('hair_long01', rep['removed'])
        self.assertAlmostEqual(rep['scale'], 1.0, places=2)
        worst = max(rep['joint_err_cm'].values())
        self.assertLess(worst, 2.0, rep['joint_err_cm'])
        self.assertNotIn('hair_long01', bpy.data.objects)
        self.assertFalse(any(m.type == 'MASK' and m.vertex_group.startswith('Delete.') for m in body.modifiers))
        self.assertGreater(len(obj.data.vertices), 10000)


if __name__ == '__main__':
    unittest.main()
