"""Scene3D rooms: coarse metric blockouts of the farm locations (Blender primitives, no external assets).
Axes: X to the right of the main camera, Y away from it (depth), Z up; metres; floor z = 0.
Detail level: furniture silhouettes and key heights (seat, counter, tub rim, window sill) are real-world;
no small props except the ones that read as anchors (sink bowl, pillows, plant, pole).
Each object gets a viewport colour for the clay pass; all room objects live in collection 'room'.
"""
import math

import bpy

WOOD = (0.55, 0.38, 0.24, 1)
TILE = (0.42, 0.42, 0.44, 1)
WHITE = (0.9, 0.9, 0.88, 1)
FABRIC = (0.7, 0.55, 0.5, 1)
GLASS = (0.6, 0.7, 0.75, 1)
METAL = (0.6, 0.6, 0.62, 1)
PLANT = (0.25, 0.4, 0.25, 1)
STONE = (0.62, 0.6, 0.56, 1)


class Builder:
    def __init__(self, name):
        self.col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(self.col)

    def _own(self, ob, name, color, bevel=0.0):
        ob.name = name
        for c in ob.users_collection:
            c.objects.unlink(ob)
        self.col.objects.link(ob)
        ob.color = color
        if bevel:
            m = ob.modifiers.new("bevel", "BEVEL")
            m.width = bevel
            m.segments = 3
        return ob

    def box(self, name, x0, y0, z0, x1, y1, z1, color, bevel=0.0, rot_z=0.0):
        bpy.ops.mesh.primitive_cube_add(size=1, location=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
        ob = bpy.context.object
        ob.scale = (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
        ob.rotation_euler.z = math.radians(rot_z)
        bpy.ops.object.transform_apply(scale=True)
        return self._own(ob, name, color, bevel)

    def cyl(self, name, x, y, z0, z1, r, color, verts=32, bevel=0.0):
        bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=z1 - z0, location=(x, y, (z0 + z1) / 2))
        return self._own(bpy.context.object, name, color, bevel)

    def carve(self, target, cutter):
        """Boolean difference; the cutter stays in the file (wireframe, not rendered)."""
        m = target.modifiers.new(f"cut_{cutter.name}", "BOOLEAN")
        m.operation = "DIFFERENCE"
        m.object = cutter
        cutter.hide_render = True
        cutter.display_type = "WIRE"
        cutter["cutter"] = True

    def sphere(self, name, x, y, z, sx, sy, sz, color):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1, location=(x, y, z))
        ob = bpy.context.object
        ob.scale = (sx, sy, sz)
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.shade_smooth()
        return self._own(ob, name, color)


def mainroom1():
    """Home_Stacey_Mainroom-1: bathroom zone on the left (tiles), living zone on the right (wood),
    separated by a slatted wood screen. Returns dict of anchors (seat heights etc.) for scene specs."""
    b = Builder("room")
    W, D, H = 7.0, 7.0, 2.8
    # shell
    b.box("floor_tile", 0, -1.5, -0.05, 2.7, D, 0, TILE)
    b.box("floor_wood", 2.7, -1.5, -0.05, W, D, 0, WOOD)
    b.box("wall_left", -0.15, -1.5, 0, 0, D, H, TILE)
    b.box("wall_back", -0.15, D, 0, W, D + 0.15, H, TILE)
    b.box("wall_right", W, -1.5, 0, W + 0.15, D, H, WHITE)
    b.box("ceiling", -0.15, -1.5, H, W + 0.15, D + 0.15, H + 0.1, WHITE)
    for i, y in enumerate((1.6, 3.2, 4.8, 6.3)):
        b.box(f"beam_{i}", 2.7, y - 0.09, H - 0.2, W, y + 0.09, H, WOOD)
    # vanity: floating counter, open shelf below, vessel sink, long mirror, shelves, big plant
    b.box("vanity_counter", 0, 0.9, 0.72, 0.58, 4.6, 0.85, STONE, bevel=0.01)
    b.box("vanity_body", 0, 0.9, 0.42, 0.55, 4.6, 0.72, STONE, bevel=0.01)
    b.box("vanity_shelf", 0, 0.9, 0.12, 0.5, 4.6, 0.16, WOOD)
    b.box("vanity_post", 0.42, 0.9, 0, 0.55, 1.05, 0.72, STONE)
    b.sphere("sink_bowl", 0.3, 3.0, 0.92, 0.22, 0.32, 0.07, STONE)
    b.box("faucet", 0.06, 2.97, 0.85, 0.1, 3.03, 1.12, METAL)
    b.box("mirror_frame", 0, 1.55, 1.2, 0.05, 4.05, 1.9, WOOD, bevel=0.02)
    b.box("mirror_glass", 0.05, 1.6, 1.25, 0.055, 4.0, 1.85, GLASS)
    b.box("wall_shelf_hi", 0, 0.0, 1.45, 0.25, 1.3, 1.48, WOOD)
    b.box("wall_shelf_lo", 0, 0.0, 1.05, 0.25, 1.3, 1.08, WOOD)
    b.cyl("plant_pot", 0.45, 0.2, 0, 0.5, 0.3, STONE, bevel=0.02)
    b.sphere("plant_leaves", 0.45, 0.2, 0.95, 0.5, 0.45, 0.55, PLANT)
    # clawfoot tub across the room past the vanity
    tub = b.sphere("tub", 1.35, 5.25, 0.38, 0.85, 0.4, 0.34, WHITE)
    b.carve(tub, b.box("tub_cut_top", 0.4, 4.7, 0.66, 2.3, 5.8, 0.8, WHITE))
    b.carve(tub, b.sphere("tub_cut_inner", 1.35, 5.25, 0.42, 0.79, 0.34, 0.3, WHITE))
    for i, (x, y) in enumerate(((0.75, 5.0), (0.75, 5.5), (1.95, 5.0), (1.95, 5.5))):
        b.cyl(f"tub_foot_{i}", x, y, 0, 0.2, 0.05, METAL, verts=12)
    # glass shower in the back-left corner
    b.box("shower_tray", 0, 6.0, 0, 1.6, D, 0.04, TILE)
    b.box("shower_glass", 1.58, 6.0, 0.04, 1.6, D, 2.05, GLASS)
    b.box("shower_frame_top", 0, 5.98, 2.03, 1.6, 6.02, 2.07, METAL)
    b.box("shower_glass_front", 0, 5.99, 0.04, 1.6, 6.0, 2.05, GLASS)
    # slatted wood screen between the zones
    b.box("screen_top", 2.62, 3.5, 2.55, 2.78, D, 2.6, WOOD)
    b.box("screen_bottom", 2.62, 3.5, 0, 2.78, D, 0.05, WOOD)
    y = 3.5
    i = 0
    while y < D:
        b.box(f"screen_slat_{i}", 2.65, y, 0.05, 2.75, y + 0.045, 2.55, WOOD)
        y += 0.11
        i += 1
    # living zone: sofa with its back to the back wall, facing the camera (-Y)
    sx0, sx1, sy0, sy1 = 2.9, 5.1, 6.05, 6.9
    b.box("sofa_base", sx0, sy0, 0.08, sx1, sy1, 0.42, FABRIC, bevel=0.05)
    b.box("sofa_seat", sx0 + 0.2, sy0, 0.42, sx1 - 0.2, sy1 - 0.24, 0.48, FABRIC, bevel=0.04)
    b.box("sofa_back", sx0, sy1 - 0.24, 0.42, sx1, sy1, 0.88, FABRIC, bevel=0.06)
    b.box("sofa_arm_l", sx0, sy0, 0.42, sx0 + 0.2, sy1, 0.64, FABRIC, bevel=0.06)
    b.box("sofa_arm_r", sx1 - 0.2, sy0, 0.42, sx1, sy1, 0.64, FABRIC, bevel=0.06)
    for i, xx in enumerate((3.3, 3.65, 3.95)):
        b.sphere(f"pillow_{i}", xx, sy1 - 0.33, 0.66, 0.2, 0.08, 0.18, FABRIC)
    for i, (x, yy) in enumerate(((sx0 + 0.05, sy0 + 0.05), (sx1 - 0.05, sy0 + 0.05), (sx0 + 0.05, sy1 - 0.05), (sx1 - 0.05, sy1 - 0.05))):
        b.cyl(f"sofa_leg_{i}", x, yy, 0, 0.08, 0.025, WOOD, verts=12)
    # slatted bench in front of the sofa, next to the screen
    b.box("bench_top", 2.95, 5.1, 0.42, 3.95, 5.45, 0.46, WOOD, bevel=0.01)
    for i, (x, yy) in enumerate(((3.0, 5.15), (3.9, 5.15), (3.0, 5.4), (3.9, 5.4))):
        b.box(f"bench_leg_{i}", x - 0.025, yy - 0.025, 0, x + 0.025, yy + 0.025, 0.42, WOOD)
    # round coffee table by the window
    b.cyl("table_top", 5.75, 6.2, 0.45, 0.5, 0.42, WOOD, bevel=0.01)
    b.cyl("table_base", 5.75, 6.2, 0, 0.45, 0.2, WOOD)
    b.cyl("table_mug", 5.6, 6.1, 0.5, 0.6, 0.045, WHITE, verts=16)
    # window wall: big glazing with frame, sheer curtain on the left side
    b.box("window_frame", 4.0, D - 0.05, 0.35, 6.9, D, 2.55, (0.2, 0.2, 0.2, 1))
    b.box("window_glass", 4.08, D - 0.06, 0.43, 6.82, D - 0.05, 2.47, GLASS)
    b.box("window_mullion", 5.42, D - 0.09, 0.35, 5.48, D - 0.05, 2.55, (0.2, 0.2, 0.2, 1))
    b.box("window_sill", 4.0, D - 0.25, 0.33, 6.9, D - 0.05, 0.38, WHITE)
    b.box("curtain", 5.2, D - 0.35, 0.02, 5.6, D - 0.3, 2.6, WHITE)
    # pole at the right edge of the frame
    b.cyl("pole", 6.3, 1.4, 0, H, 0.025, METAL, verts=16)
    return {"sofa_seat_z": 0.48, "bench_z": 0.46, "counter_z": 0.85, "tub_rim_z": 0.66,
            "window_sill_z": 0.38}


ROOMS = {"mainroom1": mainroom1}
