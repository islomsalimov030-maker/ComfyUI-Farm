"""SS-456: headless Blender render of a metric articulated mannequin, no external assets.

blender --background --python tools/ss456_mannequin_blender.py -- --spec spec.json

Experimental inputs, NOT a calibrated reconstruction: camera height/horizon from the plate card;
horizontal FOV is explicit in the spec. Sitting uses a seat contact, never feet assumed to be on floor.
Outputs: transparent mannequin PNG, scene .blend, projected joints JSON (for checking H3 adherence).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


def sphere(name, xyz, scale, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=xyz)
    ob = bpy.context.object
    ob.name = name
    ob.scale = scale
    ob.data.materials.append(mat)
    return ob


def segment(name, a, b, radius, mat):
    delta = Vector(b) - Vector(a)
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=delta.length,
                                     location=(Vector(a) + Vector(b)) / 2)
    ob = bpy.context.object
    ob.name = name
    ob.rotation_euler = delta.to_track_quat('Z', 'Y').to_euler()
    ob.data.materials.append(mat)
    for n, p in enumerate((a, b)):
        sphere(f'{name}_joint{n}', p, (radius,) * 3, mat)


def skeleton(height, pose, seat_m=.45):
    """Segment lengths fixed across poses. Metres, local forward = -Y, Z up; hip is origin.
    Neutral mannequin, not a mesh of the character; proportions are explicit approximations."""
    h = height
    thigh, shin, torso = .245*h, .246*h, .29*h
    pelvis_width, shoulder_width = .16*h, .23*h
    # Standing crown: .39 + .065 above hips; feet: .245 + .246 + .054 below hips = 1.000 H.
    j = {'hip': (0, 0, 0), 'neck': (0, 0, torso), 'head': (0, 0, .39*h)}
    for sign, side in ((-1, 'L'), (1, 'R')):
        hip = Vector((sign * pelvis_width / 2, 0, 0))
        if pose == 'standing':
            knee = hip + Vector((0, 0, -thigh))
            ankle = knee + Vector((0, 0, -shin))
        elif pose == 'sitting':
            # Both feet supported on floor; hip height fixed by seat, not inferred from an image.
            knee = hip + Vector((0, -thigh, 0))
            ankle = knee + Vector((0, -math.sqrt(max(0, shin*shin - seat_m*seat_m)), -seat_m))
        elif pose == 'squat':
            # Knee flexion / spread explicit, lengths invariant; not a bbox height multiplier.
            knee = hip + Vector((sign*.12*h, -.16*h, -math.sqrt(thigh*thigh-(.12*h)**2-(.16*h)**2)))
            ankle = knee + Vector((0, .14*h, -math.sqrt(shin*shin-(.14*h)**2)))
        else:
            raise ValueError(pose)
        shoulder = Vector((sign * shoulder_width/2, 0, .25*h))
        if pose == 'squat':
            elbow = shoulder + Vector((sign*.15*h, 0, -.07*h))
            wrist = elbow + Vector((-sign*.03*h, -.01*h, .14*h))
        else:
            elbow = shoulder + Vector((sign*.035*h, -.025*h, -.17*h))
            wrist = elbow + Vector((0, -.04*h, -.145*h))
        elbow = shoulder + (elbow-shoulder).normalized() * (.186*h)
        wrist = elbow + (wrist-elbow).normalized() * (.146*h)
        for key, p in (('hip', hip), ('knee', knee), ('ankle', ankle), ('shoulder', shoulder),
                       ('elbow', elbow), ('wrist', wrist)):
            j[f'{side}_{key}'] = tuple(p)
    return j


def main():
    args = sys.argv[sys.argv.index('--') + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    spec = json.loads(Path(ap.parse_args(args).spec).read_text(encoding='utf-8'))
    out = Path(spec['out']); out.mkdir(parents=True, exist_ok=True)
    W, H = spec.get('canvas', [1888, 1056])
    for item in spec['scenes']:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene
        sc.render.engine = 'BLENDER_EEVEE'
        sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
        sc.render.film_transparent = True
        sc.render.image_settings.file_format = 'PNG'
        sc.render.image_settings.color_mode = 'RGBA'
        sc.view_settings.view_transform = 'Standard'
        mat = bpy.data.materials.new('neutral_grey')
        mat.diffuse_color = (.38, .38, .38, 1)
        cam_h, hor = item['camera_height_cm']/100, item['horizon_y']
        fov = math.radians(item.get('hfov_deg', 60))
        fx = W/(2*math.tan(fov/2))
        bpy.ops.object.camera_add(location=(0, 0, cam_h), rotation=(math.pi/2, 0, 0))
        cam = bpy.context.object
        cam.data.type = 'PERSP'; cam.data.sensor_fit = 'HORIZONTAL'
        cam.data.sensor_width = 36
        cam.data.lens = 36/(2*math.tan(fov/2))
        sc.camera = cam
        # Calibrate image horizon using Blender's own projection, avoiding shift sign assumptions.
        bpy.context.view_layer.update()
        def pixel_y():
            return (1-world_to_camera_view(sc, cam, Vector((0, 1000, cam_h))).y)*H
        y0 = pixel_y(); cam.data.shift_y = .01; y1 = pixel_y()
        cam.data.shift_y = (hor*H-y0)/((y1-y0)/.01)
        j = skeleton(spec['height_cm']/100, item['pose'], item.get('seat_cm',45)/100)
        if item['pose'] == 'sitting':
            support_z = item['seat_cm']/100
            root_z = support_z + .054*spec['height_cm']/100  # hip above seat; same foot offset yields floor contact
            root_y = fx*(cam_h-support_z)/((item['support_xy'][1]-hor)*H)
            root_x = (item['support_xy'][0]-.5)*W*root_y/fx
        else:
            support_z = 0
            root_z = -min(p[2] for k,p in j.items() if k.endswith('ankle')) + .054*spec['height_cm']/100
            root_y = fx*cam_h/((item['support_xy'][1]-hor)*H)
            root_x = (item['support_xy'][0]-.5)*W*root_y/fx
        origin = Vector((root_x, root_y, root_z))
        world = {k: tuple(Vector(v)+origin) for k,v in j.items()}
        h = spec['height_cm']/100
        sphere('torso', origin+Vector((0,0,.14*h)), (.13*h,.075*h,.16*h), mat)
        sphere('pelvis', origin, (.105*h,.07*h,.07*h), mat)
        segment('neck', origin+Vector((0,0,.27*h)), origin+Vector((0,0,.34*h)), .028*h, mat)
        sphere('head', world['head'], (.046*h,.05*h,.065*h), mat)
        # Small nose establishes facing direction without identity/texture assets.
        sphere('nose', Vector(world['head'])+Vector((0,-.053*h,0)), (.012*h,.018*h,.015*h), mat)
        for side in ('L','R'):
            for a,b,r in (('hip','knee',.055),('knee','ankle',.039),('shoulder','elbow',.033),('elbow','wrist',.026)):
                segment(side+'_'+a+'_'+b, world[f'{side}_{a}'], world[f'{side}_{b}'], r*h, mat)
            sphere(side+'_foot', Vector(world[f'{side}_ankle'])+Vector((0,-.04*h,-.027*h)), (.04*h,.07*h,.027*h), mat)
        sc.world = bpy.data.worlds.new('World'); sc.world.color = (.7,.7,.7)
        bpy.ops.object.light_add(type='AREA', location=(root_x-3,root_y-4,5))
        lamp = bpy.context.object; lamp.data.energy=450; lamp.data.shape='DISK'; lamp.data.size=5
        lamp.rotation_euler = (Vector(world['hip'])-lamp.location).to_track_quat('-Z','Y').to_euler()
        bpy.context.view_layer.update()
        projection = {}
        for k,v in world.items():
            p=world_to_camera_view(sc,cam,Vector(v));projection[k]=[round(p.x,5),round(1-p.y,5)]
        name=item['id']; sc.render.filepath=str(out/f'{name}.png')
        (out/f'{name}.json').write_text(json.dumps({'spec': item, 'height_cm':spec['height_cm'],
            'projected_joints':projection,'camera_hfov_assumed':True},indent=2),encoding='utf-8')
        bpy.ops.wm.save_as_mainfile(filepath=str(out/f'{name}.blend'))
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
