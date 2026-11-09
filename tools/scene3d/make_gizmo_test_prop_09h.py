"""Blender-only asymmetric off-centre GLB for transform regression; own geometry."""
import argparse
import json
from pathlib import Path
import sys


def main():
    import bpy
    from mathutils import Vector
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    objects=[]
    for location,scale in [((.28,-.12,.19),(.09,.045,.12)),((.4,-.12,.11),(.06,.045,.04))]:
        bpy.ops.mesh.primitive_cube_add(size=2,location=location)
        ob=bpy.context.object;ob.scale=scale;objects.append(ob)
    bpy.ops.object.select_all(action='DESELECT')
    for ob in objects:ob.select_set(True)
    bpy.context.view_layer.objects.active=objects[0];bpy.ops.object.join()
    ob=bpy.context.object
    # Bake all offsets into vertices, leaving the GLB root at the origin.
    ob.data.transform(ob.matrix_world);ob.matrix_world.identity();ob.name='asymmetric_offcentre'
    bpy.context.view_layer.update()
    points=[ob.matrix_world@Vector(c) for c in ob.bound_box]
    bounds=[[min(v[i] for v in points) for i in range(3)],[max(v[i] for v in points) for i in range(3)]]
    a.out.mkdir(parents=True,exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(a.out/'model.glb'),export_format='GLB',use_selection=True,export_animations=False)
    (a.out/'prop.json').write_text(json.dumps(dict(name='Asymmetric off-centre test',model='model.glb',size_m=.3,
        description_en='one small asymmetric calibration block',prop_strength=.35,license_status='MIT',
        import_facts=dict(source_bbox_m=bounds,units='meters',origin='deliberately off-centre')),indent=2),encoding='utf-8')


if __name__=='__main__':main()
