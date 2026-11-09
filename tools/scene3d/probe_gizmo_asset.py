"""Read current GLB bounds with the same native Blender importer as rendering."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


if __name__=='__main__':
    import bpy
    from mathutils import Vector
    p=argparse.ArgumentParser(__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(a.model.resolve(strict=True)))
    bpy.context.view_layer.update()
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if not meshes:raise ValueError('GLB has no mesh')
    points=[o.matrix_world@Vector(v) for o in meshes for v in o.bound_box]
    bbox=[[min(v[i] for v in points) for i in range(3)],[max(v[i] for v in points) for i in range(3)]]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(schema=1,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
        bbox_world_zup=bbox,mesh_count=len(meshes),blender_version=bpy.app.version_string),indent=2),encoding='utf-8')
