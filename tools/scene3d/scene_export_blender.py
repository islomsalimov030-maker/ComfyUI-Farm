"""Export the cached placed body and props as named selectable GLB roots."""
import json
from pathlib import Path
import sys


def main():
    import bpy
    import numpy as np
    from mathutils import Vector
    request = json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text(encoding='utf-8'))
    sys.path[:0] = [str(Path(request['repo'])/'tools/scene3d'), str(Path(request['repo'])/'tools/scene3d/plate')]
    from prop_render import load_props, anchors_from_mhr70
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    with np.load(request['joints'], allow_pickle=False) as d:
        kp=d['kp']
    pivot = (kp[9]+kp[10])/2
    # The legacy preview OBJ rounds vertices to five decimal places. Use the
    # authoritative NPZ coordinates and OBJ topology, preserving vertex order.
    vertices=[]
    faces=[]
    for line in Path(request['body_obj']).read_text().splitlines():
        if line.startswith('v '):
            vertices.append([float(v) for v in line.split()[1:]])
        elif line.startswith('f '):
            faces.append([int(v.split('/')[0])-1 for v in line.split()[1:]])
    if request.get('body_cache'):
        with np.load(request['body_cache'],allow_pickle=False) as d:
            vertices=d['X'].tolist()
    mesh=bpy.data.meshes.new('3DA_body_mesh')
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    body=bpy.data.objects.new('3DA_body',mesh)
    bpy.context.scene.collection.objects.link(body)
    for v in body.data.vertices:
        v.co -= Vector(pivot.tolist())
    body.location = Vector(pivot.tolist())
    material=bpy.data.materials.new('3DA body clay')
    material.diffuse_color=(.65,.68,.72,1)
    body.data.materials.clear()
    body.data.materials.append(material)
    bpy.context.view_layer.update()
    entities=[dict(id='body', name='3DA_body', label='Персонаж', kind='body', matrix_world=[list(r) for r in body.matrix_world])]
    groups=load_props(bpy.context.scene,request['props'],anchors_from_mhr70(kp),request['library'])
    for group,prop in zip(groups,request['props']):
        root=group['root']
        entity_id='prop_'+str(group['index'])
        root.name='3DA_'+entity_id
        entities.append(dict(id=entity_id,name=root.name,label='Предмет '+str(group['index']+1)+' · '+prop['prop_id'],
                             kind='prop',prop=prop,matrix_world=[list(r) for r in root.matrix_world]))
    bpy.ops.export_scene.gltf(filepath=request['out'],export_format='GLB',export_animations=False,export_yup=True)
    Path(request['facts']).write_text(json.dumps(dict(entities=entities)),encoding='utf-8')


if __name__=='__main__':
    main()
