"""Neutral library viewport: reuse the existing normalized-prop loader/exporter.

No character, fitting, prompt generation, inference, or asset scripts.
"""
import json
from pathlib import Path
import sys


def main():
    import bpy
    request=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text(encoding='utf-8'))
    repo=Path(request['repo']);sys.path[:0]=[str(repo/'tools/scene3d'),str(repo/'tools/scene3d/plate')]
    from prop_render import load_props
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    groups=load_props(bpy.context.scene,request['props'],{'char_pelvis':[0,0,0]},request['library'])
    entities=[]
    for group,prop in zip(groups,request['props']):
        ident='prop_'+str(group['index']);root=group['root'];root.name='3DA_'+ident
        entities.append(dict(id=ident,name=root.name,label=prop.get('label',prop['prop_id']),kind='prop',
                             matrix_world=[list(r) for r in root.matrix_world]))
    bpy.ops.export_scene.gltf(filepath=request['out'],export_format='GLB',export_animations=False,export_cameras=False,export_lights=False)
    Path(request['facts']).write_text(json.dumps(dict(entities=entities)),encoding='utf-8')


if __name__=='__main__':main()
