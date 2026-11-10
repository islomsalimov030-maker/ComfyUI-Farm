"""Explicit, local PBR bindings for material-less FBX/OBJ assets.

Uses Blender's existing Principled BSDF/glTF exporter; no inferred materials.
"""
import hashlib
import json
from pathlib import Path


def material_plan(path, source):
    if path is None:
        return None
    root = Path(source).resolve(strict=True).parent
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != 1 or not isinstance(data.get('bindings'), list) or not data['bindings']:
        raise ValueError('material manifest requires schema=1 and nonempty bindings')
    names = set()
    for row in data['bindings']:
        if not isinstance(row.get('object'), str) or row['object'] in names:
            raise ValueError('material object selectors must be unique names')
        names.add(row['object'])
        if not row.get('base_color'):
            raise ValueError('base_color texture required')
        for key in ('base_color', 'normal', 'orm'):
            if not row.get(key):
                continue
            rel = Path(row[key])
            if rel.is_absolute() or ':' in row[key] or '\\' in row[key] or '..' in rel.parts:
                raise ValueError('material textures must stay inside the asset directory')
            texture = (root / rel).resolve(strict=True)
            if not texture.is_relative_to(root) or texture.suffix.lower() not in ('.png','.jpg','.jpeg','.tga','.tif','.tiff','.webp'):
                raise ValueError('unsupported or escaped material texture')
    return data


def apply_materials(objects, plan, source):
    import bpy
    root = Path(source).resolve().parent
    by_name = {o.get('source_object', o.name): o for o in objects}
    if set(by_name) != {r['object'] for r in plan['bindings']}:
        raise ValueError('material manifest must cover every mesh with exact object names')
    facts = []
    for row in plan['bindings']:
        ob = by_name[row['object']]
        if not ob.data.uv_layers.active:
            raise ValueError('explicit texture binding requires existing UVs')
        mat = bpy.data.materials.new('3DA_' + row['object']);mat.use_nodes = True
        nodes=mat.node_tree.nodes;links=mat.node_tree.links;nodes.clear()
        output=nodes.new('ShaderNodeOutputMaterial');bsdf=nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Roughness'].default_value=float(row.get('roughness',.5))
        links.new(bsdf.outputs['BSDF'],output.inputs['Surface'])
        images={}
        def image(key, color):
            path=root/row[key]
            im=bpy.data.images.load(str(path),check_existing=False);im.colorspace_settings.name=color
            im.pack();tex=nodes.new('ShaderNodeTexImage');tex.image=im
            images[key]=dict(file=row[key],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=list(im.size),colorspace=color)
            return tex
        base=image('base_color','sRGB');links.new(base.outputs['Color'],bsdf.inputs['Base Color'])
        if row.get('normal'):
            normal=image('normal','Non-Color');convert=nodes.new('ShaderNodeNormalMap')
            links.new(normal.outputs['Color'],convert.inputs['Color']);links.new(convert.outputs['Normal'],bsdf.inputs['Normal'])
        if row.get('orm'):
            orm=image('orm','Non-Color');channels=nodes.new('ShaderNodeSeparateColor');channels.mode='RGB'
            links.new(orm.outputs['Color'],channels.inputs['Color'])
            links.new(channels.outputs['Green'],bsdf.inputs['Roughness']);links.new(channels.outputs['Blue'],bsdf.inputs['Metallic'])
            # Official Blender glTF exporter recognizes this ready-made convention.
            group=bpy.data.node_groups.new('glTF Material Output','ShaderNodeTree')
            group.interface.new_socket(name='Occlusion',in_out='INPUT',socket_type='NodeSocketFloat')
            group.nodes.new('NodeGroupInput');occlusion=nodes.new('ShaderNodeGroup');occlusion.node_tree=group
            links.new(channels.outputs['Red'],occlusion.inputs['Occlusion'])
        ob.data.materials.clear();ob.data.materials.append(mat)
        for polygon in ob.data.polygons:polygon.material_index=0
        facts.append(dict(object=row['object'],material=mat.name,uv_layer=ob.data.uv_layers.active.name,images=images))
    return facts
