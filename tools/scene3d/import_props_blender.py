"""Trusted Blender helper; launch only with --factory-startup --disable-autoexec.
No .blend, scripts, asset add-ons, animations or drivers are retained in cache.
FBX unit metadata is honoured by Blender; OBJ has no units and assumes metres.
Original metre-space bbox is retained in facts; cached longest side defaults to .30 m.
"""
import argparse
import json
import math
from pathlib import Path
import sys


def normalization_transform(bbox, *, size_m=.30, scale=1.0):
    """CPU helper: centre a metre-space AABB, then size_m * scale longest side.

    Returns (centre_xyz, uniform_factor, centred_scaled_bbox). Input is [lo, hi].
    Rotation/translation are deliberately left to the renderer.
    """
    size_m, scale = float(size_m), float(scale)
    if len(bbox) != 2 or any(len(point) != 3 for point in bbox):
        raise ValueError("bbox must be [lo_xyz, hi_xyz]")
    lo, hi = [[float(v) for v in point] for point in bbox]
    if not all(math.isfinite(v) for v in lo + hi + [size_m, scale]):
        raise ValueError("bbox, size_m and scale must be finite")
    dimensions = [hi[i] - lo[i] for i in range(3)]
    longest = max(dimensions)
    if min(dimensions) < 0 or longest <= 1e-9 or size_m <= 0 or scale <= 0:
        raise ValueError("bbox must be nonempty, size_m and scale > 0")
    factor = size_m / longest * scale
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("normalization factor must be finite and > 0")
    centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
    normalized = [[-d * factor / 2 for d in dimensions], [d * factor / 2 for d in dimensions]]
    return centre, factor, normalized


def main():
    import bpy
    from mathutils import Matrix, Vector

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--facts", required=True, type=Path)
    ap.add_argument("--size-m", type=float, default=.30)
    ns = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    source = ns.source.resolve(strict=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1.0
    sc.unit_settings.length_unit = "METERS"
    ext = source.suffix.lower()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(source), use_anim=False, use_image_search=False)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(source), forward_axis="NEGATIVE_Z", up_axis="Y")
    elif ext in {".gltf", ".glb"}:
        bpy.ops.import_scene.gltf(filepath=str(source))
    else:
        raise ValueError("only FBX/OBJ/glTF/GLB assets are accepted")
    # FBX already packs embedded images; repacking reads their missing disk path.
    embedded_images = [dict(image=im.name, size=list(im.size)) for im in bpy.data.images if im.packed_file]
    imported = list(sc.objects)
    # Static geometry only. A downloaded file never controls Python or Blender drivers.
    for ob in imported:
        ob.animation_data_clear()
        if getattr(ob, "data", None) and hasattr(ob.data, "animation_data_clear"):
            ob.data.animation_data_clear()
    for mat in bpy.data.materials:
        if mat.node_tree:
            mat.node_tree.animation_data_clear()
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    meshes = []
    for ob in imported:
        if ob.type != "MESH":
            continue
        evaluated = ob.evaluated_get(deps)
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=deps)
        # Bake hierarchy and unit transforms into vertices, retaining UVs/material slots.
        mesh.transform(ob.matrix_world)
        item = bpy.data.objects.new("prop_" + ob.name, mesh)
        sc.collection.objects.link(item)
        meshes.append(item)
    for ob in imported:
        bpy.data.objects.remove(ob, do_unlink=True)
    if not meshes:
        raise ValueError("asset has no mesh")
    bpy.context.view_layer.update()
    pts = [Vector(c) for ob in meshes for c in ob.bound_box]
    lo = Vector(tuple(min(p[i] for p in pts) for i in range(3)))
    hi = Vector(tuple(max(p[i] for p in pts) for i in range(3)))
    source_bbox = [list(lo), list(hi)]
    centre, factor, normalized_bbox = normalization_transform(source_bbox, size_m=ns.size_m)
    dimensions = hi - lo
    longest = max(dimensions)
    for ob in meshes:
        ob.data.transform(Matrix.Scale(factor, 4) @ Matrix.Translation(-Vector(centre)))
    bpy.context.view_layer.update()
    missing = []
    for image in bpy.data.images:
        if image.source == "FILE" and not image.packed_file:
            try:
                image.pack()
            except RuntimeError:
                missing.append(image.name)
    bpy.ops.object.select_all(action="DESELECT")
    for ob in meshes:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    temp = ns.out.with_name("model.pending.glb")
    bpy.ops.export_scene.gltf(filepath=str(temp), export_format="GLB", use_selection=True,
                              export_animations=False, export_cameras=False, export_lights=False)
    temp.replace(ns.out)
    facts = {"preserved_embedded_images": embedded_images,
             "longest_side_m": longest, "dimensions_m": list(dimensions), "source_bbox_center_m": centre,
             "source_bbox_m": source_bbox, "normalized_bbox_m": normalized_bbox,
             "normalized_dimensions_m": [normalized_bbox[1][i] - normalized_bbox[0][i] for i in range(3)],
             "normalized_longest_side_m": ns.size_m, "normalization_factor": factor,
             "origin": "bbox_center", "units": "meters", "unit_policy": "Blender import metadata; OBJ assumes meters",
             "mesh_count": len(meshes), "vertices": sum(len(o.data.vertices) for o in meshes),
             "triangles": sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes),
             "material_count": len({m.name for o in meshes for m in o.data.materials if m}),
             "missing_images": missing, "static_cache": True, "embedded_scripts_executed": False,
             "blender_version": bpy.app.version_string}
    ns.facts.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    print("PROP_IMPORT_OK", json.dumps(facts))


if __name__ == "__main__":
    main()
