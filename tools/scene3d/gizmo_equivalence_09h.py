"""Compare actual browser capture and actual depth-occluded Blender prop mask."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt
from scipy.spatial import cKDTree
from PIL import Image


def bbox(mask):
    y,x=np.nonzero(mask)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    root=a.out;history=json.loads((root/(a.name+'_backend/history.json')).read_text(encoding='utf-8'))
    exported=json.loads((root/(a.name+'_export.json')).read_text(encoding='utf-8'))['output']
    prop_inputs=next(n['inputs'] for n in exported.values() if n['class_type']=='3DAProp')
    gizmo_connected='model_3d_info' in prop_inputs
    mode_correct=not gizmo_connected if a.name.startswith('numeric_') else gizmo_connected
    status_id=next(k for k,n in exported.items() if n.get('_meta',{}).get('title')=='Статус · опора / p95 / углы')
    text=history['outputs'][status_id]['text'][0]
    if text.startswith('status=error'):raise RuntimeError(text[:200])
    detail=json.loads(text.split('\n',1)[1]);report=detail['report'];prop=report['props'][0]
    native=Image.open(prop['silhouette']).convert('L')
    ref=np.asarray(native.resize((1344,768),Image.Resampling.LANCZOS))>127
    captured=np.asarray(Image.open(root/(a.name+'_mask.png')).convert('RGBA').resize((1344,768),Image.Resampling.LANCZOS))[...,3]>127
    ab,bb=bbox(ref),bbox(captured)
    delta=[abs(x-y) for x,y in zip(ab,bb)] if ab and bb else [999]*4
    union=np.logical_or(ref,captured).sum();iou=float(np.logical_and(ref,captured).sum()/union) if union else 0.
    ra=ref&~binary_erosion(ref);rb=captured&~binary_erosion(captured)
    distances=np.r_[distance_transform_edt(~ra)[rb],distance_transform_edt(~rb)[ra]]
    contour_max=float(distances.max()) if len(distances) else float('inf')
    state=json.loads((root/(a.name+'_state.json')).read_text(encoding='utf-8'))
    vp=np.asarray(state.get('vertices_world_zup',[]));bp=np.asarray(prop.get('vertex_probe_world',[]))
    vertex_max=float(max(cKDTree(vp).query(bp)[0].max(),cKDTree(bp).query(vp)[0].max())) if len(vp) and len(bp) else None
    result=dict(name=a.name,raster=[1344,768],blender_bbox=ab,viewport_bbox=bb,boundary_delta_px=delta,
        max_delta_px=max(delta),silhouette_iou=iou,contour_max_distance_px=contour_max,
        vertex_probe_symmetric_max_m=vertex_max,
        gizmo_input_connected=gizmo_connected,mode_correct=mode_correct,
        passed=mode_correct and max(delta)<=4 and contour_max<=4 and (not prop.get('vertex_probe_complete') or vertex_max is not None and vertex_max<1e-5),
        iou_095_advisory_passed=iou>=.95,
        geometry_status=text.split('\n')[0],timings=detail['timings'],prop=prop,
        local_prompt_id=history['prompt'][1],source='real UI serializer→Load3DAdvanced backend→Prop→PlacePose→Blender; no interception')
    (root/(a.name+'_equivalence.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('name','max_delta_px','silhouette_iou','contour_max_distance_px','vertex_probe_symmetric_max_m','passed','geometry_status','timings')},ensure_ascii=True))


if __name__=='__main__':main()
