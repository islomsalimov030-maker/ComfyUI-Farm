"""Build a PRIVATE studio data bundle; no credentials, models or network calls.

Copies prepared profiles/plates, mannequin and explicitly selected normalized
props. The output belongs outside git. Linux paths remain administrator checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import yaml


def prepare(config_file, out, selected_props, *, prop_library=None):
    cfg=yaml.safe_load(Path(config_file).read_text(encoding='utf-8'));out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    root=out/'data';farm=Path(cfg['farm_root']);library=Path(prop_library) if prop_library else farm/'props'
    entries=[]
    def copy(source,relative):
        source=Path(source).resolve(strict=True);dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);entries.append(dict(path=relative.as_posix(),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    for loc,folder in cfg['location_asset_dirs'].items():
        if Path(loc).name!=loc:raise ValueError('invalid location id')
        profile=farm/'locations'/loc
        data=json.loads((profile/'profile.json').read_text(encoding='utf-8'))
        dependencies={'profile.json'}
        for ob in data['objects']:
            dependencies.update(ob[k] for k in ['top_grid_npz','pts_npz','backrest_npz','mask_png'] if ob.get(k))
        # Profiles are small. Include local sidecars, never photos outside them.
        dependencies.update(p.name for p in profile.iterdir() if p.is_file() and p.suffix.lower() in ('.json','.npz','.png'))
        for name in sorted(dependencies):
            f=(profile/name).resolve(strict=True)
            if not f.is_relative_to(profile.resolve()):raise ValueError('profile dependency escapes location')
            copy(f,Path('farm-state/locations')/loc/name)
        for name in ['plate.json','plate_rgb.png','plate_mesh.obj','depth_loc.png']:
            copy(Path(folder)/name,Path('farm-data/plate')/loc/name)
    copy(cfg['mannequin'],Path('farm-data/mannequin_162.blend'))
    for ident in selected_props:
        if not ident or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in ident):raise ValueError('invalid prop id')
        directory=library/ident;meta=json.loads((directory/'prop.json').read_text(encoding='utf-8'))
        model=Path(meta.get('model','model.glb'))
        if model.name!=str(model) or model.suffix.lower()!='.glb':raise ValueError('invalid normalized prop file')
        copy(directory/model,Path('farm-state/props')/ident/model)
        if meta.get('full_model'):
            full=Path(meta['full_model'])
            if full.name!=str(full) or full.suffix.lower()!='.glb':raise ValueError('invalid full prop file')
            copy(directory/full,Path('farm-state/props')/ident/full)
        copy(directory/'prop.json',Path('farm-state/props')/ident/'prop.json')
    # Only allowlisted runtime fields: never copy h3_remote/auth/source dirs.
    linux=dict(repo='/root/ComfyUI/custom_nodes/ComfyUI-Farm',python='/workspace/venvs/3da-farm/bin/python',
        geometry_python='/workspace/venvs/3da-geometry/bin/python',comfy='/root/ComfyUI',farm_root='/workspace/farm-state',
        blender='/workspace/bin/blender',mannequin='/workspace/farm-data/mannequin_162.blend',cell_px=cfg['cell_px'],
        location_asset_dirs={loc:'/workspace/farm-data/plate/'+loc for loc in cfg['location_asset_dirs']},
        location_hints_by_id=cfg.get('location_hints_by_id',{}),prompt_context_by_location=cfg.get('prompt_context_by_location',{}))
    (out/'3da_linux.private.yaml').write_text(yaml.safe_dump(linux,allow_unicode=True,sort_keys=False),encoding='utf-8')
    manifest=dict(schema=1,entries=entries,total_bytes=sum(e['bytes'] for e in entries),source_code_in_git=True,
        credentials_included=False,models_included=False,runtime_installed=False,uploaded=False,
        excluded=['geometry/reference caches','runtime environments','model weights','source photographs','RefMod files on studio'],
        path_mapping={'data/farm-state':'/workspace/farm-state','data/farm-data':'/workspace/farm-data'},
        geometry_forward='NOT_RUN_ON_LINUX',production_ready=False)
    (out/'transfer_manifest.private.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=out/'3DA_studio_data.private.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for e in entries:z.write(root/e['path'],'data/'+e['path'])
        for name in ['3da_linux.private.yaml','transfer_manifest.private.json']:z.write(out/name,name)
    return dict(files=len(entries),uncompressed_bytes=manifest['total_bytes'],archive_bytes=archive.stat().st_size,uploaded=False,production_ready=False)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--prop-library',type=Path);p.add_argument('--prop',action='append',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.config,a.out,a.prop,prop_library=a.prop_library)))


if __name__=='__main__':main()
