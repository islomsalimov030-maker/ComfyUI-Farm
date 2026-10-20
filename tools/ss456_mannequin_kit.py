"""SS-456 H2: composite Blender mannequin renders over existing plates and prepare a 6-frame A/B round.

python tools/ss456_mannequin_kit.py <round_d3.yaml> <mannequin_spec.json> <round_h2.yaml>

No new models/nodes; Blender primitives rendered separately. Camera FOV and sofa seat location are hypotheses,
not measurements. Compare raw H2 against H1/D3; do not use bone_height as an absolute acceptance metric.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from farm.prompts.lint import lint, errors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('round'); ap.add_argument('spec'); ap.add_argument('out')
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.round).read_text(encoding='utf-8'))
    spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
    root = Path(cfg['src']); renders = Path(spec['out'])
    out = renders.parent / 'kit_h2'
    src = copy.deepcopy(cfg)
    cfg.update(round=Path(a.out).stem, kit_poses={}, poses=[], seeds={}, variants=[])
    pic1 = ('<Picture 1> is the empty room that the image is built on: its layout, camera position, framing and light '
            'are kept exactly.')
    rule = ('The grey articulated mannequin in <Picture 1> is only a placement and body-pose guide, not another '
            'person. Replace that mannequin with her at exactly its location and projected size, preserving '
            'the position of her head, torso, hips, knees and feet and their contact with the floor or seat. '
            'Keep the room and furniture in their original places. No mannequin remains in the photograph.')
    for scene in spec['scenes']:
        name = scene['id']; old = src['kit_poses'][name+'_plate']
        plate = Image.open(root/old['plate']).convert('RGBA')
        render = Image.open(renders/f'{name}.png').convert('RGBA')
        if render.size != plate.size:
            raise ValueError(f'{name}: render and plate sizes differ')
        # Baseline omits rectangular occluder restoration: in H1 it erased body and introduced a confound.
        # Visible occlusion/contact accuracy must be inspected in results; no claim of correct depth ordering.
        composite = Image.alpha_composite(plate, render)
        dest = out/name; dest.mkdir(parents=True, exist_ok=True)
        composite.convert('RGB').save(dest/'plate.png')
        prompt = (root/old['prompt']).read_text(encoding='utf-8')
        if pic1 not in prompt or ' Her face is sharp and natural.' not in prompt:
            raise ValueError(name+': prompt anchors missing')
        prompt = prompt.replace(pic1, pic1[:-1]+'; a grey articulated mannequin marks her exact placement, size '
                               'and body pose.', 1)
        prompt = prompt.replace(' Her face is sharp and natural.', ' '+rule+' Her face is sharp and natural.', 1)
        issues = errors(lint(prompt, 'frame', {'connected':{'<Picture 1>'}}))
        if issues:
            raise ValueError([(x.code,x.msg) for x in issues])
        (dest/'prompt.txt').write_text(prompt, encoding='utf-8')
        pose = name+'_mannequin'
        cfg['kit_poses'][pose] = dict(old, plate=(dest/'plate.png').relative_to(root).as_posix(),
                                     prompt=(dest/'prompt.txt').relative_to(root).as_posix())
        seeds = list(src['seeds'][name+'_plate'])
        cfg['poses'].append(pose); cfg['seeds'][pose]=seeds
        var = copy.deepcopy(next(v for v in src['variants'] if v['id']=='plate_'+name))
        var.update(id='mannequin_'+name, poses=[pose], seeds={pose:seeds})
        cfg['variants'].append(var)
    Path(a.out).write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(a.out, 'frames',sum(len(x) for x in cfg['seeds'].values()))


if __name__ == '__main__':
    main()
