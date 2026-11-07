"""Run the AIO PHOTO block (V2.2 or server V2.3, API JSON exported by the server frontend) on one SFW table row.

  python aio_run.py --api workflows/aio/AIO_manual_V2.3.photo.api.json --row S3D_STD01_window_cup --3d on --tag v23
Sets inputs by node TITLE, uploads them, refuses an occupied queue, posts one prompt, waits, downloads the outputs to
$FARM_COMFY/output/<subfolder>. Mods from FARM_MODS (face, character). Never edits the workflow files.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_ab as ab  # noqa: E402
import farm_env  # noqa: E402
import quality_round as qr  # noqa: E402
import remote_sync  # noqa: E402
import table_round as tr  # noqa: E402

ROWS = {'S3D_STD01_window_cup': 1, 'S3D_STD02_bench_look_back': 1, 'S3D_SIT01_bed_edge': 3, 'S3D_LIE01_bed_read': 3,
        'S3D_CRO01_mat_shoelace': 2}


def by_title(g, prefix):
    hits = [k for k, v in g.items() if v.get('_meta', {}).get('title', '').startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f'title {prefix!r}: {len(hits)} nodes')
    return g[hits[0]]


def legacy_prompt(spec, scene):
    """V2.2 PHOTO has no <Video 1> and no <Picture 2> (identity = RefMods): the table text without those references."""
    return '\n'.join([
        'subject_definitions:',
        '<Subject 1> is the adult woman.',
        '<Picture 1> is the exact room composition, furnishing, materials and lighting of the target photograph.',
        'summary:',
        '[reference generation] One single photorealistic static image of <Subject 1> in the room from <Picture 1>.',
        'retention_analysis:',
        '<Subject 1>: wear the specified white T-shirt and light-blue jeans.',
        '<Picture 1>: fully_preserved - retain room materials, furnishings, camera and lighting.',
        'detailed_description:',
        f'A photorealistic live-action still. [Shot 1] {ab.STATIC} The room is {spec["room"]}. '
        f'<Subject 1> is {scene["action"]}. She wears an opaque white short-sleeved cotton T-shirt, light-blue '
        'straight-leg jeans and white lace-up sneakers. The clothing is fully opaque and covers her torso and legs. '
        f'True-to-life scale: she is never enlarged to fill the frame. Scale anchor: {scene["anchor"]}. '
        f'{qr.light_sentence(scene.get("light") or spec.get("light"))} '
        'One continuous shot with no cuts and no close-up inserts.',
        'Timeline: From 0.00 to 0.17 seconds the subject holds the starting pose without moving.',
        'The lighting is preserved unchanged throughout.',
        'Colors and saturation stay unchanged from the reference image.',
        'overall_soundscape:', 'Quiet room tone.', 'non_diegetic_music:', 'N/A'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--api', required=True)
    ap.add_argument('--row', required=True)
    ap.add_argument('--3d', dest='d3', choices=['on', 'off'], default='off')
    ap.add_argument('--face-denoise', type=float, default=0.35)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--build-only', action='store_true')
    a = ap.parse_args()
    g = json.loads(Path(a.api).read_text(encoding='utf-8'))
    n = ROWS[a.row]
    loc = f'table3d_m{n}'
    spec = json.loads((farm_env.ROOT / f'table_specs/mainroom{n}.json').read_text(encoding='utf-8'))
    scene = next(s for s in spec['scenes'] if s['name'] == a.row)
    mods = [m.strip() for m in farm_env.MODS.split(',') if m.strip() and farm_env.MODS != 'none']
    files = {'room': f'{loc}_rgb.png', 'face': '0003.png',
             'clay': f'{loc}_s2/{a.row}/playblast.png', 'shadow': f'{loc}_s2/{a.row}/shadow.png'}
    by_title(g, 'ФОТО · локация или лист ракурсов')['inputs']['image'] = files['room']
    by_title(g, 'ФОТО · промпт кадра')['inputs']['value'] = legacy_prompt(spec, scene)
    for t, names in (('ФОТО · моды: 1', mods[:2]), ('ФОТО · мод лица для прохода', mods[:1])):
        node = by_title(g, t)
        for i in range(1, 9):
            node['inputs'][f'mod_{i}'] = names[i - 1] if i <= len(names) else '(none)'
    panel = by_title(g, 'ПУЛЬТ ФОТО')['inputs']
    panel.update(face_denoise=a.face_denoise, seed=a.seed)
    by_title(g, 'МАСКА ВКЛ')['inputs']['value'] = False
    by_title(g, 'имя:')['inputs']['string_a'] = f'plate3d_aio/{a.tag}_{a.row}_'
    if any(v['_meta']['title'].startswith('3D ·') for v in g.values()):
        by_title(g, '3D · ПУЛЬТ / 3D ВКЛ')['inputs']['value'] = a.d3 == 'on'
        by_title(g, '3D · ПУЛЬТ / seed')['inputs']['value'] = a.seed
        for t, f in (('3D · ВХОД / <Picture 1>', 'room'), ('3D · ВХОД / <Picture 2>', 'face'),
                     ('3D · ВХОД / playblast', 'clay'), ('3D · ВХОД / shadow', 'shadow')):
            by_title(g, t)['inputs']['image'] = files[f]
        ab.LOC[:] = [loc]
        by_title(g, '3D · ПУЛЬТ / SFW ref prompt')['inputs']['value'] = tr.prompt(spec, scene)
    elif a.d3 == 'on':
        raise SystemExit('this API has no 3D branch')
    out = farm_env.ROOT / 'aio_run'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{a.tag}_{a.row}.api.json').write_text(json.dumps(g, indent=1, ensure_ascii=False), encoding='utf-8')
    if a.build_only:
        return
    for f in files.values():
        p = farm_env.COMFY / 'input' / f
        remote_sync.upload(p, Path(f).parent.as_posix() if '/' in f else '')
    q = qr.api('/queue')
    if q['queue_running'] or q['queue_pending']:
        raise SystemExit('Queue is occupied; refusing to add work')
    t0 = time.monotonic()
    pid = ab.post(g)
    print('[aio] submitted', a.tag, a.row, pid, flush=True)
    dropped = 0
    while time.monotonic() - t0 < 1500:
        try:                                    # 07.11e: обрыв связи / 5xx (перезапуск сервера) не роняет скрипт
            h = qr.api('/history/' + pid)
            if dropped and pid not in h and not qr._queued(pid):
                raise SystemExit(f'[aio] lost: задачи {pid} нет ни в истории, ни в очереди (сервер перезапускался)')
            dropped = 0
        except (OSError, ValueError) as e:
            dropped += 1
            print(f'[aio] connection problem {dropped}: {e}', flush=True)
            time.sleep(15)
            continue
        if pid in h:
            st = h[pid].get('status', {})
            if st.get('status_str') == 'error':
                msg = [m for m in st.get('messages', []) if m[0] == 'execution_error']
                raise SystemExit(f'[aio] error: {json.dumps(msg[-1][1] if msg else st)[:1500]}')
            if st.get('completed'):
                imgs = [im for o in h[pid].get('outputs', {}).values() for im in o.get('images', [])]
                farm_env.download_outputs(imgs)
                sec = round(time.monotonic() - t0, 1)
                rec = dict(tag=a.tag, row=a.row, d3=a.d3, seconds=sec, prompt_id=pid,
                           images=[f"{i.get('subfolder', '')}/{i['filename']}" for i in imgs if i.get('type') == 'output'])
                (out / f'{a.tag}_{a.row}.run.json').write_text(json.dumps(rec, indent=1))
                print(f'[aio] done {sec}s', rec['images'], flush=True)
                return
        time.sleep(4)
    raise SystemExit('[aio] timeout')


if __name__ == '__main__':
    main()
