"""Deterministic offline UI-only AIO V2.2 -> V2.3 patch. Never submits a prompt.

Existing prompt literals are opaque: copied, never logged or used as templates.
Run from repo: .venv/Scripts/python tools/scene3d/patch_aio_v23.py
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from wfkit import G, check_ui, _norm

PREFIX = '3D · '
# ui-2 (07.11c): SERVER variant (studio ComfyUI 0.34.6) = winning recipe S3 comp2 only: playblast <Video 1>, base ref2va,
# SAM3 person, person over photo x (1 - 0.8*shadow) inside the graph. No Fun CN and no harmonisation (V2.3-pod later).
VERSION = 'aio-v23-ui-3-server'

# New SFW text only; no interpolation of inherited literals.
PROMPT = '''subject_definitions:
<Subject 1> is the adult woman whose facial identity is supplied by <Picture 2>.
<Picture 1> is the room photograph defining the camera, furniture, materials and lighting.
<Picture 2> is only an identity reference, never a layout.
<Video 1> is the shadowed clay playblast defining pose, size, placement and contact with furniture.
summary:
[reference generation] One single photograph. Replace the clay figure with <Subject 1> in an opaque grey T-shirt, full-length navy trousers and white sneakers, preserving the exact room composition.
retention_analysis:
<Subject 1>: attribute_transfer - facial identity from <Picture 2>.
<Picture 1>: fully_preserved - room composition, materials and lighting.
<Picture 2>: attribute_transfer - facial identity only.
<Video 1>: partially_preserved - geometry, pose, size and placement, not clay appearance.
detailed_description:
Photorealistic live-action still. [Shot 1] Preserve the exact pose, projected size and position from <Video 1>. True-to-life scale: keep the furniture-to-body proportions of the playblast; never enlarge her to fill the frame. She is lit by the same light as the room and casts a soft natural shadow on the floor and furniture beneath her. Preserve feet contact and the contact with furniture. No mannequin, clay texture or skeleton overlay.
Filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, with no camera movement, no zoom and no pan.
One continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen.
The lighting is preserved unchanged throughout.
Colors and saturation stay unchanged from the reference image.
overall_soundscape:
Quiet room tone.
non_diegetic_music:
N/A'''


def graphs(root):
    return [root] + root.get('definitions', {}).get('subgraphs', [])


def find_type(g, typ):
    hits = [n for n in g['nodes'] if n['type'] == typ]
    if len(hits) != 1:
        raise ValueError(f'Expected one {typ}, got {len(hits)}')
    return hits[0]


def patch(source):
    r = copy.deepcopy(source)
    if r.get('extra', {}).get('aio_v23'):
        raise ValueError('Already patched; input must be original V2.2')
    top = G(r, r, top=True)
    photo = next(s for s in r['definitions']['subgraphs'] if any(n.get('title') == 'gen · sampler pass 1' for n in s['nodes']))
    pg = G(photo, r)
    inst = find_type(r, photo['id'])
    old = {n['title']: n for n in photo['nodes'] if n.get('title')}
    added = []
    controls = {}

    def add(g, typ, label, inputs=(), outputs=(), widgets=None, pos=(0, 0), size=(320, 100)):
        n = g.add_node(typ, PREFIX + label, pos=pos, size=size, inputs=inputs, outputs=outputs, widgets=widgets,
                       props={'Node name for S&R': typ, 'aio_v23': VERSION})
        added.append(n['id'])
        return n

    def clone(template, label, x, y, widgets=None):
        n = pg.clone_node(template, PREFIX + label, pos=(x, y))
        n['mode'] = 0
        n['properties']['aio_v23'] = VERSION
        if widgets is not None:
            n['widgets_values'] = widgets
            n.pop('widgets_values_named', None)
        added.append(n['id'])
        return n

    def conn(src, dst, name, slot=0):
        return pg.connect(src, slot, dst, name)

    def switch(label, a, b, toggle, x, y, a_slot=0, b_slot=0):
        n = clone(old['маска: итог · ref2v'], label, x, y, [False])
        conn(a, n, 'on_false', a_slot)
        conn(b, n, 'on_true', b_slot)
        conn(toggle, n, 'switch')
        return n

    def control(key, typ, value, outtype, x, y, label):
        outname = 'value' if typ in ('FloatConstant', 'INTConstant') else outtype
        n = add(top, typ, 'ПУЛЬТ / ' + label, outputs=[(outname, outtype)], widgets={'value': value}, pos=(x, y), size=(380, 82))
        idx = pg.add_sub_input('v23_' + key, outtype)
        # Stable interface UUIDs; update both definition and instance, never definition alone.
        photo['inputs'][idx]['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, VERSION + '/' + key))
        photo['inputs'][idx]['pos'] = [16, 2510 + (idx-20)*20]
        inst['inputs'].append({'name': 'v23_' + key, 'type': outtype, 'link': None})
        top.connect(n, 0, inst, 'v23_' + key)
        # Typed reroute via a ready-made lazy switch with identical branches is unnecessary:
        # return subgraph virtual source plus input index for local connections.
        controls[key] = {'title': n['title'], 'default': value, 'input': 'v23_' + key, 'type': outtype}
        return idx

    enabled = control('enabled', 'PrimitiveBoolean', False, 'BOOLEAN', -320, 1940, '3D ВКЛ (default OFF)')
    usevideo = control('video1', 'PrimitiveBoolean', True, 'BOOLEAN', 90, 1940, '<Video 1> ВКЛ (S3 comp2)')
    steps = control('steps', 'INTConstant', 25, 'INT', 90, 2200, 'base steps (no turbo)')
    seed = control('seed', 'INTConstant', 101, 'INT', 500, 2200, 'seed (fixed)')
    # Expose local IMAGE inputs with their own LoadImage nodes; no dependence on muted PHOTO loaders.
    def image_input(key, filename, x, label):
        n = add(top, 'LoadImage', 'ВХОД / ' + label, outputs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')],
                widgets={'image': filename, 'upload': 'image'}, pos=(x, 2570), size=(380, 320))
        idx = pg.add_sub_input('v23_' + key, 'IMAGE')
        photo['inputs'][idx]['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, VERSION + '/' + key))
        photo['inputs'][idx]['pos'] = [16, 2510 + (idx-20)*20]
        inst['inputs'].append({'name': 'v23_' + key, 'type': 'IMAGE', 'link': None})
        top.connect(n, 0, inst, 'v23_' + key)
        return idx
    plate = image_input('plate', 'example.png', -320, '<Picture 1> фотоплита (загрузить)')
    identity = image_input('identity', 'example.png', 90, '<Picture 2> лицо SFW (загрузить)')
    clay = image_input('clay', 'example.png', 500, 'playblast.png с тенями (загрузить)')
    shadow_in = image_input('shadow', 'example.png', 910, 'shadow.png тень вне фигуры (загрузить)')
    prompt_idx = control('prompt', 'PrimitiveStringMultiline', PROMPT, 'STRING', -320, 2970, 'SFW ref prompt (отдельно от V2.2)')

    def wire(idx, dst, name):
        pg.connect(-10, idx, dst, name)

    def toggled(label, a, b, idx, x, y, a_slot=0, b_slot=0):
        n = clone(old['маска: итог · ref2v'], label, x, y, [False])
        conn(a, n, 'on_false', a_slot); conn(b, n, 'on_true', b_slot); wire(idx, n, 'switch')
        return n

    # Engine lives below the legacy photo engine, not over its nodes.
    model = add(pg, 'UNETLoader', 'MODEL / base ref2va (no turbo)', outputs=[('MODEL', 'MODEL')],
                widgets={'unet_name': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'weight_dtype': 'default'}, pos=(100, 5000))
    shift = clone(old['gen · shift'], 'MODEL / shift', 500, 5000, [12, 3]); conn(model, shift, 'model')
    # Remove inherited shift input socket; the new shift has its own unchanged 12/3 widgets.
    shift['inputs'] = [i for i in shift['inputs'] if i['name'] != 'shift_video']
    clip = clone(old['gen · Qwen3-VL'], 'MODEL / Qwen3-VL', 100, 5180,
                 ['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'minimax', 'default'])
    vae = clone(old['gen · H3 video VAE'], 'MODEL / H3 video VAE', 100, 5350)
    avae = clone(old['w7 · H3 audio VAE'], 'MODEL / H3 audio VAE', 100, 5480)
    def scale(idx, label, x, y):
        n = add(pg, 'ImageScale', label, inputs=[('image','IMAGE')], outputs=[('IMAGE','IMAGE')],
                widgets={'upscale_method':'lanczos','width':1344,'height':768,'crop':'disabled'}, pos=(x,y))
        wire(idx, n, 'image'); return n
    bg = scale(plate, 'INPUT / фотоплита 1344x768', 500, 5200)
    pb = scale(clay, 'INPUT / playblast 1344x768', 500, 5400)
    repeat = add(pg,'RepeatImageBatch','INPUT / playblast ×5 frames @24fps', inputs=[('image','IMAGE')], outputs=[('IMAGE','IMAGE')],widgets={'amount':5},pos=(900,5400))
    conn(pb,repeat,'image')
    r2v = clone(old['w7 · location sheet → условие'], 'COND / без Video 1', 1300, 5000, ['',1344,768,5,'match'])
    r2vv = clone(r2v,'COND / с Video 1 (R1s)', 1800,5000)
    for n in (r2v,r2vv):
        conn(clip,n,'clip'); conn(vae,n,'vae'); conn(avae,n,'audio_vae')
        conn(bg,n,'ref_images.ref_image_0'); wire(identity,n,'ref_images.ref_image_1')
    # This diagnostic fallback must not reference a Video that is not connected.
    no_video_prompt = '\n'.join(line for line in PROMPT.splitlines()
                                if not line.startswith('<Video 1>'))
    no_video_prompt = no_video_prompt.replace('<Video 1>', '<Picture 1>').replace('the playblast', 'the room photograph')
    no_video_prompt = no_video_prompt.replace('Replace the clay figure', 'Place the woman naturally in the room')
    pn = add(pg, 'PrimitiveStringMultiline', 'COND / SFW fallback without Video1',
             outputs=[('STRING','STRING')], widgets={'value':no_video_prompt}, pos=(1300,6300), size=(400,300))
    conn(pn,r2v,'prompt'); wire(prompt_idx,r2vv,'prompt')
    conn(repeat,r2vv,'ref_videos.ref_video_0')
    cond = toggled('SELECT / Video1 conditioning',r2v,r2vv,usevideo,2300,5000)
    lat = toggled('SELECT / Video1 latent',r2v,r2vv,usevideo,2300,5200,1,1)
    mods = clone(old['w7 · RefMod → условие'],'COND / existing RefMod (unchanged recipe)',2700,5000)
    conn(cond,mods,'conditioning'); pg.connect(-10,9,mods,'mods')
    neg = clone(old['gen · пустой negative'],'COND / negative',3100,5000); conn(mods,neg,'conditioning')
    ks = clone(old['gen · sampler pass 1'],'SAMPLE / base 25 steps cfg1',3500,5000,['enable',101,'fixed',25,1,'euler','simple',0,10000,'disable'])
    for src,name in ((shift,'model'),(mods,'positive'),(neg,'negative'),(lat,'latent_image')):conn(src,ks,name)
    wire(steps,ks,'steps');wire(seed,ks,'noise_seed')
    dec = clone(old['gen · decode'],'DECODE / R1s',4000,5000);conn(ks,dec,'samples');conn(vae,dec,'vae')
    sel = clone(old['gen · выбор кадра'],'SELECT / generated frame',4500,5000);conn(dec,sel,'frames');conn(dec,sel,'recommended_index',3)
    # PERSON: SERVER RMBG schema (no model_name widget; the FACE pass SAM3 of V2.2 uses the same old layout).
    person = add(pg,'SAM3Segment','COMPOSITE / SAM3 person', inputs=[('image','IMAGE')],
                 outputs=[('IMAGE','IMAGE'),('MASK','MASK'),('MASK_IMAGE','IMAGE')],
                 widgets={'prompt':'person','output_mode':'Merged','confidence_threshold':0.5,'max_segments':1,'segment_pick':0,'mask_blur':0,'mask_offset':0,'device':'Auto','invert_output':False,'unload_model':False,'background':'Alpha','background_color':'#222222'},pos=(5000,5000),size=(350,430))
    # Output size = V2.2 size (07.11d): room photo -> V2.2 megapixels (2 MP, own aspect); the 1344x768 generation,
    # the shadow and the photo are composited at that size (the 1344x768 aspect squeeze is undone by the upscale).
    mp = clone(old['gen · плита → мегапиксели'],'COMPOSITE / фото → мегапиксели V2.2',4300,4700)
    wire(plate,mp,'image'); wire(next(i for i,x in enumerate(photo['inputs']) if x['name']=='megapixels'),mp,'megapixels')
    size = clone(old['11190:11116'],'COMPOSITE / размер выхода',4700,4700); conn(mp,size,'image')
    def to_out(label,x,y,src=None,idx=None):
        n = clone(old['маска: к размеру исходника · ref2v'],label,x,y,['lanczos',512,512,'disabled'])
        if src is not None: conn(src,n,'image')
        else: wire(idx,n,'image')
        conn(size,n,'width',0); conn(size,n,'height',1)
        return n
    big = to_out('COMPOSITE / кадр → размер V2.2',4700,5000,src=sel)
    conn(big,person,'image')
    # S3 comp2 feather scaled to ~2 MP (x1.43): grow 6 px, blur 9 px
    blur = clone(old['shm'],'COMPOSITE / feather person',5500,5000,[6,0,True,False,9,1,1,False]);conn(person,blur,'mask',1)
    # Cast/contact shadow on the photo: photo x (1 - 0.8*shadow) = ImageBlend(photo, invert(shadow), 0.8, multiply)
    # (core: image1*(1-f) + image1*image2*f). shadow.png comes from render_plate_scene (figure rim already removed).
    sh = to_out('COMPOSITE / shadow → размер V2.2', 5000, 5500, idx=shadow_in)
    inv = add(pg,'ImageInvert','COMPOSITE / 1 - shadow',inputs=[('image','IMAGE')],outputs=[('IMAGE','IMAGE')],pos=(5500,5500),size=(260,60))
    conn(sh,inv,'image')
    blend = add(pg,'ImageBlend','COMPOSITE / photo x (1 - 0.8 shadow)',inputs=[('image1','IMAGE'),('image2','IMAGE')],
                outputs=[('IMAGE','IMAGE')],widgets={'blend_factor':0.8,'blend_mode':'multiply'},pos=(5900,5500),size=(320,110))
    conn(mp,blend,'image1');conn(inv,blend,'image2')
    composite = clone(old['f85:shc'],'COMPOSITE / person onto photo',6400,5000,[0,0,False])
    conn(blend,composite,'destination');conn(big,composite,'source');conn(blur,composite,'mask')
    beforeface = toggled('SELECT / 3D OFF legacy frame',old['gen · выбор кадра'],composite,enabled,9000,5000)
    # Only two legacy IMAGE consumers are redirected; all face-pass widgets/conditioning untouched.
    rewired = []
    for target,name in ((old['11190:11128'],'image'),(old['11190:11114'],'image')):
        inp = next(i for i in target['inputs'] if i['name']==name)
        original = copy.deepcopy(pg.link(inp['link']))
        conn(beforeface,target,name)
        rewired.append(original)
    photo.setdefault('groups',[]).append({'id':max([g.get('id',0) for g in photo.get('groups',[])]+[0])+1,'title':'3D · движок (не трогать)','bounding':[50,4900,9450,1500],'color':'#31526b','flags':{}})
    inst['size'][1] += (len(photo['inputs'])-20)*20
    r['groups'].append({'id':max(g['id'] for g in r['groups'])+1,'title':'3D · отдельный пульт (default OFF)','bounding':[-350,1870,1270,1600],'color':'#31526b','flags':{}})
    note = add(top,'MarkdownNote','README / рецепты и ограничения',widgets={'text':'''SERVER variant (studio ComfyUI 0.34.6): recipe S3 comp2 only. No Fun CN, no harmonisation (both doubled the face / changed the room in tests). Fun CN variant = V2.3-pod later.
3D default OFF: legacy V2.2 photo/video/face controls are unchanged. Enable PHOTO as in V2.2, then 3D ВКЛ.
Upload: Picture1 room photo, Picture2 SFW identity, playblast.png (Video 1, clay + shadows, render_plate_scene.py) and shadow.png from the same render folder. Generation 1344x768; composite and output at the V2.2 megapixels size (2 MP); 5 frames @24fps; base ref2va 25 steps cfg1, no turbo.
Chain: generation -> upscale to V2.2 size -> SAM3 person (grow 6 / blur 9) -> person over photo x (1 - 0.8 shadow) -> existing FACE pass (its denoise unchanged; profile / face down: 0.35 only).
Existing PHOTO RefMod loader is reused (set the Stacy mods there). Legacy mask editor applies after face pass; disable mask mode for 3D acceptance.
Troubleshooting: pose drifts -> check the playblast; pasted edge -> PERSON mask/feather; shadow lost -> shadow.png must be from the SAME render as playblast.png.'''},pos=(-1050,1940),size=(650,850))
    # Preserve legacy graph state counters except newly extended photo state.
    for s0 in source['definitions']['subgraphs']:
        s = next(s for s in r['definitions']['subgraphs'] if s['id']==s0['id'])
        if s is not photo: s['state'] = copy.deepcopy(s0['state'])
    r.setdefault('extra',{})['aio_v23'] = {'version':VERSION,'source_sha256':hashlib.sha256(json.dumps(source,ensure_ascii=False,indent=1).encode()).hexdigest(),'photo_id':photo['id'],'added_nodes':added,'controls':controls,'rewired_legacy_links':rewired,'winner':None,'runtime_tested':False}
    return r


def validate(r):
    errors = list(check_ui(r))
    subs = {s['id']:s for s in r['definitions']['subgraphs']}
    added = set(r['extra']['aio_v23']['added_nodes'])
    titles = set()
    def compatible(a,b):
        return '*' in (a,b) or bool(set(a.split(',')) & set(b.split(',')))
    for g in graphs(r):
        nodes={n['id']:n for n in g['nodes']}
        if len(nodes)!=len(g['nodes']):errors.append('duplicate node id')
        links={_norm(l)['id']:_norm(l) for l in g['links']}
        if len(links)!=len(g['links']):errors.append('duplicate link id')
        for n in g['nodes']:
            if n['id'] in added:
                if n.get('title') in titles:errors.append('duplicate new title')
                titles.add(n.get('title'))
            if n['type'] in subs:
                s=subs[n['type']]
                for k in ('inputs','outputs'):
                    if [(x['name'],x['type']) for x in n[k]] != [(x['name'],x['type']) for x in s[k]]:errors.append('instance interface mismatch')
        for l in links.values():
            try:
                a=(g['inputs'][l['origin_slot']] if l['origin_id']==-10 else nodes[l['origin_id']]['outputs'][l['origin_slot']])
                b=(g['outputs'][l['target_slot']] if l['target_id']==-20 else nodes[l['target_id']]['inputs'][l['target_slot']])
                if not compatible(a['type'],b['type']) or not compatible(a['type'],l['type']) or not compatible(b['type'],l['type']):errors.append(f'type mismatch link {l["id"]}')
            except (KeyError,IndexError):errors.append('bad endpoint/slot')
        # Check reverse entries exactly, not just existence of an arbitrary id.
        for n in g['nodes']:
            for k,i in enumerate(n.get('inputs',[])):
                if i.get('link') is not None:
                    l=links[i['link']]
                    if (l['target_id'],l['target_slot'])!=(n['id'],k):errors.append('input reverse endpoint mismatch')
            for k,o in enumerate(n.get('outputs',[])):
                for lid in o.get('links') or []:
                    l=links[lid]
                    if (l['origin_id'],l['origin_slot'])!=(n['id'],k):errors.append('output reverse endpoint mismatch')
        for direction,endpoint,key in (('inputs',-10,'origin'),('outputs',-20,'target')):
            ids=[]
            for k,item in enumerate(g.get(direction,[]) if g is not r else []):
                ids.append(item['id'])
                for lid in item.get('linkIds',[]):
                    l=links[lid]
                    if (l[key+'_id'],l[key+'_slot'])!=(endpoint,k):errors.append('interface linkIds endpoint mismatch')
            if len(ids)!=len(set(ids)):errors.append('duplicate interface UUID')
    return errors


def off_compare(source,r):
    """Undo only declared additions/rewires; compare all legacy data, including opaque literals.

    Structural proof, NOT frontend export or runtime execution.
    """
    q=copy.deepcopy(r); meta=q['extra'].pop('aio_v23'); added=set(meta['added_nodes'])
    photo=next(s for s in q['definitions']['subgraphs'] if s['id']==meta['photo_id']); pg=G(photo,q)
    original=next(s for s in source['definitions']['subgraphs'] if s['id']==meta['photo_id'])
    # Verify OFF switch actually chooses the original selector, not merely an inverse patch.
    sw=next(n for n in photo['nodes'] if n.get('title')==PREFIX+'SELECT / 3D OFF legacy frame')
    false=pg.link(next(i for i in sw['inputs'] if i['name']=='on_false')['link'])
    legacy=next(n for n in original['nodes'] if n.get('title')=='gen · выбор кадра')
    assert (false['origin_id'],false['origin_slot'])==(legacy['id'],0)
    ctl=meta['controls']['enabled']; assert ctl['default'] is False
    actual_ctl=next(n for n in q['nodes'] if n.get('title')==ctl['title'])
    assert actual_ctl['widgets_values']==[False], '3D default must remain OFF'
    toggle=pg.link(next(i for i in sw['inputs'] if i['name']=='switch')['link'])
    assert toggle['origin_id']==-10 and photo['inputs'][toggle['origin_slot']]['name']==ctl['input']
    inst=next(n for n in q['nodes'] if n['type']==photo['id'])
    tg=G(q,q,True)
    for n in list(q['nodes']):
        if n['id'] in added:tg.remove_node(n['id'])
    for n in list(photo['nodes']):
        if n['id'] in added:pg.remove_node(n['id'])
    for l in meta['rewired_legacy_links']:
        # restore exact original ids and endpoint references
        photo['links'].append(copy.deepcopy(l))
        pg.node(l['origin_id'])['outputs'][l['origin_slot']]['links'].append(l['id'])
        pg.node(l['target_id'])['inputs'][l['target_slot']]['link']=l['id']
    photo['inputs']=photo['inputs'][:len(original['inputs'])]
    inst0=next(n for n in source['nodes'] if n['type']==photo['id'])
    inst['inputs']=inst['inputs'][:len(inst0['inputs'])]; inst['size']=copy.deepcopy(inst0['size'])
    q['groups']=copy.deepcopy(source['groups']); photo['groups']=copy.deepcopy(original['groups'])
    q['last_node_id']=source['last_node_id'];q['last_link_id']=source['last_link_id']
    photo['state']=copy.deepcopy(original['state'])
    # Canonicalize connection ordering only (fanout order has no execution meaning).
    def canonical(x):
        x=copy.deepcopy(x)
        for g in graphs(x):
            g['links']=sorted(g['links'],key=lambda l:_norm(l)['id'])
            for n in g['nodes']:
                for o in n.get('outputs',[]):
                    if isinstance(o.get('links'),list):o['links'].sort()
            for k in ('inputs','outputs'):
                for i in g.get(k,[]) if g is not x else []:
                    if 'linkIds' in i:i['linkIds'].sort()
        return x
    assert canonical(source)==canonical(q), 'Legacy data changed beyond declared insertions'
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=ROOT/'workflows/aio/AIO_manual_V2.2.ui.json')
    ap.add_argument('--output',type=Path,default=ROOT/'workflows/aio/AIO_manual_V2.3.ui.json')
    ap.add_argument('--check',action='store_true',help='Check existing output; do not write')
    args=ap.parse_args()
    if args.source.resolve()==args.output.resolve():raise SystemExit('Refusing to overwrite source')
    before=args.source.read_bytes(); source=json.loads(before)
    r=patch(source)
    errs=validate(r); off_compare(source,r)
    if errs:raise SystemExit('UI validation failed: '+repr(errs[:20]))
    if args.check:
        assert json.loads(args.output.read_text(encoding='utf-8'))==r,'Output differs from deterministic patch'
    else:
        args.output.write_text(json.dumps(r,ensure_ascii=False,indent=1),encoding='utf-8',newline='\n')
    assert args.source.read_bytes()==before,'Source mutated'
    print(json.dumps({'version':VERSION,'added_nodes':len(r['extra']['aio_v23']['added_nodes']),'graphs':len(graphs(r)),'nodes':sum(len(g['nodes']) for g in graphs(r)),'links':sum(len(g['links']) for g in graphs(r)),'link_errors':len(errs),'off_structural_comparison':True,'source_unchanged':True,'runtime_tested':False}))

if __name__=='__main__':
    main()
