"""Build AIO 3DA from the user's exact AIO, preserving its opaque legacy data.

Offline only. Live frontend exports and execution remain separate checks.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import wfkit
from scene3d.aio_preflight import clean

SOURCE_SHA256 = '5b6e88d9f743e516d9f059e50081513c788a9bfd1a57f21aa22f8e775e7cd1b6'
VERSION = '3DA_AIO_v1.0'
POSE_TEXT = 'Wide full-length photograph of one adult woman wearing an opaque grey T-shirt, navy trousers and white sneakers. The frame contains space above her head and floor below her shoes. Her entire head, both hands and both feet are visible with generous empty margins. She is standing and raises her right forearm vertically, with her right hand beside her head; her left arm hangs relaxed. Natural human proportions, eye-level fixed camera, neutral room background.'
QWEN_TEXT = 'Create one full-body photograph. Picture 1 supplies the adult character appearance and opaque clothing. Picture 2 supplies the room and composition. Place the same adult in that room, standing with one hand raised and the other relaxed. Both feet and both hands are visible. The desired pose comes from this instruction; do not copy the pose from either input. Keep natural body proportions.'


def pack_schema():
    p=ROOT/'comfyui/ComfyUI-3DA/nodes.py'
    spec=importlib.util.spec_from_file_location('aio_pack_schema',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return {name:dict(input=cls.INPUT_TYPES(),output=cls.RETURN_TYPES,output_name=cls.RETURN_NAMES,
                      output_node=getattr(cls,'OUTPUT_NODE',False)) for name,cls in mod.NODE_CLASS_MAPPINGS.items()}


class Build:
    def __init__(self, workflow, schema):
        self.w=workflow;self.oi=schema;self.top=wfkit.G(workflow,workflow,top=True)

    def add(self,g,typ,title,values=None,pos=(0,0),size=(350,160),mode=0):
        info=self.oi[typ];schema={**info.get('input',{}).get('required',{}),**info.get('input',{}).get('optional',{})}
        values=values or {};inputs=[];named={};widgets=[]
        for field,entry in schema.items():
            t=entry[0];opts=entry[1] if len(entry)>1 else {}
            is_widget=isinstance(t,list) or t in ('STRING','INT','FLOAT','BOOLEAN','COMBO')
            if is_widget and not opts.get('forceInput'):
                choices=t if isinstance(t,list) else opts.get('options',[])
                default=opts.get('default',choices[0] if choices else '' if t in ('STRING','COMBO') else False if t=='BOOLEAN' else 0)
                value=values.get(field,default);named[field]=value;widgets.append(value)
                if opts.get('control_after_generate') or field in ('seed','noise_seed'):
                    widgets.append('fixed');named['control_after_generate']='fixed'
            elif field in info.get('input',{}).get('required',{}):inputs.append((field,t))
        for key,value in values.items():
            if key not in named: named[key]=value
        n=g.add_node(typ,title,inputs=inputs,outputs=list(zip(info.get('output_name',info['output']),info['output'])),
                     pos=pos,size=size,mode=mode)
        n['widgets_values']=widgets;n['widgets_values_named']=named
        if typ=='Load3DAdvanced':
            n['inputs']=[i for i in n['inputs'] if i['name']!='viewport_state']
            n['widgets_values']=[named['model_file'],values.get('viewport_state',{}),named['width'],named['height']]
        n['properties']['3DA AIO']=VERSION
        return n

    def connect(self,g,src,slot,dst,name):
        schema={**self.oi.get(dst['type'],{}).get('input',{}).get('required',{}),
                **self.oi.get(dst['type'],{}).get('input',{}).get('optional',{})}
        entry=schema.get(name,[]);opts=entry[1] if len(entry)>1 else {}
        typ=src['outputs'][slot]['type'] if src!=-10 else g.g['inputs'][slot]['type']
        widget=bool(entry and (isinstance(entry[0],list) or entry[0] in ('STRING','INT','FLOAT','BOOLEAN','COMBO')) and not opts.get('forceInput'))
        return g.connect(src,slot,dst,name,type_=typ,widget=widget)

    def switch(self,g,title,false,true,control,pos):
        n=self.add(g,'LazySwitchKJ',title,{'switch':False},pos)
        for src,field in [(false,'on_false'),(true,'on_true'),(control,'switch')]:self.connect(g,*src,n,field)
        return n

    def image(self,title,pos):return self.add(self.top,'LoadImage',title,{'image':'','upload':'image'},pos,(350,300))

    def mode_switch(self,g,title,false,true,control,pos):
        n=self.add(g,'ImpactSwitch',title,{'select':2,'sel_mode':True},pos)
        for index,src in enumerate((false,true),1):self.connect(g,*src,n,'input'+str(index))
        self.connect(g,*control,n,'select')
        return n


def wrap(w, selected, name, pos):
    """Move nodes into one standard subgraph, preserving every boundary link."""
    top=wfkit.G(w,w,top=True);sid=str(uuid.uuid5(uuid.NAMESPACE_URL,VERSION+'/'+name))
    selected=set(selected);original=list(w['nodes']);nodes={n['id']:n for n in original}
    sub=dict(id=sid,name=name,version=1,state={},inputNode={'id':-10,'bounding':[-400,0,240,700]},
             outputNode={'id':-20,'bounding':[4200,0,240,700]},inputs=[],outputs=[],nodes=[n for n in original if n['id'] in selected],
             links=[],groups=[],widgets=[],config={},extra={})
    w.setdefault('definitions',{}).setdefault('subgraphs',[]).append(sub)
    instance=top.add_node(sid,name,pos=pos,size=(470,420),inputs=[],outputs=[],widgets={})
    incoming={};outgoing={};newlinks=[]
    for link in list(w['links']):
        lid,src,so,dst,di,t=link;si=src in selected;ti=dst in selected
        if si and ti:sub['links'].append(dict(id=lid,origin_id=src,origin_slot=so,target_id=dst,target_slot=di,type=t))
        elif ti:
            key=(src,so)
            if key not in incoming:
                idx=len(incoming);incoming[key]=idx;outer=top._new_link_id();label=nodes[src].get('title',str(src))+' / '+str(so)
                sub['inputs'].append(dict(id=str(uuid.uuid4()),name=label,type=t,linkIds=[]))
                instance['inputs'].append(dict(name=label,type=t,link=outer));newlinks.append([outer,src,so,instance['id'],idx,t])
                old=nodes[src]['outputs'][so]['links'];nodes[src]['outputs'][so]['links']=[i for i in old if i!=lid]+[outer]
            else:
                nodes[src]['outputs'][so]['links']=[i for i in nodes[src]['outputs'][so]['links'] if i!=lid]
            idx=incoming[key];sub['inputs'][idx]['linkIds'].append(lid)
            sub['links'].append(dict(id=lid,origin_id=-10,origin_slot=idx,target_id=dst,target_slot=di,type=t))
        elif si:
            key=(src,so)
            if key not in outgoing:
                idx=len(outgoing);outgoing[key]=idx;inner=top._new_link_id();label=nodes[src].get('title',str(src))+' / '+str(so)
                sub['outputs'].append(dict(id=str(uuid.uuid4()),name=label,type=t,linkIds=[inner]))
                instance['outputs'].append(dict(name=label,type=t,links=[]))
                sub['links'].append(dict(id=inner,origin_id=src,origin_slot=so,target_id=-20,target_slot=idx,type=t))
                old=nodes[src]['outputs'][so]['links'];nodes[src]['outputs'][so]['links']=[i for i in old if i!=lid]+[inner]
            else:nodes[src]['outputs'][so]['links']=[i for i in nodes[src]['outputs'][so]['links'] if i!=lid]
            idx=outgoing[key];instance['outputs'][idx]['links'].append(lid);newlinks.append([lid,instance['id'],idx,dst,di,t])
        else:newlinks.append(link)
    w['nodes']=[n for n in w['nodes'] if n['id'] not in selected];w['links']=newlinks
    return instance


def pose_sources(b):
    g=b.top;before={n['id'] for n in b.w['nodes']}
    upload=b.image('3DA · Поза: загруженный референс',(0,1800))
    char=b.image('3DA · Qwen Picture 1: персонаж и внешность',(0,2200))
    room=b.image('3DA · Qwen Picture 2: локация и композиция',(400,2200))
    source=b.add(g,'ImpactSwitch','3DA · pose_source: 1=фото / 2=Z-Image / 3=Qwen2511',{'select':1,'sel_mode':True},(850,1800))
    for i,label in enumerate(['Загруженный референс','Z-Image T2I','Qwen2511 I2I (2 фото)'],1):
        source['inputs'].append(dict(name='input'+str(i),label=label,type='IMAGE',link=None))
    def branch(prefix,model,clip_name,clip_type,vae_name,prompt,steps,cfg,sampler,scheduler,x):
        unet=b.add(g,'UNETLoader',prefix+' · model',{'unet_name':model,'weight_dtype':'default'},(x,3000))
        clip=b.add(g,'CLIPLoader',prefix+' · text encoder',{'clip_name':clip_name,'type':clip_type,'device':'default'},(x,3200))
        vae=b.add(g,'VAELoader',prefix+' · VAE',{'vae_name':vae_name},(x,3400))
        lat=b.add(g,'EmptySD3LatentImage',prefix+' · размер pose-reference',{'width':768 if prefix=='Z-Image' else 1024,'height':1024,'batch_size':1},(x+400,3400))
        enc=b.add(g,'TextEncodeQwenImageEditPlus' if prefix=='Qwen2511' else 'CLIPTextEncode',prefix+' · желаемая поза (текст)',{'prompt':prompt} if prefix=='Qwen2511' else {'text':prompt},(x+400,3000),(410,260))
        b.connect(g,clip,0,enc,'clip')
        if prefix=='Qwen2511':
            for s,slot,target in [(vae,0,'vae'),(char,0,'image1'),(room,0,'image2')]:b.connect(g,s,slot,enc,target)
            neg=b.add(g,'TextEncodeQwenImageEditPlus',prefix+' · negative с теми же двумя refs',{'prompt':''},(x+850,3000))
            for s,slot,target in [(clip,0,'clip'),(vae,0,'vae'),(char,0,'image1'),(room,0,'image2')]:b.connect(g,s,slot,neg,target)
            shift=b.add(g,'ModelSamplingAuraFlow',prefix+' · shift',{'shift':3.1},(x+400,2800));b.connect(g,unet,0,shift,'model');model_node=shift
        else:
            neg=b.add(g,'ConditioningZeroOut',prefix+' · CFG1 negative',{},(x+850,3000));b.connect(g,enc,0,neg,'conditioning');model_node=unet
        sample=b.add(g,'KSampler',prefix+' · перегенерировать seed',{'seed':101,'steps':steps,'cfg':cfg,'sampler_name':sampler,'scheduler':scheduler,'denoise':1.0},(x+1300,3000),(350,300))
        for s,slot,target in [(model_node,0,'model'),(enc,0,'positive'),(neg,0,'negative'),(lat,0,'latent_image')]:b.connect(g,s,slot,sample,target)
        dec=b.add(g,'VAEDecode',prefix+' · pose-reference IMAGE',{},(x+1700,3000));b.connect(g,sample,0,dec,'samples');b.connect(g,vae,0,dec,'vae')
        return dec
    z=branch('Z-Image','z_image_turbo_bf16.safetensors','qwen_3_4b.safetensors','lumina2','ae.safetensors',POSE_TEXT,9,1.,'euler','simple',0)
    q=branch('Qwen2511','qwen_image_edit_2511_fp8mixed.safetensors','qwen_2.5_vl_7b_fp8_scaled.safetensors','qwen_image','qwen_image_vae.safetensors',QWEN_TEXT,40,4.,'euler','simple',2300)
    for i,s in enumerate([upload,z,q],1):b.connect(g,s,0,source,'input'+str(i))
    preview=b.add(g,'PreviewImage','3DA · Просмотр выбранного pose-reference',{},(1300,1800),(430,360));b.connect(g,source,0,preview,'images')
    # Keep the three explicit LoadImage inputs and the selector outside. Models
    # and samplers are ordinary native nodes in a readable engine subgraph.
    outside={n['id'] for n in [upload,char,room,source,preview]}
    wrap(b.w,{n['id'] for n in b.w['nodes'] if n['id'] not in before|outside},'3DA · Источники позы (не трогать)',(850,2200))
    return source,preview


def integrate(source, oi):
    w=copy.deepcopy(source);w['id']=str(uuid.uuid5(uuid.NAMESPACE_URL,VERSION));w.setdefault('extra',{})['3DA AIO']=dict(version=VERSION,source_sha256=SOURCE_SHA256)
    b=Build(w,oi);g=b.top;original={n['id'] for n in w['nodes']}
    # The added console occupies the left side; preserve the old AIO layout
    # together on the right instead of overlaying its controls.
    for n in w['nodes']:n['pos'][0]+=8000
    for group in w.get('groups',[]):group['bounding'][0]+=8000
    pose,pose_preview=pose_sources(b)
    plate=b.image('3DA · Фотоплита комнаты',(0,1250));identity=b.image('3DA · Персонаж: эталон внешности SFW',(400,1250))
    enabled=b.add(g,'ImpactInt','3DA · Режим фото: 1=исходный AIO / 2=постановка',{'value':2},(850,750))
    native_steps=b.add(g,'INTConstant','3DA · H3 base steps (без turbo)',{'value':25},(1250,750))
    loc=b.add(g,'3DALocation','3DA · Локация',{'location_id':'mainroom1'},(850,1000))
    b.connect(g,plate,0,loc,'plate')
    prop=b.add(g,'3DAProp','3DA · Предметы для подготовки',{'prop_id':'','anchor':'world','size_m':.30},(0,2700),(400,450))
    # No prop is implicitly loaded. Connect this optional chain after selecting
    # a registered prop; the preparation node defaults to an empty prop list.
    prep=b.add(g,'3DAPlacePose','3DA · Подготовка персонажа',{'pose_class':'stand','target_object':'floor','block_on_fail':False,'seed':101},(850,2750),(470,650))
    b.connect(g,loc,0,prep,'location');b.connect(g,pose,0,prep,'pose_ref')
    export=b.add(g,'3DASceneExport','3DA · Подготовить сцену гизмо',{},(1350,2750),mode=2);b.connect(g,prep,9,export,'body_guides')
    viewport=b.add(g,'Load3DAdvanced','3DA · Сцена: персонаж и предметы',{'model_file':'none','viewport_state':{},'width':1344,'height':768},(0,3500),(850,760))
    edit=b.add(g,'3DASceneEdits','3DA · Применить правки сцены',{'manifest':''},(900,3500))
    for s,slot,target in [(viewport,0,'model_3d'),(viewport,1,'model_3d_info'),(viewport,2,'camera_info'),(loc,0,'location')]:b.connect(g,s,slot,edit,target)
    place=b.add(g,'3DAPlacePose','3DA · Итог: повторная проверка сцены',{'pose_class':'stand','target_object':'floor','block_on_fail':True,'seed':101},(1350,3500),(470,650))
    for s,slot,target in [(loc,0,'location'),(pose,0,'pose_ref'),(edit,0,'body_transform'),(edit,1,'props')]:b.connect(g,s,slot,place,target)
    guides=b.add(g,'3DANativeGuides','3DA · Допуск сцены до загрузки H3',{},(1850,3500));b.connect(g,place,9,guides,'body_guides')
    # Add a native H3 branch to the EXISTING photo engine. Its RefMod input,
    # face pass, colour matching and original switches remain connected.
    photo=next(s for s in w['definitions']['subgraphs'] if any(n.get('title')=='gen · sampler pass 1' for n in s['nodes']))
    pg=wfkit.G(photo,w);inst=next(n for n in w['nodes'] if n['type']==photo['id']);by={n.get('title'):n for n in photo['nodes']}
    def inward(src,slot,label):
        typ=src['outputs'][slot]['type'];idx=pg.add_sub_input('3da_'+label,typ);inst['inputs'].append(dict(name='3da_'+label,type=typ,link=None));g.connect(src,slot,inst,'3da_'+label,type_=typ);return (-10,idx)
    enabled_in=inward(enabled,0,'enabled')
    steps_in=inward(native_steps,0,'native_steps')
    external={key:inward(guides,slot,key) for key,slot in [('plate',0),('playblast',1),('init',2),('strength',3),('prompt',4),('width',5),('height',6)]}
    external['identity']=inward(identity,0,'identity')
    # Reuse the verified existing native server graph and its exact samplers.
    api=json.loads((ROOT/'workflows/3DA_h3_server_api.json').read_text(encoding='utf-8'))
    wanted=['unet','shift','regional_model','r2v','D1_lat','D1_noise','D1_guider','D1_sampler','D1_sample','D1_decode','D1_select','playblast5']
    built={}
    refs={'clip':by['gen · Qwen3-VL'],'vae':by['gen · H3 video VAE'],'avae':by['w7 · H3 audio VAE']}
    alias={'ref_room':'plate','playblast':'playblast','playblast_input':'playblast','init_input':'init','strength_mask':'strength','prompt_text':'prompt','ref_char':'identity'}
    for index,key in enumerate(wanted):
        a=api[key];values={k:v for k,v in a['inputs'].items() if not isinstance(v,list)}
        built[key]=b.add(pg,a['class_type'],'3DA H3 · '+key,values,(100+(index%4)*440,5200+(index//4)*400),(400,270))
    for key,n in built.items():
        for field,value in api[key]['inputs'].items():
            if not isinstance(value,list):continue
            src,slot=value
            if src in built:origin=(built[src],slot)
            elif src in refs:origin=(refs[src],slot)
            elif src in alias:origin=external[alias[src]]
            else:raise ValueError('Unmapped native H3 input '+src)
            b.connect(pg,*origin,n,field)
    for name in ('width','height'):b.connect(pg,*external[name],built['r2v'],name)
    b.connect(pg,*steps_in,built['D1_lat'],'steps')
    apply=b.add(pg,'MiniMaxH3RefModApply','3DA H3 · действующие AIO RefMods',
                copy.deepcopy(by['w7 · RefMod → условие'].get('widgets_values_named',{})),(1900,5200))
    b.connect(pg,built['r2v'],0,apply,'conditioning')
    mods_idx=next(i for i,n in enumerate(photo['inputs']) if n['name']=='mods');b.connect(pg,-10,mods_idx,apply,'mods');b.connect(pg,apply,0,built['D1_guider'],'conditioning')
    raw=by['gen · выбор кадра'];consumers=[]
    for l in list(photo['links']):
        d=wfkit._norm(l)
        if d['origin_id']==raw['id'] and d['origin_slot']==0:consumers.append((pg.node(d['target_id']),d['target_slot']))
    chosen=b.mode_switch(pg,'3DA · фото перед существующим face pass',(raw,0),(built['D1_select'],0),enabled_in,(2200,6500))
    for n,slot in consumers:b.connect(pg,chosen,0,n,n['inputs'][slot]['name'])
    after_face=by['f85:shc'];legacy_final=by['маска: итог · ref2v']
    photo_mode=b.mode_switch(pg,'3DA · сохранить исходную маску / передать face-pass на строгую вклейку',
                             (legacy_final,0),(after_face,0),enabled_in,(2700,6500))
    pg.connect(photo_mode,0,-20,3,type_='IMAGE')
    # Composite only AFTER the entire photo engine / face pass. Geometry gate
    # is a lazy barrier around the whole photo subtree, including its loaders.
    photo_out=next(n for n in w['nodes'] if n.get('title')=='ИТОГ фото · ref2v')
    old_link=g.link(photo_out['inputs'][0]['link']);photo_slot=old_link['origin_slot']
    barrier=b.add(g,'LazySwitchKJ','3DA · Geometry gate → native фото',{'switch':False},(1900,4100))
    for s,slot,target in [(guides,7,'switch'),(plate,0,'on_false'),(inst,photo_slot,'on_true')]:b.connect(g,s,slot,barrier,target)
    composite=b.add(g,'3DANativeComposite','3DA · Финал после face pass',{},(2300,4100))
    for s,slot,target in [(barrier,0,'image'),(place,9,'body_guides'),(place,7,'prop_repair'),(place,8,'prop_repair_mask')]:b.connect(g,s,slot,composite,target)
    final=b.mode_switch(g,'3DA · обычное фото / 3D-постановка',(inst,photo_slot),(composite,0),(enabled,0),(2700,4100));b.connect(g,final,0,photo_out,'images')
    preview=b.add(g,'PreviewImage','3DA · Проверить новый рендер до H3',{},(1850,2750),(430,360));b.connect(g,place,0,preview,'images')
    status=b.add(g,'PreviewAny','3DA · Контакты / кадр / статус',{},(2300,2750),(430,360));b.connect(g,place,5,status,'source')
    metric=b.add(g,'PreviewAny','3DA · Метрики финальной вклейки',{},(2700,4400));b.connect(g,composite,2,metric,'source')
    # Preserve every existing top-level node and four legacy subgraphs.
    assert original <= {n['id'] for n in w['nodes']}
    note=b.add(g,'MarkdownNote','README',{},(0,0),(820,620));note['widgets_values']=[README];note.pop('widgets_values_named',None)
    cheat=b.add(g,'MarkdownNote','Симптом → крутилка',{},(900,0),(700,620));cheat['widgets_values']=[CHEAT];cheat.pop('widgets_values_named',None)
    w['groups'] += [dict(id=1001,title='1 · Вход / 3DA',bounding=[-40,1200,850,1450],color='#3f789e',font_size=24,flags={}),
                    dict(id=1002,title='2 · ПУЛЬТ / 3DA',bounding=[820,700,1050,3450],color='#9c7442',font_size=24,flags={}),
                    dict(id=1003,title='3 · Движок / 3DA',bounding=[-40,2650,850,1650],color='#4d556b',font_size=24,flags={}),
                    dict(id=1004,title='4 · Результат / 3DA',bounding=[1850,2650,1300,2100],color='#427b61',font_size=24,flags={})]
    for n in (pose_preview,preview,status,metric):n['mode']=2
    return w


README = '''3DA AIO · два этапа
1. Сначала откройте prepare.ui.json: заполните фотоплиту и pose_source. 1=загруженный референс; 2=Z-Image T2I; 3=Qwen2511: два независимых LoadImage, персонаж+комната, позу задаёт текст. Неактивные ветки отрезаются ImpactSwitch select_on_prompt ДО валидации: пустые чужие LoadImage не нужны.
2. Очередь «Подготовить сцену гизмо» строит настоящие меши. Extension загружает GLB в общий viewport, камеру комнаты и manifest. Тело — геометрия SAM, не плоская картинка. Только комната остаётся фотоплитой. Предметы по желанию: выберите prop_id и соедините Prop→Подготовка.props до подготовки.
3. В viewport выберите «Персонаж» или «Предмет», рядом — «Перемещение / Вращение / Масштаб». Любая ручка масштаба персонажа равномерно меняет его размер; предмет допускает масштаб по осям. Backend повторно проверяет размер/контакты/кадр, не принимает отрицательный scale, растяжение тела или shear. Камера фиксирована, Fit заблокирован. Локация восстанавливается из подготовленной плиты. Это корень тела, не редактор суставов.
4. Сохраните UI после подготовки. Для правок/финала включите итоговую проверку и фото, отключите SceneExport (он заново готовит исходную сцену). При изменении позы, роста, комнаты или состава предметов подготовьте сцену заново.
5. Итог повторно проверяет меш, контакты, кадр и предметы. Непринятая сцена блокирует всю native H3-ветку. На том же сервере нет Remote/self-submit. AIO RefMods применяются к новому conditioning; существующий face pass идёт до последнего восстановления комнаты/предметов. Видео и face-pass готового видео сохранены.
6. Для preview используйте preview.api.json, для фото photo.api.json. API требует подставить входы/manifest подготовленной сцены. Пустой manifest — ожидаемая ошибка, не готовый финал. Требуется установленная серверная геометрия; наличие нод её не доказывает.
7. Числовые body_* сохранены для прежних workflow; при подключённом гизмо тела действует его матрица вместо этих ручек. При смене pose_source измените select самого ImpactSwitch (1–3), sel_mode оставьте true.'''
CHEAT = '''Камера двигается → убедитесь, что загружен 3DA extension; backend guard не отключать.
Двигается другой объект → ПКМ viewport → Выбрать · Персонаж / Предмет.
Нужна другая поза → pose_source и текст в источнике; TRS не сгибают суставы.
Масштаб тела → height_m в подготовке И итоговой постановке, затем новая сцена.
body_dx/dy/dz → метры в right/depth/up базисе комнаты; body_dyaw/dpitch/droll → градусы вокруг таза. В numeric режиме после правки Queue и новая проверка. В scene режиме числовые поправки заменены гизмо.
Изменили pose_class/опору/height/seed → одинаково в подготовке и итоге; иначе stale-scene guard.
Плохой кадр/контакт → исправить гизмо, смотреть новый playblast и статус; H3 закрыт при fail.
Нужен только pose-reference → отдельный pose.api.json, без геометрии и H3.
Слабая идентичность → реальные фото/RefMod в сохранённых AIO loader; builder не обучает LoRA.
Нужен финал без видео → photo.api.json или mute групп ВИДЕО и ПРОХОД ЛИЦА готового видео.'''


def refmod_builder(oi, count=1):
    w=dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL,VERSION+'/refmod')),version=.4,last_node_id=0,last_link_id=0,nodes=[],links=[],groups=[],config={},extra={})
    b=Build(w,oi);g=b.top;images=[b.image('Фото '+str(i+1)+' · '+('подключено' if i<count else 'необязательно: загрузить и подключить'),((i%4)*400,(i//4)*400)) for i in range(8)]
    vae=b.add(g,'VAELoader','RefMod · H3 video VAE',{'vae_name':'minimax_h3_video_vae_fp16.safetensors'},(0,950))
    extract=b.add(g,'MiniMaxH3RefModExtract','RefMod · 1–8 фото → reference adapter',dict(name='my_concept',concept_type='identity',mode='Full Reference',extraction_preset='manual',ref_resolution=512,
                  latent_frames=1,identity=0,merge=False,motion_only=False,multiplier=1,max_tokens=16384,budget_policy='error',save=False,subfolder='farm'),(450,950),(450,650))
    for i,n in enumerate(images[:count]):b.connect(g,n,0,extract,'refs_image.ref_image_'+str(i))
    b.connect(g,vae,0,extract,'vae')
    save=b.add(g,'MiniMaxH3RefModSave','RefMod · Save в models/refmods/farm',{'filename_prefix':'','subfolder':'farm'},(1000,950));b.connect(g,extract,0,save,'mods')
    inspect=b.add(g,'MiniMaxH3RefModInspect','RefMod · Inspect / preview сохранённого',{'index':0,'preview':'stored','strength':1.,'visual_preview':'first_frame'},(1000,1250));b.connect(g,save,0,inspect,'mods');b.connect(g,vae,0,inspect,'vae')
    for node,slot,title,pos in [(save,1,'RefMod · Путь результата',(1450,950)),(inspect,0,'RefMod · Метаданные',(1450,1150))]:
        view=b.add(g,'PreviewAny',title,{},pos);b.connect(g,node,slot,view,'source')
    view=b.add(g,'PreviewImage','RefMod · Превью',{},(1450,1400),(400,350));b.connect(g,inspect,1,view,'images')
    note=b.add(g,'MarkdownNote','README',{},(0,-550),(1500,500));note['widgets_values']=['1–8 РАЗНЫХ фотографий одной концепции. Фото 1 подключено; дополнительные LoadImage необязательны: загрузите нужные и соедините с автоматически растущими refs_image. Неподключённые пустые фото не выполняются. Не смешивайте разных людей в identity.\nИмя my_concept → AIO MiniMaxH3RefModsLoader mod_1=farm/my_concept после Refresh. Save заменяет одноимённый файл: задавайте новое имя для версии.\nFull Reference + merge=false сохраняет отдельные refs (рекомендуется для лица). Compressed Reference + merge=false сжимает каждый ref отдельно; merge=true строит общий grid (теряет детали, refinement=0..2000). Это RefMod, не обучение LoRA.\nPreset manual оставляет ручки доступными. identity_encode принудительно меняет их на Full Reference/1024/0 steps/merge off: 8 фото потребуют большего token budget. По умолчанию resolution512, max_tokens16384, budget_policy=error: нет скрытого усечения фото. При memory error уменьшите разрешение или количество фото; не увеличивайте copies.\nСначала Inspect Save, затем загрузите полученный файл в действующий AIO Loader и проверьте Apply→conditioning→генерацию. Сохранение/превью сами по себе не доказывают качество личности.'];note.pop('widgets_values_named',None)
    w['groups']=[dict(id=i+1,title=t,bounding=box,color=c,font_size=24,flags={}) for i,(t,box,c) in enumerate([
        ('1 · Вход',[-40,-40,1600,850],'#3f789e'),('2 · ПУЛЬТ',[400,900,520,760],'#9c7442'),('3 · Движок',[-40,900,420,360],'#4d556b'),('4 · Результат',[960,900,920,950],'#427b61')])]
    return w


def save(w,path):
    errors=wfkit.check_ui(w)
    if errors:raise ValueError(errors[:20])
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(w,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--object-info',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if hashlib.sha256(a.source.read_bytes()).hexdigest()!=SOURCE_SHA256:raise ValueError('Not the supplied AIO')
    oi=json.loads(a.object_info.read_text(encoding='utf-8'));oi.update(pack_schema());oi['MarkdownNote']=dict(input={},output=[],output_name=[])
    w=integrate(json.loads(a.source.read_text(encoding='utf-8-sig')),oi)
    # Named stage files avoid re-exporting/resetting a scene during an edit.
    photo_ui=copy.deepcopy(w)
    for n in photo_ui['nodes']:
        if n['type'] in ('8c87aaa9-03cc-4a78-b986-9dbd7df00baf','eeafa102-d4f0-4f51-a26e-715929de67b7','VHS_VideoCombine'):n['mode']=2
    save(photo_ui,a.out/(VERSION+'.ui.json'))
    prepare=copy.deepcopy(w)
    for n in prepare['nodes']:
        if n['type'] in {s['id'] for s in w.get('definitions',{}).get('subgraphs',[]) if 'Источники позы' not in s['name']} or n['type'] in ('SaveImage','VHS_VideoCombine','3DANativeComposite','3DANativeGuides','3DASceneEdits') or n.get('title') in ('3DA · Итог: повторная проверка сцены','3DA · Проверить новый рендер до H3','3DA · Контакты / кадр / статус','3DA · Метрики финальной вклейки'):n['mode']=2
        if n['type']=='3DASceneExport' or n.get('title')=='3DA · Просмотр выбранного pose-reference':n['mode']=0
    save(prepare,a.out/(VERSION+'.prepare.ui.json'))
    preview_ui=copy.deepcopy(photo_ui)
    for n in preview_ui['nodes']:
        if n['type']=='SaveImage' or n['type']=='43521446-c686-4e05-84c5-16aa0a0e24a9':n['mode']=2
        if n.get('title') in ('3DA · Проверить новый рендер до H3','3DA · Контакты / кадр / статус'):n['mode']=0
    save(preview_ui,a.out/(VERSION+'.preview.ui.json'))
    # Export API templates independently of stage muting. Frontend roundtrip
    # and server prompt validation are recorded separately, never inferred.
    export_w=copy.deepcopy(w)
    for n in export_w['nodes']:
        if n.get('title') in ('3DA · Проверить новый рендер до H3','3DA · Контакты / кадр / статус','3DA · Подготовить сцену гизмо','3DA · Просмотр выбранного pose-reference'):n['mode']=0
    api=clean(wfkit.to_api(export_w),oi)
    stages={'photo':['ИТОГ фото · ref2v'],'video':['ИТОГ видео FullHD · ref2v'],'face':['ИТОГ face pass готового видео'],
            'preview':['3DA · Проверить новый рендер до H3','3DA · Контакты / кадр / статус'],'prepare':['3DA · Подготовить сцену гизмо']}
    for name,titles in stages.items():
        graph=wfkit.prune(api,titles);(a.out/(VERSION+'.'+name+'.api.json')).write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    for select in (1,2,3):
        graph=wfkit.prune(api,['3DA · Просмотр выбранного pose-reference'])
        next(n for n in graph.values() if n['_meta']['title'].startswith('3DA · pose_source'))['inputs']['select']=select
        (a.out/f'3DA_pose_source.{select}.api.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    for count in (1,2,8):
        builder=refmod_builder(oi,count);save(builder,a.out/f'3DA_RefMod_1-8.{count}.ui.json')
        graph=wfkit.prune(clean(wfkit.to_api(builder),oi),['RefMod · Превью','RefMod · Путь результата','RefMod · Метаданные'])
        (a.out/f'3DA_RefMod_1-8.{count}.api.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
        if count==1:save(builder,a.out/'3DA_RefMod_1-8.ui.json')
    (a.out/'README.md').write_text(README+'\n\n'+CHEAT+'\n',encoding='utf-8')
    print('Built AIO UI + prepare UI + RefMod 1/2/8 UI. Runtime exports still required.')


if __name__=='__main__':main()
