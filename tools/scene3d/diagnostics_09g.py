"""09g read-only cached geometry diagnostics; no place/refine or annotation changes."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/scene3d'),str(REPO/'tools/scene3d/posev2')]
import pack_cli

COLORS=['#2a78d6','#eb6834','#1baf7a']

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else float(x)),encoding='utf-8')
def font(s):return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',s)
def pelvis(k):return(k[9]+k[10])/2

def measure(k,ori,up=None):
    up=np.array([0,0,1.]) if up is None else np.asarray(up)
    p=pelvis(k);a=(k[5]+k[6])/2-p;b=k[69]-p
    return dict(pelvis=p,shoulder_axis=a,neck_axis=b,
      shoulder_tilt_deg=float(np.degrees(np.arccos(np.clip(a@up/np.linalg.norm(a),-1,1)))),
      neck_tilt_deg=float(np.degrees(np.arccos(np.clip(b@up/np.linalg.norm(b),-1,1)))),
      roll_deg=float(ori.roll_deg(k,'sit_tucked')))


def setup(old,shot,kind):
    req=load(old/f'{shot}_{kind}_request.json');pre=load(old/f'{shot}_{kind}_preview.json');cfg=req['config']
    pf,profiles=pack_cli.modules(cfg)
    import place,orient,manual_pose
    from shot_spec import ShotSpec
    profile=profiles.LocationProfile(req['location'])
    spec=ShotSpec(**pre['report']['shot_spec'])
    seed=pack_cli.seed_for(profile,spec,cfg)
    plate=pf.Plate(pack_cli.assets_for(cfg,profile.data['loc_id'],profile.dir))
    cache=Path(cfg['farm_root'])/'cache/placement'/pre['report']['fit_hash']
    base=manual_pose.load_placement(cache,pre['report']['fit_hash'])
    refpath=Path(cfg['farm_root'])/'cache/pose_ref'/hashlib.sha256(Path(req['pose_ref']).read_bytes()).hexdigest()/'pose.ref.npz'
    ref=pf.load_ref(refpath)
    ref.update(path=str(refpath),body_cache_dir=str(refpath.parent/'body_geometry'))
    return req,pre,cfg,pf,place,orient,profile,spec,seed,plate,base,ref,cache


def dots(im,u,v,step=12,color='#52514e'):
    d=ImageDraw.Draw(im)
    for x,y in zip(u[::step],v[::step]):
        if np.isfinite(x+y) and 0<=x<im.width and 0<=y<im.height:
            d.ellipse((float(x)-1,float(y)-1,float(x)+1,float(y)+1),fill=color)


def axes(im,k,project):
    pts=np.array([pelvis(k),(k[5]+k[6])/2,k[69]])
    u,v,_=project(pts);d=ImageDraw.Draw(im)
    for i,col in zip((1,2),COLORS):
        d.line((float(u[0]),float(v[0]),float(u[i]),float(v[i])),fill=col,width=4)
        d.ellipse((u[i]-5,v[i]-5,u[i]+5,v[i]+5),fill=col,outline='white',width=1)
    return im


def t01(old,out):
    req,pre,cfg,pf,pl,ori,profile,spec,seed,plate,base,ref,cache=setup(old,'T01','pose')
    dst=out/'T01_stages';dst.mkdir(exist_ok=True)
    shutil=__import__('shutil')
    shutil.copy2(pre['playblast'],dst/'final09f2_playblast.png')
    # This existing helper does not run fitting; scope caches ref plane extraction.
    with pf.ref_plane_scope():B=pl.prepare_body(ref,spec,req['height_m'])
    preB=copy.deepcopy(B);pl.orient_body(B,spec)
    stages=[('a_source',ref['V'],ref['kp']),('b_location_preorient',preB['V'],preB['kp']),
            ('c_root_orient',B['V'],B['kp']),('d_final',base['X'],base['KP'])]
    def project_ref(X):
        return ref['f']*X[:,0]/X[:,2]+ref['cx'],ref['f']*X[:,1]/X[:,2]+ref['cy'],X[:,2]
    orig=Image.open(req['pose_ref']).convert('RGB');images=[];facts={}
    previous_vertices=ref['V']
    for name,V,K in stages:
        matrix,shift=pl.similarity(ref['V'],V)
        incremental_matrix,incremental_shift=pl.similarity(previous_vertices,V)
        previous_vertices=V
        if name=='a_source':
            im=orig.copy();u,v,z=project_ref(V);dots(im,u,v,step=3,color='#52514e');axes(im,K,project_ref)
            # Roll is measured after expressing the keypoints in the inferred
            # source upright basis; raw CV Y-down is not world up.
            met=measure(K@preB['Rref'].T,ori)
            met['angle_basis']='inferred source Z-up Rref; not independent source gravity'
            met['raw_cv_pelvis']=pelvis(K)
            met['raw_cv_shoulder_axis']=(K[5]+K[6])/2-pelvis(K)
            met['raw_cv_neck_axis']=K[69]-pelvis(K)
            met['source_up_cv']=preB['Rref'][2]
        else:
            im=plate.rgb.copy()
            # Pre-placement arrays have no translation into the room yet.
            # For display ONLY align pelvis to final; persisted stage arrays stay raw.
            display_shift=pelvis(base['KP'])-pelvis(K) if name!='d_final' else np.zeros(3)
            Vshow=V+display_shift;Kshow=K+display_shift
            u,v,z=plate.project(Vshow);dots(im,u,v,step=3,color='#52514e');axes(im,Kshow,plate.project)
            met=measure(K,ori)
            met['display_only_translation']=display_shift
        im.save(dst/(name+'.png'));np.savez_compressed(dst/(name+'.npz'),V=V,KP=K)
        facts[name]=dict(measurements=met,matrix_from_source=matrix,shift_from_source=shift,
                        matrix_from_previous=incremental_matrix,shift_from_previous=incremental_shift,
                        affine_fit_max_residual_m=float(np.max(np.linalg.norm(ref['V']@matrix.T+shift-V,axis=1))))
        images.append(im)
    facts['coordinate_conversion']=dict(Rref=preB['Rref'],height_scale=preB['s'],up_source=preB['up_src'],n_planes=preB['n_planes'])
    facts['orient_info']=B['orient']
    sheet=Image.new('RGB',(1600,840),'#fcfcfb');d=ImageDraw.Draw(sheet)
    labels=['а SAM/MHR + исходный реф','б Z-up до orient (реконструкция)','в root/orient (реконструкция)','г финал09f2 (точный кэш)']
    for i,(im,label,(name,_,_)) in enumerate(zip(images,labels,stages)):
        x=(i%2)*800;y=(i//2)*400
        thumb=im.copy();thumb.thumbnail((790,300));sheet.paste(thumb,(x+(800-thumb.width)//2,y+35))
        d.text((x+8,y+3),label,fill='#0b0b0b',font=font(19))
        m=facts[name]['measurements'];d.text((x+8,y+342),f"tilt shoulders {m['shoulder_tilt_deg']:.2f}°; neck {m['neck_tilt_deg']:.2f}°; roll {m['roll_deg']:.2f}°",fill='#0b0b0b',font=font(17))
    d.text((10,802),'Оси: синий таз→середина плеч; оранжевый таз→шея. б/в: только display shift к финальному тазу.',fill='#0b0b0b',font=font(19))
    sheet.save(out/'t01_stages_g.jpg',quality=94)
    facts['limitation']='b/c reconstructed by unchanged prepare_body/orient_body, not captured during09f2; b is inferred Z-up basis, not an already placed mesh. Display translation is not placement. Source absolute world-up independent of inferred planes is unavailable.'
    facts['conclusion']='Source body already has the same inferred tilt as stage b (neck52.7deg). No new backwards tilt is introduced by b coordinate conversion. Stage c clamps roll, not the sit_tucked neck tilt52.7deg; final51.1deg. Cannot distinguish SAM reconstruction tilt from wrong inferred source up without independently measured source gravity/camera orientation.'
    save(out/'t01_diagnostics.json',facts)
    return facts


def q04(old,out):
    req,pre,cfg,pf,pl,ori,profile,spec,seed,plate,base,ref,cache=setup(old,'Q04','prop')
    import top_region as tr
    treg,freg,_,oi=pl.resolve_regions(plate,spec,seed,pre['report']['body_dims']['H'],profile)
    treg=tr.pose_region(treg,spec.pose_class);treg.object_id=oi.get('object_id',spec.target_object)
    S=pl.build_surfaces(plate,treg,freg)
    groups=list(base['groups']);present={g[0] for g in groups if g[3]=='support'}
    for name in spec.support_regions():
        idx=np.flatnonzero(base['reg']==name)
        if name not in present and len(idx):groups.append((name,pl._lowest(base['X'],idx),'target','support'))
    names=['pelvis','thigh.L','thigh.R'];facts={};im=plate.rgb.copy();draw=ImageDraw.Draw(im)
    X=base['X'];u,v,z=plate.project(X)
    for n,col in zip(names,COLORS):
        allidx=np.flatnonzero(base['reg']==n)
        dots(im,u[allidx],v[allidx],step=1,color=col)
        g=next(g for g in groups if g[0]==n and g[3]=='support');idx=g[1]
        points=X[idx];tree,P,N=S[g[2]];_,j=tree.query(points);near=P[j]
        gap=pl.support_distances(S,g[2],points)
        order=np.argsort(gap);medidx=order[(len(order)-1)//2:(len(order)//2)+1]
        pu,pv,_=plate.project(points);qu,qv,_=plate.project(near)
        for a,b,c,d in zip(pu[::max(1,len(idx)//50)],pv[::max(1,len(idx)//50)],qu[::max(1,len(idx)//50)],qv[::max(1,len(idx)//50)]):
            draw.line((float(a),float(b),float(c),float(d)),fill=col,width=1)
        for k in medidx:
            draw.line((float(pu[k]),float(pv[k]),float(qu[k]),float(qv[k])),fill=col,width=4)
            draw.ellipse((pu[k]-6,pv[k]-6,pu[k]+6,pv[k]+6),fill=col,outline='white',width=2)
        reported=pre['report']['support_contact']['part_gaps_m'][n]
        measured=float(np.median(gap));assert abs(measured-reported)<1e-7,(n,measured,reported)
        facts[n]=dict(anatomical_side=n,source='cached support group' if n in present else 'evaluate._lowest(final mesh, region) fallback',
                      vertex_ids=idx,points_world=points,nearest_surface_points_world=near,signed_gaps_m=gap,
                      median_vertex_ids=idx[medidx],median_gap_m=measured,reported_gap_m=reported,
                      points_image_px=np.column_stack([pu,pv]),surface_image_px=np.column_stack([qu,qv]))
    __import__('shutil').copy2(pre['playblast'],out/'Q04_contacts_playblast.png')
    im.save(out/'Q04_contact_points.png')
    refim=Image.open(req['pose_ref']).convert('RGB')
    # Anatomical thigh axes on source: MHR hipL9/kneeL11, hipR10/kneeR12.
    rd=ImageDraw.Draw(refim)
    for name,col,pair in [('thigh.L',COLORS[1],(9,11)),('thigh.R',COLORS[2],(10,12))]:
        k=ref['kp'][list(pair)];a=ref['f']*k[:,0]/k[:,2]+ref['cx'];b=ref['f']*k[:,1]/k[:,2]+ref['cy']
        rd.line((float(a[0]),float(b[0]),float(a[1]),float(b[1])),fill=col,width=5)
        rd.text((float(a.mean()),float(b.mean())),name,fill='#0b0b0b',stroke_width=2,stroke_fill='white',font=font(18))
    sheet=Image.new('RGB',(1800,710),'#fcfcfb');d=ImageDraw.Draw(sheet)
    for i,(pic,label) in enumerate([(refim,'Референс: анатомические L/R'),(im,'Финал09f2: регионы и точки signed gap')]):
        pic.thumbnail((890,520));sheet.paste(pic,(i*900+(900-pic.width)//2,40));d.text((i*900+10,8),label,fill='#0b0b0b',font=font(22))
    for i,n in enumerate(names):
        d.rectangle((10,570+i*35,28,588+i*35),fill=COLORS[i]);d.text((40,566+i*35),f'{n}: gap median {facts[n]["median_gap_m"]:.6f} м; {len(facts[n]["vertex_ids"])} точек; жирная линия = медиана',fill='#0b0b0b',font=font(20))
    d.text((10,677),'thigh.L поднято, thigh.R опорное. Обе стороны объявлены support_parts; геометрия/пороги не изменялись.',fill='#0b0b0b',font=font(21))
    sheet.save(out/'q04_contacts_g.jpg',quality=94)
    facts['support_parts']=spec.support_parts;facts['support_surface']=pre['report']['surface_match']['support_surface_id']
    facts['conclusion']='Anatomical thigh.L raised (gap .225540m), thigh.R supporting (.031976m); both are declared support_parts. Left was absent from original low-band group and gets evaluate._lowest fallback on final mesh; not an instruction to lower body or adjust thresholds.'
    # Native 4x contact crop, supplementary to the full-room overview.
    all_points=np.concatenate([np.asarray(facts[n]['points_image_px']) for n in names])
    lo=np.floor(all_points.min(0)-25).astype(int);hi=np.ceil(all_points.max(0)+25).astype(int)
    box=(max(0,int(lo[0])),max(0,int(lo[1])),min(im.width,int(hi[0])),min(im.height,int(hi[1])))
    contact_crop=im.crop(box)
    contact_crop.resize((contact_crop.width*4,contact_crop.height*4),Image.Resampling.NEAREST).save(out/'q04_contacts_crop4x.png')
    facts['contact_crop_bbox_px']=box
    save(out/'q04_diagnostics.json',facts)
    return facts


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=True);t=time.monotonic();results={}
    for shot,fn in [('T01',t01),('Q04',q04)]:
        if time.monotonic()-t>2700:results[shot]={'status':'skipped','reason':'45min timebox'};continue
        try:results[shot]={'status':'measured','conclusion':fn(a.old,a.out)['conclusion']}
        except Exception as e:results[shot]={'status':'error','reason':str(e)};__import__('traceback').print_exc()
    save(a.out/'diagnostics.json',results);print(json.dumps(results,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
