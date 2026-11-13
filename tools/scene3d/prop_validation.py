"""Final scene validity includes props, independently of body placement.

13.11 «КАДР ЛЮБОЙ»: предмет или тело частично (или предмет целиком) вне кадра — предупреждение,
не отказ. Блокируют H3 только геометрия (пересечения, проникновение, нет опоры) и тело без единой
видимой точки (статус сцены out_of_frame).
"""
import copy

FRAME_FLAGS = ('prop_out_of_frame', 'out_of_frame')
BLOCKING_PROP_FLAGS = ('prop_hits_body', 'prop_hits_room', 'prop_penetration', 'prop_floating')


def blocking_reasons(report):
    reasons=[]
    if report.get('status') not in ('ok','partial'):
        reasons.append('scene status='+str(report.get('status')))
    for index,prop in enumerate(report.get('props',[])):
        flags=prop.get('flags',{})
        for name in BLOCKING_PROP_FLAGS:
            if flags.get(name):
                reasons.append(f'prop[{index}] {name}')
    return reasons


def frame_warnings(report):
    out=[]
    body_flags=report.get('body_flags',report.get('flags',[]))
    if 'body_out_of_frame' in body_flags or 'out_of_frame' in body_flags:
        out.append('body_partly_out_of_frame')
    for index,prop in enumerate(report.get('props',[])):
        if any(prop.get('flags',{}).get(name) for name in FRAME_FLAGS):
            out.append(f'prop[{index}] out_of_frame')
    return out


def include_props(report):
    """Keep body evidence; promote a failed prop into the final scene status."""
    props=report.get('props',[])
    checks=[]
    for index,prop in enumerate(props):
        flags=prop.get('flags',{})
        for name in BLOCKING_PROP_FLAGS:
            if name in flags:
                checks.append(dict(check=f'prop[{index}]:{name}',ok=not flags.get(name,False),
                                   unit='intersection',excess=float(bool(flags.get(name)))))
    if not props:
        return report
    report.setdefault('body_flags',list(report.get('flags',[])))
    report['flags']=sorted(set(report.get('flags',[])+[name for prop in props for name,active in prop.get('flags',{}).items() if active]))
    report['body_status']=report['status']
    body=copy.deepcopy(report.get('validation',{}))
    report['body_validation']=body
    failure=next((c for c in checks if not c['ok']),None)
    if failure and report['status'] in ('ok','partial'):
        report['status']='contact_fail'
    report['validation']={**body,'body_ok':body.get('ok',False),'status':report['status'],'flags':report['flags'],
        'ok':body.get('ok',False) and failure is None,'checks':body.get('checks',[])+checks,
        'first_failure':body.get('first_failure') or failure,'warnings':frame_warnings(report)}
    report['h3_eligibility']=dict(ok=not blocking_reasons(report),reasons=blocking_reasons(report),
                                  warnings=frame_warnings(report))
    return report


def keep_props_in_frame(props, kp, project, size, margin=.06):
    """0.3.4: предмет по смещению от якоря (без матрицы 3D-окна), попавший вне кадра, придвигается к якорю
    по той же линии, пока не войдёт в кадр с полем margin. Якорь сам вне кадра — предмет не трогаем.
    project(points Nx3) -> (u, v, z) в пикселях; size = (W, H). Возвращает (props, заметки по-русски)."""
    import numpy as np
    from prop_render import anchors_from_mhr70
    anchors = anchors_from_mhr70(kp)
    width, height = size

    def inside(point):
        u, v, z = project(np.asarray([point], dtype=float))
        return bool(z[0] > 0 and margin <= u[0] / width <= 1 - margin and margin <= v[0] / height <= 1 - margin)

    out, notes = [], []
    for index, prop in enumerate(props):
        prop = dict(prop)
        if 'absolute_matrix_world' not in prop:
            anchor = prop.get('anchor', 'world')
            base = np.asarray(anchors['char_pelvis' if anchor == 'world' else anchor], dtype=float)
            offset = np.asarray([float(prop.get(axis, 0.)) for axis in 'xyz'], dtype=float)
            if offset.any() and inside(base) and not inside(base + offset):
                low, high = 0., 1.
                for _ in range(24):
                    mid = (low + high) / 2
                    low, high = (mid, high) if inside(base + offset * mid) else (low, mid)
                prop.update(zip('xyz', (offset * low).tolist()))
                notes.append(f'предмет {index + 1} был вне кадра — придвинут к персонажу')
        out.append(prop)
    return out, notes
