"""Final scene validity includes props, independently of body placement."""
import copy


def blocking_reasons(report):
    reasons=[]
    if report.get('status') not in ('ok','partial'):
        reasons.append('scene status='+str(report.get('status')))
    for index,prop in enumerate(report.get('props',[])):
        flags=prop.get('flags',{})
        for name in ('prop_out_of_frame','out_of_frame','prop_hits_body','prop_hits_room'):
            if flags.get(name):
                reasons.append(f'prop[{index}] {name}')
    return reasons


def include_props(report):
    """Keep body evidence; promote a failed prop into the final scene status."""
    props=report.get('props',[])
    checks=[]
    for index,prop in enumerate(props):
        flags=prop.get('flags',{})
        checks.append(dict(check=f'prop[{index}]:frame',ok=not(flags.get('out_of_frame') or flags.get('prop_out_of_frame')),
            unit='frame',excess=float(bool(flags.get('out_of_frame') or flags.get('prop_out_of_frame')))))
        for name in ('prop_hits_body','prop_hits_room'):
            checks.append(dict(check=f'prop[{index}]:{name}',ok=not flags.get(name,False),unit='intersection',excess=float(bool(flags.get(name)))))
    if not checks:
        return report
    report['flags']=sorted(set(report.get('flags',[])+[name for prop in props for name,active in prop.get('flags',{}).items() if active]))
    report['body_status']=report['status']
    body=copy.deepcopy(report.get('validation',{}))
    report['body_validation']=body
    failure=next((c for c in checks if not c['ok']),None)
    if failure and report['status'] in ('ok','partial'):
        report['status']='out_of_frame' if failure['unit']=='frame' else 'contact_fail'
    report['validation']={**body,'body_ok':body.get('ok',False),'status':report['status'],'flags':report['flags'],
        'ok':body.get('ok',False) and failure is None,'checks':body.get('checks',[])+checks,
        'first_failure':body.get('first_failure') or failure}
    report['h3_eligibility']=dict(ok=not blocking_reasons(report),reasons=blocking_reasons(report))
    return report
