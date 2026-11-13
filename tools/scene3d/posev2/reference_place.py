"""Automatic adapter to the existing rigid reference fitter; no pose-class clamp.

Support is observed geometry, never an assumed floor or a per-room hint.
An unsuccessful fit still returns the reconstructed mesh for the editor.
"""
import copy
import numpy as np
from scipy.spatial import cKDTree

from placement_config import section
import pose_fit as pf
import place as legacy
import profiles
from body import height_est, regions


def candidates(profile, plate):
    found = []
    for obj in profile.data['objects']:
        if 'top_grid_npz' not in obj:
            continue
        reg = profile.region(obj)
        if not reg.mask.any():
            continue
        target = np.asarray(obj.get('seed_world', reg.points3d().mean(0)), float)
        u, v, z = plate.project(target[None])
        if z[0] <= 0:
            continue
        # Choose visible observed support patches; no furniture/floor labels.
        score = (float(u[0] / plate.Wp - .5) ** 2 + float(v[0] / plate.Hp - .7) ** 2)
        found.append((score, obj, reg, target))
    return sorted(found, key=lambda item: item[0])


def evaluate(X, KP, groups, plate, profile, spec, report):
    cfg = section('place')
    support = [g[1] for g in groups if g[3] == 'support']
    indices = np.unique(np.concatenate(support)) if support else np.array([], int)
    options = candidates(profile, plate)
    nearest = []
    for _, obj, reg, _ in options:
        distances = cKDTree(reg.points3d()).query(X[indices])[0] if len(indices) else np.array([np.inf])
        observed = reg.in_mask(X[indices]) if len(indices) else np.array([], bool)
        if observed.any():
            i, j = reg.cell_of(X[indices][observed])
            # Signed normal gap to the sampled surface, not distance to the
            # centre of a 2cm cell (which can be 1.4cm on a perfect contact).
            gap = float(np.abs(X[indices][observed, 2]-reg.height[i, j]).min())
        else:
            gap = float(distances.min())
        nearest.append((gap, obj, reg, distances))
    gap, obj, reg, distances = min(nearest, key=lambda t: t[0]) if nearest else (float('inf'), None, None, np.array([]))
    if reg is not None:
        reg.diagnostic_surfaces = [dict(id=o['id'], region=r, soft=bool(o.get('soft', False))) for _, o, r, _ in options]
        reg.wall_planes = profile.walls()
    pen = legacy.penetration_stats(plate, X, reg, bool(obj and obj.get('soft')), 'auto')
    frame = legacy.framing(X, plate)
    # The same observed footprint gate used by manual placement. Never fill a
    # missing floor or accept a contact merely because the mesh is visible.
    from support_geometry import hull_metrics
    footprint = hull_metrics(X[indices], reg) if reg is not None and len(indices) else {}
    fraction = footprint.get('in_mask_fraction') or 0.
    checks = [dict(check='observed_support', ok=obj is not None, excess=0 if obj else 1, unit='surface'),
              dict(check='support_contact', ok=gap <= cfg['contact_tol'], excess=max(0., gap-cfg['contact_tol']), unit='m'),
              dict(check='footprint', ok=fraction >= cfg['hull_in_mask_min'], excess=max(0., cfg['hull_in_mask_min']-fraction), unit='area_fraction'),
              dict(check='frame', ok=frame.get('visible', frame['ok']), excess=int(not frame.get('visible', frame['ok'])), unit='frame'),
              dict(check='reference_pose', ok=report.get('reference_pose_error_m', 0.) <= section('reference_registration')['pose_error_m'],
                   excess=report.get('reference_pose_error_m', 0.), unit='m')]
    checks.extend(dict(check='penetration_surface:'+name, ok=value['ok'], excess=max(0., value['p95_m']-value['threshold_m']), unit='m')
                  for name, value in pen['surfaces'].items())
    failed = [c for c in checks if not c['ok']]
    status = 'ok' if not failed else 'out_of_frame' if not frame.get('visible', frame['ok']) else 'contact_fail'
    report.update(status=status, validation=dict(ok=not failed, status=status, first_failure=failed[0] if failed else None, checks=checks),
                  shot_spec=spec.to_dict(), body_dims=profiles.body_dims_from_kp(KP), frame=frame,
                  penetration=pen, footprint=dict(ok=fraction >= cfg['hull_in_mask_min'], support_hull=footprint),
                  target_info=dict(id=obj['id'], name='observed support surface') if obj else {},
                  support_object=obj['id'] if obj else 'unresolved', support_gap_cm=gap*100,
                  flags=[c['check'] for c in failed]+([] if frame['ok'] else ['out_of_frame']), max_penetration_cm=pen['max_cm'],
                  support_parts_on=[g[0] for g in groups if g[3] == 'support'],
                  zone_diagnostics=dict(target_object='auto', observed_support=obj['id'] if obj else None),
                  placement_mode='reference_rigid', pose_class_inferred=False, no_floor_required=True)
    return report


def keypoints25(KP, plate):
    ku, kv, kz = plate.project(KP[pf.MHR_TO_B25])
    kzp = plate.depth_at(ku, kv)
    kp25 = [[round(float(a), 1), round(float(b), 1),
             int(0 <= a < plate.Wp and 0 <= b < plate.Hp and z > 0 and z <= zp + 0.02)]
            for a, b, z, zp in zip(ku, kv, kz, kzp)] + [[0.0, 0.0, 0]] * 30
    return dict(keypoints_body25_px=kp25, px_size=[plate.Wp, plate.Hp])


def diagnostic(ref, plate, height):
    scale = height / height_est(ref['kp'])
    X = ref['V'] @ plate.R.T * scale
    KP = ref['kp'] @ plate.R.T * scale
    up = np.array([0., 0., 1.])
    indices = np.argsort(X @ up)[:max(20, len(X)//50)]
    anchor = plate.pixel_point(.5, .7)
    shift = anchor - X[indices].mean(0)
    return X+shift, KP+shift, indices


def place(ref, plate, profile, spec, height):
    cfg = section('reference_registration')
    attempts = []
    for _, obj, reg, target in candidates(profile, plate)[:cfg['candidate_count']]:
        u, v, _ = plate.project(target[None])
        X, raw, labels = pf.fit(ref, plate, (float(u[0]/plate.Wp), float(v[0]/plate.Hp)), height,
                               target_world=target, max_nfev=cfg['max_nfev'], check_source_vertical=True)
        st = raw.pop('_state')
        KP = ref['kp'] @ st['M'].T + st['b']
        groups = [(name, values[0], values[1], 'support' if name in st['support'] else 'secondary')
                  for name, values in st['contacts'].items()]
        if not any(g[3] == 'support' for g in groups):
            groups.append(('lowest', st['sup_idx'], 'h', 'support'))
        error = float(np.linalg.norm(X-(ref['V'] @ st['M'].T+st['b']), axis=1).max())
        report = evaluate(X, KP, groups, plate, profile, spec, {**raw, 'reference_pose_error_m': error})
        state = dict(M=st['M'], b=st['b'], target=target, groups=groups)
        result = dict(status=report['status'], rep=report, X=X, KP=KP, reg_lbl=labels, state=state, final=dict(stage='fit'))
        attempts.append(result)
        print('[reference placement]', obj['id'], report['status'], 'gap_cm', report['support_gap_cm'], flush=True)
        if report['validation']['ok']:
            break
    if attempts:
        return min(attempts, key=lambda r: (len([c for c in r['rep']['validation']['checks'] if not c['ok']]),
                    r['rep']['penetration']['excess_m'], r['rep']['support_gap_cm']))
    X, KP, indices = diagnostic(ref, plate, height)
    groups = [('lowest', indices, 'any', 'support')]
    # 0.3.16: те же точки BODY-25 в пикселях фото, что пишет pose_fit, — их ждёт рендер сцены (pose25.json).
    report = evaluate(X, KP, groups, plate, profile, spec, dict(contacts={}, reference_pose_error_m=0.,
                                                                **keypoints25(KP, plate)))
    M, b = legacy.similarity(ref['V'], X)
    return dict(status=report['status'], rep=report, X=X, KP=KP, reg_lbl=regions(ref['V'], ref['kp']),
                state=dict(M=M, b=b, target=X[indices].mean(0), groups=groups), final=dict(stage='fit'))


def revalidate(base, plate, profile, spec, values):
    from manual_pose import transform
    X, KP, actual = transform(base['X'], base['KP'], plate, values)
    report = copy.deepcopy(base['report'])
    report.update(body_transform=actual, transform_applied=True)
    return X, KP, evaluate(X, KP, base['groups'], plate, profile, spec, report)
