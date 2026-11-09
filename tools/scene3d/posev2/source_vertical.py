"""Independent image-line check on depth-derived source gravity (CPU only).

Architectural verticals validate a MoGe plane normal. With insufficient lines
the established estimate is retained. For eligible poses a contradiction falls
back to the explicit camera prior. Reclines preserve unresolved depth pitch;
image-line directions alone do not prove the complete 3D gravity vector.
"""
import numpy as np
from placement_config import section


def detect(path):
    import cv2
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError('Cannot read source image for vertical validation')
    found = cv2.createLineSegmentDetector().detect(gray)[0]
    return np.empty((0, 4)) if found is None else found.reshape(-1, 4).astype(float)


def check(ref, up, allow_camera_prior=True):
    cfg = section('source_vertical')
    path = ref.get('image_path')
    if not path:
        return up, None
    try:
        lines = detect(path)
    except ImportError:
        if not ref.get('vertical_python'):
            return up, {'status': 'unavailable', 'reason': 'opencv unavailable'}
        # Existing farm env has OpenCV; geometry env stays unchanged.
        import subprocess, json
        from pathlib import Path
        result = subprocess.run([ref['vertical_python'], str(Path(__file__).resolve()), str(path)],
            capture_output=True, text=True, timeout=30, check=True)
        lines = np.array(json.loads(result.stdout), dtype=float).reshape(-1, 4)
    if len(lines) == 0:
        return up, {'status': 'insufficient_lines'}
    delta = lines[:, 2:] - lines[:, :2]
    length = np.linalg.norm(delta, axis=1)
    # Exclude the projected person bbox. Clothes/limbs are not vertical evidence.
    kp = ref['kp2d']
    lo, hi = np.min(kp, axis=0), np.max(kp, axis=0)
    mid = (lines[:, :2]+lines[:, 2:])/2
    outside = (mid[:, 0] < lo[0]) | (mid[:, 0] > hi[0]) | (mid[:, 1] < lo[1]) | (mid[:, 1] > hi[1])
    keep = outside & (length >= cfg['min_line_height']*ref['H']) & (np.abs(delta[:, 0]) <= np.abs(delta[:, 1])*cfg['max_vertical_slope'])
    lines, delta, length, mid = lines[keep], delta[keep], length[keep], mid[keep]
    if len(lines) < cfg['min_lines'] or np.ptp(mid[:, 0]) < cfg['min_span_width']*ref['W']:
        return up, {'status': 'insufficient_lines', 'count': len(lines)}
    def errors(u):
        direction = ref['f']*u[:2] - (mid-[ref['cx'], ref['cy']])*u[2]
        cos = np.abs(np.sum(direction*delta, axis=1)) / np.maximum(np.linalg.norm(direction,axis=1)*length, 1e-12)
        return np.degrees(np.arccos(np.clip(cos, 0, 1)))
    def weighted_median(a):
        order = np.argsort(a)
        return float(a[order[np.searchsorted(np.cumsum(length[order]), length.sum()/2)]])
    camera = np.array([0., -1., 0.])
    plane_error, camera_error = weighted_median(errors(up)), weighted_median(errors(camera))
    rejected = plane_error > cfg['reject_angle_deg'] and plane_error-camera_error > cfg['improvement_deg']
    if rejected and not allow_camera_prior:
        # Image directions reject a normal but do not uniquely determine its
        # depth component. A level-camera prior destroyed supported reclines
        # in the full regression render; preserve that unresolved ambiguity.
        return up,dict(status='contradiction_prior_not_applicable',lines=lines.tolist(),
            plane_error_deg=plane_error,camera_error_deg=camera_error,
            source_up_before=np.asarray(up).tolist(),source_up_after=np.asarray(up).tolist(),
            reason='Recline depth pitch is not established by image-line directions; preserve source estimate')
    return (camera if rejected else up), dict(status='rejected_depth_vertical' if rejected else 'consistent',
        lines=lines.tolist(), plane_error_deg=plane_error, camera_error_deg=camera_error,
        source_up_before=np.asarray(up).tolist(), source_up_after=(camera if rejected else up).tolist())


if __name__ == '__main__':
    import sys, json
    print(json.dumps(detect(sys.argv[1]).tolist()))
