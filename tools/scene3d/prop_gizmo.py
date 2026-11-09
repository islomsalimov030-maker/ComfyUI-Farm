"""Core Load3DAdvanced adapter. Matrices act on the unmodified library GLB.

Three uses Y-up; Blender/plate use Z-up. No anchor or size normalization is
applied in this absolute mode, including for off-centre and nonuniform assets.
"""
import math
import numpy as np

B = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]])


def model_matrix(models):
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError('Гизмо: требуется ровно один отдельный GLB предмета')
    item = models[0]
    def vector(key, axes):
        value = item.get(key, {})
        try:
            a = np.array([value[k] for k in axes], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise ValueError('Гизмо: отсутствует ' + key) from None
        if not np.isfinite(a).all():
            raise ValueError('Гизмо: нечисловой ' + key)
        return a
    p, q, s = vector('position', 'xyz'), vector('quaternion', 'xyzw'), vector('scale', 'xyz')
    if np.any(s <= 0) or abs(np.linalg.norm(q) - 1) > 1e-4:
        raise ValueError('Гизмо: положительный scale и единичный quaternion обязательны')
    q /= np.linalg.norm(q)
    x, y, z, w = q
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    m = np.eye(4)
    m[:3, :3] = B.T @ r @ np.diag(s) @ B
    m[:3, 3] = B.T @ p
    return m.tolist()


def validate_matrix(value):
    m = np.asarray(value, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1]):
        raise ValueError('absolute_matrix_world must be finite affine 4x4')
    if np.linalg.det(m[:3, :3]) <= 0:
        raise ValueError('absolute_matrix_world must preserve handedness and positive scale')
    return m


def camera_config(plate, raster):
    """Match the renderer's horizontal sensor fit, including anisotropic resize."""
    w, h = map(int, raster)
    if w <= 0 or h <= 0:
        raise ValueError('Invalid raster')
    # The renderer uses lens/sensor_width and does not implement lens shift.
    intr = plate.get('intrinsics_norm', {})
    if isinstance(intr, list):
        intr = dict(cx=intr[0][2], cy=intr[1][2])
    for key in ('shift_x', 'shift_y', 'lens_shift_x', 'lens_shift_y'):
        if abs(float(plate.get(key, 0))) > 1e-8:
            raise ValueError('Гизмо: lens shift пока не поддерживается')
    for key in ('cx', 'cy'):
        if key in intr and abs(float(intr[key]) - .5) > 1e-5:
            raise ValueError('Гизмо: off-centre principal point / lens shift не поддерживается')
    r = np.asarray(plate['R_world_from_cv'], dtype=float)
    if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), atol=1e-5):
        raise ValueError('Invalid plate camera rotation')
    pos = B @ np.array([0., 0., float(plate['cam_height_m'])])
    vec = lambda v: dict(zip('xyz', v.tolist()))
    fov_y = math.degrees(2 * math.atan(float(plate['sensor_width_mm']) / float(plate['lens_mm']) * h / w / 2))
    return dict(position=vec(pos), target=vec(pos+B@r[:, 2]), zoom=1,
                cameraType='perspective', fov=fov_y, useCustomUp=True, customUp=vec(-B@r[:, 1]))


def validate_camera(actual, expected, limits):
    """A 2D background cannot reveal that core Fit/orbit changed the camera."""
    def vec(value):
        try:a=np.asarray([value[k] for k in 'xyz'],dtype=float)
        except (TypeError,KeyError,ValueError):raise ValueError('Гизмо: неполные данные камеры') from None
        if not np.isfinite(a).all():raise ValueError('Гизмо: нечисловая камера')
        return a
    def angle(a,b):
        length=np.linalg.norm(a)*np.linalg.norm(b)
        if length<1e-12:raise ValueError('Гизмо: нулевое направление камеры')
        return math.degrees(math.acos(float(np.clip(np.dot(a,b)/length,-1,1))))
    if not isinstance(actual,dict):raise ValueError('Гизмо: подключите camera_info')
    p,q=vec(actual.get('position')),vec(expected['position'])
    direction=vec(actual.get('target'))-p;desired=vec(expected['target'])-q
    up=vec(actual.get('customUp')) if actual.get('useCustomUp') else np.array([0.,1.,0.])
    values=np.asarray([actual.get('fov',float('nan')),actual.get('zoom',1)],dtype=float)
    mismatch=(not np.isfinite(values).all() or actual.get('cameraType')!='perspective' or
        np.linalg.norm(p-q)>limits['position_m'] or angle(direction,desired)>limits['direction_deg'] or
        angle(up,vec(expected['customUp']))>limits['direction_deg'] or
        abs(values[0]-expected['fov'])>limits['fov_deg'] or abs(values[1]-1)>limits['zoom'])
    if mismatch:
        raise ValueError('Камера гизмо отличается от камеры комнаты. Подогнать к просмотру/Fit меняет также масштаб предмета. '
                         'Откройте исходный safe-пресет для восстановления камеры и размера; двигайте предмет стрелками гизмо.')
