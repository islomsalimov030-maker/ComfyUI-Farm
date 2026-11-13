"""Растр рендера до H3 и H3 (0.2.0 «б»): мегапиксели из пульта, кадр как у плиты.

Геометрия плиты считается в 1344×768 (нормированные интринсики не зависят от разрешения),
поэтому рендер в любом растре того же соотношения сторон даёт тот же кадр.
0 / пусто — прежние 1344×768.
"""
import math

BASE = (1344, 768)
MP_MIN, MP_MAX = 0.5, 4.2


def raster_for(megapixels=0.0, step=16):
    try:
        mp = float(megapixels or 0)
    except (TypeError, ValueError):
        raise ValueError('3DA: megapixels должно быть числом') from None
    if not math.isfinite(mp) or mp <= 0:
        return BASE
    mp = min(max(mp, MP_MIN), MP_MAX)
    aspect = BASE[0] / BASE[1]
    width = max(step, round(math.sqrt(mp * 1e6 * aspect) / step) * step)
    height = max(step, round(width / aspect / step) * step)
    return int(width), int(height)


def parse(text):
    if not text:
        return None
    width, height = (int(v) for v in str(text).lower().split('x'))
    if width < 64 or height < 64 or width > 8192 or height > 8192:
        raise ValueError('raster out of range: ' + str(text))
    return width, height
