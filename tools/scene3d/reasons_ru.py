"""Понятные русские причины отказа геометрии (0.2.0 «д»).

Из отчёта постановки (fit.json) — только то, что блокирует H3: статус сцены не ok/partial и
блокирующие флаги предметов (prop_validation.BLOCKING_PROP_FLAGS). Кадр — предупреждение
(13.11 «КАДР ЛЮБОЙ»), здесь отказом не считается, кроме «в кадре нет ни одной части тела».
"""
import math

STATUS_RU = {
    'contact_fail': 'тело не касается опоры',
    'wrong_surface': 'тело стоит не на выбранной поверхности',
    'out_of_frame': 'в кадре нет ни одной части тела',
    'no_target': 'выбранная поверхность не найдена на фото',
    'error': 'ошибка подготовки сцены',
}

CHECK_RU = {
    'tilt': 'наклон корпуса вне допустимого для позы',
    'roll': 'крен тела вне допустимого',
    'camera_side': 'тело повёрнуто к камере не той стороной',
    'yaw_window': 'поворот тела вне допустимого для позы',
    'support_target': 'опора — не выбранная поверхность',
    'observed_support': 'опора под телом не видна на фото',
    'back_contact': 'спина не касается спинки',
    'height_contact': 'рост не сходится с опорой (ноги или голова уходят в поверхность)',
    'cells': 'тело не помещается на поверхности',
    'refine_limits': 'подгонка позы упёрлась в пределы',
    'frame': 'в кадре нет ни одной части тела',
}


def cm(value_m):
    # 0.3.15: «не найдено» (inf или 1e9 из JSON) — «?», а не OverflowError / 100000000000 см.
    value = float(value_m)
    if not math.isfinite(value) or abs(value) >= 1e3:
        return '?'
    return max(0, round(value * 100))


def surface_ru(name):
    """'hard:floor' → 'пола', 'soft:bed_0' → 'мягкой поверхности bed_0' (родительный падеж)."""
    kind, _, ident = str(name).partition(':')
    if not ident:
        kind, ident = '', kind
    low = ident.lower()
    if low in ('floor', 'pol') or low.startswith('floor'):
        return 'пола'
    if low.startswith('wall'):
        return 'стены'
    if low in ('pillows', 'backrest'):
        return 'подушек/спинки'
    if low in ('any', 'room', 'scene'):
        return 'стены/мебели'
    return ('мягкой поверхности ' if kind == 'soft' else 'поверхности ') + ident


def _body_reasons(fit):
    status = fit.get('body_status', fit.get('status'))
    if status in ('ok', 'partial'):
        return []
    validation = fit.get('body_validation') or fit.get('validation') or {}
    surfaces = (fit.get('penetration') or {}).get('surfaces', {})
    out = []
    for check in validation.get('checks', []):
        if check.get('ok', True) or str(check.get('check', '')).startswith('prop['):
            continue
        name = str(check['check'])
        excess = float(check.get('excess') or 0.)
        if name.startswith('penetration_surface:'):
            surface = name.split(':', 1)[1]
            info = surfaces.get(surface, {})
            depth = info.get('p95_m', excess + float(info.get('threshold_m', 0.)))
            text = f'Персонаж: внутри {surface_ru(surface)} на {cm(depth)} см'
            if 'threshold_m' in info:
                text += f' (допуск {cm(info["threshold_m"])} см)'
            out.append(text)
        elif name.startswith('penetration:'):
            out.append(f'Персонаж: проникновение в {"мягкую" if name.endswith("soft") else "жёсткую"} '
                       f'поверхность больше допуска на {cm(excess)} см')
        elif name == 'support_contact':
            gap = fit.get('support_gap_cm')
            if gap is not None and float(gap) < 0:
                out.append(f'Персонаж: вдавлен в опору на {round(-float(gap))} см')
            else:
                hang = float(gap) / 100 if gap is not None else excess
                out.append(f'Персонаж: нет опоры — висит на {cm(hang)} см')
        elif name.startswith('support_gap:'):
            out.append(f'Персонаж: {name.split(":", 1)[1]} не касается опоры — зазор больше допуска на {cm(excess)} см')
        elif name == 'footprint':
            out.append(f'Персонаж: опора выходит за край поверхности (не хватает {round(excess * 100)} % площади)')
        elif name == 'reference_pose':
            out.append(f'Персонаж: поза отличается от референса на {cm(excess)} см')
        else:
            out.append('Персонаж: ' + CHECK_RU.get(name, name))
    if not out:
        out.append('Персонаж: ' + STATUS_RU.get(str(status), f'статус {status}'))
    return out


def _prop_reasons(fit):
    out = []
    for index, prop in enumerate(fit.get('props', [])):
        flags = prop.get('flags', {})
        label = f'Предмет {index + 1}'
        if prop.get('prop_id'):
            label += f' ({prop["prop_id"]})'
        checks = prop.get('checks', {})
        if flags.get('prop_hits_body'):
            out.append(f'{label}: пересекает тело персонажа')
        if flags.get('prop_hits_room') and not flags.get('prop_penetration'):
            depth = checks.get('room_depth_m')
            out.append(f'{label}: внутри стены/мебели' + (f' на {cm(depth)} см' if depth else ''))
        if flags.get('prop_penetration'):
            info = checks.get('penetration', {})
            out.append(f'{label}: внутри {surface_ru(info.get("surface", "any"))} на {cm(info.get("depth_m", 0.))} см')
        if flags.get('prop_floating'):
            info = checks.get('support', {})
            out.append(f'{label}: нет опоры — висит на {cm(info.get("gap_m", 0.))} см')
    return out


def explain(fit):
    """Список русских причин, по которым H3 не отправляется; пусто — геометрия принята."""
    return _body_reasons(fit) + _prop_reasons(fit)
