"""CPU-only 09f2 metrics and evidence; missing shots never stop other shots.

Run from the repository environment, e.g.:
    .venv/Scripts/python -I tools/scene3d/evidence_09f2.py --shots Q01
No submissions, network calls, workflow changes or asset writes are performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

REPO = Path(__file__).resolve().parents[2]
STATE = Path('C:/work/scene3d_work/farm/props_09f2')
POSE_SHOTS = ('Q04', 'Q12', 'T01', 'T05')
SIZE = (1344, 768)
sys.path.insert(0, str(REPO / 'tools/scene3d'))
sys.path.insert(0, str(REPO / 'tools/scene3d/plate'))
from props_09f2 import composite_mask
from pose_gate_v26 import gate_metric


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def font(size):
    for path in ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def public(value):
    """Last defensive pass: no absolute asset paths or credential fields in JSON."""
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items()
                if not re.search(r'token|password|secret|authorization|api_key', k, re.I)}
    if isinstance(value, (list, tuple)):
        return [public(v) for v in value]
    if isinstance(value, str):
        if re.search(r'[A-Za-z]:[\\/]|(?:^|\s)/(?:workspace|home|Users|work|tmp)/', value):
            return '[local asset]'
        return value
    return value


def flags(report):
    value = report.get('flags', [])
    return [str(k) for k, v in value.items() if v] if isinstance(value, dict) else list(value)


def support_summary(preview):
    report = preview.get('report', {})
    match = report.get('surface_match', {}) or report.get('fit', {}).get('surface_match', {}) or {}
    target = report.get('target_info', {}) or {}
    contact = report.get('support_contact', {}) or {}
    return {
        'status': preview.get('status', report.get('status')),
        'support_surface': match.get('support_surface_id', target.get('support_surface', target.get('object_id', report.get('support_object')))),
        'soft_surfaces': match.get('soft_surface_ids', target.get('soft_surfaces', [])),
        'soft_sink_m': match.get('soft_sink_m', target.get('soft_sink_m')),
        'surface_match_status': match.get('status'),
        'max_support_gap_m': contact.get('max_support_gap_m', report.get('max_support_gap_m')),
        'gap_parts': contact.get('gap_parts', []),
        'part_gaps_m': contact.get('part_gaps_m', {}),
        'support_parts': report.get('support_parts_on', report.get('support', [])),
        'flags': flags(report),
        'orientation': {k: report.get('orient', {}).get(k)
                        for k in ('tilt_final_deg', 'roll_final_deg')},
        'frame': report.get('frame', {}),
        'timings': preview.get('timings', {}),
        'preview_wall_s': preview.get('preview_wall_s'),
    }


def image(path, mode='RGB'):
    with Image.open(path) as source:
        result = source.convert(mode)
    if result.size != SIZE:
        raise ValueError('expected full-resolution 1344x768 input')
    return result


def prop_metrics(state, shot, preview):
    final = image(state / f'{shot}_remote/final.png')
    plate = image(Path(preview['init']).parent / 'plate.png')
    strength = np.asarray(image(preview['strength'], 'L'))
    alpha = composite_mask(strength)
    outside = alpha == 0
    if not outside.any():
        raise ValueError('empty outside-composite region')
    diff = np.abs(np.asarray(final, dtype=np.int16) - np.asarray(plate, dtype=np.int16))
    changed = np.any(diff != 0, axis=2)
    score = gate_metric(final, plate, outside)
    count = int(changed[outside].sum())
    props = []
    for prop in preview['report'].get('props', []):
        entry = {k: prop[k] for k in (
            'index', 'prop_id', 'size_m', 'scale', 'prop_strength',
            'source_bbox_m', 'bbox_source_m', 'source_dimensions_m',
            'final_bbox_m', 'bbox_final_m', 'final_dimensions_m',
            'frame_inside_fraction', 'flags') if k in prop}
        # Cross-connected erosion, at native resolution, not on a thumbnail.
        with Image.open(prop['silhouette']) as source:
            sil = np.asarray(source.convert('L').resize(SIZE, Image.Resampling.NEAREST)) > 127
        core = binary_erosion(sil, iterations=8)
        n = int(core.sum())
        opaque = int(np.any(diff > 10, axis=2)[core].sum())
        fraction = opaque / n if n else None
        entry.update(core_pixels=n, opaque_core_pixels=opaque,
                     opacity_fraction=fraction, opacity_passed=n > 0 and fraction >= .9)
        props.append(entry)
    result = support_summary(preview)
    result.update(
        image_size=list(SIZE), ssim_outside=score, outside_pixels=int(outside.sum()),
        outside_changed_pixels=count, outside_passed=score >= .9999 and count == 0,
        props=props, opacity_passed=bool(props) and all(p['opacity_passed'] for p in props),
        composite_reference='S>0; cross dilation 16px; Gaussian sigma=8 radius=8 mirror',
        opacity_reference='silhouette>127; cross erosion 8px at 1344x768; any(abs(final-plate)>10), int16 difference',
    )
    result['passed'] = result['outside_passed'] and result['opacity_passed']
    rr = state / f'{shot}_remote/result.json'
    if rr.is_file():
        remote = load(rr)
        result['remote_wall_s'] = remote.get('total_remote_wall_s')
    return result


def fmt(value):
    if isinstance(value, float):
        return f'{value:.5f}'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(fmt(v) for v in value) + ']'
    return str(value)


def wrap(text, width, draw, face):
    """Pixel-based wrapping, including long no-space lists/flags."""
    lines, current = [], ''
    for word in str(text).split():
        trial = (current + ' ' + word).strip()
        if draw.textlength(trial, font=face) <= width:
            current = trial
            continue
        if current:
            lines.append(current)
        current = ''
        for char in word:
            if current and draw.textlength(current + char, font=face) > width:
                lines.append(current)
                current = ''
            current += char
    if current:
        lines.append(current)
    return lines or ['']


def text_lines(items, width, face):
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    return [line for item in items for line in wrap(public(item), width, draw, face)]


def tile(sheet, path, column, y, width, height, label, face):
    draw = ImageDraw.Draw(sheet)
    draw.text((column * width + 8, y + 5), label, font=face, fill='#dddddd')
    if path is not None and Path(path).is_file():
        with Image.open(path) as source:
            thumb = source.convert('RGB')
        thumb.thumbnail((width - 12, height - 35))
        sheet.paste(thumb, (column * width + (width - thumb.width) // 2,
                            y + 30 + (height - 35 - thumb.height) // 2))
    else:
        draw.text((column * width + 10, y + 100), 'НЕТ ФАЙЛА', fill='#efc58e', font=face)


def save_sheet(sheet, name, output, downloads):
    target = output / name
    sheet.save(target, quality=93)
    shutil.copy2(target, downloads / name)
    return target.relative_to(REPO).as_posix() if target.is_relative_to(REPO) else name


def prop_sheet(state, shot, preview, metrics, output, downloads):
    face = font(18)
    body_frame = metrics['frame'].get('body', {})
    items = [
        f"{shot} · {metrics['status']} · SSIM outside {metrics['ssim_outside']:.8f} · outside changed {metrics['outside_changed_pixels']} / {metrics['outside_pixels']} · passed {metrics['passed']}",
        f"preview {metrics['timings'].get('total_s')} s · wall {metrics['preview_wall_s']} s · body frame fraction {body_frame.get('inside_fraction')} · out_of_frame {body_frame.get('out_of_frame')}",
        f"support {metrics['support_surface']} · gap {fmt(metrics['max_support_gap_m'])} m · flags {', '.join(metrics['flags']) or 'none'}",
    ]
    for prop in metrics['props']:
        items.extend([
            f"prop {prop.get('index')} · opacity {fmt(prop['opacity_fraction'])} ({prop['opaque_core_pixels']}/{prop['core_pixels']}) ≥0.9: {prop['opacity_passed']} · size {prop.get('size_m')} m · strength {prop.get('prop_strength')}",
            f"source bbox m {fmt(prop.get('source_bbox_m', prop.get('bbox_source_m')))} · dims {fmt(prop.get('source_dimensions_m'))}",
            f"final bbox m {fmt(prop.get('final_bbox_m', prop.get('bbox_final_m')))} · dims {fmt(prop.get('final_dimensions_m'))}",
            f"prop frame fraction {prop.get('frame_inside_fraction')} · flags {prop.get('flags', {})}",
        ])
    lines = text_lines(items, 1780, face)
    sheet = Image.new('RGB', (1800, 410 + len(lines) * 25 + 12), '#161616')
    for j, (label, path) in enumerate((('плейбласт', preview['playblast']), ('init', preview['init']),
                                       ('final', state / f'{shot}_remote/final.png'))):
        tile(sheet, path, j, 0, 600, 405, label, face)
    draw = ImageDraw.Draw(sheet)
    for i, line in enumerate(lines):
        draw.text((8, 410 + i * 25), line, font=face, fill='white')
    return save_sheet(sheet, f'props_{shot}_f.jpg', output, downloads)


def pose_sheet(state, output, downloads):
    rows, facts = [], {}
    face, small = font(17), font(15)
    for shot in POSE_SHOTS:
        preview_file = state / f'{shot}_pose_preview.json'
        request_file = state / f'{shot}_pose_request.json'
        if not preview_file.is_file():
            facts[shot] = {'status': 'pending', 'reason': 'pose preview missing'}
            continue
        try:
            preview = load(preview_file)
            if preview.get('status') == 'error':
                facts[shot] = {'status': 'error', 'reason': 'pose preview reports error'}
                continue
            summary = support_summary(preview)
            request = load(request_file) if request_file.is_file() else {}
            ref = Path(request['pose_ref']) if request.get('pose_ref') else None
            refs_dir = state.parent.parent / 'refs_v2_npz'
            ref_body = refs_dir / (ref.stem + '.ref_check.jpg') if ref else None
            # Job layout: jobs/<id>/render/pose/init.png and jobs/<id>/fit/overlay.jpg.
            render = Path(preview.get('init', preview['playblast'])).parent
            overlay = render.parent.parent / 'fit/overlay.jpg'
            if not overlay.is_file():
                overlay = Path(preview['debug']) if preview.get('debug') else None
            files = [ref, ref_body, overlay, Path(preview['playblast'])]
            missing = [name for name, path in zip(('reference', 'ref_body', 'fit_or_debug', 'playblast'), files)
                       if path is None or not path.is_file()]
            summary.update(freshness='NEW', missing_images=missing)
            facts[shot] = summary
            items = [
                f"NEW · {shot} · status {summary['status']} · support {summary['support_surface']} · soft {summary['soft_surfaces']}",
                f"gap {fmt(summary['max_support_gap_m'])} m · gap_parts {summary['gap_parts']} · soft_sink {fmt(summary['soft_sink_m'])} m · surface_match {summary['surface_match_status']}",
                f"flags {', '.join(summary['flags']) or 'none'} · tilt {fmt(summary['orientation']['tilt_final_deg'])}° · roll {fmt(summary['orientation']['roll_final_deg'])}°",
            ]
            if missing:
                items.append('missing images: ' + ', '.join(missing))
            lines = text_lines(items, 1580, face)
            rows.append((files, lines))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            facts[shot] = {'status': 'error', 'reason': type(exc).__name__}
    if rows:
        heights = [270 + 24 * len(lines) + 15 for _, lines in rows]
        sheet = Image.new('RGB', (1600, sum(heights)), '#161616')
        y = 0
        for (files, lines), height in zip(rows, heights):
            for j, (label, path) in enumerate(zip(('реф', 'реф + тело', 'постановка', 'плейбласт'), files)):
                tile(sheet, path, j, y, 400, 260, label, small)
            draw = ImageDraw.Draw(sheet)
            for i, line in enumerate(lines):
                draw.text((8, y + 270 + i * 24), line, fill='white', font=face)
            y += height
        save_sheet(sheet, 'poses_v27.jpg', output, downloads)
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=STATE)
    parser.add_argument('--shots', nargs='+', choices=('Q01', 'Q04'), default=['Q01', 'Q04'])
    parser.add_argument('--skip-poses', action='store_true')
    args = parser.parse_args()
    output = REPO / 'docs/experiments/img'
    downloads = Path.home() / 'Downloads/3DA_PROPS_2026-11-09f2'
    output.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)
    target = REPO / 'docs/experiments/3DA_PROPS_2026-11-09f2.json'
    facts = load(target) if target.is_file() else {'experiment': '3DA_PROPS_2026-11-09f2', 'props': {}, 'poses': {}}
    for shot in args.shots:
        preview_file = args.state / f'{shot}_prop_preview.json'
        if not preview_file.is_file() or not (args.state / f'{shot}_remote/final.png').is_file():
            facts['props'][shot] = {'status': 'pending', 'reason': 'prop preview or remote final missing'}
            continue
        try:
            preview = load(preview_file)
            if preview.get('status') == 'error':
                facts['props'][shot] = {'status': 'error', 'reason': 'prop preview reports error'}
                continue
            metrics = prop_metrics(args.state, shot, preview)
            metrics['evidence'] = prop_sheet(args.state, shot, preview, metrics, output, downloads)
            facts['props'][shot] = metrics
        except (OSError, ValueError, KeyError, TypeError) as exc:
            facts['props'][shot] = {'status': 'error', 'reason': type(exc).__name__}
    if not args.skip_poses:
        facts['poses'] = pose_sheet(args.state, output, downloads)
    target.write_text(json.dumps(public(facts), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    shutil.copy2(target, downloads / target.name)
    print(json.dumps({k: {field: value for field, value in v.items() if field in (
        'status', 'reason', 'ssim_outside', 'outside_changed_pixels', 'opacity_passed', 'passed')}
        for k, v in facts['props'].items()}, ensure_ascii=True))
    print('poses: ' + ', '.join(f'{shot}={row["status"]}' for shot, row in facts['poses'].items()))
    return 0 if not any(r.get('status') == 'error' for r in facts['props'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
