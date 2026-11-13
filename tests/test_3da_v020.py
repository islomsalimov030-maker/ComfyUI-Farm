"""0.2.0 часть 2 (б–е) без GPU/Blender: растр из МП, одежда/промпт кадра, кадр-предупреждение,
русские причины отказа, серверная проверка предметов."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/scene3d'), str(ROOT / 'tools/scene3d/posev2')]
import prompt_h3
import prop_checks
import raster
import reasons_ru
import scene_edit
from prop_validation import blocking_reasons, include_props


def nodes():
    spec = importlib.util.spec_from_file_location('v020_nodes', ROOT / 'comfyui/ComfyUI-3DA/nodes.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# б) растр из мегапикселей пульта
def test_raster_from_panel_megapixels_keeps_plate_framing():
    assert raster.raster_for(0) == (1344, 768) and raster.raster_for(None) == (1344, 768)
    for mp in (1, 2, 3):
        w, h = raster.raster_for(mp)
        assert w % 16 == 0 and h % 16 == 0
        assert abs(w * h / 1e6 - mp) / mp < .03 and abs(w / h - 1344 / 768) < .02
    assert raster.raster_for(99) == raster.raster_for(raster.MP_MAX)
    with pytest.raises(ValueError):
        raster.raster_for('много')


# в) одежда — не 3DA; промпт кадра пользователя поверх промпта сцены
FIT = {'shot_spec': {'pose_class': 'stand', 'target_object': 'floor', 'secondary_contacts': []}, 'target_info': {}}


def scene_prompt():
    return prompt_h3.prompt_d(prompt_h3.table_prompt({'room': 'the room in <Picture 1>'},
        {'action': 'holding the body pose shown in <Video 1>', 'anchor': 'the bed reaches her knees'}), FIT)


def test_scene_prompt_has_no_hardcoded_clothing():
    text = scene_prompt()
    for word in ('T-shirt', 'jeans', 'sneakers', 'trousers', 'wears'):
        assert word not in text
    assert prompt_h3.lint(text, True) == []


def test_frame_prompt_merges_into_description_with_shifted_pictures():
    base = scene_prompt()
    assert prompt_h3.merge_frame_prompt(base, '  ') == base
    merged = prompt_h3.merge_frame_prompt(base, 'She wears a long red wool coat as in image 2.')
    desc = merged.split('detailed_description:')[1]
    assert 'red wool coat as in <Picture 3>' in desc and desc.index('static') < desc.index('red wool coat')
    assert '<Picture 3> is a reference image' in merged and '<Picture 3>: attribute_transfer' in merged
    assert prompt_h3.lint(merged, True, {'<Picture 1>', '<Picture 2>', '<Picture 3>', '<Video 1>'}) == []
    with pytest.raises(ValueError, match='подключено 1'):
        prompt_h3.merge_frame_prompt(base, 'outfit from image 3', user_refs=1)


def test_structured_frame_prompt_merges_by_section():
    user = ('subject_definitions:\n<Subject 2> is the knitted scarf in <Picture 2>.\nsummary:\n[reference generation] '
            'A calm evening.\nretention_analysis:\n<Subject 2>: attribute_transfer - scarf.\ndetailed_description:\n'
            '[Shot 1] She wears <Subject 2>.\noverall_soundscape:\nSoft rain outside.\nnon_diegetic_music:\nN/A')
    merged = prompt_h3.merge_frame_prompt(scene_prompt(), user)
    assert merged.count('subject_definitions:') == 1 and '<Subject 2> is the knitted scarf in <Picture 3>.' in merged
    assert 'A calm evening.' in merged.split('retention_analysis:')[0]
    assert 'Soft rain outside.' in merged and merged.index('She wears <Subject 2>') > merged.index('[Shot 1]')


# г) кадр — предупреждение, отказ только без единой видимой точки тела
def test_partly_out_of_frame_is_not_a_status():
    import place
    plate = SimpleNamespace(Wp=100, Hp=100, project=lambda X: (X[:, 0], X[:, 1], np.ones(len(X))))
    partly = place.framing(np.array([[50., 50, 0], [150, 50, 0]]), plate)
    gone = place.framing(np.array([[150., 50, 0], [160, 50, 0]]), plate)
    assert not partly['ok'] and partly['visible'] and not gone['visible']
    ev = dict(support_object='floor', target_instance='floor', support_ok=True, frame=partly,
              penetration={'ok': True}, footprint={'ok': True}, cells_ok=True, support_part_gaps_m={})
    spec = SimpleNamespace(target_object='floor')
    assert place.status_of(ev, spec) == 'ok'
    assert place.status_of({**ev, 'frame': gone}, spec) == 'out_of_frame'


# д) русские причины
def test_russian_reasons_for_rejected_geometry():
    fit = dict(status='contact_fail', support_gap_cm=12.4,
               validation=dict(ok=False, checks=[dict(check='support_contact', ok=False, excess=.12),
                                                 dict(check='penetration_surface:hard:floor', ok=False, excess=.12)]),
               penetration=dict(surfaces={'hard:floor': dict(p95_m=.15, threshold_m=.03)}))
    assert reasons_ru.explain(fit) == ['Персонаж: нет опоры — висит на 12 см',
                                       'Персонаж: внутри пола на 15 см (допуск 3 см)']
    fit = dict(status='partial', validation=dict(ok=False, checks=[dict(check='tilt', ok=False)]),
               props=[dict(prop_id='stick', flags=dict(prop_floating=True, prop_out_of_frame=True),
                           checks=dict(support=dict(gap_m=.51)))])
    assert reasons_ru.explain(fit) == ['Предмет 1 (stick): нет опоры — висит на 51 см']
    assert reasons_ru.surface_ru('soft:bed_0') == 'мягкой поверхности bed_0'


def test_rejection_message_and_status_line():
    mod = nodes()
    error = mod.rejected(['Персонаж: внутри пола на 15 см'])
    assert isinstance(error, mod.GeometryRejected) and str(error) == '3DA: H3 не отправлен — Персонаж: внутри пола на 15 см'
    line = mod.human_status(dict(reasons_ru=[], warnings=['body_partly_out_of_frame', 'prop[0] out_of_frame']))
    assert line == ' — Предупреждение: тело частично вне кадра; предмет 1 частично вне кадра'
    with pytest.raises(mod.GeometryRejected, match='Сначала выполните группу 1'):
        mod.accepted_guides({})


def test_empty_manifest_asks_for_group_1(tmp_path):
    for value in ('', '   ', str(tmp_path / 'nope/manifest.json')):
        with pytest.raises(ValueError, match='Сначала выполните группу 1'):
            scene_edit.read_manifest(value, {'farm_root': str(tmp_path)})
    with pytest.raises(ValueError, match='Сначала выполните группу 1'):
        scene_edit.check_file(None, {'sha256': 'x'})


# е) серверная проверка предметов
def floor_plate(with_floor=True):
    xs, ys = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(1, 3, 41))
    P = np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)] if with_floor else np.zeros((0, 3))
    N = np.tile([0., 0, 1], (len(P), 1))
    tree = cKDTree(P) if len(P) else None
    return SimpleNamespace(trees={'h': (tree, P, N), 'any': (tree, P, N)})


def cube(z0, size=.1, x=0., y=2.):
    g = np.linspace(0, size, 6)
    return np.array([[x + a, y + b, z0 + c] for a in g for b in g for c in g])


def below_floor(plate, X):
    return np.maximum(-X[:, 2], 0)


def test_prop_floating_penetrating_and_resting():
    plate = floor_plate()
    flags, info = prop_checks.check(plate, cube(.30), depth_fn=below_floor)
    assert flags['prop_floating'] and abs(info['support']['gap_m'] - .30) < 1e-6 and not flags['prop_penetration']
    flags, info = prop_checks.check(plate, cube(.02), depth_fn=below_floor)
    assert not any(flags.values()), (flags, info)
    flags, info = prop_checks.check(plate, cube(-.09), overlap_pairs=12, depth_fn=below_floor)
    assert flags['prop_penetration'] and flags['prop_hits_room'] and info['penetration']['surface'] == 'hard:floor'
    flags, _ = prop_checks.check(plate, cube(-.01), overlap_pairs=4, depth_fn=below_floor)
    assert not flags['prop_hits_room'], 'touching contact without depth is not a rejection'


def test_prop_support_skipped_when_held_or_unseen():
    held = prop_checks.check(floor_plate(), cube(.5), body=cube(.6, .2), depth_fn=below_floor)[1]['support']
    assert held['checked'] is False and held['reason'] == 'touches_body'
    unseen = prop_checks.check(floor_plate(False), cube(.5), depth_fn=below_floor)[1]['support']
    assert unseen['checked'] is False and unseen['reason'] == 'no_visible_support'


def test_new_prop_flags_block_but_frame_does_not():
    report = dict(status='ok', validation=dict(ok=True, checks=[]),
                  props=[dict(flags=dict(prop_floating=True, prop_out_of_frame=True))])
    result = include_props(report)
    assert blocking_reasons(result) == ['scene status=contact_fail', 'prop[0] prop_floating']


def test_no_prop_choice_passes_the_chain_through(tmp_path):
    mod = nodes()
    from unittest.mock import patch
    with patch.object(mod, 'config', return_value={'farm_root': str(tmp_path)}):
        assert mod.prop_choices()[0] == mod.NO_PROP
        out = mod.Prop().execute(mod.NO_PROP, 'world', 0., 0., 0., 0., 0., 0., 1.)
    assert out == ('[]',)
