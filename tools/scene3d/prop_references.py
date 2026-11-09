"""Optional multi-view prop references; no geometry, material or default changes.

Picture indices are supplied by the actual compacted Remote connections.
Test-object-specific descriptions belong to experiment callers, not this module.
"""
import re

SECTIONS = ('subject_definitions', 'summary', 'retention_analysis', 'detailed_description',
            'overall_soundscape', 'non_diegetic_music')


def picture_list(pictures):
    labels = [f'<Picture {n}>' for n in pictures]
    if len(labels) == 1:
        return labels[0]
    return ', '.join(labels[:-1]) + (' and ' if len(labels) == 2 else ' and ') + labels[-1]


def with_prop_references(text, pictures, *, definition=None):
    """No references is a byte-identical no-op; otherwise define one Subject 2.

    Existing experiment-supplied Subject 2 is retained, but its actual media
    connections must still be checked by the caller's linter.
    """
    if not pictures:
        return text
    if re.search(r'(?m)^<Subject 2> is ', text):
        return text
    parts = re.split(r'(?m)^(' + '|'.join(SECTIONS) + r'):\s*\n', text)
    if parts[1::2] != list(SECTIONS):
        raise ValueError('Prop references require all six H3 sections in order')
    bodies = dict(zip(parts[1::2], parts[2::2]))
    source = picture_list(pictures)
    if definition is None:
        definition = (f'<Subject 2> is one floating prop shown from {len(pictures)} viewpoint(s) in {source}. '
                      'These images show the same single object and define its surface and visible details. '
                      'Its position, scale and viewing direction come from the placed mesh in the init image.')
    bodies['subject_definitions'] = bodies['subject_definitions'].rstrip() + '\n' + definition + '\n'
    bodies['summary'] = bodies['summary'].rstrip() + (
        ' Transfer the appearance of <Subject 2> from its reference views while preserving its placement from the init image.\n')
    bodies['retention_analysis'] = bodies['retention_analysis'].rstrip() + (
        '\n<Subject 2>: attribute_transfer - transfer surface appearance and visible details from '
        f'{source}; retain position, projected size and orientation from the mesh in the init image.\n')
    # Remove only the old prop clause, never the character's pose/RefMod clauses.
    bodies['detailed_description'] = re.sub(
        r'The [^\n.!?]+ floats motionless in the air; preserve its exact shape, color and material as in the init image\.',
        '', bodies['detailed_description']).rstrip()
    bodies['detailed_description'] += (
        '\nExactly one instance of <Subject 2> is visible. Preserve the mesh-defined position, projected size and orientation. '
        'The reference backgrounds and additional reference views do not appear in the photograph. '
        f'Transfer the surface appearance and visible details of <Subject 2> from {source}, not the neutral material in the init. '
        'The object floats motionless, with highlights consistent with the unchanged room lighting.\n')
    return ''.join(f'{name}:\n{bodies[name].strip()}\n' for name in SECTIONS)


def connected_labels(mapping, *, video=True):
    return {f'<Picture {row["picture"]}>' for row in mapping} | ({'<Video 1>'} if video else set())


def check_media(text, connected):
    used = set(re.findall(r'<(?:Picture|Video|Audio) \d+>', text))
    missing = sorted(used - set(connected))
    if missing:
        raise ValueError('H3 references are not connected: ' + ', '.join(missing))


def require_prompt_lint(text, connected):
    from farm.prompts.lint import lint
    issues = lint(text, 'frame', {'connected': set(connected)})
    errors = [i.code for i in issues if i.severity == 'error']
    if errors:
        raise ValueError(f'H3 lint failed: {errors}; no GPU submission')
    return [f'{i.severity}:{i.code}' for i in issues]
