"""H3 prompt text for 3DA placement (standalone copy of the step-9 'D' recipe).

Pure text functions extracted from the research scripts (quality_round.clay_prompt,
table_round.prompt, pose_h3_v25.prompt_b, pose_h3_v26.prompt_d) so that the pack
does not import the experiment chain. Output must stay byte-identical to them.
"""
import re

STATIC = ("filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, "
          "with no camera movement, no zoom and no pan.")
POSE = {
    "kneel": "She kneels upright on top of the {obj}: both knees and shins rest on its surface, her thighs and torso "
             "are vertical.",
    "lie": "She lies on top of the {obj}; her whole body, legs and feet included, rests on its surface and nothing "
           "hangs over the edge.",
    "recline": "She reclines on the {obj}: her hips rest on its surface and her back and shoulders lean against the "
               "{backrest}, her head higher than her hips.",
    "sit_edge": "She sits on the edge of the {obj}: her hips rest on its surface and her feet stand on the floor.",
    "sit_tucked": "She sits on top of the {obj} with her legs folded beside her, resting on its surface.",
    "sit_floor": "She sits on the floor.",
    "stand": "She stands on the floor.",
    "crouch": "She crouches on the floor.",
    "all_fours": "She is on all fours on the floor, her hands and knees resting on it.",
}
MANNEQUIN = "Her body pose and placement exactly match the grey mannequin in {pb}, and the mannequin is fully replaced by her."
ROOM = "The room, the {obj} and all furniture stay exactly as in {plate}, with the same shape, height, size and position."
PB_LABEL, PLATE_LABEL = "<Video 1>", "<Picture 1>"
TRANSFER = "Only her body pose and placement are taken from <Video 1>; the grey mannequin itself never appears in the shot."


def light_sentence(light):
    if not light:
        return ''
    az = float(light.get('azimuth_deg', 0)) % 360
    side = ('from the right of the frame' if 30 <= az < 150 else 'from behind her, against the camera' if 150 <= az < 210
            else 'from the left of the frame' if 210 <= az < 330 else 'from the camera side')
    if 100 <= az < 150 or 210 <= az < 260:
        side += ' and slightly behind her'
    r, g, b = (light.get('color') or [1, 1, 1])[:3]
    warm = 'warm golden' if r - b > 0.3 else 'cool bluish' if b - r > 0.1 else 'neutral white'
    el = float(light.get('elevation_deg', 30))
    low = 'low ' if el < 25 else ''
    hard = 'crisp' if float(light.get('softness_deg', 8)) <= 5 else 'soft-edged'
    return (f'The key light is the {low}{warm} daylight of the room coming {side}: it lights the same side of her body '
            f'and hair as it lights the furniture, the opposite side of her body falls into the same soft shade as the '
            f'room, and her body casts a {hard} shadow on the floor and furniture in the same direction as the shadows '
            f'of the furniture in <Picture 1>.')


def clay_prompt(spec, scene, shadow=False):
    outfit = ('She wears an opaque grey athletic T-shirt and opaque full-length navy trousers.' if not shadow else
              'She wears an opaque grey athletic T-shirt, opaque full-length navy trousers and white sneakers. ' +
              (light_sentence(scene.get('light') or spec.get('light')) or
               'She is lit by the same light as the room in <Picture 1> and casts a soft natural shadow '
               'on the floor and furniture beneath her.'))
    return '\n'.join([
        'subject_definitions:',
        '<Subject 1> is the adult woman whose identity is shown in <Picture 2>; replace the clay figure in <Video 1> with her.',
        '<Picture 1> is the exact room composition, furnishing, materials and lighting of the target photograph.',
        '<Picture 2> supplies only the facial identity and never appears as a separate image in the shot.',
        '<Video 1> is a clay playblast defining the camera, body pose, projected size and precise position of <Subject 1>.',
        'summary:',
        '[reference generation] One single photorealistic photograph. Reproduce the composition and body pose of <Video 1>, '
        'using the real room from <Picture 1> and the facial identity from <Picture 2>.',
        'retention_analysis:',
        '<Video 1>: partially_preserved - preserve geometry, pose, projected body scale and camera, replace only clay appearance.',
        '<Picture 1>: fully_preserved - retain room materials, furnishings and lighting.',
        '<Picture 2>: attribute_transfer - transfer facial identity to <Subject 1>.',
        'detailed_description:',
        f'A photorealistic live-action still. [Shot 1] {STATIC} The room is {spec["room"]}. '
        f'<Subject 1> is {scene["action"]}. {outfit} '
        f'Preserve the exact pose and placement from <Video 1>. Scale anchor: {scene["anchor"]}. '
        'No clay materials, no mannequin, no skeleton overlay. One continuous shot with no cuts and no close-up inserts; '
        '<Picture 2> is only a reference for her face and never appears on screen.',
        'Timeline: From 0.00 to 0.17 seconds the subject holds the starting pose without moving.',
        'The lighting is preserved unchanged throughout.',
        'Colors and saturation stay unchanged from the reference image.',
        'overall_soundscape:', 'Quiet room tone.', 'non_diegetic_music:', 'N/A'])


def table_prompt(spec, scene):
    text = clay_prompt(spec, scene, shadow=True)
    old = 'She wears an opaque grey athletic T-shirt, opaque full-length navy trousers and white sneakers.'
    new = ('She wears an opaque white short-sleeved cotton T-shirt, light-blue straight-leg jeans '
           'and white lace-up sneakers. The clothing is fully opaque and covers her torso and legs.')
    if old not in text:
        raise ValueError('outfit sentence missing')
    text = text.replace(old, new).replace('One single photorealistic photograph.',
        'One single photorealistic static image. <Picture 2> is only an identity reference sheet, not a layout.')
    text = text.replace('Scale anchor:', 'True-to-life scale: she is never enlarged to fill the frame. Scale anchor:')
    text = text.replace('retention_analysis:\n', 'retention_analysis:\n<Subject 1>: attribute_transfer - facial identity; wear the specified white T-shirt and light-blue jeans.\n')
    sections = ['subject_definitions:', 'summary:', 'retention_analysis:', 'detailed_description:',
                'overall_soundscape:', 'non_diegetic_music:']
    if [text.index(s) for s in sections] != sorted(text.index(s) for s in sections):
        raise ValueError('prompt sections out of order')
    return text


def prompt_b(text_a, fit):
    ss = fit["shot_spec"]
    ti = fit.get("target_info") or {}
    obj = ti.get("name") or ss["target_object"]
    pose = POSE[ss["pose_class"]].format(obj=obj, backrest=ti.get("backrest_name") or "backrest")
    if ss["pose_class"] == "stand" and any(c["surface"] == "wall" for c in ss.get("secondary_contacts", [])):
        pose = pose[:-1] + ", her hands resting on the wall."
    add = " ".join([pose, MANNEQUIN.format(pb=PB_LABEL), ROOM.format(obj=obj, plate=PLATE_LABEL)])
    t = text_a.replace("One single photorealistic static image.", "One single photograph.")
    m = re.search(r"\[Shot 1\] (.*?\.)(\s)", t, re.S)
    if not m:
        raise ValueError("no [Shot 1] in prompt A")
    return t[:m.end(1)] + " " + add + t[m.end(1):]


def prompt_d(text_a, fit):
    text = prompt_b(text_a, fit)
    text = re.sub(r"(?m)^<Video 1>:.*$", "<Video 1>: attribute_transfer - transfer only body pose and placement to <Subject 1>.", text)
    start = text.index("detailed_description:") + len("detailed_description:")
    return text[:start] + "\n" + TRANSFER + text[start:]


def lint(text, video, connected=None):
    from farm.prompts.lint import lint as _lint
    if connected is None:
        connected = {"<Picture 1>", "<Picture 2>"} | ({"<Video 1>"} if video else set())
    return [f"{i.severity}:{i.code}" for i in _lint(text, "frame", {"connected": set(connected)})]


def require_lint(text, video, connected=None):
    issues = lint(text, video, connected=connected)
    if any(i.startswith("error:") for i in issues):
        raise ValueError(f"H3 lint failed: {issues}; no GPU submission")
    return issues
