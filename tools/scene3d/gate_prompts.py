"""Scene3D 3D-A: SFW H3 prompts for the gate scenes.

python tools/scene3d/gate_prompts.py            # print them
python tools/scene3d/gate_prompts.py --json     # as JSON, for the workflow builder

One prompt per scene, ref-format for a single still frame (tag [reference generation]).

Rules that the linter in the H3 skill enforces (all checked here):
  - the six sections in order: subject_definitions, summary, retention_analysis,
    detailed_description, overall_soundscape, non_diegetic_music
  - the camera line: "filmed from one fixed position; the camera holds a completely static,
    locked-off shot throughout, with no camera movement, no zoom and no pan"
  - the lighting line: "The lighting is preserved unchanged throughout."
  - the colour line: "Colors and saturation stay unchanged from the reference image."
  - the word "studio" is forbidden
  - scale anchors by furniture, not centimetres

The location sheet is <Picture 1> (the reference sheet of the room), the face sheet is
<Picture 2>. Both are referenced from <Subject N> lines; neither is a frame anchor here -
the 3D maps carry the layout.
"""

SCENES = {
    "S1_sofa_sit": {
        "action": "sitting upright on the sofa with her back against the sofa backrest and her hands resting on her lap",
        "place": "in the living zone of the room, on the sofa",
    },
    "S2_bench_sit": {
        "action": "sitting on the wooden bench, leaning back slightly with her hands resting on her thighs",
        "place": "in the living zone of the room, on the wooden bench",
    },
    "S3_vanity_stand": {
        "action": "standing next to the bathroom vanity, one hand resting on her hip and the other arm hanging naturally at her side",
        "place": "in the bathroom zone of the room, beside the vanity counter and the mirror",
    },
    "S4_window_stand": {
        "action": "standing by the window with her head turned to look outside, her arms relaxed at her sides",
        "place": "in the living zone of the room, next to the window",
    },
    "S5_room_stand": {
        "action": "standing in the middle of the room in a relaxed pose with her arms hanging naturally at her sides",
        "place": "in the middle of the living zone of the room",
    },
}

CAMERA = ("filmed from one fixed position; the camera holds a completely static, locked-off "
          "shot throughout, with no camera movement, no zoom and no pan")
LIGHT = "The lighting is preserved unchanged throughout."
COLOR = "Colors and saturation stay unchanged from the reference image."

TEMPLATE = """subject_definitions:
<Subject 1> is St4cy_H3, the young woman in the character sheet in <Picture 2>, with exactly the face, hair, skin tone and body proportions shown there.
<Subject 2> is the home interior in <Picture 1>, with the tiled bathroom zone, the wooden living zone, the slatted screen, the bathtub, the vanity with the vessel sink and the long mirror, the sofa, the wooden bench, the window and the ceiling beams, exactly as they appear there.
<Picture 1> is the location sheet of the room and is used only as a reference for its layout, furniture and materials; it never appears on screen.
<Picture 2> is an identity reference sheet of the face of <Subject 1> and never appears on screen.
summary:
[reference generation] One single photograph of <Subject 1> {action} {place}. The shot reproduces the pose, the framing and the scale of the reference layout, with <Subject 1> placed exactly where the layout puts her.
retention_analysis:
<Subject 1> (appears in the frame): fully_preserved - her face, hair, skin tone, body proportions and outfit are retained.
<Subject 2> (appears in the frame): fully_preserved - the room, its furniture, its layout and its materials are retained.
<Picture 1>: fully_preserved - the layout, the furniture and the materials of the room are retained, the sheet itself is not shown.
<Picture 2>: fully_preserved - the facial identity of <Subject 1> is retained.
detailed_description:
The target image is one photorealistic live-action still, {camera}. The room is lit by soft daylight from the window and warm interior light. {light} {color} True-to-life scale: the height of <Subject 1> matches the height of the sofa backrest and the height of the mirror, and she is never enlarged to fill the frame. Scale anchor: the sofa seat reaches her upper thigh and the long mirror reaches above her head. One continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen.
overall_soundscape:
N/A
non_diegetic_music:
N/A
"""


def build(scene):
    s = SCENES[scene]
    return TEMPLATE.format(action=s["action"], place=s["place"],
                           camera=CAMERA, light=LIGHT, color=COLOR).strip()


def lint(p):
    """The checks from the H3 prompt skill. Returns a list of problems."""
    bad = []
    order = ["subject_definitions:", "summary:", "retention_analysis:",
             "detailed_description:", "overall_soundscape:", "non_diegetic_music:"]
    pos = [p.find(s) for s in order]
    if any(x < 0 for x in pos):
        bad.append("missing section")
    elif pos != sorted(pos):
        bad.append("sections out of order")
    if "studio" in p.lower():
        bad.append("the word 'studio' is forbidden")
    if CAMERA not in p:
        bad.append("camera line missing")
    if LIGHT not in p:
        bad.append("lighting line missing")
    if COLOR not in p:
        bad.append("colour line missing")
    for w in ("pan", "zoom", "dolly", "tilt", "truck", "arc", "tracking"):
        if f" {w}" in p.lower() and "no zoom and no pan" not in p.lower():
            bad.append(f"camera movement word: {w}")
    if p.count("[Shot 1]"):
        bad.append("[Shot 1] with a timecode")
    return bad


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = {}
    for name in SCENES:
        p = build(name)
        problems = lint(p)
        out[name] = p
        if problems:
            print(f"LINT FAIL {name}: {problems}")
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        for name, p in out.items():
            print(f"########## {name}\n{p}\n")


if __name__ == "__main__":
    main()
