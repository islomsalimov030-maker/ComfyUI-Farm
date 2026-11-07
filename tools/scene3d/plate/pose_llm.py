"""Pose round, method A: prompt -> local LLM (Ollama, qwen3:8b, Apache-2.0) -> pose JSON for render_plate_scene.

  python pose_llm.py --prompts pose_round_prompts.json --out $FARM_ROOT/pose_round/A [--url http://127.0.0.1:11435]
The model gets the MPFB rig conventions (the same as tools/scene3d/poses.py), the joint limits and three worked
examples; it answers with {"steps": [...], "contacts": [...]} (Ollama structured output). Nothing is executed here:
render_plate_scene.py applies the steps, clamps the joints and solves the contacts.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

SYSTEM = """You pose a 3D mannequin (MPFB rig, adult woman 1.62 m) for a still photo. Answer ONLY with JSON.
Rest pose: A-pose, standing, she faces -Y, her left side is +X, up is +Z. Arms hang ~45 deg out to the sides.
A step is [bone, axis, degrees]: rotate that bone (and everything below it) about the ARMATURE axis X, Y or Z
through the bone head. Steps run in order, parents first. Bones:
 root (whole body, pivot at the pelvis), spine05 (lowest) spine04 spine03 spine02 spine01 (chest), neck01, head,
 upperleg01.L/R (hip), lowerleg01.L/R (knee), foot.L/R (ankle), upperarm01.L/R (shoulder), lowerarm01.L/R (elbow),
 wrist.L/R.
Conventions (verified):
 thigh forward (hip flexion) = upperleg01 X NEGATIVE; knee bend = lowerleg01 X POSITIVE (after the thigh);
 thigh out to her side = upperleg01.L Z NEGATIVE, upperleg01.R Z POSITIVE;
 spine / head bend FORWARD = X POSITIVE, lean back = X NEGATIVE; head turn to her left = head Z POSITIVE;
 arm down further = upperarm01.L Y POSITIVE / upperarm01.R Y NEGATIVE; arm forward = upperarm01 X NEGATIVE;
 forearm forward/bend = lowerarm01 X NEGATIVE; foot point down = foot X POSITIVE.
 WHOLE BODY: root X +90 = body pitched forward horizontal, face down (all fours: + thighs X -90, knees X +90);
 root X -90 = lying on her back; root Y +90 = lying on her LEFT side, root Y -90 = on her right side.
Hands are best placed with a reach step: ["@reach", "L" or "R", anchor_bone, [x, y, z]] moves that arm so the
middle-finger base lands at anchor_bone head + offset in metres (ARMATURE axes: x her left, y BACK (-y = in front of
her), z up; x is mirrored for "R", so negative x = towards her midline). Examples of anchors: "head" [0.075, -0.08,
-0.03] = right hand at the cheek; "upperleg01.L" [0.07, -0.06, 0.13] = hand on the hip; "lowerleg01.L" [0, 0.03,
0.13] = hand on the thigh; "upperarm01.L" [0, 0, 0.55] = arm straight up; "foot.R" [0.03, -0.10, 0.04] = hand at the
shoe.
Joint limits (deg): elbow 0..150, knee 0..150, hip flexion -30..135, hip abduction -30..50, shoulder raise 0..180,
spine bend <= 80, spine twist <= 50, neck bend <= 70, head turn <= 80. Keep every joint inside.
Contacts: list the body parts that must touch a surface: part in pelvis, back, head, foot.L, foot.R, hand.L, hand.R,
knee.L, knee.R, elbow.L, elbow.R, shin.L, shin.R, thigh.L, thigh.R; surface = floor, bed, bench, wall or mat.
The placement code moves the body onto the surface; you only pose it. Keep the face towards the camera (-Y) unless
the prompt says otherwise.
Examples:
"sitting upright on a chair, hands on the thighs" -> {"steps": [["upperleg01.L","X",-88],["upperleg01.R","X",-88],
 ["lowerleg01.L","X",88],["lowerleg01.R","X",88],["@reach","L","lowerleg01.L",[0,0.03,0.13]],
 ["@reach","R","lowerleg01.R",[0,0.03,0.13]]], "contacts": [{"part":"pelvis","surface":"bench"},
 {"part":"foot.L","surface":"floor"},{"part":"foot.R","surface":"floor"}]}
"crouching, both hands at the right shoe" -> {"steps": [["upperleg01.L","X",-100],["upperleg01.R","X",-100],
 ["upperleg01.L","Z",-12],["upperleg01.R","Z",12],["lowerleg01.L","X",128],["lowerleg01.R","X",128],
 ["foot.L","X",-28],["foot.R","X",-28],["spine05","X",38],["head","X",18],["@reach","R","foot.R",[0.03,-0.10,0.04]],
 ["@reach","L","foot.R",[-0.06,-0.10,0.05]]], "contacts": [{"part":"foot.L","surface":"floor"},{"part":"foot.R","surface":"floor"}]}
"standing, right hand at the cheek" -> {"steps": [["@reach","L","upperleg01.L",[0.11,0,-0.09]],
 ["@reach","R","head",[0.075,-0.08,-0.03]],["head","Z",-6]], "contacts": [{"part":"foot.L","surface":"floor"},
 {"part":"foot.R","surface":"floor"}]}"""

SCHEMA = {"type": "object", "required": ["steps", "contacts"], "properties": {
    "steps": {"type": "array", "items": {"type": "array"}},
    "contacts": {"type": "array", "items": {"type": "object", "required": ["part", "surface"],
                                            "properties": {"part": {"type": "string"}, "surface": {"type": "string"}}}}}}


def ask(url, model, text):
    body = {"model": model, "stream": False, "think": False, "format": SCHEMA,
            "options": {"temperature": 0.2, "num_ctx": 8192},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f'"{text}" ->'}]}
    req = urllib.request.Request(url + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.load(r)["message"]["content"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11435")
    ap.add_argument("--model", default="qwen3:8b")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in json.loads(Path(a.prompts).read_text(encoding="utf-8"))["prompts"]:
        t0 = time.monotonic()
        try:
            spec = ask(a.url, a.model, p["action"])
        except Exception as e:  # keep the round going; a failed answer is a result too
            spec = {"steps": [], "contacts": [], "error": str(e)[:300]}
        spec.update(contact="solve", method="A", model=a.model, prompt=p["action"], seconds=round(time.monotonic() - t0, 1))
        if not spec.get("contacts"):
            spec["contacts"] = p["contacts"]
        (out / f"{p['id']}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        print(f"[A] {p['id']}: {len(spec.get('steps', []))} steps, {len(spec.get('contacts', []))} contacts, "
              f"{spec['seconds']} s{' ERROR ' + spec['error'] if 'error' in spec else ''}", flush=True)


if __name__ == "__main__":
    main()
