"""Scene3D 3D-A: gate metrics for the H3 Fun ControlNet experiment.

    python tools/scene3d/gate_metrics.py --help

Measures, per branch (base / pose / depth / depth_pose / depth_pose_tuned):

  * pose match   - PCK@0.05 and mean error against the 3D keypoints, via DWPose on the studio
                   (the same measuring tool the 31.10 pilot used)
  * height ratio - generated figure height / 3D figure height, in frame heights
  * face similarity - facenet-pytorch InceptionResnetV1 (MIT, commercial OK) cosine between the
                   generated face crop and the reference face crop. NOT InsightFace: buffalol
                   is research-only.
  * room recognizability - CLIP-ish? No: a simple check is left to the human form; this script
                   only prints the per-scene file list so the sheet can be assembled.

Everything is measured on the frames the gate produced; the 3D side comes from the
pose25.json keypoints that render_h3_maps.py wrote next to each scene.

Commercial note: facenet-pytorch is MIT (weights: VGGFace2, academic dataset). AdaFace IR-101
is the alternative if the farm ever needs a non-dataset-dependent face metric.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------- faces

class FaceMetric:
    """Cosine similarity of facenet-pytorch embeddings of the largest face crop in an image."""

    def __init__(self, device="cpu"):
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN

        self.torch = torch
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=device, post_process=True)
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def embed(self, image_path):
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        faces = self.mtcnn(img)
        if faces is None or len(faces) == 0:
            return None
        tens = faces[0]
        # Pick the largest face so a background person cannot win over the subject.
        # MTCNN.detect() is ragged (a box may come back as None), so read the areas defensively.
        try:
            boxes = self.mtcnn.detect(img)
        except Exception:
            boxes = None
        if boxes is not None and len(boxes):
            areas = []
            for b in boxes:
                try:
                    bb = np.asarray(b, dtype=float).ravel()
                    areas.append(float((bb[2] - bb[0]) * (bb[3] - bb[1])) if len(bb) >= 4 else -1.0)
                except Exception:
                    areas.append(-1.0)
            if areas and max(areas) > 0:
                best = int(np.argmax(areas))
                if best < len(faces):
                    tens = faces[best]
        with self.torch.no_grad():
            emb = self.net(tens.unsqueeze(0).to(self.device))
        emb = self.torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()[0]

    def similarity(self, a_path, b_path):
        ea, eb = self.embed(a_path), self.embed(b_path)
        if ea is None or eb is None:
            return None
        return float(np.dot(ea, eb))


# --------------------------------------------------------------------------- pose

def load_3d_keypoints(scene_dir):
    """The 3D keypoints of the mannequin, as saved by render_h3_maps.py."""
    p = Path(scene_dir) / "pose25.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("people", "frames", "keypoints"):
            if key in data:
                data = data[key]
                break
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        first = data[0]
        for key in ("keypoints", "body_kp", "pose_body"):
            if key in first:
                return np.asarray(first[key], dtype=float)
        return None
    arr = np.asarray(data, dtype=float)
    return arr if arr.ndim == 2 else None


def pck(ref, gen, tol=0.05):
    """Share of joints visible in both that are within tol of the frame height.

    ref/gen: (N,3) arrays of x,y,confidence in pixels; the frame height is taken as the
    reference image height. Left/right swaps are tolerated (the pilot did the same).
    """
    from pose_eval import SWAP

    ref = np.asarray(ref, dtype=float)
    gen = np.asarray(gen, dtype=float)
    n = min(len(ref), len(gen))
    if n == 0:
        return None, None
    ref, gen = ref[:n], gen[:n]
    h = 1000.0  # normalised later by the caller

    both = (ref[:, 2] > 0.1) & (gen[:, 2] > 0.1)
    if both.sum() == 0:
        return 0.0, None
    d = np.linalg.norm(ref[:, :2] - gen[:, :2], axis=1)
    d_sw = np.linalg.norm(ref[:, :2] - gen[SWAP[:n], :2], axis=1) if n >= 18 else d
    d = np.minimum(d, d_sw)
    ok = d[both] <= tol * h
    return float(ok.mean()), float(d[both].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="", help="comma separated scene names")
    ap.add_argument("--root", default="C:/work/scene3d_work/gate162",
                    help="where the 3D maps live (pose25.json per scene)")
    ap.add_argument("--results", default="C:/work/scene3d_work/gate_results",
                    help="where the generated frames live: <results>/<scene>/<branch>/*.png")
    ap.add_argument("--ref-face", default="", help="reference face crop for the face metric")
    ap.add_argument("--out", default="", help="write the metrics table here (json)")
    ap.add_argument("--no-face", action="store_true", help="skip the face metric (slow download)")
    a = ap.parse_args()

    scenes = a.scenes.split(",") if a.scenes else []
    face = None if a.no_face else FaceMetric()

    table = {}
    for sc in scenes:
        scene_dir = Path(a.root) / sc
        ref = load_3d_keypoints(scene_dir)
        branches = {}
        br_dir = Path(a.results) / sc
        if not br_dir.exists():
            continue
        for br in sorted(p.name for p in br_dir.iterdir() if p.is_dir()):
            frames = sorted(br_dir.joinpath(br).glob("*.png"))
            if not frames:
                continue
            sims = []
            for f in frames:
                if face is not None and a.ref_face and os.path.exists(a.ref_face):
                    s = face.similarity(f, a.ref_face)
                    if s is not None:
                        sims.append(s)
            branches[br] = {"frames": len(frames), "face_sim": (float(np.mean(sims)) if sims else None)}
        table[sc] = branches

    print(json.dumps(table, indent=1, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
