"""CPU-only DWPose PCK measurement using Apache-2.0 rtmlib and explicit local ONNX weights.

Diagnostic use only: UBody training-data terms have not been confirmed for production.
No ComfyUI extension, OpenPose code or model download is invoked here.
PCK follows pose_eval.py (body joints visible in both, best direct/LR-swapped labels)
with coverage and stricter PCK over all target-visible joints reported separately.
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

# COCO17 -> OpenPose18. Neck is the midpoint of the COCO shoulders.
COCO = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
BODY = list(range(14))
SWAP = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]


def openpose18(points, scores):
    out = []
    for i in COCO:
        if i is None:
            out.append([*(points[5] + points[6]) / 2, min(scores[5], scores[6])])
        else:
            out.append([*points[i], scores[i]])
    return np.asarray(out)


def score(target, detected, height, confidence, threshold):
    visible = np.flatnonzero(target[:14, 2] > 0)
    if detected is None:
        return dict(pck=0.0, pck_strict=0.0, coverage=0.0, joints=0, target_joints=len(visible), lr_swap=False)
    options = []
    for swapped, det in ((False, detected), (True, detected[SWAP])):
        keep = [i for i in visible if det[i, 2] > confidence]
        errors = np.linalg.norm(target[keep, :2] - det[keep, :2], axis=1) / height
        hits = int(np.sum(errors < threshold))
        options.append(dict(pck=hits / len(keep) if keep else 0.0,
                            pck_strict=hits / len(visible) if len(visible) else None,
                            coverage=len(keep) / len(visible) if len(visible) else 0.0,
                            joints=len(keep), target_joints=len(visible), lr_swap=swapped,
                            mean_error=float(np.mean(errors)) if len(errors) else None))
    # Unlike sum(error), a low-coverage swap cannot win merely by losing joints.
    return max(options, key=lambda x: (x['pck_strict'] or 0, x['coverage']))


def main():
    from rtmlib import Wholebody

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glob', required=True)
    parser.add_argument('--maps', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--weights', default='/workspace/models/dwpose')
    parser.add_argument('--thresholds', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.thresholds).read_text())['qa']
    weights = Path(args.weights)
    for name in ('yolox_l.onnx', 'dw-ll_ucoco_384.onnx'):
        if not (weights / name).is_file():
            raise SystemExit(f'Missing local weight {name}; no auto-download')
    model = Wholebody(det=str(weights / 'yolox_l.onnx'), det_input_size=(640, 640),
                      pose=str(weights / 'dw-ll_ucoco_384.onnx'), pose_input_size=(288, 384),
                      to_openpose=False, backend='onnxruntime', device='cpu')
    model.det_model.det_mode = 'multiclass'
    rows = []
    for filename in sorted(glob.glob(args.glob)):
        path = Path(filename)
        target_path = Path(args.maps) / path.parent.name / 'pose25.json'
        if not target_path.exists():
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read {path}')
        data = json.loads(target_path.read_text())
        target = np.asarray(data['keypoints_body25'][:18], dtype=float)
        target[:, 0] *= image.shape[1] / data['size'][0]
        target[:, 1] *= image.shape[0] / data['size'][1]
        # Call detector separately: RTMPose may infer the full frame on an empty bbox list.
        # rtmlib 0.0.16 'human' mode returns boxes of all 80 COCO classes: keep class 0 (person) only.
        boxes, classes = model.det_model(image)
        boxes = np.asarray(boxes).reshape(-1, 4)[np.asarray(classes).reshape(-1) == 0]
        n = len(boxes)
        det = None
        selected_box = None
        if n:
            points, scores = model.pose_model(image, bboxes=boxes)
            if points.shape[1] != 133:
                raise ValueError(f'Expected COCO-WholeBody 133 points, got {points.shape}')
            # Select nearest detected person to the target projection, not best PCK.
            idx = min(range(n), key=lambda i: np.linalg.norm((boxes[i, :2] + boxes[i, 2:]) / 2
                                                            - target[target[:, 2] > 0, :2].mean(axis=0)))
            det = openpose18(points[idx], scores[idx])
            selected_box = boxes[idx].tolist()
        result = score(target, det, image.shape[0], config['confidence'], config['pck_distance'])
        row = dict(file=str(path), scene=path.parent.name, people=n, person_box=selected_box,
                   detected_body=None if det is None else det.tolist(), **result)
        rows.append(row)
        print('[pose]', path.parent.name, path.name, n, 'people',
              f"PCK {result['pck']:.3f} strict {result['pck_strict']} coverage {result['coverage']:.3f}", flush=True)
        dest = Path(args.out)
        temp = dest.with_suffix('.tmp')
        temp.write_text(json.dumps(rows, indent=2))
        temp.replace(dest)
    print('[pose] DONE', len(rows), flush=True)


if __name__ == '__main__':
    main()
