"""3D-A 08.11a п.11г: DWPose (rtmlib, Apache-2.0; веса локально, только диагностика) по списку картинок -> JSON
{файл: [{box, body18: [[x, y, score] × 18]}]} — все люди (класс person), точки в OpenPose-18 (как pose_metrics.py).
  $FARM_PY_POSE -I tools/scene3d/plate/dwpose_detect.py --list files.txt --out det.json --weights $FARM_DWPOSE
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COCO = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]


def openpose18(points, scores):
    out = []
    for i in COCO:
        if i is None:
            out.append([*((points[5] + points[6]) / 2).tolist(), float(min(scores[5], scores[6]))])
        else:
            out.append([float(points[i][0]), float(points[i][1]), float(scores[i])])
    return out


def main():
    from rtmlib import Wholebody
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    a = ap.parse_args()
    w = Path(a.weights)
    model = Wholebody(det=str(w / "yolox_l.onnx"), det_input_size=(640, 640), pose=str(w / "dw-ll_ucoco_384.onnx"),
                      pose_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cpu")
    model.det_model.det_mode = "multiclass"
    files = [x.strip() for x in Path(a.list).read_text(encoding="utf-8").splitlines() if x.strip()]
    res = json.loads(Path(a.out).read_text(encoding="utf-8")) if Path(a.out).is_file() else {}
    for f in files:
        if f in res:
            continue
        img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
        boxes, classes = model.det_model(img)
        boxes = np.asarray(boxes).reshape(-1, 4)[np.asarray(classes).reshape(-1) == 0]
        people = []
        if len(boxes):
            pts, sc = model.pose_model(img, bboxes=boxes)
            for k in range(len(boxes)):
                people.append({"box": boxes[k].tolist(), "body18": openpose18(pts[k], sc[k])})
        res[f] = {"size": [img.shape[1], img.shape[0]], "people": people}
        Path(a.out).write_text(json.dumps(res), encoding="utf-8")
        print("[dw]", Path(f).name, len(people), flush=True)


if __name__ == "__main__":
    main()
