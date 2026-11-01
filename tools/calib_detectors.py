"""WS-15: калибровка MediaPipe против YOLO (face_yolov8m / hand_yolov8s) на кадрах Стейси.
Запуск: <venv с ultralytics+mediapipe>/python calib_detectors.py <папка кадров> <путь к StacyLoop>."""
import glob, json, os, sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
frames, sl = sys.argv[1], sys.argv[2]
sys.path.insert(0, sl)
import mp_detect as M  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from ultralytics import YOLO  # noqa: E402

yf = YOLO(hf_hub_download("Bingsu/adetailer", "face_yolov8m.pt"))
yh = YOLO(hf_hub_download("Bingsu/adetailer", "hand_yolov8s.pt"))
mf, mh = M.load(M.FACE), M.load(M.HAND)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def cover(a, b):   # доля центра MP-рамки внутри YOLO
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return a[0] <= cx <= a[2] and a[1] <= cy <= a[3]


rows, hands = [], []
for p in sorted(glob.glob(os.path.join(frames, '*.jpg'))):
    bgr = cv2.imread(p)
    ry = yf.predict(bgr, conf=0.15, verbose=False)[0]
    rm = mf.predict(bgr, conf=0.05)[0]
    yb, yc = ry.boxes.xyxy.tolist(), ry.boxes.conf.tolist()
    mb, mc = rm.boxes.xyxy.tolist(), rm.boxes.conf.tolist()
    r = {"f": os.path.basename(p), "H": bgr.shape[0], "yolo_max": max(yc, default=0), "mp_max": max(mc, default=0),
         "n_y": len(yb), "n_m": len(mb)}
    if yb and mb:
        i = int(np.argmax(yc)); a = yb[i]
        j = int(np.argmax([iou(a, b) for b in mb])); b = mb[j]
        r.update(iou=iou(a, b), center_in=cover(a, b), rw=(a[2] - a[0]) / max(1, b[2] - b[0]),
                 rh=(a[3] - a[1]) / max(1, b[3] - b[1]), dcy=((a[1] + a[3]) - (b[1] + b[3])) / 2 / max(1, b[3] - b[1]),
                 yolo_h=a[3] - a[1], yolo_conf=yc[i], mp_conf=mc[j])
    rows.append(r)
    hy = yh.predict(bgr, conf=0.3, verbose=False)[0]
    hm = mh.predict(bgr, conf=0.3)[0]
    hands.append({"f": r["f"], "n_y": len(hy.boxes), "n_m": len(hm.boxes)})

m = [r for r in rows if "iou" in r]
q = lambda k: np.percentile([r[k] for r in m], [10, 50, 90]).round(3).tolist() if m else None
print(json.dumps({
    "frames": len(rows), "face_yolo": sum(r["n_y"] > 0 for r in rows), "face_mp": sum(r["n_m"] > 0 for r in rows),
    "matched": len(m), "center_in": sum(r["center_in"] for r in m),
    "iou_p10_50_90": q("iou"), "w_ratio_yolo_over_mp": q("rw"), "h_ratio_yolo_over_mp": q("rh"),
    "centre_shift_y_in_mp_heights": q("dcy"), "yolo_conf": q("yolo_conf"), "mp_conf": q("mp_conf"),
    "yolo_face_h_px": q("yolo_h"),
    "miss_mp_when_yolo": [r["f"] for r in rows if r["n_y"] and not r["n_m"]][:10],
    "miss_yolo_when_mp": [r["f"] for r in rows if r["n_m"] and not r["n_y"]][:10],
    "hands_frames_yolo": sum(h["n_y"] > 0 for h in hands), "hands_frames_mp": sum(h["n_m"] > 0 for h in hands),
    "hands_both": sum(h["n_y"] > 0 and h["n_m"] > 0 for h in hands),
}, ensure_ascii=False, indent=1))
json.dump({"faces": rows, "hands": hands}, open(os.path.join(frames, 'calib.json'), 'w'), indent=1)
