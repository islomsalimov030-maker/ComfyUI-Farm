"""Scene3D pilot: how well generated frames follow the 3D scene pose (DWPose on the studio vs render_scene keypoints).
python tools/scene3d/pose_eval.py C:/work/scene3d_work/scenes C:/work/scene3d_work/gen [--variants A,D4R55]
Per image: PCK@0.05 (share of joints visible in both within 5 % of the frame height), mean error in % of height,
face flip (3D face hidden but detected, or the other way round), body span (neck->ankles) ratio generated / 3D.
DWPose (IDEA-Research, Apache-2.0) is used only as a measuring tool here.
"""
import argparse
import json
import math
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zimage_cn import session, upload  # noqa: E402

BODY = list(range(14))  # COCO-18 without eyes/ears
FACE = [0, 14, 15]
SWAP = {2: 5, 3: 6, 4: 7, 5: 2, 6: 3, 7: 4, 8: 11, 9: 12, 10: 13, 11: 8, 12: 9, 13: 10, 14: 15, 15: 14, 16: 17, 17: 16}


def detect(s, url, img_name, prefix):
    g = {"img": {"class_type": "LoadImage", "inputs": {"image": img_name}},
         "dw": {"class_type": "DWPreprocessor", "inputs": {"image": ["img", 0], "detect_hand": "disable",
                                                           "detect_body": "enable", "detect_face": "disable",
                                                           "resolution": 864, "bbox_detector": "yolox_l.onnx",
                                                           "pose_estimator": "dw-ll_ucoco_384.onnx",
                                                           "scale_stick_for_xinsr_cn": "disable"}},
         "save": {"class_type": "SavePoseKpsAsJsonFile", "inputs": {"pose_kps": ["dw", 1], "filename_prefix": prefix}}}
    r = s.post(f"{url}/prompt", json={"prompt": g, "client_id": "scene3d"}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt {r.status_code}: {r.text[:400]}")
    pid = r.json()["prompt_id"]
    for _ in range(200):
        time.sleep(2)
        h = s.get(f"{url}/history/{pid}", timeout=60).json().get(pid)
        if not h:
            continue
        st = h.get("status", {})
        if st.get("status_str") == "error":
            raise RuntimeError(json.dumps(st.get("messages"))[:600])
        if st.get("completed"):
            break
    sub, base = (prefix.rsplit("/", 1) + [""])[:2] if "/" in prefix else ("", prefix)
    for n in range(1, 6):
        for fname in (f"{base}_{n:05d}.json", f"{base}_{n:05d}_.json", f"{base}.json"):
            q = urllib.parse.urlencode({"filename": fname, "subfolder": sub, "type": "output"})
            v = s.get(f"{url}/view?{q}", timeout=60)
            if v.status_code == 200 and v.content[:1] in (b"[", b"{"):
                return v.json()
    raise RuntimeError(f"pose json not found for {prefix}")


def best_person(kps):
    frame = kps[0] if isinstance(kps, list) else kps
    w, h = frame.get("canvas_width", 1), frame.get("canvas_height", 1)
    people = frame.get("people", [])
    if not people:
        return None
    p = max(people, key=lambda p: sum(p["pose_keypoints_2d"][2::3]))
    k = p["pose_keypoints_2d"]
    pts = [(k[3 * i], k[3 * i + 1], k[3 * i + 2]) for i in range(18)]
    if max(max(x, y) for x, y, _ in pts) <= 1.5:  # normalised coordinates
        pts = [(x * w, y * h, c) for x, y, c in pts]
    return pts


def score(target, det, size):
    h = size[1]
    if det is None:
        return {"pck": 0.0, "err": None, "face_flip": None, "span": None, "n": 0}
    def joint_errs(d):
        out = []
        for i in BODY:
            tx, ty, tv = target[i]
            dx, dy, dc = d[i]
            if tv and dc > 0.3:
                out.append(math.hypot(tx - dx, ty - dy) / h)
        return out
    # seen from behind, DWPose often swaps left/right: score the better of direct and mirrored labels
    swapped = [det[SWAP.get(i, i)] for i in range(18)]
    errs, errs_sw = joint_errs(det), joint_errs(swapped)
    lr_swap = bool(errs_sw) and sum(errs_sw) < sum(errs)
    if lr_swap:
        errs = errs_sw
    t_face = sum(target[i][2] for i in FACE) >= 2
    d_face = sum(det[i][2] > 0.3 for i in FACE) >= 2

    def span(pts, vis):
        ank = [pts[i] for i in (10, 13) if vis(i)]
        if not vis(1) or not ank:
            return None
        return math.hypot(pts[1][0] - sum(a[0] for a in ank) / len(ank), pts[1][1] - sum(a[1] for a in ank) / len(ank))
    ts = span(target, lambda i: target[i][2])
    ds = span(det, lambda i: det[i][2] > 0.3)
    return {"pck": round(sum(e < 0.05 for e in errs) / max(1, len(errs)), 2),
            "err": round(100 * sum(errs) / len(errs), 1) if errs else None,
            "face_flip": t_face != d_face, "lr_swap": lr_swap,
            "span": round(ds / ts, 2) if ts and ds else None, "n": len(errs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes")
    ap.add_argument("gen")
    ap.add_argument("--variants", default="A,D4R55")
    a = ap.parse_args()
    s, url = session()
    rows = []
    for sd in sorted(p for p in Path(a.gen).iterdir() if p.is_dir()):
        info = json.loads((Path(a.scenes) / sd.name / "pose.json").read_text(encoding="utf-8"))
        target = info["keypoints_coco18"]
        for v in a.variants.split(","):
            for img in sorted(sd.glob(f"{v}_s*.png")):
                name = upload(s, url, img, f"eval_{sd.name}_{img.name}")
                det = best_person(detect(s, url, name, f"scene3d_eval/{sd.name}_{img.stem}"))
                r = {"scene": sd.name, "variant": v, "file": img.name, **score(target, det, info["size"])}
                rows.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)
    (Path(a.gen) / "pose_eval.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for v in a.variants.split(","):
        rs = [r for r in rows if r["variant"] == v]
        pck = sum(r["pck"] for r in rs) / max(1, len(rs))
        errs = [r["err"] for r in rs if r["err"] is not None]
        flips = sum(bool(r["face_flip"]) for r in rs)
        spans = [r["span"] for r in rs if r["span"]]
        print(f"[eval] {v}: n={len(rs)} PCK@0.05 {pck:.2f}, mean err {sum(errs) / max(1, len(errs)):.1f} % H, "
              f"face flips {flips}, span ratio {min(spans or [0]):.2f}..{max(spans or [0]):.2f}")


if __name__ == "__main__":
    main()
