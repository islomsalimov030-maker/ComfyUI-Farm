"""Pose round: the best poses of each method through H3 with the photo recipe S3 comp2 (table_server.graph: L_C2 =
playblast <Video 1> + base ref2va 25 steps + SAM3 person in one job) -> comp2 locally. One server job per pose.

  python pose_h3.py --picks A/Q01_stand_hand_face,B/Q03_bed_kneel,... [--seed 101]
Outputs: $FARM_COMFY/output/plate3d_poses/<M>_<id>/L_C2_s<seed>_*.png, mask, comp2_s<seed>.png; runs.json.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_ab as ab  # noqa: E402
import composite_round as cr  # noqa: E402
import farm_env  # noqa: E402
import pose_round as pr  # noqa: E402
import quality_round as qr  # noqa: E402
import remote_sync  # noqa: E402
import table_server as ts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--seed", type=int, default=101)
    a = ap.parse_args()
    ps = {p["id"]: p for p in pr.prompts()}
    inp, outdir = farm_env.COMFY / "input", farm_env.COMFY / "output"
    runs_path = pr.ROUND / "h3_runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    for pick in a.picks.split(","):
        m, pid = pick.split("/")
        p = ps[pid]
        loc = f"pose_{m}_m{p['loc']}"
        spec = json.loads((pr.ROUND / m / f"spec_m{p['loc']}.json").read_text(encoding="utf-8"))
        scene = next(s for s in spec["scenes"] if s["name"] == pid)
        shutil.copy(inp / f"{loc}_s2" / "_plate" / "plate_rgb.png", inp / f"{loc}_rgb.png")
        g = ts.graph(loc, spec, scene, a.seed)
        pre = f"plate3d_poses/{m}_{pid}/L_C2"
        g["save"]["inputs"]["filename_prefix"] = f"{pre}_s{a.seed}"
        g["sam_save"]["inputs"]["filename_prefix"] = f"{pre}_mask_s{a.seed}"
        name = f"{m}_{pid}_s{a.seed}"
        if any(r["name"] == name and r["status"] == "success" for r in runs):
            print("[poseH3] skip", name)
            continue
        remote_sync.upload(inp / f"{loc}_rgb.png", "")
        remote_sync.upload(inp / f"{loc}_s2" / pid / "playblast.png", f"{loc}_s2/{pid}")
        q = qr.api("/queue")
        if q["queue_running"] or q["queue_pending"]:
            raise SystemExit("Queue is occupied; refusing to add work")
        t0 = time.monotonic()
        pid_ = ab.post(g)
        status, result = qr.wait(pid_, 1200)
        row = dict(name=name, method=m, id=pid, status=status, seconds=round(time.monotonic() - t0, 1))
        runs.append(row)
        runs_path.write_text(json.dumps(runs, indent=1))
        print(f"[poseH3] {name}: {status} {row['seconds']}s", flush=True)
        if status != "success":
            raise SystemExit("stop after non-success")
        d = outdir / "plate3d_poses" / f"{m}_{pid}"
        cr.comp2(inp / f"{loc}_rgb.png", inp / f"{loc}_s2" / pid / "shadow.png",
                 sorted(d.glob(f"L_C2_s{a.seed}_*.png"))[-1], sorted(d.glob(f"L_C2_mask_s{a.seed}_*.png"))[-1],
                 d / f"comp2_s{a.seed}.png")


if __name__ == "__main__":
    main()
