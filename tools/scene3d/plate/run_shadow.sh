#!/bin/bash
# Step 2 shadows on the pod: S_R1s (+ harmonisation 0.30/0.45) and S_R1sR3 on P4/P8/P9/M3_bed_sit x 101/202,
# then pose/face metrics on the new frames only. Exit code -> /workspace/logs/shadow_exit.log
R=/workspace/farm/scene3d/repo; D=$R/tools/scene3d/plate; L=/workspace/logs
STATE=/workspace/farm/scene3d/quality_round; O=/workspace/ComfyUI-next/output/plate3d_quality
P=/workspace/ComfyUI-next/venv/bin/python; POSE=/workspace/venvs/pose_eval/bin/python
finish() { printf 'SHADOW_EXIT=%s\n' "$?" > "$L/shadow_exit.log"; }
trap finish EXIT
set -e
cd "$D"
$P -u quality_round.py --group S
"$POSE" -I -u "$D/pose_metrics.py" --glob "$O/P*/S_*.png" --maps /workspace/ComfyUI-next/input/plate3d \
  --thresholds "$D/quality_round.yaml" --out "$STATE/pose_S_main.json"
"$POSE" -I -u "$D/pose_metrics.py" --glob "$O/M3*/S_*.png" --maps /workspace/ComfyUI-next/input/plate3d_m3 \
  --thresholds "$D/quality_round.yaml" --out "$STATE/pose_S_m3.json"
export CUDA_VISIBLE_DEVICES=''; export PYTHONPATH=/workspace/venvs/face_pkgs
$P -u "$D/quality_report.py" --poses "$STATE/pose_S_main.json" --maps /workspace/ComfyUI-next/input/plate3d \
  --group S --out /workspace/farm/scene3d/reports/quality/S/main
$P -u "$D/quality_report.py" --poses "$STATE/pose_S_m3.json" --maps /workspace/ComfyUI-next/input/plate3d_m3 \
  --group S --out /workspace/farm/scene3d/reports/quality/S/m3
echo SHADOW_DONE
