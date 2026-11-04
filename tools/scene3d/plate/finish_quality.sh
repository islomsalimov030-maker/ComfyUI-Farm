#!/bin/bash
# Bounded follow-up for the current quality batch; CPU-only diagnostics, no new generation.
set -euo pipefail
R=/workspace/farm/scene3d/repo
D=$R/tools/scene3d/plate
L=/workspace/logs
STATE=/workspace/farm/scene3d/quality_round
P=/workspace/ComfyUI-next/venv/bin/python
POSE=/workspace/venvs/pose_eval/bin/python
finish() { printf 'METRICS_EXIT=%s\n' "$?" > "$L/quality_metrics_exit.log"; }
trap finish EXIT
for ((i=0; i<720; i++)); do
  if ! tmux has-session -t quality_batch 2>/dev/null; then break; fi
  sleep 10
done
if tmux has-session -t quality_batch 2>/dev/null; then
  printf 'Batch did not finish within two hours; no generation stopped by this observer.\n'
  exit 2
fi
cd "$R"
"$POSE" -I -u "$D/pose_metrics.py" \
  --glob '/workspace/ComfyUI-next/output/plate3d_quality/P*/*.png' \
  --maps /workspace/ComfyUI-next/input/plate3d --thresholds "$D/quality_round.yaml" \
  --out "$STATE/pose_all_main.json"
"$POSE" -I -u "$D/pose_metrics.py" \
  --glob '/workspace/ComfyUI-next/output/plate3d_quality/M3*/*.png' \
  --maps /workspace/ComfyUI-next/input/plate3d_m3 --thresholds "$D/quality_round.yaml" \
  --out "$STATE/pose_all_m3.json"
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=/workspace/venvs/face_pkgs
for group in Q1 Q2 R1 Q3 Q4 R2 R3; do
  "$P" -u "$D/quality_report.py" --poses "$STATE/pose_all_main.json" \
    --maps /workspace/ComfyUI-next/input/plate3d --group "$group" \
    --out "/workspace/farm/scene3d/reports/quality/$group/main"
  "$P" -u "$D/quality_report.py" --poses "$STATE/pose_all_m3.json" \
    --maps /workspace/ComfyUI-next/input/plate3d_m3 --group "$group" \
    --out "/workspace/farm/scene3d/reports/quality/$group/m3"
done
printf 'QUALITY_METRICS_DONE\n'
