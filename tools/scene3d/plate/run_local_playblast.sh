#!/bin/bash
# 07.11 local PC (Blender 5.0, Cycles OptiX on the 3060 Ti): maps + playblasts for P4/P8 (mainroom1) and M3_bed_sit
# (mainroom3) into $FARM_COMFY/input: plate3d[_m3] (plain maps, no playblast, + draw_h3_maps), <maps>_s
# (--photo-emission 0, as 05.11), <maps>_s2 (--photo-emission ${EMIT2:-2.5}). source farm_env.sh first.
set -eu
R=$(cd "$(dirname "$0")/../../.." && pwd); I=$FARM_COMFY/input; W=C:/work/scene3d_work/plate
LOG=$FARM_ROOT/logs/playblast_local.log; mkdir -p "$I" "$(dirname "$LOG")"
cd "$R"
render() {  # plate spec out only extra...
  local plate=$1 spec=$2 out=$3 only=$4; shift 4
  mkdir -p "$out/_plate"; cp "$plate/plate_rgb.png" "$plate/plate.json" "$out/_plate/"
  local t0=$(date +%s)
  "$FARM_BLENDER" -b --python tools/scene3d/plate/render_plate_scene.py -- --plate "$plate" --spec "$spec" \
    --mannequin "$FARM_MANNEQUIN" --out "$out" --only "$only" "$@" 2>&1 \
    | grep --line-buffered "^\[plate\]\|^\[poses\]\|Error\|Traceback\|SystemExit" || true
  echo "[local] $(basename "$out") $* : $(( $(date +%s) - t0 )) s" | tee -a "$LOG"
}
M1=tools/scene3d/plate/scenes_mainroom1_r4.json; M3=tools/scene3d/plate/scenes_mainroom3.json
STEPS=${STEPS:-plain s s2}
for step in $STEPS; do
  case $step in
    plain) render $W/mainroom1 $M1 $I/plate3d P4_table_stand,P8_hand_face --no-playblast
           render $W/mainroom3 $M3 $I/plate3d_m3 M3_bed_sit --no-playblast
           "$FARM_PY" tools/scene3d/draw_h3_maps.py $I/plate3d 2>&1 | grep -v "^$" | tail -3
           "$FARM_PY" tools/scene3d/draw_h3_maps.py $I/plate3d_m3 2>&1 | grep -v "^$" | tail -3 ;;
    s)     render $W/mainroom1 $M1 $I/plate3d_s P4_table_stand,P8_hand_face --photo-emission 0
           render $W/mainroom3 $M3 $I/plate3d_m3_s M3_bed_sit --photo-emission 0 ;;
    s2)    render $W/mainroom1 $M1 $I/plate3d_s2 P4_table_stand,P8_hand_face --photo-emission ${EMIT2:-2.5}
           render $W/mainroom3 $M3 $I/plate3d_m3_s2 M3_bed_sit --photo-emission ${EMIT2:-2.5} ;;
  esac
done
echo PLAYBLAST_DONE | tee -a "$LOG"
