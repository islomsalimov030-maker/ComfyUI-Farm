#!/bin/bash
# Round 4b on the pod: render the maps with the bpy wheel (no PC round trip), draw H3 maps, run A/B + faces.
set -u
R=/workspace/farm/scene3d/repo; I=/workspace/ComfyUI-next/input; P=/workspace/ComfyUI-next/venv/bin/python
BP=/workspace/venvs/bpy/bin/python; M=/workspace/farm/scene3d/stacy_mannequin_162_pod.blend; L=/workspace/logs
cd $R
render() {  # plate_dir spec out_root
  $BP -u tools/scene3d/plate/render_plate_scene.py -- --plate $1 --spec $2 --mannequin $M --out $3 2>&1 \
    | grep --line-buffered "^\[plate\]\|^\[poses\]\|Error\|Traceback\|SystemExit"
  $P tools/scene3d/draw_h3_maps.py $3 2>&1 | grep -v "^$"
  mv $3/_sheet_*.jpg /workspace/farm/scene3d/sheets/ 2>/dev/null
}
render $I/plate3d/_plate tools/scene3d/plate/scenes_mainroom1_r4.json $I/plate3d > $L/r4b_render.log
mkdir -p $I/plate3d_m2/_plate $I/plate3d_m3/_plate
cp /workspace/farm/scene3d/plate/mainroom2/plate_rgb.png /workspace/farm/scene3d/plate/mainroom2/plate.json $I/plate3d_m2/_plate/
cp /workspace/farm/scene3d/plate/mainroom3/plate_rgb.png /workspace/farm/scene3d/plate/mainroom3/plate.json $I/plate3d_m3/_plate/
render /workspace/farm/scene3d/plate/mainroom2 tools/scene3d/plate/scenes_mainroom2.json $I/plate3d_m2 >> $L/r4b_render.log
render /workspace/farm/scene3d/plate/mainroom3 tools/scene3d/plate/scenes_mainroom3.json $I/plate3d_m3 >> $L/r4b_render.log
echo RENDER_DONE >> $L/r4b_render.log
B="$P tools/scene3d/plate/build_ab.py"
$B --maps $I/plate3d --spec tools/scene3d/plate/scenes_mainroom1_r4.json --variants A5,A7,A8,A5h --seeds 101,202 --sheet-tag r4m1 --face > $L/r4m1.log 2>&1
$B --maps $I/plate3d_m2 --spec tools/scene3d/plate/scenes_mainroom2.json --variants A5,A7 --seeds 101,202 --sheet-tag r4m2 --face > $L/r4m2.log 2>&1
$B --maps $I/plate3d_m3 --spec tools/scene3d/plate/scenes_mainroom3.json --variants A5,A7 --seeds 101,202 --sheet-tag r4m3 --face > $L/r4m3.log 2>&1
echo ALL_DONE >> $L/r4b_render.log
