#!/usr/bin/env python3
"""WS-10b (23.10): video_deflicker@1 — убрать дрожь пикселей по всему кадру готового видео.

    python tools/build_video_deflicker.py      → workflows/video_deflicker@1.api.json

Готовая нода StacyLoop `StacyTemporalStabilize` (та же, что «анти-дрожь до RTX» в video_h3_loop@1): соседние кадры
выравниваются оптическим потоком и усредняются там, где совпадают (порог threshold), микроконтраст возвращается.
Вход — загрузка видео кнопкой; итог — mp4 с fps исходника в output/deflicker/.
"""
from __future__ import annotations

import json
from pathlib import Path

WF = Path(__file__).resolve().parents[1] / "workflows"


def build() -> dict:
    def node(cls, inputs, title, role):
        return {"class_type": cls, "inputs": inputs, "_meta": {"title": title, "role": role}}
    return {
        "IN": node("StacyVideoInput", {"video": "(none)"}, "ВИДЕО — загрузить кнопкой (choose video to upload)", "deflicker:in"),
        "ST": node("StacyTemporalStabilize", {"images": ["IN", 0], "strength": 1.0, "radius": 2, "threshold": 4.0,
                                              "loop": True, "motion_compensation": True, "keep_sharpness": True},
                   "АНТИ-ДРОЖЬ — strength 1; сильнее: radius 3, threshold 6; петля фермы — loop вкл", "deflicker:knob"),
        "OUT": node("VHS_VideoCombine", {"images": ["ST", 0], "frame_rate": ["IN", 2], "loop_count": 0,
                                         "filename_prefix": "deflicker/deflicker", "format": "video/h264-mp4",
                                         "pix_fmt": "yuv420p", "crf": 12, "save_metadata": False,
                                         "trim_to_audio": False, "pingpong": False, "save_output": True},
                    "ИТОГ — output/deflicker/deflicker_NNNNN.mp4", "deflicker:out"),
        "REP": node("PreviewAny", {"source": ["ST", 1]}, "ОТЧЁТ стабилизатора", "deflicker:out"),
    }


def main() -> int:
    g = build()
    (WF / "video_deflicker@1.api.json").write_text(json.dumps(g, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("video_deflicker@1", len(g), "нод")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
