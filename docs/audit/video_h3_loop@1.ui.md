# Инвентарь: `video_h3_loop@1.ui.json`

Сгенерировано `tools/wf_inventory.py`. Узлов верхнего уровня: 33, сабграфов: 2, frontend 1.53.6.

## верхний уровень

Группы: «1 · Вход: кадр, эталон лица, видео-референс, промпты», «2 · ПУЛЬТ», «ПРЕВЬЮ», «3 · Движки», «4 · Результат: генерация», «face pass готового видео»

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| 1 · Вход: кадр, эталон лица, видео-референс, промпты | 7 | LoadImage | IN keyframe · ключевой кадр (первый = последний) | on | image=stacy_kf_Sitting-8.png |
| 1 · Вход: кадр, эталон лица, видео-референс, промпты | 8 | LoadImage | IN face_ref · эталон лица <Picture 2> | on | image=5.png |
| 1 · Вход: кадр, эталон лица, видео-референс, промпты | 12 | PrimitiveStringMultiline | IN prompt · промпт генерации | on | value=<текст, 3769 симв.> |
| 1 · Вход: кадр, эталон лица, видео-референс, промпты | 13 | PrimitiveStringMultiline | IN face_prompt · промпт face pass | on | value=<текст, 286 симв.> |
| 1 · Вход: кадр, эталон лица, видео-референс, промпты | 14 | LoadImage | IN end_frame · конечный кадр <Picture 3> (loop ВЫКЛ + end_frame ВКЛ) | on | image=stacy_kf_Sitting-8.png |
| 1 · Вход: кадр, эталон лица, видео-референс, промпты | 15 | StacyRefVideo | IN motion_ref · видео-референс <Video 1> (необязательно) | on | video=(none); start_sec=0; length_sec=0 |
| 2 · ПУЛЬТ | 11 | StacyPanel | PANEL | on | mode=true; seed=3875932223; fixed=fixed; duration_sec=7; loop=true; end_frame=false; free_vram=auto; face_pass=auto; face_denoise_auto=true; face_denoise=0.3; face_lora=1; face_lock_temporal=0.8; char_lora=1.0 |
| 2 · ПУЛЬТ | 130 | PrimitiveInt | KNOB steps | on | value=8; control_after_generate=fixed |
| 2 · ПУЛЬТ | 131 | PrimitiveFloat | KNOB shift | on | value=12.0 |
| 2 · ПУЛЬТ | 132 | PrimitiveFloat | KNOB color_lock | on | value=-1.0 |
| 2 · ПУЛЬТ | 133 | PrimitiveInt | KNOB seam_crossfade | on | value=-1; control_after_generate=fixed |
| 2 · ПУЛЬТ | 134 | PrimitiveFloat | KNOB temporal_strength | on | value=1.0 |
| 2 · ПУЛЬТ | 135 | PrimitiveFloat | KNOB face_confidence | on | value=-1.0 |
| 2 · ПУЛЬТ | 136 | PrimitiveFloat | KNOB crop_factor | on | value=2.5 |
| 2 · ПУЛЬТ | 137 | PrimitiveFloat | KNOB large_face_mult | on | value=0.35 |
| 2 · ПУЛЬТ | 138 | PrimitiveFloat | KNOB face_lock | on | value=1.0 |
| 2 · ПУЛЬТ | 139 | PrimitiveFloat | KNOB hand_strength | on | value=0.2 |
| 3 · Движки | 1 | ⟦ДВИЖОК: генерация + FullHD + face pass (не трогать)⟧ | ДВИЖОК: генерация + FullHD + face pass (не трогать) | on | switch=; prompt_1=; string_b= |
| 4 · Результат: генерация | 9 | VHS_VideoCombine | OUT raw · 1344x768 | on | frame_rate=24; loop_count=0; filename_prefix=stacy_h3/Stacy_raw; format=video/h264-mp4; pix_fmt=yuv420p; crf=12; save_metadata=true; trim_to_audio=false; pingpong=false; save_output=true |
| 4 · Результат: генерация | 10 | VHS_VideoCombine | OUT video · итог FullHD 24 fps | on | frame_rate=24; loop_count=0; filename_prefix=stacy_h3/Stacy; format=video/h264-mp4; pix_fmt=yuv420p; crf=12; save_metadata=true; trim_to_audio=false; pingpong=false; save_output=true |
| 4 · Результат: генерация | 16 | SaveImage | OUT last_frame · последний кадр | on | filename_prefix=stacy_h3/Stacy_last |
| 4 · Результат: генерация | 17 | PreviewAny | OUT report · отчёт генерации | on |  |
| face pass готового видео | 2 | ⟦ДВИЖОК: face pass готового видео (не трогать)⟧ | ДВИЖОК: face pass готового видео (не трогать) | on | prompt=; string_b= |
| face pass готового видео | 18 | StacyVideoInput | ВИДЕО для face pass (загрузить) | on | video=(none) |
| face pass готового видео | 19 | VHS_VideoCombine | ИТОГ face pass готового видео | on | frame_rate=24; loop_count=0; filename_prefix=stacy_h3/Stacy_facepass; format=video/h264-mp4; pix_fmt=yuv420p; crf=12; save_metadata=true; trim_to_audio=false; pingpong=false; save_output=true |
| face pass готового видео | 20 | PreviewAny | ОТЧЁТ face pass видео | on |  |
| ПРЕВЬЮ | 21 | StacyModelByMode | модель активного режима → превью | on |  |
| ПРЕВЬЮ | 22 | ModelPreviewOverrideKJ | ПРЕВЬЮ (генерация / face pass видео) | on | max_resolution=1024; jpeg_quality=92; suppress_default_preview=true; preview_frames=1024; preview_fps=24; tiny_vae=taeh3.safetensors; preview= |
| — | 3 | MarkdownNote | README | on |  |
| — | 4 | MarkdownNote | Шпаргалка | on |  |
| — | 5 | MarkdownNote | ПРОМПТ-ГАЙД | on |  |
| — | 6 | MarkdownNote | Образец промпта (Sitting-8, 8 с, петля) | on |  |
| — | 129 | Fast Groups Muter (rgthree) |  | on |  |

## сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1)

Группы: «Модели», «Генерация H3 ref2va», «FullHD», «Face pass», «Режим»

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| Face pass | 64 | StacyLoopExtend | loop extend (+head, grid) | on | min_extra=24; h3_grid=true; is_loop=true |
| Face pass | 65 | StacyAutoConfidence | порог детектора лица (AUTO) | on | confidence=-1 |
| Face pass | 66 | H3FaceTrackCrop | face track + crop | on | detector=face_yolov8m.pt; confidence=0.35; crop_factor=2.5; canvas_width=768; canvas_height=768; canvas_mode=manual; smooth_window=21; size_smooth_window=51; smooth_method=gaussian; size_mode=per_frame; identity_track=false; identity_threshold=0.28; select=largest_face; fallback_detector=none; fallback_head_frac=0.5; select_index=0; identity_model=insightface; cut_detection=none; cut_threshold=3; … |
| Face pass | 67 | MiniMaxH3ReferenceToVideo | face conditioning | on | prompt=; width=1344; height=768; length=124; ref_image_size=match |
| Face pass | 68 | BasicScheduler | legacy denoise | on | scheduler=simple; steps=4; denoise=1 |
| Face pass | 69 | H3InjectVideoLatent | inject crops | on |  |
| Face pass | 70 | H3PerFrameDenoise | per-frame denoise | on | denoise_multiplier_small_face=1; denoise_multiplier_large_face=0.35; scale_mode=absolute_px; face_px_small=100; face_px_large=300; gamma=1; smooth_frames=9 |
| Face pass | 71 | StacyOcclusionDenoise | gentler under hands | on | hand_model=hand_yolov8s.pt; confidence=0.3; hand_strength=0.2; spread=0.25; time_sigma=2; expr_strength=0.3; expr_frames=6 |
| Face pass | 72 | BasicGuider | face guider | on |  |
| Face pass | 73 | RandomNoise | face seed | on | noise_seed=0; control_after_generate=randomize |
| Face pass | 74 | KSamplerSelect | euler (face) | on | sampler_name=euler |
| Face pass | 75 | SamplerCustomAdvanced | face sample | on |  |
| Face pass | 76 | VAEDecode | face decode | on |  |
| Face pass | 77 | StacyFlowAlign | lock face to source motion | on | align=1; align_blur=0.02; temporal=0.6; flow_size=384; temporal_mode=ema; radius=4 |
| Face pass | 78 | H3FaceStitch | stitch face | on | paste_region=face_only; mask_dilation=24; feather=6; colour_match=1; blend=1; undetected_frames=fade_out; feather_scales_with_crop=false |
| Face pass | 79 | StacyLoopFold | loop fold | on | original_frames=0; fade=16; curve=smootherstep; is_loop=true |
| Face pass | 80 | LazySwitchKJ | face pass? итог | on | switch=false |
| Face pass | 81 | StacyReport | сводный отчёт | on |  |
| Face pass | 82 | StringConcatenate | отчёт + анти-дрожь | on | string_a=; string_b=; delimiter=⏎⏎ |
| Face pass | 83 | LazySwitchKJ | отчёт цвета/шва | on | switch=false |
| Face pass | 84 | LazySwitchKJ | AUTO-отчёт (+порог детектора при face pass) | on | switch=false |
| Face pass | 85 | StringConcatenate | AUTO-настройки | on | string_a=; string_b=; delimiter=⏎ |
| Face pass | 86 | StringConcatenate | AUTO + отчёт | on | string_a=; string_b=; delimiter=⏎⏎ |
| Face pass | 87 | StringConcatenate | отчёт: тайминг промпта + видео-референс | on | string_a=; string_b=; delimiter=⏎ |
| Face pass | 88 | StringConcatenate | отчёт + тайминг промпта | on | string_a=; string_b=; delimiter=⏎⏎ |
| FullHD | 62 | RTXVideoSuperResolution | RTX VSR | on | resize_type=target dimensions; resize_type.width=1920; resize_type.height=1096; quality=ULTRA |
| FullHD | 63 | StacyFitFrame | exact 1920x1080 | on | width=1920; height=1080; method=bicubic; anchor_y=0.5 |
| Генерация H3 ref2va | 33 | StacyFitFrame | keyframe 1344x768 | on | width=1344; height=768; method=lanczos; anchor_y=0.5 |
| Генерация H3 ref2va | 34 | StacyFitFrame | конечный кадр → 1344x768 | on | width=1344; height=768; method=lanczos; anchor_y=0.5 |
| Генерация H3 ref2va | 35 | LazySwitchKJ | последний кадр = ключевой (петля) или конечный | on | switch=false |
| Генерация H3 ref2va | 36 | StacyPromptRetime | промпт ← длительность и петля | on |  |
| Генерация H3 ref2va | 37 | MiniMaxH3ReferenceToVideo | ref2va conditioning | on | prompt=; width=1344; height=768; length=124; ref_image_size=match |
| Генерация H3 ref2va | 38 | MiniMaxH3ReferenceToVideo | ref2va conditioning + <Video 1> | on | prompt=; width=1344; height=768; length=124; ref_image_size=match |
| Генерация H3 ref2va | 39 | MiniMaxH3ReferenceToVideo | ref2va conditioning + <Picture 3> (конечный кадр) | on | prompt=; width=1344; height=768; length=124; ref_image_size=match |
| Генерация H3 ref2va | 40 | MiniMaxH3ReferenceToVideo | ref2va conditioning + <Picture 3> + <Video 1> | on | prompt=; width=1344; height=768; length=124; ref_image_size=match |
| Генерация H3 ref2va | 41 | LazySwitchKJ | конечный кадр <Picture 3>? conditioning (без видео) | on | switch=false |
| Генерация H3 ref2va | 42 | LazySwitchKJ | конечный кадр <Picture 3>? conditioning (с видео) | on | switch=false |
| Генерация H3 ref2va | 43 | LazySwitchKJ | видео-референс? conditioning | on | switch=false |
| Генерация H3 ref2va | 44 | LazySwitchKJ | конечный кадр <Picture 3>? latent (без видео) | on | switch=false |
| Генерация H3 ref2va | 45 | LazySwitchKJ | конечный кадр <Picture 3>? latent (с видео) | on | switch=false |
| Генерация H3 ref2va | 46 | LazySwitchKJ | видео-референс? latent | on | switch=false |
| Генерация H3 ref2va | 47 | MiniMaxH3AddGuide | anchor frame 0 | on | frame_idx=0 |
| Генерация H3 ref2va | 48 | MiniMaxH3AddGuide | anchor last frame | on | frame_idx=-1 |
| Генерация H3 ref2va | 49 | LazySwitchKJ | якорь последнего кадра? (петля или конечный кадр) | on | switch=false |
| Генерация H3 ref2va | 50 | BasicGuider | guider (cfg 1) | on |  |
| Генерация H3 ref2va | 51 | RandomNoise | seed | on | noise_seed=0; control_after_generate=randomize |
| Генерация H3 ref2va | 52 | KSamplerSelect | res_multistep | on | sampler_name=res_multistep |
| Генерация H3 ref2va | 53 | BasicScheduler | simple | on | scheduler=simple; steps=8; denoise=1 |
| Генерация H3 ref2va | 54 | SamplerCustomAdvanced | sample | on |  |
| Генерация H3 ref2va | 55 | VAEDecode | decode | on |  |
| Генерация H3 ref2va | 56 | StacyFreeVRAM | освободить VRAM (до цвета и апскейла) | on | enabled=true |
| Генерация H3 ref2va | 57 | StacyColorLock | цвет → как у ключевого кадра | on | strength=1; chunk=16 |
| Генерация H3 ref2va | 58 | StacyLoopSeam | loop seam | on | crossfade=8; curve=smootherstep |
| Генерация H3 ref2va | 59 | LazySwitchKJ | петля? шов | on | switch=false |
| Генерация H3 ref2va | 60 | StacyTemporalStabilize | анти-дрожь до RTX (выравнивание по движению) | on | strength=1; radius=2; threshold=4; loop=true; motion_compensation=true; keep_sharpness=true |
| Генерация H3 ref2va | 61 | ImageFromBatch | последний кадр | on | batch_index=4095; length=1 |
| Модели | 23 | UNETLoader | ARM model · H3 ref2va | on | unet_name=Minimax-h3_Singularity_ref2va_Pruned_v1.…; weight_dtype=default |
| Модели | 24 | LoraLoaderModelOnly | ARM lora · LoRA персонажа (сила = PANEL char_lora) | on | lora_name=stacy_h3_A_stills_000001750.safetensors; strength_model=1 |
| Модели | 25 | MiniMaxH3SigmaShift | shift | on | shift_video=12; shift_audio=3 |
| Модели | 26 | ModelAttentionBackend | attention | on | attention=comfy kitchen attention |
| Модели | 27 | CLIPLoader | Qwen3-VL | on | clip_name=qwen3vl_32b_h3_ultra_uncensored_heretic_…; type=minimax; device=default |
| Модели | 28 | VAELoader | H3 video VAE | on | vae_name=minimax_h3_video_vae_fp16.safetensors |
| Модели | 29 | VAELoader | H3 audio VAE | on | vae_name=minimax_h3_audio_vae_fp32.safetensors |
| Модели | 30 | UNETLoader | H3 FL2VA (face) | on | unet_name=minimax_h3_fl2va_pruned_int8_convrot.saf…; weight_dtype=default |
| Модели | 31 | LoraLoaderModelOnly | FL2V turbo 4-step | on | lora_name=minimax_h3_fl2v_turbo_4step_v1.1_768p_co…; strength_model=1 |
| Модели | 32 | LoraLoaderModelOnly | Stacy face LoRA | on | lora_name=stacy_h3_face_v1_000001800.safetensors; strength_model=1 |
| Режим | 89 | StacySettings | пульт → значения (+ AUTO по ключевому кадру) | on | branch=generate |
| Режим | 90 | StacyGate | режим: генерация | on |  |
| Режим | 91 | StacyGate | режим: генерация (raw) | on |  |
| Режим | 92 | StacyGate | режим: генерация (кадр) | on |  |
| Режим | 93 | StacyGate | режим: генерация (отчёт) | on |  |

## сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1)

Группы: «Модели», «Видео», «Face pass», «Отчёт и режим»

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| Face pass | 105 | MiniMaxH3ReferenceToVideo | [видео] face conditioning | on | prompt=; width=1344; height=768; length=124; ref_image_size=match |
| Face pass | 106 | BasicScheduler | [видео] legacy denoise | on | scheduler=simple; steps=4; denoise=1 |
| Face pass | 107 | H3InjectVideoLatent | [видео] inject crops | on |  |
| Face pass | 108 | H3PerFrameDenoise | [видео] per-frame denoise | on | denoise_multiplier_small_face=1; denoise_multiplier_large_face=0.35; scale_mode=absolute_px; face_px_small=100; face_px_large=300; gamma=1; smooth_frames=9 |
| Face pass | 109 | StacyOcclusionDenoise | [видео] gentler under hands | on | hand_model=hand_yolov8s.pt; confidence=0.3; hand_strength=0.2; spread=0.25; time_sigma=2; expr_strength=0.3; expr_frames=6 |
| Face pass | 110 | BasicGuider | [видео] face guider | on |  |
| Face pass | 111 | RandomNoise | [видео] face seed | on | noise_seed=0; control_after_generate=randomize |
| Face pass | 112 | KSamplerSelect | [видео] euler (face) | on | sampler_name=euler |
| Face pass | 113 | SamplerCustomAdvanced | [видео] face sample | on |  |
| Face pass | 114 | VAEDecode | [видео] face decode | on |  |
| Face pass | 115 | StacyFlowAlign | [видео] lock face to source motion | on | align=1; align_blur=0.02; temporal=0.6; flow_size=384; temporal_mode=ema; radius=4 |
| Face pass | 116 | H3FaceStitch | [видео] stitch face | on | paste_region=face_only; mask_dilation=24; feather=6; colour_match=1; blend=1; undetected_frames=fade_out; feather_scales_with_crop=false |
| Face pass | 117 | StacyLoopFold | [видео] loop fold | on | original_frames=0; fade=16; curve=smootherstep; is_loop=true |
| Видео | 100 | StacyFitFrame | [видео] ровно 1920x1080 | on | width=1920; height=1080; method=bicubic; anchor_y=0.5 |
| Видео | 101 | ImageFromBatch | [видео] первый кадр для AUTO | on | batch_index=0; length=1 |
| Видео | 102 | StacyLoopExtend | [видео] loop extend (+head, grid) | on | min_extra=24; h3_grid=true; is_loop=true |
| Видео | 103 | StacyAutoConfidence | [видео] порог детектора лица (AUTO) | on | confidence=-1 |
| Видео | 104 | H3FaceTrackCrop | [видео] face track + crop | on | detector=face_yolov8m.pt; confidence=0.35; crop_factor=2.5; canvas_width=768; canvas_height=768; canvas_mode=manual; smooth_window=21; size_smooth_window=51; smooth_method=gaussian; size_mode=per_frame; identity_track=false; identity_threshold=0.28; select=largest_face; fallback_detector=none; fallback_head_frac=0.5; select_index=0; identity_model=insightface; cut_detection=none; cut_threshold=3; … |
| Модели | 94 | CLIPLoader | [видео] Qwen3-VL | on | clip_name=qwen3vl_32b_h3_ultra_uncensored_heretic_…; type=minimax; device=default |
| Модели | 95 | VAELoader | [видео] H3 video VAE | on | vae_name=minimax_h3_video_vae_fp16.safetensors |
| Модели | 96 | VAELoader | [видео] H3 audio VAE | on | vae_name=minimax_h3_audio_vae_fp32.safetensors |
| Модели | 97 | UNETLoader | [видео] H3 FL2VA (face) | on | unet_name=minimax_h3_fl2va_pruned_int8_convrot.saf…; weight_dtype=default |
| Модели | 98 | LoraLoaderModelOnly | [видео] FL2V turbo 4-step | on | lora_name=minimax_h3_fl2v_turbo_4step_v1.1_768p_co…; strength_model=1 |
| Модели | 99 | LoraLoaderModelOnly | [видео] Stacy face LoRA | on | lora_name=stacy_h3_face_v1_000001800.safetensors; strength_model=1 |
| Отчёт и режим | 118 | StacySettings | [видео] пульт → значения (+ AUTO по видео) | on | branch=face pass video |
| Отчёт и режим | 119 | StringConcatenate | [видео] отчёт 0 | on | string_a=; string_b=; delimiter=⏎⏎ |
| Отчёт и режим | 120 | StringConcatenate | [видео] отчёт 1 | on | string_a=; string_b=; delimiter=⏎⏎=== FACE TRACK ===⏎ |
| Отчёт и режим | 121 | StringConcatenate | [видео] отчёт 2 | on | string_a=; string_b=; delimiter=⏎⏎=== PER-FRAME DENOISE ===⏎ |
| Отчёт и режим | 122 | StringConcatenate | [видео] отчёт 3 | on | string_a=; string_b=; delimiter=⏎⏎=== HANDS NEAR THE FACE ===⏎ |
| Отчёт и режим | 123 | StringConcatenate | [видео] отчёт 4 | on | string_a=; string_b=; delimiter=⏎⏎=== FACE LOCK ===⏎ |
| Отчёт и режим | 124 | StacyGate | режим: face pass видео | on |  |
| Отчёт и режим | 125 | StacyGate | режим: face pass видео (отчёт) | on |  |

## Модели и LoRA

| граф | id | тип | title | виджет | файл | mode |
|---|---|---|---|---|---|---|
| верхний уровень | 22 | ModelPreviewOverrideKJ | ПРЕВЬЮ (генерация / face pass видео) | tiny_vae | taeh3.safetensors | on |
| сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1) | 66 | H3FaceTrackCrop | face track + crop | detector | face_yolov8m.pt | on |
| сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1) | 71 | StacyOcclusionDenoise | gentler under hands | hand_model | hand_yolov8s.pt | on |
| сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1) | 23 | UNETLoader | ARM model · H3 ref2va | unet_name | Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors | on |
| сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1) | 24 | LoraLoaderModelOnly | ARM lora · LoRA персонажа (сила = PANEL char_lora) | lora_name | stacy_h3_A_stills_000001750.safetensors | on |
| сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1) | 27 | CLIPLoader | Qwen3-VL | clip_name | qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors | on |
| сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1) | 28 | VAELoader | H3 video VAE | vae_name | minimax_h3_video_vae_fp16.safetensors | on |
| сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1) | 29 | VAELoader | H3 audio VAE | vae_name | minimax_h3_audio_vae_fp32.safetensors | on |
| сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1) | 30 | UNETLoader | H3 FL2VA (face) | unet_name | minimax_h3_fl2va_pruned_int8_convrot.safetensors | on |
| сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1) | 31 | LoraLoaderModelOnly | FL2V turbo 4-step | lora_name | minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors | on |
| сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1) | 32 | LoraLoaderModelOnly | Stacy face LoRA | lora_name | stacy_h3_face_v1_000001800.safetensors | on |
| сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1) | 109 | StacyOcclusionDenoise | [видео] gentler under hands | hand_model | hand_yolov8s.pt | on |
| сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1) | 104 | H3FaceTrackCrop | [видео] face track + crop | detector | face_yolov8m.pt | on |
| сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1) | 94 | CLIPLoader | [видео] Qwen3-VL | clip_name | qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors | on |
| сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1) | 95 | VAELoader | [видео] H3 video VAE | vae_name | minimax_h3_video_vae_fp16.safetensors | on |
| сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1) | 96 | VAELoader | [видео] H3 audio VAE | vae_name | minimax_h3_audio_vae_fp32.safetensors | on |
| сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1) | 97 | UNETLoader | [видео] H3 FL2VA (face) | unet_name | minimax_h3_fl2va_pruned_int8_convrot.safetensors | on |
| сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1) | 98 | LoraLoaderModelOnly | [видео] FL2V turbo 4-step | lora_name | minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors | on |
| сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1) | 99 | LoraLoaderModelOnly | [видео] Stacy face LoRA | lora_name | stacy_h3_face_v1_000001800.safetensors | on |

## Пакеты нод

| пакет | лицензия | ноды |
|---|---|---|
| comfy-core | GPL-3.0 (ComfyUI) | 3e522116-de38-40b4-a934-f0ee49b32b33, 42e92e31-e745-40ea-b12b-2d7fe51d2fde, BasicGuider, BasicScheduler, CLIPLoader, ImageFromBatch, KSamplerSelect, LoadImage, LoraLoaderModelOnly, MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift, ModelAttentionBackend, PreviewAny, PrimitiveFloat, PrimitiveInt, PrimitiveStringMultiline, RandomNoise, SamplerCustomAdvanced, SaveImage, StringConcatenate, UNETLoader, VAEDecode, VAELoader |
| comfyui-h3-facerefine | MIT ⚠ identity_track=insightface тянет buffalo_l (NC) | H3FaceStitch, H3FaceTrackCrop, H3InjectVideoLatent, H3PerFrameDenoise |
| comfyui-kjnodes | GPL-3.0 | LazySwitchKJ, ModelPreviewOverrideKJ |
| comfyui-videohelpersuite | GPL-3.0 | VHS_VideoCombine |
| comfyui_nvidia_rtx_nodes | Apache-2.0 (код; RTX VSR — проприетарная NVIDIA SDK под капотом) | RTXVideoSuperResolution |
| islomsalimov030-maker/ComfyUI-StacyLoop | ⚠ нет LICENSE (свой репо — добавить MIT); зависит от ultralytics (AGPL-3.0) | StacyAutoConfidence, StacyColorLock, StacyFitFrame, StacyFlowAlign, StacyFreeVRAM, StacyGate, StacyLoopExtend, StacyLoopFold, StacyLoopSeam, StacyModelByMode, StacyOcclusionDenoise, StacyPanel, StacyPromptRetime, StacyRefVideo, StacyReport, StacySettings, StacyTemporalStabilize, StacyVideoInput |

## Узлы без потребителей (кроме выходных/UI)


## Неподключённые выходы StacySettings (ручки пульта, которые ни на что не влияют)

- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `steps`
- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `shift`
- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `color_lock`
- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `seam_crossfade`
- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `large_face_mult`
- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `face_lock`
- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `hand_strength`
- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `face_confidence`
- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `crop_factor`
- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `video_mode`
- сабграф «ДВИЖОК: генерация + FullHD + face pass (не трогать)» (3e522116, экземпляров: 1): #89 «пульт → значения (+ AUTO по ключевому кадру)» → `video`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `length`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `seconds`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `seed`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `steps`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `shift`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `color_lock`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `seam_crossfade`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `free_vram`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `face_pass`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `large_face_mult`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `face_lock`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `hand_strength`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `face_confidence`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `crop_factor`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `generate`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `video`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `last_anchor`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `char_lora`
- сабграф «ДВИЖОК: face pass готового видео (не трогать)» (42e92e31, экземпляров: 1): #118 «[видео] пульт → значения (+ AUTO по видео)» → `end_ref`
