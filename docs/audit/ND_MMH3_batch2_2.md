# Инвентарь: `ND_MMH3_batch2_2.json`

Сгенерировано `tools/wf_inventory.py`. Узлов верхнего уровня: 60, сабграфов: 4, frontend 1.53.6.

## верхний уровень

Группы: «Minimax I2I», «Варианты LoRA (каждый промпт × каждый вариант)», «Body Pass Zimage», «Face Pass Zimage», «Face_Pass_Minimax»

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| Body Pass Zimage | 11345 | SeedNode |  | on | seed=612651749928607; fixed=fixed |
| Body Pass Zimage | 11346 | Note | Triggerwords | on |  |
| Body Pass Zimage | 11347 | ⟦New Subgraph⟧ | New Subgraph | on | seed=685683109897719; denoise=0.5; text=Stacy_v1 |
| Body Pass Zimage | 11348 | Image Comparer (rgthree) |  | on |  |
| Body Pass Zimage | 11351 | PreviewImage |  | on |  |
| Face Pass Zimage | 11358 | Note | Triggerwords | on |  |
| Face Pass Zimage | 11359 | ⟦New Subgraph⟧ | New Subgraph | on | denoise=0.6; value=1328; text=Stacy_v1, brown eyes; manual_mode=false; seed=993866159869322 |
| Face Pass Zimage | 11361 | SeedNode |  | on | seed=953552952704593; fixed=fixed |
| Face Pass Zimage | 11365 | Image Comparer (rgthree) |  | on |  |
| Face Pass Zimage | 11369 | PreviewImage |  | on |  |
| Face_Pass_Minimax | 11354 | PrimitiveFloat | Megapixels | on | value=4 |
| Face_Pass_Minimax | 11355 | mxSlider | Denoise лица | on | Xi=0; Xf=0.85; isfloatX=1 |
| Face_Pass_Minimax | 11356 | Text Multiline | Промпт (триггер St4cy_H3) | on | text=<текст, 161 симв.> |
| Face_Pass_Minimax | 11357 | mxSlider | Коррекция цвета под оригинал | on | Xi=0; Xf=0.9; isfloatX=1 |
| Face_Pass_Minimax | 11360 | Note | Triggerwords | on |  |
| Face_Pass_Minimax | 11362 | LoraLoaderModelOnly |  | on | lora_name=stacy_h3_face_v1.safetensors; strength_model=1 |
| Face_Pass_Minimax | 11363 | SeedNode |  | on | seed=953552952704593; fixed=fixed |
| Face_Pass_Minimax | 11364 | ⟦New Subgraph⟧ | New Subgraph | on | strength=0.8; denoise=0.55; noise_seed=42; prompt=; megapixels=2; manual_mode=false |
| Face_Pass_Minimax | 11368 | ModelPreviewOverrideKJ | MiniMax Sampling Preview | on | max_resolution=1024; jpeg_quality=80; suppress_default_preview=true; preview_frames=1; preview_fps=1; tiny_vae=taeh3.safetensors; preview= |
| Face_Pass_Minimax | 11370 | PreviewImage |  | on |  |
| Face_Pass_Minimax | 11371 | Image Comparer (rgthree) |  | on |  |
| Minimax I2I | 11271 | Note |  | on |  |
| Minimax I2I | 11272 | Note |  | on |  |
| Minimax I2I | 11280 | LoadImage |  | on | image=ND_MMH3__00727_.png |
| Minimax I2I | 11282 | LoadImage |  | on | image=ComfyUI_temp_osloj_00024_ (1).png |
| Minimax I2I | 11283 | PrimitiveFloat | Megapixels | on | value=2 |
| Minimax I2I | 11284 | LoadImage |  | BYPASS | image=girlsgold_ru_img_11.jpg |
| Minimax I2I | 11286 | LoadImage |  | BYPASS | image=коллаж-Стейси-2.png |
| Minimax I2I | 11288 | SimplePromptBatcher | ПРОМПТЫ: по одному на строку (каждая строка = отдельная картинка) | on | prepend=; prompts=<текст, 1117 симв.>; append= |
| Minimax I2I | 11289 | SeedNode |  | on | seed=786675712436384; fixed=randomize |
| Minimax I2I | 11291 | ModelPreviewOverrideKJ | MiniMax Sampling Preview | on | max_resolution=1024; jpeg_quality=80; suppress_default_preview=true; preview_frames=1; preview_fps=1; tiny_vae=taeh3.safetensors; preview= |
| Minimax I2I | 11293 | PrimitiveBoolean | 10ErosMax | on | value=true |
| Minimax I2I | 11298 | mxSlider | Second Pass | Thick Details --------> Thin Details | on | Xi=40; Xf=40; isfloatX=0 |
| Minimax I2I | 11299 | mxSlider | Ref Identity Strenght | on | Xi=1; Xf=1; isfloatX=1 |
| Minimax I2I | 11300 | mxSlider | First Pass |  Steps | on | Xi=6; Xf=6; isfloatX=0 |
| Minimax I2I | 11301 | mxSlider | Second Pass | Steps  (More = More Changes) | on | Xi=8; Xf=8; isfloatX=0 |
| Minimax I2I | 11306 | ResolutionSelector |  | on | aspect_ratio=16:9 (Widescreen); megapixels=4; multiple=32 |
| Варианты LoRA (каждый промпт × каждый вариант) | 11292 | SaveImage | V1 · результат | on | filename_prefix=ND_MMH3_v1 |
| Варианты LoRA (каждый промпт × каждый вариант) | 11330 | ⟦New Subgraph⟧ | Движок V1 (+ общая модель для всех колонок) | on | megapixels=2; string=; switch=true; values.a=22; width=1376; height=768; source_fidelity=0.05; noise_seed=42; steps=6 |
| Варианты LoRA (каждый промпт × каждый вариант) | 11332 | Power Lora Loader (rgthree) | V1 · без LoRA | on | ➕ Add Lora= |
| Варианты LoRA (каждый промпт × каждый вариант) | 11333 | Power Lora Loader (rgthree) | V2 · Stacy | on | ➕ Add Lora= |
| Варианты LoRA (каждый промпт × каждый вариант) | 11334 | ⟦New Subgraph⟧ | Движок V2 | BYPASS | megapixels=2; string=; switch=true; values.a=22; width=1376; height=768; source_fidelity=0.05; noise_seed=42; steps=6 |
| Варианты LoRA (каждый промпт × каждый вариант) | 11335 | SaveImage | V2 · результат | BYPASS | filename_prefix=ND_MMH3_v2 |
| Варианты LoRA (каждый промпт × каждый вариант) | 11336 | Power Lora Loader (rgthree) | V3 · Stacy + NSFW AIO | on | ➕ Add Lora= |
| Варианты LoRA (каждый промпт × каждый вариант) | 11337 | ⟦New Subgraph⟧ | Движок V3 | on | megapixels=2; string=; switch=true; values.a=22; width=1376; height=768; source_fidelity=0.05; noise_seed=42; steps=6 |
| Варианты LoRA (каждый промпт × каждый вариант) | 11338 | SaveImage | V3 · результат | on | filename_prefix=ND_MMH3_v3 |
| Варианты LoRA (каждый промпт × каждый вариант) | 11339 | Power Lora Loader (rgthree) | V4 · Stacy + NSFW AIO + motion + penis | on | ➕ Add Lora= |
| Варианты LoRA (каждый промпт × каждый вариант) | 11340 | ⟦New Subgraph⟧ | Движок V4 | on | megapixels=2; string=; switch=true; values.a=22; width=1376; height=768; source_fidelity=0.05; noise_seed=42; steps=6 |
| Варианты LoRA (каждый промпт × каждый вариант) | 11341 | SaveImage | V4 · результат | on | filename_prefix=ND_MMH3_v4 |
| — | 11270 | Note |  | on |  |
| — | 11331 | Note |  | on |  |
| — | 11342 | Note |  | on |  |
| — | 11343 | easy boolean | Ручная маска если лицо автоматом не нашлось | on | value=false |
| — | 11344 | LoadImage | Исходник  (для ручной маски: ПКМ → Open in MaskEditor → закрась лицо) | on | image=clipspace-painted-masked-1790907713660.p… |
| — | 11349 | Fast Groups Bypasser (rgthree) |  | on |  |
| — | 11350 | Note | Triggerwords | on |  |
| — | 11352 | Fast Groups Muter (rgthree) |  | on |  |
| — | 11353 | Fast Groups Muter (rgthree) |  | on |  |
| — | 11366 | Note | Triggerwords | on |  |
| — | 11367 | Note | Triggerwords | on |  |

## сабграф «New Subgraph» (4b3acdbc, экземпляров: 4)

Группы: «📝 Промпт и исходник», «🎲 Сэмплинг», «🔗 Референсы», «🧩 Подготовка латента», «🖨️ Декод и сборка»

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| — | 10323 | DiffusionModelLoaderKJ | Model: 10Eros Max (тумблер ON) | on | model_name=10Eros_Max_h3_TURBO-hybrid_beta4_int8_co…; weight_dtype=default; compute_dtype=default; patch_cublaslinear=false; sage_attention=disabled; enable_fp16_accumulation=false |
| — | 11278 | DiffusionModelLoaderKJ | Model: MiniMax H3 fl2va (тумблер OFF) | on | model_name=minimax_h3_fl2va_pruned_int8_convrot.saf…; weight_dtype=default; compute_dtype=default; patch_cublaslinear=false; sage_attention=disabled; enable_fp16_accumulation=false |
| — | 11279 | ComfySwitchNode | 10ErosMax ON → Eros | OFF → fl2va | on | switch=true |
| — | 11302 | MiniMaxH3SigmaShift |  | on | shift_video=12; shift_audio=3 |
| — | 11303 | CLIPLoader |  | on | clip_name=qwen3vl_32b_h3_ultra_uncensored_heretic_…; type=minimax; device=default |
| — | 11304 | VAELoader | Load VAE Video | on | vae_name=minimax_h3_video_vae_fp16.safetensors |
| 🎲 Сэмплинг | 11308 | ComfyMathExpression | Start Step Math | on | expression=a - b |
| 🎲 Сэмплинг | 11313 | KSamplerAdvanced | Sampler | First Pass | on | add_noise=enable; noise_seed=42; control_after_generate=fixed; steps=6; cfg=1; sampler_name=euler; scheduler=simple; start_at_step=0; end_at_step=10000; return_with_leftover_noise=disable |
| 🎲 Сэмплинг | 11314 | easy cleanGpuUsed |  | on |  |
| 🎲 Сэмплинг | 11315 | KSamplerAdvanced | Sampler | Second  Pass | on | add_noise=enable; noise_seed=42; control_after_generate=fixed; steps=22; cfg=1; sampler_name=euler; scheduler=simple; start_at_step=16; end_at_step=10000; return_with_leftover_noise=disable |
| 📝 Промпт и исходник | 11305 | ImageScaleToTotalPixels |  | on | upscale_method=lanczos; megapixels=2; resolution_steps=32 |
| 📝 Промпт и исходник | 11307 | RegexReplace | Image N → <Picture N> | on | string=; regex_pattern=\bimage\s*([1-9])\b; replace=<Picture \1>; case_insensitive=true; multiline=false; dotall=false; count=0 |
| 🔗 Референсы | 11309 | MiniMaxH3PromptBuilder |  | on | prompt_text=<текст, 148 симв.>; builder_state=<текст, 265 симв.>; Edit prompt…=null; + Media loader=null; mmh3_summary= |
| 🔗 Референсы | 11310 | MiniMaxH3ReferenceSplitter |  | on |  |
| 🖨️ Декод и сборка | 11316 | H3ImageDecode |  | on |  |
| 🖨️ Декод и сборка | 11317 | H3ImageFrameSelector |  | on | strategy=decode_recommended; manual_index=0; skip_first_frames=0; candidate_start=0; candidate_end=1; similarity_weight=0.25; top_k=4; emit_candidate_batch=false |
| 🧩 Подготовка латента | 11311 | H3ReferenceEditPrepare |  | on | edit_instruction=; width=1376; height=768; quality_profile=recommended ¦ 5 frames; source_fidelity=0.05; source_fit=contain_pad; reference_detail=max_identity_2048; optimize_for_still=true |
| 🧩 Подготовка латента | 11312 | ConditioningZeroOut |  | on |  |

## сабграф «New Subgraph» (c723a27b, экземпляров: 1)

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| — | 11068 | ConditioningZeroOut |  | on |  |
| — | 11069 | ModelSamplingAuraFlow |  | on | shift=3 |
| — | 11070 | VAEEncodeTiled |  | on | tile_size=4096; overlap=64; temporal_size=64; temporal_overlap=8 |
| — | 11071 | ColorMatch | Подогнать цвет B под A | on | method=mkl; strength=1; multithread=true |
| — | 11072 | DyPE_FLUX |  | on | width=2048; height=2048; model_type=auto; method=vision_yarn; yarn_alt_scaling=true; enable_dype=true; base_resolution=1024; dype_start_sigma=1; dype_scale=2; dype_exponent=2; base_shift=0.5; max_shift=1.15 |
| — | 11073 | GetImageSize | размер A | on |  |
| — | 11074 | MaskComposite | Маска A + маска B (объединение) | on | x=0; y=0; operation=add |
| — | 11075 | ImageCompositeMasked | Подставить персонажа из B в A | on | x=0; y=0; resize_source=false |
| — | 11076 | GrowMaskWithBlur | Растушёвка края | on | expand=10; incremental_expandrate=0; tapered_corners=true; flip_input=false; blur_radius=6; lerp_alpha=1; decay_factor=1; fill_holes=true |
| — | 11077 | CLIPTextEncode |  | on | text=Stacy_v1 |
| — | 11078 | VAEDecodeTiled |  | on | tile_size=4096; overlap=64; temporal_size=64; temporal_overlap=8 |
| — | 11079 | Reroute |  | on |  |
| — | 11080 | SAM3Segment |  | on | prompt=woman; output_mode=Merged; confidence_threshold=0.5; max_segments=1; segment_pick=0; mask_blur=0; mask_offset=0; device=Auto; invert_output=false; unload_model=false; background=Alpha; background_color=#222222 |
| — | 11081 | ImageScale | B -> размер A | on | upscale_method=lanczos; width=1024; height=1024; crop=disabled |
| — | 11082 | SAM3Segment |  | on | prompt=woman; output_mode=Merged; confidence_threshold=0.5; max_segments=1; segment_pick=0; mask_blur=0; mask_offset=0; device=Auto; invert_output=false; unload_model=false; background=Alpha; background_color=#222222 |
| — | 11083 | InpaintStitchImproved | Сшить обратно (после модели) | on |  |
| — | 11084 | SAM3Segment |  | on | prompt=woman; output_mode=Merged; confidence_threshold=0.5; max_segments=1; segment_pick=0; mask_blur=0; mask_offset=0; device=Auto; invert_output=false; unload_model=false; background=Alpha; background_color=#222222 |
| — | 11085 | GrowMaskWithBlur | Расширить + смягчить + залить дыры | on | expand=0; incremental_expandrate=0; tapered_corners=true; flip_input=false; blur_radius=8; lerp_alpha=1; decay_factor=1; fill_holes=true |
| — | 11086 | KSampler |  | on | seed=685683109897719; control_after_generate=randomize; steps=9; cfg=1; sampler_name=euler; scheduler=simple; denoise=0.8 |
| — | 11087 | CLIPLoader |  | on | clip_name=qwen_3_4b.safetensors; type=lumina2; device=default |
| — | 11088 | VAELoader |  | on | vae_name=flux_ultra_vae.safetensors.safetensors |
| — | 11089 | UNETLoader |  | on | unet_name=z_image_turbo_bf16.safetensors; weight_dtype=default |
| — | 11090 | InpaintCropImproved | Кроп по маске -> 1024 | on | downscale_algorithm=lanczos; upscale_algorithm=lanczos; preresize=false; preresize_mode=ensure minimum and maximum resolution; preresize_min_width=1024; preresize_min_height=1024; preresize_max_width=1024; preresize_max_height=1024; mask_fill_holes=true; mask_expand_pixels=0; mask_invert=false; mask_blend_pixels=40; mask_hipass_filter=0; extend_for_outpainting=false; extend_up_factor=1; extend_dow… |
| — | 11091 | LoraLoaderModelOnly |  | on | lora_name=Stacy_v1_z-image-base-prose_3500.safeten…; strength_model=1.5 |

## сабграф «New Subgraph» (0cb1226f, экземпляров: 1)

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| — | 11130 | MaskToImage | Маска детектора -> картинка | on |  |
| — | 11131 | CLIPLoader |  | on | clip_name=qwen_3_4b.safetensors; type=lumina2; device=default |
| — | 11132 | VAELoader |  | on | vae_name=ae.safetensors |
| — | 11133 | BboxDetectorCombined_v2 |  | on | threshold=0.25; dilation=10 |
| — | 11134 | VAEDecode |  | on |  |
| — | 11135 | ImageBlend |  | on | blend_factor=0.7; blend_mode=normal |
| — | 11136 | VAEEncode |  | on |  |
| — | 11137 | UltralyticsDetectorProvider |  | on | model_name=bbox/face_yolov8m.pt |
| — | 11138 | InpaintStitchImproved |  | on |  |
| — | 11139 | ColorMatchV2 |  | on | method=mkl; strength=0.75; multithread=true |
| — | 11140 | CLIPTextEncode |  | on | text=Stacy_v1, brown eyes |
| — | 11141 | Note | Подсказка: лицо | on |  |
| — | 11142 | KSampler | Face pass — KSampler | on | seed=993866159869322; control_after_generate=randomize; steps=12; cfg=1; sampler_name=euler; scheduler=simple; denoise=0.8 |
| — | 11143 | InpaintCropImproved |  | on | downscale_algorithm=lanczos; upscale_algorithm=lanczos; preresize=false; preresize_mode=ensure minimum and maximum resolution; preresize_min_width=1024; preresize_min_height=1024; preresize_max_width=1024; preresize_max_height=1024; mask_fill_holes=true; mask_expand_pixels=0; mask_invert=false; mask_blend_pixels=64; mask_hipass_filter=0; extend_for_outpainting=false; extend_up_factor=1; extend_dow… |
| — | 11144 | easy int |  | on | value=1328 |
| — | 11145 | UNETLoader |  | on | unet_name=z_image_turbo_bf16.safetensors; weight_dtype=default |
| — | 11146 | LoraLoaderModelOnly |  | on | lora_name=Stacy_v1_z-image-base-prose_3500.safeten…; strength_model=1.2 |
| — | 11147 | ComfySwitchNode | Маска: OFF = авто (YOLO face) | ON = ручная | on | switch=false |

## сабграф «New Subgraph» (0dddbb3b, экземпляров: 1)

Группы: «🖨️ Декод → Color Match → вклейка», «🎲 Подготовка и img2img сэмплинг», «✂️ Маска (авто SAM3 / ручная) и кроп лица», «🧠 Модели (LoRA подключается снаружи)»

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| ✂️ Маска (авто SAM3 / ручная) и кроп лица | 11110 | Get resolution [Crystools] |  | on |  |
| ✂️ Маска (авто SAM3 / ручная) и кроп лица | 11111 | CR Color Panel |  | on | panel_width=512; panel_height=512; fill_color=white; fill_color_hex=#000000 |
| ✂️ Маска (авто SAM3 / ручная) и кроп лица | 11112 | ImageToMask |  | on | channel=red |
| ✂️ Маска (авто SAM3 / ручная) и кроп лица | 11113 | ImpactConditionalBranch | маска пустая → весь кадр | есть → лицо | on | cond=false |
| ✂️ Маска (авто SAM3 / ручная) и кроп лица | 11114 | InpaintCropImproved | Кроп лица из ОРИГИНАЛА (без ресайза кадра) | on | downscale_algorithm=lanczos; upscale_algorithm=lanczos; preresize=false; preresize_mode=ensure minimum and maximum resolution; preresize_min_width=1024; preresize_min_height=1024; preresize_max_width=1024; preresize_max_height=1024; mask_fill_holes=true; mask_expand_pixels=0; mask_invert=false; mask_blend_pixels=40; mask_hipass_filter=0; extend_for_outpainting=false; extend_up_factor=1; extend_dow… |
| ✂️ Маска (авто SAM3 / ручная) и кроп лица | 11115 | ImageScaleToTotalPixels | Кроп → Megapixels (только для генерации) | on | upscale_method=lanczos; megapixels=2; resolution_steps=32 |
| ✂️ Маска (авто SAM3 / ручная) и кроп лица | 11127 | GrowMask | Расширить маску лица | on | expand=12; tapered_corners=true |
| ✂️ Маска (авто SAM3 / ручная) и кроп лица | 11128 | SAM3Segment | Авто-детект лица (SAM3) | on | prompt=face; output_mode=Merged; confidence_threshold=0.5; max_segments=1; segment_pick=0; mask_blur=0; mask_offset=0; device=Auto; invert_output=false; unload_model=false; background=Alpha; background_color=#222222 |
| ✂️ Маска (авто SAM3 / ручная) и кроп лица | 11129 | ComfySwitchNode | Маска: OFF = авто (SAM3) | ON = ручная | on | switch=false |
| 🎲 Подготовка и img2img сэмплинг | 11116 | GetImageSize |  | on |  |
| 🎲 Подготовка и img2img сэмплинг | 11117 | H3TextToImagePrepare | H3 Text to Image prepare (только промпт + LoRA) | on | prompt=; width=1344; height=768; quality_profile=recommended ¦ 5 frames; optimize_for_still=true |
| 🎲 Подготовка и img2img сэмплинг | 11118 | H3LatentFromImage | H3 img2img: латент из кропа + расписание денойза | on | steps=10; scheduler=simple; denoise=0.55 |
| 🎲 Подготовка и img2img сэмплинг | 11119 | RandomNoise |  | on | noise_seed=42; control_after_generate=fixed |
| 🎲 Подготовка и img2img сэмплинг | 11120 | BasicGuider |  | on |  |
| 🎲 Подготовка и img2img сэмплинг | 11121 | KSamplerSelect |  | on | sampler_name=euler |
| 🎲 Подготовка и img2img сэмплинг | 11122 | SamplerCustomAdvanced | Sampler | img2img | on |  |
| 🖨️ Декод → Color Match → вклейка | 11123 | H3ImageDecode |  | on |  |
| 🖨️ Декод → Color Match → вклейка | 11124 | H3ImageFrameSelector |  | on | strategy=manual_index; manual_index=2; skip_first_frames=0; candidate_start=0; candidate_end=1; similarity_weight=0.25; top_k=4; emit_candidate_batch=false |
| 🖨️ Декод → Color Match → вклейка | 11125 | InpaintStitchImproved | Вклейка обратно в оригинал (исходный размер) | on |  |
| 🖨️ Декод → Color Match → вклейка | 11126 | ColorMatch | Color Match: сгенерированное лицо → исходный кроп | on | method=mkl; strength=0.8; multithread=true |
| 🧠 Модели (LoRA подключается снаружи) | 11105 | DiffusionModelLoaderKJ |  | on | model_name=minimax_h3_fl2va_pruned_int8_convrot.saf…; weight_dtype=default; compute_dtype=default; patch_cublaslinear=false; sage_attention=disabled; enable_fp16_accumulation=false |
| 🧠 Модели (LoRA подключается снаружи) | 11106 | CLIPLoader |  | on | clip_name=qwen3vl_32b_h3_ultra_uncensored_heretic_…; type=minimax; device=default |
| 🧠 Модели (LoRA подключается снаружи) | 11107 | VAELoader | Load VAE Video | on | vae_name=minimax_h3_video_vae_fp16.safetensors |
| 🧠 Модели (LoRA подключается снаружи) | 11108 | MiniMaxH3SigmaShift |  | on | shift_video=12; shift_audio=3 |
| 🧠 Модели (LoRA подключается снаружи) | 11109 | easy isMaskEmpty | маска пустая? | on |  |

## Модели и LoRA

| граф | id | тип | title | виджет | файл | mode |
|---|---|---|---|---|---|---|
| верхний уровень | 11362 | LoraLoaderModelOnly |  | lora_name | stacy_h3_face_v1.safetensors | on |
| верхний уровень | 11368 | ModelPreviewOverrideKJ | MiniMax Sampling Preview | tiny_vae | taeh3.safetensors | on |
| верхний уровень | 11291 | ModelPreviewOverrideKJ | MiniMax Sampling Preview | tiny_vae | taeh3.safetensors | on |
| верхний уровень | 11332 | Power Lora Loader (rgthree) | V1 · без LoRA | lora_1 | stacy_h3_A_stills_000001750.safetensors @1 on=False | on |
| верхний уровень | 11332 | Power Lora Loader (rgthree) | V1 · без LoRA | lora_2 | hmnsfwAIOV25.C2Qs.safetensors @0.5 on=False | on |
| верхний уровень | 11332 | Power Lora Loader (rgthree) | V1 · без LoRA | lora_3 | hmmotion.nsfwAIOV2.minimaxh3.wCeX.safetensors @0.5 on=False | on |
| верхний уровень | 11332 | Power Lora Loader (rgthree) | V1 · без LoRA | lora_4 | penisv2MinimaxH3.HbNF.safetensors @0.6 on=False | on |
| верхний уровень | 11333 | Power Lora Loader (rgthree) | V2 · Stacy | lora_1 | stacy_h3_A_stills_000001750.safetensors @1 on=False | on |
| верхний уровень | 11333 | Power Lora Loader (rgthree) | V2 · Stacy | lora_2 | hmnsfwAIOV25.C2Qs.safetensors @0.5 on=False | on |
| верхний уровень | 11333 | Power Lora Loader (rgthree) | V2 · Stacy | lora_3 | hmmotion.nsfwAIOV2.minimaxh3.wCeX.safetensors @0.5 on=False | on |
| верхний уровень | 11333 | Power Lora Loader (rgthree) | V2 · Stacy | lora_4 | penisv2MinimaxH3.HbNF.safetensors @0.6 on=False | on |
| верхний уровень | 11336 | Power Lora Loader (rgthree) | V3 · Stacy + NSFW AIO | lora_1 | stacy_h3_A_stills_000001750.safetensors @1 on=False | on |
| верхний уровень | 11336 | Power Lora Loader (rgthree) | V3 · Stacy + NSFW AIO | lora_2 | hmnsfwAIOV25.C2Qs.safetensors @0.5 on=True | on |
| верхний уровень | 11336 | Power Lora Loader (rgthree) | V3 · Stacy + NSFW AIO | lora_3 | hmmotion.nsfwAIOV2.minimaxh3.wCeX.safetensors @0.5 on=False | on |
| верхний уровень | 11336 | Power Lora Loader (rgthree) | V3 · Stacy + NSFW AIO | lora_4 | penisv2MinimaxH3.HbNF.safetensors @0.6 on=False | on |
| верхний уровень | 11339 | Power Lora Loader (rgthree) | V4 · Stacy + NSFW AIO + motion + penis | lora_1 | stacy_h3_A_stills_000001750.safetensors @1 on=False | on |
| верхний уровень | 11339 | Power Lora Loader (rgthree) | V4 · Stacy + NSFW AIO + motion + penis | lora_2 | hmnsfwAIOV25.C2Qs.safetensors @0.5 on=True | on |
| верхний уровень | 11339 | Power Lora Loader (rgthree) | V4 · Stacy + NSFW AIO + motion + penis | lora_3 | hmmotion.nsfwAIOV2.minimaxh3.wCeX.safetensors @0.5 on=True | on |
| верхний уровень | 11339 | Power Lora Loader (rgthree) | V4 · Stacy + NSFW AIO + motion + penis | lora_4 | penisv2MinimaxH3.HbNF.safetensors @0.6 on=True | on |
| сабграф «New Subgraph» (4b3acdbc, экземпляров: 4) | 10323 | DiffusionModelLoaderKJ | Model: 10Eros Max (тумблер ON) | model_name | 10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors | on |
| сабграф «New Subgraph» (4b3acdbc, экземпляров: 4) | 11278 | DiffusionModelLoaderKJ | Model: MiniMax H3 fl2va (тумблер OFF) | model_name | minimax_h3_fl2va_pruned_int8_convrot.safetensors | on |
| сабграф «New Subgraph» (4b3acdbc, экземпляров: 4) | 11303 | CLIPLoader |  | clip_name | qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors | on |
| сабграф «New Subgraph» (4b3acdbc, экземпляров: 4) | 11304 | VAELoader | Load VAE Video | vae_name | minimax_h3_video_vae_fp16.safetensors | on |
| сабграф «New Subgraph» (c723a27b, экземпляров: 1) | 11087 | CLIPLoader |  | clip_name | qwen_3_4b.safetensors | on |
| сабграф «New Subgraph» (c723a27b, экземпляров: 1) | 11088 | VAELoader |  | vae_name | flux_ultra_vae.safetensors.safetensors | on |
| сабграф «New Subgraph» (c723a27b, экземпляров: 1) | 11089 | UNETLoader |  | unet_name | z_image_turbo_bf16.safetensors | on |
| сабграф «New Subgraph» (c723a27b, экземпляров: 1) | 11091 | LoraLoaderModelOnly |  | lora_name | Stacy_v1_z-image-base-prose_3500.safetensors | on |
| сабграф «New Subgraph» (0cb1226f, экземпляров: 1) | 11131 | CLIPLoader |  | clip_name | qwen_3_4b.safetensors | on |
| сабграф «New Subgraph» (0cb1226f, экземпляров: 1) | 11132 | VAELoader |  | vae_name | ae.safetensors | on |
| сабграф «New Subgraph» (0cb1226f, экземпляров: 1) | 11137 | UltralyticsDetectorProvider |  | model_name | bbox/face_yolov8m.pt | on |
| сабграф «New Subgraph» (0cb1226f, экземпляров: 1) | 11145 | UNETLoader |  | unet_name | z_image_turbo_bf16.safetensors | on |
| сабграф «New Subgraph» (0cb1226f, экземпляров: 1) | 11146 | LoraLoaderModelOnly |  | lora_name | Stacy_v1_z-image-base-prose_3500.safetensors | on |
| сабграф «New Subgraph» (0dddbb3b, экземпляров: 1) | 11105 | DiffusionModelLoaderKJ |  | model_name | minimax_h3_fl2va_pruned_int8_convrot.safetensors | on |
| сабграф «New Subgraph» (0dddbb3b, экземпляров: 1) | 11106 | CLIPLoader |  | clip_name | qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors | on |
| сабграф «New Subgraph» (0dddbb3b, экземпляров: 1) | 11107 | VAELoader | Load VAE Video | vae_name | minimax_h3_video_vae_fp16.safetensors | on |

## Пакеты нод

| пакет | лицензия | ноды |
|---|---|---|
| comfy-core | GPL-3.0 (ComfyUI) | 0cb1226f-ea56-45c8-b1b2-190da28b5f30, 0dddbb3b-ec56-447a-9030-5d4f858973b4, 4b3acdbc-009e-4fbe-9722-4b132fbf993f, BasicGuider, CLIPLoader, CLIPTextEncode, ComfyMathExpression, ComfySwitchNode, ConditioningZeroOut, GetImageSize, GrowMask, ImageBlend, ImageCompositeMasked, ImageScale, ImageScaleToTotalPixels, ImageToMask, KSampler, KSamplerAdvanced, KSamplerSelect, LoadImage, LoraLoaderModelOnly, MaskComposite, MaskToImage, MiniMaxH3SigmaShift, ModelSamplingAuraFlow, PreviewImage, PrimitiveBoolean, PrimitiveFloat, RandomNoise, RegexReplace, ResolutionSelector, SamplerCustomAdvanced, SaveImage, SeedNode, UNETLoader, VAEDecode, VAEDecodeTiled, VAEEncode, VAEEncodeTiled, VAELoader, c723a27b-4cc3-41da-90b1-1643d57dfa89 |
| ComfyUI-Crystools | MIT | Get resolution [Crystools] |
| ComfyUI-DyPE | Apache-2.0 | DyPE_FLUX |
| comfyui-easy-use | GPL-3.0 | easy boolean, easy cleanGpuUsed, easy int, easy isMaskEmpty |
| comfyui-impact-pack | GPL-3.0 | BboxDetectorCombined_v2, ImpactConditionalBranch |
| comfyui-impact-subpack | ⚠ AGPL-3.0 (+ ultralytics AGPL) | UltralyticsDetectorProvider |
| comfyui-inpaint-cropandstitch | GPL-3.0 | InpaintCropImproved, InpaintStitchImproved |
| comfyui-kjnodes | GPL-3.0 | ColorMatch, ColorMatchV2, DiffusionModelLoaderKJ, GrowMaskWithBlur, ModelPreviewOverrideKJ |
| comfyui-mxtoolkit | GPL-3.0 | mxSlider |
| comfyui-rmbg | GPL-3.0 (код); SAM3 веса — SAM License Meta, проверить | SAM3Segment |
| comfyui-simple-prompt-batcher | ⚠ нет LICENSE | SimplePromptBatcher |
| ComfyUI_Comfyroll_CustomNodes | ⚠ нет LICENSE | CR Color Panel |
| fantastic-minimax-h3-prompt-builder | MIT | MiniMaxH3PromptBuilder, MiniMaxH3ReferenceSplitter |
| islomsalimov030-maker/h3_img2img_latent | ⚠ нет LICENSE (свой репо — добавить MIT) | H3LatentFromImage |
| minimax-h3-image-studio | Unlicense | H3ImageDecode, H3ImageFrameSelector, H3ReferenceEditPrepare, H3TextToImagePrepare |
| rgthree-comfy | MIT | Image Comparer (rgthree), Power Lora Loader (rgthree) |
| was-ns | MIT | Text Multiline |

## Узлы без потребителей (кроме выходных/UI)

- сабграф «New Subgraph» (0cb1226f, экземпляров: 1): #11130 `MaskToImage` «Маска детектора -> картинка»
- сабграф «New Subgraph» (0cb1226f, экземпляров: 1): #11139 `ColorMatchV2` «»
