# Инвентарь: `frame_h3@1.ui.json`

Сгенерировано `tools/wf_inventory.py`. Узлов верхнего уровня: 37, сабграфов: 3, frontend 1.53.6.

## верхний уровень

Группы: «1 · Вход: плита, лист персонажа, одежда, поза, промпты», «2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки», «3 · Движки (не трогать)», «4 · Результат»

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| 1 · Вход: плита, лист персонажа, одежда, поза, промпты | 55 | LoadImage | IN plate · плита / исходник <Picture 1> | on | image=plate.png |
| 1 · Вход: плита, лист персонажа, одежда, поза, промпты | 56 | LoadImage | IN sheet · лист персонажа <Picture 2> | on | image=stacy_sheet.png |
| 1 · Вход: плита, лист персонажа, одежда, поза, промпты | 57 | LoadImage | IN outfit · одежда <Picture 3> (необязательно) | on | image=outfit.png |
| 1 · Вход: плита, лист персонажа, одежда, поза, промпты | 58 | LoadImage | IN pose · поза <Picture 4> (необязательно) | on | image=pose.png |
| 1 · Вход: плита, лист персонажа, одежда, поза, промпты | 59 | PrimitiveStringMultiline | IN prompt · промпт кадра (ref-edit) | on | value= |
| 1 · Вход: плита, лист персонажа, одежда, поза, промпты | 60 | PrimitiveStringMultiline | IN face_prompt_z · промпт лица Z-image (Stacy_v1) | on | value=<текст, 48 симв.> |
| 1 · Вход: плита, лист персонажа, одежда, поза, промпты | 61 | PrimitiveStringMultiline | IN face_prompt_h3 · промпт лица H3 (St4cy_H3) | on | value=<текст, 161 симв.> |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 62 | DiffusionModelLoaderKJ | ARM model · модель H3 | on | model_name=10Eros_Max_h3_TURBO-hybrid_beta4_int8_co…; weight_dtype=default; compute_dtype=default; patch_cublaslinear=false; sage_attention=disabled; enable_fp16_accumulation=false |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 63 | Power Lora Loader (rgthree) | ARM loras · LoRA руки | on | ➕ Add Lora= |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 64 | ModelPreviewOverrideKJ | ПРЕВЬЮ сэмплинга | on | max_resolution=1024; jpeg_quality=80; suppress_default_preview=true; preview_frames=1; preview_fps=1; tiny_vae=taeh3.safetensors; preview= |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 65 | PrimitiveInt | KNOB seed | on | value=42; control_after_generate=fixed |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 66 | PrimitiveFloat | KNOB megapixels | on | value=2.0 |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 67 | PrimitiveFloat | KNOB ref_identity | on | value=1.0 |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 68 | PrimitiveInt | KNOB pass1_steps | on | value=6; control_after_generate=fixed |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 69 | PrimitiveInt | KNOB pass2_total | on | value=40; control_after_generate=fixed |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 70 | PrimitiveInt | KNOB pass2_steps | on | value=8; control_after_generate=fixed |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 71 | PrimitiveFloat | KNOB shift | on | value=12.0 |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 72 | ResolutionSelector | разрешение (16:9) | on | aspect_ratio=16:9 (Widescreen); megapixels=4; multiple=32 |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 73 | PrimitiveInt | KNOB face_seed | on | value=953552952704593; control_after_generate=fixed |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 74 | PrimitiveBoolean | KNOB face_z | on | value=true |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 75 | PrimitiveFloat | KNOB face_z_denoise | on | value=0.6 |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 76 | PrimitiveFloat | KNOB face_z_lora | on | value=1.2 |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 77 | PrimitiveInt | KNOB face_z_size | on | value=1328; control_after_generate=fixed |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 78 | PrimitiveBoolean | KNOB face_h3 | on | value=true |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 79 | PrimitiveFloat | KNOB face_h3_denoise | on | value=0.45 |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 80 | PrimitiveFloat | KNOB face_h3_lora | on | value=1.0 |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 81 | PrimitiveFloat | KNOB face_h3_color | on | value=0.95 |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 82 | PrimitiveInt | KNOB face_h3_max_crop | on | value=1024; control_after_generate=fixed |
| 2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки | 83 | PrimitiveFloat | KNOB face_h3_megapixels | on | value=4.0 |
| 3 · Движки (не трогать) | 84 | ⟦ДВИЖОК кадра: H3 Reference Edit (не трогать)⟧ | Движок кадра (не трогать) | on |  |
| 3 · Движки (не трогать) | 85 | ⟦ДВИЖОК лица Z-image (не трогать)⟧ | Лицо Z-image (не трогать) | on |  |
| 3 · Движки (не трогать) | 86 | ⟦ДВИЖОК лица H3 (не трогать)⟧ | Лицо H3 (не трогать) | on |  |
| 4 · Результат | 87 | SaveImage | OUT raw · кадр до доработки лица | on | filename_prefix=farm/frame_raw |
| 4 · Результат | 88 | SaveImage | OUT frame · итоговый кадр | on | filename_prefix=farm/frame |
| 4 · Результат | 89 | Image Comparer (rgthree) | сравнение: до / после лица | on |  |
| — | 90 | MarkdownNote | README | on |  |
| — | 91 | MarkdownNote | Шпаргалка | on |  |

## сабграф «ДВИЖОК кадра: H3 Reference Edit (не трогать)» (eca45c2d, экземпляров: 1)

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| — | 1 | CLIPLoader | gen · Qwen3-VL | on | clip_name=qwen3vl_32b_h3_ultra_uncensored_heretic_…; type=minimax; device=default |
| — | 2 | VAELoader | gen · H3 video VAE | on | vae_name=minimax_h3_video_vae_fp16.safetensors |
| — | 3 | MiniMaxH3SigmaShift | gen · shift | on | shift_video=12; shift_audio=3 |
| — | 4 | ImageScaleToTotalPixels | gen · плита → мегапиксели | on | upscale_method=lanczos; megapixels=2; resolution_steps=32 |
| — | 5 | RegexReplace | gen · Image N → <Picture N> | on | string=; regex_pattern=\bimage\s*([1-9])\b; replace=<Picture \1>; case_insensitive=true; multiline=false; dotall=false; count=0 |
| — | 6 | H3ReferenceEditPrepare | gen · H3ReferenceEditPrepare | on | edit_instruction=; width=1376; height=768; quality_profile=recommended ¦ 5 frames; source_fidelity=0.05; source_fit=contain_pad; reference_detail=max_identity_2048; optimize_for_still=true |
| — | 7 | ConditioningZeroOut | gen · пустой negative | on |  |
| — | 8 | ComfyMathExpression | gen · старт 2-го прохода = total - steps | on | expression=a - b |
| — | 9 | KSamplerAdvanced | gen · sampler pass 1 | on | add_noise=enable; noise_seed=42; control_after_generate=fixed; steps=6; cfg=1; sampler_name=euler; scheduler=simple; start_at_step=0; end_at_step=10000; return_with_leftover_noise=disable |
| — | 10 | KSamplerAdvanced | gen · sampler pass 2 | on | add_noise=enable; noise_seed=42; control_after_generate=fixed; steps=22; cfg=1; sampler_name=euler; scheduler=simple; start_at_step=16; end_at_step=10000; return_with_leftover_noise=disable |
| — | 11 | H3ImageDecode | gen · decode | on |  |
| — | 12 | H3ImageFrameSelector | gen · выбор кадра | on | strategy=decode_recommended; manual_index=0; skip_first_frames=0; candidate_start=0; candidate_end=1; similarity_weight=0.25; top_k=4; emit_candidate_batch=false |

## сабграф «ДВИЖОК лица Z-image (не трогать)» (e0f9a455, экземпляров: 1)

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| — | 13 | UltralyticsDetectorProvider | faceZ · YOLO face | on | model_name=bbox/face_yolov8m.pt |
| — | 14 | BboxDetectorCombined_v2 | faceZ · маска лица | on | threshold=0.25; dilation=10 |
| — | 15 | easy isMaskEmpty | faceZ · лица нет? | on |  |
| — | 16 | LazySwitchKJ | faceZ · нет лица → без изменений | on | switch=false |
| — | 17 | LazySwitchKJ | faceZ · проход включён? | on | switch=false |
| — | 18 | InpaintCropImproved | faceZ · кроп лица | on | downscale_algorithm=lanczos; upscale_algorithm=lanczos; preresize=false; preresize_mode=ensure minimum and maximum resolution; preresize_min_width=1024; preresize_min_height=1024; preresize_max_width=1024; preresize_max_height=1024; mask_fill_holes=true; mask_expand_pixels=0; mask_invert=false; mask_blend_pixels=64; mask_hipass_filter=0; extend_for_outpainting=false; extend_up_factor=1; extend_dow… |
| — | 19 | UNETLoader | faceZ · Z-image turbo | on | unet_name=z_image_turbo_bf16.safetensors; weight_dtype=default |
| — | 20 | LoraLoaderModelOnly | faceZ · LoRA Stacy_v1 (Z) | on | lora_name=Stacy_v1_z-image-base-prose_3500.safeten…; strength_model=1.2 |
| — | 21 | CLIPLoader | faceZ · Qwen3-4B | on | clip_name=qwen_3_4b.safetensors; type=lumina2; device=default |
| — | 22 | VAELoader | faceZ · VAE ae | on | vae_name=ae.safetensors |
| — | 23 | CLIPTextEncode | faceZ · промпт | on | text=Stacy_v1, brown eyes |
| — | 24 | VAEEncode | faceZ · encode | on |  |
| — | 25 | KSampler | faceZ · sampler | on | seed=993866159869322; control_after_generate=randomize; steps=12; cfg=1; sampler_name=euler; scheduler=simple; denoise=0.8 |
| — | 26 | VAEDecode | faceZ · decode | on |  |
| — | 27 | ImageBlend | faceZ · смешать 70% нового | on | blend_factor=0.7; blend_mode=normal |
| — | 28 | InpaintStitchImproved | faceZ · вклейка | on |  |

## сабграф «ДВИЖОК лица H3 (не трогать)» (aa39cea7, экземпляров: 1)

| группа | id | тип | title | mode | виджеты |
|---|---|---|---|---|---|
| — | 29 | SAM3Segment | faceH3 · SAM3 face | on | prompt=face; output_mode=Merged; confidence_threshold=0.5; max_segments=1; segment_pick=0; mask_blur=0; mask_offset=0; device=Auto; invert_output=false; unload_model=false; background=Alpha; background_color=#222222 |
| — | 30 | GrowMask | faceH3 · расширить маску | on | expand=12; tapered_corners=true |
| — | 31 | easy isMaskEmpty | faceH3 · лица нет? | on |  |
| — | 32 | InpaintCropImproved | faceH3 · кроп лица (контекст 1.6) | on | downscale_algorithm=lanczos; upscale_algorithm=lanczos; preresize=false; preresize_mode=ensure minimum and maximum resolution; preresize_min_width=1024; preresize_min_height=1024; preresize_max_width=1024; preresize_max_height=1024; mask_fill_holes=true; mask_expand_pixels=0; mask_invert=false; mask_blend_pixels=40; mask_hipass_filter=0; extend_for_outpainting=false; extend_up_factor=1; extend_dow… |
| — | 33 | GetImageSize | faceH3 · размер кропа | on |  |
| — | 34 | ComfyMathExpression | faceH3 · кроп больше порога? | on | expression=a > b |
| — | 35 | LazySwitchKJ | faceH3 · крупное лицо → без изменений | on | switch=false |
| — | 36 | LazySwitchKJ | faceH3 · нет лица → без изменений | on | switch=false |
| — | 37 | LazySwitchKJ | faceH3 · проход включён? | on | switch=false |
| — | 38 | DiffusionModelLoaderKJ | faceH3 · H3 fl2va | on | model_name=minimax_h3_fl2va_pruned_int8_convrot.saf…; weight_dtype=default; compute_dtype=default; patch_cublaslinear=false; sage_attention=disabled; enable_fp16_accumulation=false |
| — | 39 | LoraLoaderModelOnly | faceH3 · LoRA лица St4cy_H3 | on | lora_name=stacy_h3_face_v1.safetensors; strength_model=1 |
| — | 40 | MiniMaxH3SigmaShift | faceH3 · shift | on | shift_video=12; shift_audio=3 |
| — | 41 | CLIPLoader | faceH3 · Qwen3-VL | on | clip_name=qwen3vl_32b_h3_ultra_uncensored_heretic_…; type=minimax; device=default |
| — | 42 | VAELoader | faceH3 · H3 video VAE | on | vae_name=minimax_h3_video_vae_fp16.safetensors |
| — | 43 | ImageScaleToTotalPixels | faceH3 · кроп → мегапиксели | on | upscale_method=lanczos; megapixels=2; resolution_steps=32 |
| — | 44 | GetImageSize | faceH3 · размер для генерации | on |  |
| — | 45 | H3TextToImagePrepare | faceH3 · промпт → латент | on | prompt=; width=1344; height=768; quality_profile=recommended ¦ 5 frames; optimize_for_still=true |
| — | 46 | H3LatentFromImage | faceH3 · img2img латент | on | steps=10; scheduler=simple; denoise=0.55 |
| — | 47 | RandomNoise | faceH3 · seed | on | noise_seed=42; control_after_generate=fixed |
| — | 48 | BasicGuider | faceH3 · guider | on |  |
| — | 49 | KSamplerSelect | faceH3 · euler | on | sampler_name=euler |
| — | 50 | SamplerCustomAdvanced | faceH3 · sampler | on |  |
| — | 51 | H3ImageDecode | faceH3 · decode | on |  |
| — | 52 | H3ImageFrameSelector | faceH3 · выбор кадра | on | strategy=manual_index; manual_index=2; skip_first_frames=0; candidate_start=0; candidate_end=1; similarity_weight=0.25; top_k=4; emit_candidate_batch=false |
| — | 53 | ColorMatch | faceH3 · цвет к исходному кропу | on | method=mkl; strength=0.8; multithread=true |
| — | 54 | InpaintStitchImproved | faceH3 · вклейка | on |  |

## Модели и LoRA

| граф | id | тип | title | виджет | файл | mode |
|---|---|---|---|---|---|---|
| верхний уровень | 62 | DiffusionModelLoaderKJ | ARM model · модель H3 | model_name | 10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors | on |
| верхний уровень | 63 | Power Lora Loader (rgthree) | ARM loras · LoRA руки | lora_1 | stacy_h3_A_stills_000001750.safetensors @1 on=False | on |
| верхний уровень | 64 | ModelPreviewOverrideKJ | ПРЕВЬЮ сэмплинга | tiny_vae | taeh3.safetensors | on |
| сабграф «ДВИЖОК кадра: H3 Reference Edit (не трогать)» (eca45c2d, экземпляров: 1) | 1 | CLIPLoader | gen · Qwen3-VL | clip_name | qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors | on |
| сабграф «ДВИЖОК кадра: H3 Reference Edit (не трогать)» (eca45c2d, экземпляров: 1) | 2 | VAELoader | gen · H3 video VAE | vae_name | minimax_h3_video_vae_fp16.safetensors | on |
| сабграф «ДВИЖОК лица Z-image (не трогать)» (e0f9a455, экземпляров: 1) | 13 | UltralyticsDetectorProvider | faceZ · YOLO face | model_name | bbox/face_yolov8m.pt | on |
| сабграф «ДВИЖОК лица Z-image (не трогать)» (e0f9a455, экземпляров: 1) | 19 | UNETLoader | faceZ · Z-image turbo | unet_name | z_image_turbo_bf16.safetensors | on |
| сабграф «ДВИЖОК лица Z-image (не трогать)» (e0f9a455, экземпляров: 1) | 20 | LoraLoaderModelOnly | faceZ · LoRA Stacy_v1 (Z) | lora_name | Stacy_v1_z-image-base-prose_3500.safetensors | on |
| сабграф «ДВИЖОК лица Z-image (не трогать)» (e0f9a455, экземпляров: 1) | 21 | CLIPLoader | faceZ · Qwen3-4B | clip_name | qwen_3_4b.safetensors | on |
| сабграф «ДВИЖОК лица Z-image (не трогать)» (e0f9a455, экземпляров: 1) | 22 | VAELoader | faceZ · VAE ae | vae_name | ae.safetensors | on |
| сабграф «ДВИЖОК лица H3 (не трогать)» (aa39cea7, экземпляров: 1) | 38 | DiffusionModelLoaderKJ | faceH3 · H3 fl2va | model_name | minimax_h3_fl2va_pruned_int8_convrot.safetensors | on |
| сабграф «ДВИЖОК лица H3 (не трогать)» (aa39cea7, экземпляров: 1) | 39 | LoraLoaderModelOnly | faceH3 · LoRA лица St4cy_H3 | lora_name | stacy_h3_face_v1.safetensors | on |
| сабграф «ДВИЖОК лица H3 (не трогать)» (aa39cea7, экземпляров: 1) | 41 | CLIPLoader | faceH3 · Qwen3-VL | clip_name | qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors | on |
| сабграф «ДВИЖОК лица H3 (не трогать)» (aa39cea7, экземпляров: 1) | 42 | VAELoader | faceH3 · H3 video VAE | vae_name | minimax_h3_video_vae_fp16.safetensors | on |

## Пакеты нод

| пакет | лицензия | ноды |
|---|---|---|
| comfy-core | GPL-3.0 (ComfyUI) | BasicGuider, CLIPLoader, CLIPTextEncode, ComfyMathExpression, ConditioningZeroOut, GetImageSize, GrowMask, ImageBlend, ImageScaleToTotalPixels, KSampler, KSamplerAdvanced, KSamplerSelect, LoadImage, LoraLoaderModelOnly, MiniMaxH3SigmaShift, PrimitiveBoolean, PrimitiveFloat, PrimitiveInt, PrimitiveStringMultiline, RandomNoise, RegexReplace, ResolutionSelector, SamplerCustomAdvanced, SaveImage, UNETLoader, VAEDecode, VAEEncode, VAELoader, aa39cea7-986c-4379-8556-007ad89a9903, e0f9a455-8552-4991-9448-3324f3e80e49, eca45c2d-e262-484f-841d-b62b17f9263f |
| comfyui-easy-use | GPL-3.0 | easy isMaskEmpty |
| comfyui-impact-pack | GPL-3.0 | BboxDetectorCombined_v2 |
| comfyui-impact-subpack | ⚠ AGPL-3.0 (+ ultralytics AGPL) | UltralyticsDetectorProvider |
| comfyui-inpaint-cropandstitch | GPL-3.0 | InpaintCropImproved, InpaintStitchImproved |
| comfyui-kjnodes | GPL-3.0 | ColorMatch, DiffusionModelLoaderKJ, LazySwitchKJ, ModelPreviewOverrideKJ |
| comfyui-rmbg | GPL-3.0 (код); SAM3 веса — SAM License Meta, проверить | SAM3Segment |
| islomsalimov030-maker/h3_img2img_latent | ⚠ нет LICENSE (свой репо — добавить MIT) | H3LatentFromImage |
| minimax-h3-image-studio | Unlicense | H3ImageDecode, H3ImageFrameSelector, H3ReferenceEditPrepare, H3TextToImagePrepare |
| rgthree-comfy | MIT | Image Comparer (rgthree), Power Lora Loader (rgthree) |

## Узлы без потребителей (кроме выходных/UI)

