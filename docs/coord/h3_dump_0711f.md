# H3 дамп шага 7 (07.11f) — по /history сервера студии

## V2_Q03_bed_kneel_full_s101
prompt_id `0fcd3ba9-7394-4057-ace6-d1cbf946af94`; разрешение 1344×768, length 5, ref_image_size match
Подано:
- <Picture 1> ← `ref_images.ref_image_0`: LoadImage `pose_V2_m3_full_rgb.png`
- <Picture 2> ← `ref_images.ref_image_1`: LoadImage `0003.png`
- <Video 1> ← `ref_videos.ref_video_0`: LoadImage `pose_V2_m3_full_s2/Q03_bed_kneel/playblast.png` → ImageScale → RepeatImageBatch
Сэмплер: `ks` KSamplerAdvanced noise_seed=101, steps=25, cfg=1.0, sampler_name=euler, scheduler=simple, start_at_step=0
Модель/шифт: `unet` DiffusionModelLoaderKJ {"model_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default", "compute_dtype": "default", "patch_cublaslinear": false, "sage_attention": "disabled", "enable_fp16_accumulation": false}; `shift` MiniMaxH3SigmaShift {"shift_video": 12.0, "shift_audio": 3.0}
Узлы: CLIPLoader, ConditioningZeroOut, DiffusionModelLoaderKJ, GrowMaskWithBlur, H3ImageDecode, H3ImageFrameSelector, ImageScale, KSamplerAdvanced, LoadImage, MaskToImage, MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift, RepeatImageBatch, SAM3Segment, SaveImage, VAELoader
Выходы: plate3d_poses/V2_Q03_bed_kneel_full/L_C2_s101_00003_.png, plate3d_poses/V2_Q03_bed_kneel_full/L_C2_mask_s101_00003_.png
Линтер (stage=frame, connected=['<Picture 1>', '<Picture 2>', '<Video 1>']): ['error:E_MUST_SINGLE_PHOTO', 'warn:W_SCALE_ANCHOR']
retention_analysis:
```
<Subject 1>: attribute_transfer - facial identity; wear the specified white T-shirt and light-blue jeans.
<Video 1>: partially_preserved - preserve geometry, pose, projected body scale and camera, replace only clay appearance.
<Picture 1>: fully_preserved - retain room materials, furnishings and lighting.
<Picture 2>: attribute_transfer - transfer facial identity to <Subject 1>.
```
Промпт целиком:
```
subject_definitions:
<Subject 1> is the adult woman whose identity is shown in <Picture 2>; replace the clay figure in <Video 1> with her.
<Picture 1> is the exact room composition, furnishing, materials and lighting of the target photograph.
<Picture 2> supplies only the facial identity and never appears as a separate image in the shot.
<Video 1> is a clay playblast defining the camera, body pose, projected size and precise position of <Subject 1>.
summary:
[reference generation] One single photorealistic static image. <Picture 2> is only an identity reference sheet, not a layout. Reproduce the composition and body pose of <Video 1>, using the real room from <Picture 1> and the facial identity from <Picture 2>.
retention_analysis:
<Subject 1>: attribute_transfer - facial identity; wear the specified white T-shirt and light-blue jeans.
<Video 1>: partially_preserved - preserve geometry, pose, projected body scale and camera, replace only clay appearance.
<Picture 1>: fully_preserved - retain room materials, furnishings and lighting.
<Picture 2>: attribute_transfer - transfer facial identity to <Subject 1>.
detailed_description:
A photorealistic live-action still. [Shot 1] filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, with no camera movement, no zoom and no pan. The room is a bedroom with lilac walls lit by low warm sunlight with window shadows, a low wide bed with beige and pink pillows, two wooden nightstands with glowing rattan lamps and a light woven floor. <Subject 1> is kneeling upright on the bed, sitting back on her heels, both hands resting on her thighs. She wears an opaque white short-sleeved cotton T-shirt, light-blue straight-leg jeans and white lace-up sneakers. The clothing is fully opaque and covers her torso and legs. The key light is the warm golden daylight of the room coming from the left of the frame: it lights the same side of her body and hair as it lights the furniture, the opposite side of her body falls into the same soft shade as the room, and her body casts a crisp shadow on the floor and furniture in the same direction as the shadows of the furniture in <Picture 1>. Preserve the exact pose and placement from <Video 1>. True-to-life scale: she is never enlarged to fill the frame. Scale anchor: the bed is wider than she is tall. No clay materials, no mannequin, no skeleton overlay. One continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen.
Timeline: From 0.00 to 0.17 seconds the subject holds the starting pose without moving.
The lighting is preserved unchanged throughout.
Colors and saturation stay unchanged from the reference image.
overall_soundscape:
Quiet room tone.
non_diegetic_music:
N/A
```

## V2_T03_armchair_side_sit_full_s101
prompt_id `c66c52a0-41b9-4895-8415-8f23bc11b68f`; разрешение 1344×768, length 5, ref_image_size match
Подано:
- <Picture 1> ← `ref_images.ref_image_0`: LoadImage `pose_V2_m3_full_rgb.png`
- <Picture 2> ← `ref_images.ref_image_1`: LoadImage `0003.png`
- <Video 1> ← `ref_videos.ref_video_0`: LoadImage `pose_V2_m3_full_s2/T03_armchair_side_sit/playblast.png` → ImageScale → RepeatImageBatch
Сэмплер: `ks` KSamplerAdvanced noise_seed=101, steps=25, cfg=1.0, sampler_name=euler, scheduler=simple, start_at_step=0
Модель/шифт: `unet` DiffusionModelLoaderKJ {"model_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default", "compute_dtype": "default", "patch_cublaslinear": false, "sage_attention": "disabled", "enable_fp16_accumulation": false}; `shift` MiniMaxH3SigmaShift {"shift_video": 12.0, "shift_audio": 3.0}
Узлы: CLIPLoader, ConditioningZeroOut, DiffusionModelLoaderKJ, GrowMaskWithBlur, H3ImageDecode, H3ImageFrameSelector, ImageScale, KSamplerAdvanced, LoadImage, MaskToImage, MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift, RepeatImageBatch, SAM3Segment, SaveImage, VAELoader
Выходы: plate3d_poses/V2_T03_armchair_side_sit_full/L_C2_s101_00002_.png, plate3d_poses/V2_T03_armchair_side_sit_full/L_C2_mask_s101_00002_.png
Линтер (stage=frame, connected=['<Picture 1>', '<Picture 2>', '<Video 1>']): ['error:E_MUST_SINGLE_PHOTO']
retention_analysis:
```
<Subject 1>: attribute_transfer - facial identity; wear the specified white T-shirt and light-blue jeans.
<Video 1>: partially_preserved - preserve geometry, pose, projected body scale and camera, replace only clay appearance.
<Picture 1>: fully_preserved - retain room materials, furnishings and lighting.
<Picture 2>: attribute_transfer - transfer facial identity to <Subject 1>.
```
Промпт целиком:
```
subject_definitions:
<Subject 1> is the adult woman whose identity is shown in <Picture 2>; replace the clay figure in <Video 1> with her.
<Picture 1> is the exact room composition, furnishing, materials and lighting of the target photograph.
<Picture 2> supplies only the facial identity and never appears as a separate image in the shot.
<Video 1> is a clay playblast defining the camera, body pose, projected size and precise position of <Subject 1>.
summary:
[reference generation] One single photorealistic static image. <Picture 2> is only an identity reference sheet, not a layout. Reproduce the composition and body pose of <Video 1>, using the real room from <Picture 1> and the facial identity from <Picture 2>.
retention_analysis:
<Subject 1>: attribute_transfer - facial identity; wear the specified white T-shirt and light-blue jeans.
<Video 1>: partially_preserved - preserve geometry, pose, projected body scale and camera, replace only clay appearance.
<Picture 1>: fully_preserved - retain room materials, furnishings and lighting.
<Picture 2>: attribute_transfer - transfer facial identity to <Subject 1>.
detailed_description:
A photorealistic live-action still. [Shot 1] filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, with no camera movement, no zoom and no pan. The room is a bedroom with lilac walls lit by low warm sunlight with window shadows, a low wide bed with beige and pink pillows, two wooden nightstands with glowing rattan lamps and a light woven floor. <Subject 1> is sitting sideways on the edge of the bed, leaning back on one elbow, her legs stretched out and crossed at the ankles. She wears an opaque white short-sleeved cotton T-shirt, light-blue straight-leg jeans and white lace-up sneakers. The clothing is fully opaque and covers her torso and legs. The key light is the warm golden daylight of the room coming from the left of the frame: it lights the same side of her body and hair as it lights the furniture, the opposite side of her body falls into the same soft shade as the room, and her body casts a crisp shadow on the floor and furniture in the same direction as the shadows of the furniture in <Picture 1>. Preserve the exact pose and placement from <Video 1>. True-to-life scale: she is never enlarged to fill the frame. Scale anchor: the low bed reaches just below her knees. No clay materials, no mannequin, no skeleton overlay. One continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen.
Timeline: From 0.00 to 0.17 seconds the subject holds the starting pose without moving.
The lighting is preserved unchanged throughout.
Colors and saturation stay unchanged from the reference image.
overall_soundscape:
Quiet room tone.
non_diegetic_music:
N/A
```

## V2_Q08_bed_recline_elbows_full_s101
prompt_id `4091355a-1d4a-482b-bf8d-938ce76660d9`; разрешение 1344×768, length 5, ref_image_size match
Подано:
- <Picture 1> ← `ref_images.ref_image_0`: LoadImage `pose_V2_m3_full_rgb.png`
- <Picture 2> ← `ref_images.ref_image_1`: LoadImage `0003.png`
- <Video 1> ← `ref_videos.ref_video_0`: LoadImage `pose_V2_m3_full_s2/Q08_bed_recline_elbows/playblast.png` → ImageScale → RepeatImageBatch
Сэмплер: `ks` KSamplerAdvanced noise_seed=101, steps=25, cfg=1.0, sampler_name=euler, scheduler=simple, start_at_step=0
Модель/шифт: `unet` DiffusionModelLoaderKJ {"model_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default", "compute_dtype": "default", "patch_cublaslinear": false, "sage_attention": "disabled", "enable_fp16_accumulation": false}; `shift` MiniMaxH3SigmaShift {"shift_video": 12.0, "shift_audio": 3.0}
Узлы: CLIPLoader, ConditioningZeroOut, DiffusionModelLoaderKJ, GrowMaskWithBlur, H3ImageDecode, H3ImageFrameSelector, ImageScale, KSamplerAdvanced, LoadImage, MaskToImage, MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift, RepeatImageBatch, SAM3Segment, SaveImage, VAELoader
Выходы: plate3d_poses/V2_Q08_bed_recline_elbows_full/L_C2_s101_00002_.png, plate3d_poses/V2_Q08_bed_recline_elbows_full/L_C2_mask_s101_00002_.png
Линтер (stage=frame, connected=['<Picture 1>', '<Picture 2>', '<Video 1>']): ['error:E_MUST_SINGLE_PHOTO', 'warn:W_SCALE_ANCHOR']
retention_analysis:
```
<Subject 1>: attribute_transfer - facial identity; wear the specified white T-shirt and light-blue jeans.
<Video 1>: partially_preserved - preserve geometry, pose, projected body scale and camera, replace only clay appearance.
<Picture 1>: fully_preserved - retain room materials, furnishings and lighting.
<Picture 2>: attribute_transfer - transfer facial identity to <Subject 1>.
```
Промпт целиком:
```
subject_definitions:
<Subject 1> is the adult woman whose identity is shown in <Picture 2>; replace the clay figure in <Video 1> with her.
<Picture 1> is the exact room composition, furnishing, materials and lighting of the target photograph.
<Picture 2> supplies only the facial identity and never appears as a separate image in the shot.
<Video 1> is a clay playblast defining the camera, body pose, projected size and precise position of <Subject 1>.
summary:
[reference generation] One single photorealistic static image. <Picture 2> is only an identity reference sheet, not a layout. Reproduce the composition and body pose of <Video 1>, using the real room from <Picture 1> and the facial identity from <Picture 2>.
retention_analysis:
<Subject 1>: attribute_transfer - facial identity; wear the specified white T-shirt and light-blue jeans.
<Video 1>: partially_preserved - preserve geometry, pose, projected body scale and camera, replace only clay appearance.
<Picture 1>: fully_preserved - retain room materials, furnishings and lighting.
<Picture 2>: attribute_transfer - transfer facial identity to <Subject 1>.
detailed_description:
A photorealistic live-action still. [Shot 1] filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, with no camera movement, no zoom and no pan. The room is a bedroom with lilac walls lit by low warm sunlight with window shadows, a low wide bed with beige and pink pillows, two wooden nightstands with glowing rattan lamps and a light woven floor. <Subject 1> is half-lying on the bed, leaning back on both elbows, legs stretched out towards the camera. She wears an opaque white short-sleeved cotton T-shirt, light-blue straight-leg jeans and white lace-up sneakers. The clothing is fully opaque and covers her torso and legs. The key light is the warm golden daylight of the room coming from the left of the frame: it lights the same side of her body and hair as it lights the furniture, the opposite side of her body falls into the same soft shade as the room, and her body casts a crisp shadow on the floor and furniture in the same direction as the shadows of the furniture in <Picture 1>. Preserve the exact pose and placement from <Video 1>. True-to-life scale: she is never enlarged to fill the frame. Scale anchor: the bed is wider than she is tall. No clay materials, no mannequin, no skeleton overlay. One continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen.
Timeline: From 0.00 to 0.17 seconds the subject holds the starting pose without moving.
The lighting is preserved unchanged throughout.
Colors and saturation stay unchanged from the reference image.
overall_soundscape:
Quiet room tone.
non_diegetic_music:
N/A
```

## V2_Q04_bed_side_lie_full_s101
prompt_id `b9f8701b-a7fa-40f2-96a2-e0a53c65d5ab`; разрешение 1344×768, length 5, ref_image_size match
Подано:
- <Picture 1> ← `ref_images.ref_image_0`: LoadImage `pose_V2_m3_full_rgb.png`
- <Picture 2> ← `ref_images.ref_image_1`: LoadImage `0003.png`
- <Video 1> ← `ref_videos.ref_video_0`: LoadImage `pose_V2_m3_full_s2/Q04_bed_side_lie/playblast.png` → ImageScale → RepeatImageBatch
Сэмплер: `ks` KSamplerAdvanced noise_seed=101, steps=25, cfg=1.0, sampler_name=euler, scheduler=simple, start_at_step=0
Модель/шифт: `unet` DiffusionModelLoaderKJ {"model_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default", "compute_dtype": "default", "patch_cublaslinear": false, "sage_attention": "disabled", "enable_fp16_accumulation": false}; `shift` MiniMaxH3SigmaShift {"shift_video": 12.0, "shift_audio": 3.0}
Узлы: CLIPLoader, ConditioningZeroOut, DiffusionModelLoaderKJ, GrowMaskWithBlur, H3ImageDecode, H3ImageFrameSelector, ImageScale, KSamplerAdvanced, LoadImage, MaskToImage, MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift, RepeatImageBatch, SAM3Segment, SaveImage, VAELoader
Выходы: plate3d_poses/V2_Q04_bed_side_lie_full/L_C2_s101_00003_.png, plate3d_poses/V2_Q04_bed_side_lie_full/L_C2_mask_s101_00003_.png
Линтер (stage=frame, connected=['<Picture 1>', '<Picture 2>', '<Video 1>']): ['error:E_MUST_SINGLE_PHOTO', 'warn:W_SCALE_ANCHOR']
retention_analysis:
```
<Subject 1>: attribute_transfer - facial identity; wear the specified white T-shirt and light-blue jeans.
<Video 1>: partially_preserved - preserve geometry, pose, projected body scale and camera, replace only clay appearance.
<Picture 1>: fully_preserved - retain room materials, furnishings and lighting.
<Picture 2>: attribute_transfer - transfer facial identity to <Subject 1>.
```
Промпт целиком:
```
subject_definitions:
<Subject 1> is the adult woman whose identity is shown in <Picture 2>; replace the clay figure in <Video 1> with her.
<Picture 1> is the exact room composition, furnishing, materials and lighting of the target photograph.
<Picture 2> supplies only the facial identity and never appears as a separate image in the shot.
<Video 1> is a clay playblast defining the camera, body pose, projected size and precise position of <Subject 1>.
summary:
[reference generation] One single photorealistic static image. <Picture 2> is only an identity reference sheet, not a layout. Reproduce the composition and body pose of <Video 1>, using the real room from <Picture 1> and the facial identity from <Picture 2>.
retention_analysis:
<Subject 1>: attribute_transfer - facial identity; wear the specified white T-shirt and light-blue jeans.
<Video 1>: partially_preserved - preserve geometry, pose, projected body scale and camera, replace only clay appearance.
<Picture 1>: fully_preserved - retain room materials, furnishings and lighting.
<Picture 2>: attribute_transfer - transfer facial identity to <Subject 1>.
detailed_description:
A photorealistic live-action still. [Shot 1] filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, with no camera movement, no zoom and no pan. The room is a bedroom with lilac walls lit by low warm sunlight with window shadows, a low wide bed with beige and pink pillows, two wooden nightstands with glowing rattan lamps and a light woven floor. <Subject 1> is lying on her side on the bed facing the camera, her head propped on her bent arm, legs slightly bent. She wears an opaque white short-sleeved cotton T-shirt, light-blue straight-leg jeans and white lace-up sneakers. The clothing is fully opaque and covers her torso and legs. The key light is the warm golden daylight of the room coming from the left of the frame: it lights the same side of her body and hair as it lights the furniture, the opposite side of her body falls into the same soft shade as the room, and her body casts a crisp shadow on the floor and furniture in the same direction as the shadows of the furniture in <Picture 1>. Preserve the exact pose and placement from <Video 1>. True-to-life scale: she is never enlarged to fill the frame. Scale anchor: the bed is wider than she is tall. No clay materials, no mannequin, no skeleton overlay. One continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen.
Timeline: From 0.00 to 0.17 seconds the subject holds the starting pose without moving.
The lighting is preserved unchanged throughout.
Colors and saturation stay unchanged from the reference image.
overall_soundscape:
Quiet room tone.
non_diegetic_music:
N/A
```

## Что делает comp2 (по коду)
- `pose_h3.py --plan full` сначала **кропит** плиту, плейбласт, shadow.png и mask.png одним прямоугольником вокруг фигуры
  (`plan_crop.py`, зум до 2.5×, кроп ≥ 40 % высоты) — H3 получает уже увеличенную плиту как `<Picture 1>` и плейбласт как `<Video 1>`.
- В той же задаче сервера — `SAM3Segment` (текст `person`) по **H3-кадру** → `GrowMaskWithBlur` (expand 4, blur 6) → маска сохраняется.
- Локально `composite_round.comp2`: **вклейка человека из H3-кадра** по этой маске поверх кропа плиты × (1 − 0.8·shadow.png)
  (тень из Blender-рендера, морфологически открыта 5 px). Вне маски человека — пиксели плиты, поэтому на листах шага 7
  комната всегда «совпадала»: перерисовку комнаты H3 (другая мебель, кровать выше) comp2 прячет.
