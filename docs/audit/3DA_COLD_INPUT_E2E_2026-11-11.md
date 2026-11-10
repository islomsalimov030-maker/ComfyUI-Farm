# AIO server inventory — 11.11.2026

Источник: свежий GET /object_info; installed comfyui-3da-scene 0.1.4 enabled. Очередь перед первым cold prompt: 0 running / 0 pending.

Все три AIO-шаблона содержат 250 узлов с учётом сабграфов. Это инвентаризация, не runtime acceptance.

| Класс | Есть |
|---|---|
| 3DALocation | да |
| 3DANativeComposite | да |
| 3DANativeGuides | да |
| 3DAPlacePose | да |
| 3DAProp | да |
| 3DASceneEdits | да |
| 3DASceneExport | да |
| 43521446-c686-4e05-84c5-16aa0a0e24a9 | да |
| 8c87aaa9-03cc-4a78-b986-9dbd7df00baf | да |
| 9089fd8f-b8d7-412b-968f-46f098c1721d | да |
| BasicGuider | да |
| BasicScheduler | да |
| CLIPLoader | да |
| CLIPTextEncode | да |
| ColorMatch | да |
| ComfyMathExpression | да |
| ConditioningZeroOut | да |
| DifferentialDiffusion | да |
| DiffusionModelLoaderKJ | да |
| EmptySD3LatentImage | да |
| Fast Groups Muter (rgthree) | нет; необязательный UI controller muted |
| GetImageSize | да |
| GrowMask | да |
| GrowMaskWithBlur | да |
| H3FaceStitch | да |
| H3FaceTrackCrop | да |
| H3ImageDecode | да |
| H3ImageFrameSelector | да |
| H3InjectVideoLatent | да |
| H3LatentFromImage | да |
| H3PerFrameDenoise | да |
| H3TextToImagePrepare | да |
| INTConstant | да |
| ImageCompositeMasked | да |
| ImageFromBatch | да |
| ImageScale | да |
| ImageScaleToTotalPixels | да |
| ImageSharpen | да |
| ImpactInt | да |
| ImpactSwitch | да |
| InpaintCropImproved | да |
| InpaintStitchImproved | да |
| KSampler | да |
| KSamplerAdvanced | да |
| KSamplerSelect | да |
| LazySwitchKJ | да |
| Load3DAdvanced | да |
| LoadImage | да |
| LoraLoaderModelOnly | да |
| MarkdownNote | да |
| MiniMaxH3AddGuide | да |
| MiniMaxH3RefModApply | да |
| MiniMaxH3RefModsLoader | да |
| MiniMaxH3ReferenceToVideo | да |
| MiniMaxH3SigmaShift | да |
| ModelAttentionBackend | да |
| ModelPreviewOverrideKJ | да |
| ModelSamplingAuraFlow | да |
| NGFarmPanelFace | да |
| NGFarmPanelPhoto | да |
| NGFarmPanelVideo | да |
| PreviewAny | да |
| PreviewImage | да |
| PrimitiveBoolean | да |
| PrimitiveStringMultiline | да |
| RTXVideoSuperResolution | да |
| RandomNoise | да |
| RegexReplace | да |
| RepeatImageBatch | да |
| SAM3Segment | да |
| SamplerCustomAdvanced | да |
| SaveImage | да |
| StacyAutoConfidence | да |
| StacyColorLock | да |
| StacyFitFrame | да |
| StacyFlowAlign | да |
| StacyFreeVRAM | да |
| StacyGate | да |
| StacyLoopExtend | да |
| StacyLoopFold | да |
| StacyLoopSeam | да |
| StacyOcclusionDenoise | да |
| StacyPanel | да |
| StacyPromptRetime | да |
| StacyRefVideo | да |
| StacyReport | да |
| StacySettings | да |
| StacyTemporalStabilize | да |
| StacyVideoInput | да |
| StringConcatenate | да |
| TextEncodeQwenImageEditPlus | да |
| UNETLoader | да |
| VAEDecode | да |
| VAELoader | да |
| VHS_VideoCombine | да |
| c6db46ef-8dc5-52b5-a187-7e04ea08b487 | да |
| eeafa102-d4f0-4f51-a26e-715929de67b7 | да |

| Loader input | Модель / мод | Есть |
|---|---|---|
| clip_name | qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors | да |
| clip_name | qwen_2.5_vl_7b_fp8_scaled.safetensors | да |
| clip_name | qwen_3_4b.safetensors | да |
| lora_name | minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors | да |
| lora_name | minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors | да |
| lora_name | stacy_h3_A_stills_000001750.safetensors | да |
| lora_name | stacy_h3_face_v1_000001800.safetensors | да |
| mod_1 | ss456/split_face | да |
| mod_2 | (none) | да |
| mod_2 | ss456/out_athleisure | да |
| mod_3 | (none) | да |
| mod_4 | (none) | да |
| mod_5 | (none) | да |
| mod_6 | (none) | да |
| mod_7 | (none) | да |
| mod_8 | (none) | да |
| model_name | 10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors | да |
| model_name | minimax_h3_fl2va_pruned_int8_convrot.safetensors | да |
| model_name | minimax_h3_ref2va_pruned_int8_convrot.safetensors | да |
| unet_name | 10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors | да |
| unet_name | minimax_h3_fl2va_pruned_int8_convrot.safetensors | да |
| unet_name | minimax_h3_ref2va_pruned_int8_convrot.safetensors | да |
| unet_name | qwen_image_edit_2511_fp8mixed.safetensors | да |
| unet_name | z_image_turbo_bf16.safetensors | да |
| vae_name | ae.safetensors | да |
| vae_name | minimax_h3_audio_vae_fp32.safetensors | да |
| vae_name | minimax_h3_video_vae_fp16.safetensors | да |
| vae_name | qwen_image_vae.safetensors | да |

Source-pose Z-Image/Qwen сохранены; выбран select=1 (загруженное фото). Видео/отдельный video face-pass остаются muted как в исходном photo stage. Фото и его face-pass сохранены. Fast Groups Muter отсутствует и mode=2; управление этапами задаётся явно.

Server prepare/preview/photo API: 26/28/104 nodes; локальный schema audit: 0 errors. Load3DAdvanced до подготовки не содержит GLB; preview/final до UI-правок не являются готовыми к выполнению graphs.
