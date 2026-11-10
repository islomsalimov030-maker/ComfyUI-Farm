# Что используется в воркфлоу farm_auto / farm_manual и лицензии (23.10.2026)

✅ — коммерческое использование разрешено · ⚠ — нужно проверить или есть условия · ❌ — нельзя (в воркфлоу не используется)

## Модели

| Что | Файл | Зачем | Лицензия |
|---|---|---|---|
| MiniMax H3 ref2va | `minimax_h3_ref2va_pruned_int8_convrot` | кадр и видео (модель ref2v) | H3 Community License (территория и условия — отдельный вопрос, ведётся отдельно) |
| MiniMax H3 fl2va | `minimax_h3_fl2va_pruned_int8_convrot` | проход лица в фото и видео | H3 Community License |
| H3 video VAE / audio VAE | `minimax_h3_video_vae_fp16`, `minimax_h3_audio_vae_fp32` | кодирование и декодирование | H3 Community License |
| Текстовый энкодер H3 | `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot` | энкодер промпта | ✅ Apache-2.0 (сборка ethanfel, база Qwen3-VL-32B — Apache-2.0; AGPL heretic — только код инструмента). Риск — не лицензия, а происхождение: сторонняя сборка. Замена `qwen3vl_32b_minimax_h3_int8_convrot` — ⚠ MiniMax H3 Community License (25.10) |
| Turbo-LoRA ref2v 8 шагов, fl2v 4 шага | `minimax_h3_ref2v_turbo_8step_v1.0…`, `minimax_h3_fl2v_turbo_4step_v1.1…` (lightx2v) | ускорение ref2v и прохода лица | ✅ Apache-2.0 |
| 10Eros Max beta5 | `10Eros_Max_h3_hybrid_beta5_int8` (TenStrip) | модель-переключатель | ⚠ H3 CL ($20 млн/год) + LTX-2 CL ($10 млн/год) + Wan 2.2 (Apache-2.0) + Krea 2 CL ($1 млн/год, + content-фильтры и Notice). Проверено 25.10 — RESEARCH |
| PinkCherry ref2va beta-0.6 | `pinkcherrry_minimax_h3_ref2va_beta-0.6_int8_convrot` (SexGod1979) | модель-переключатель | ✅ на карточке Apache-2.0; поверх действуют условия H3 |
| SAM3 | `facebook/sam3` | маска лица и тела для прохода лица | ✅ SAM License (Meta): коммерция разрешена, кроме военных и запрещённых применений |
| RTX Video Super Resolution | драйвер NVIDIA (`nvidia-vfx`) | апскейл до 1920×1080 | ⚠ проприетарная библиотека NVIDIA, бесплатна на картах RTX; для прода прочитать условия |
| LLM промптов | Ollama `qwen3-vl:8b-instruct` | пишет поля промпта из строки таблицы (только авто, только под) | ✅ Apache-2.0 |
| RefMod — моды персонажа | `farm/<персонаж>_face`, `farm/<вариант>_body` | лицо и тело в наряде | наши, собраны из своих референсов |

QA фермы работает вне воркфлоу. Все модели QA разрешены для коммерции:
- MediaPipe — Apache-2.0;
- AuraFace v1 — Apache-2.0;
- MiVOLO v2 — Apache-2.0;
- Depth Anything 3 Mono-Large / Small — Apache-2.0.

Не используются, потому что некоммерческие (❌):
- InsightFace. Коммерческую лицензию продают по запросу, но она не нужна.
- Depth Anything V2 Base и Large (CC-BY-NC).

## Пакеты нод

Все пакеты стоят и на студии, и на поде.

| Пакет | Ноды в воркфлоу | Лицензия |
|---|---|---|
| ComfyUI (ядро) | семплеры, загрузчики, MiniMaxH3ReferenceToVideo, строки, математика | GPL-3.0 ✅ (для внутреннего использования) |
| ComfyUI-MiniMaxH3Mod (Luisacaotica) | RefMod: загрузка и применение модов | MIT ✅ |
| ComfyUI-H3-FaceRefine (Carasibana) | трекинг лица, вклейка, покадровый denoise | MIT ✅ (режим `identity_track` выключен, поэтому InsightFace не грузится) |
| ComfyUI-MiniMax-H3-Image-Studio (astropuzzo) | подготовка кадра H3, декодирование, выбор кадра | Unlicense ✅ |
| KJNodes | загрузчик модели, ColorMatch, маски, ленивый переключатель | GPL-3.0 ✅ (для внутреннего использования) |
| ComfyUI-Inpaint-CropAndStitch | кроп и вклейка лица и маски | GPL-3.0 ✅ (для внутреннего использования) |
| ComfyUI-RMBG (1038lab) | SAM3Segment | GPL-3.0 ✅ (для внутреннего использования) |
| VideoHelperSuite | сохранение видео | GPL-3.0 ✅ (для внутреннего использования) |
| Nvidia RTX nodes (Comfy-Org) | RTXVideoSuperResolution | код Apache-2.0 ✅, библиотека — смотри строку RTX VSR выше |
| rgthree-comfy | Fast Groups Muter (включатели групп) | MIT ✅ |
| ComfyUI-Farm (наш) | шот из таблицы, пульты, ZIP, выбор локации, реф-видео | наш; ⚠ добавить файл LICENSE (репозиторий публичный) |
| ComfyUI-StacyLoop (наш) | движок петли: пульт, шов, стабилизатор, color lock, проход лица видео | наш; ⚠ добавить LICENSE |
| h3_img2img_latent (наш) | H3LatentFromImage | наш; ⚠ добавить LICENSE |

GPL-3.0 означает следующее. Пользоваться внутри компании можно без условий. Если распространять изменённый код нод наружу, нужно отдавать и его исходники. На сгенерированные картинки и видео GPL не распространяется.
