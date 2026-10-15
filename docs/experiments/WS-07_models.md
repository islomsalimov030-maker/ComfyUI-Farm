# WS-07 — модели и ноды для серий (проверено 13–14.10.2026)

Размеры взяты из API: HF — `lfs.size`, Civitai — `sizeKB`. Машинный список — `tools/ws07_models.yaml`, загрузчик — `tools/ws07_fetch.py`.
**Все производные H3 подчиняются лицензии MiniMax H3**: её территория исключает ЕС, UK, Корею и США, а под стоит в EU-RO-1 (RESEARCH §1). Это не новость, но на каждую строку ниже она тоже распространяется.

## Сессия 1 — скачать (≈ 51 ГБ; с запасной 10Eros ≈ 72 ГБ)

| Файл | Папка | ГБ | Источник | Лицензия | Зачем |
|---|---|---|---|---|---|
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16` | loras | 1.82 | HF lightx2v/Minimax-h3-Turbo | Apache-2.0 | turbo эталонной руки (если на томе нет) |
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16` | loras | 1.82 | HF Comfy-Org/MiniMax-H3 | MiniMax H3 CLA | s3_lora_turbo |
| `minimax_h3_t1_image_vae_step1597` | vae | 5.21 | HF Mamad8/MiniMax-H3-Image-VAE | ⚠ **лицензии нет**, обучен на Booru + фото | s1_gen_t1, только тест |
| `pinkcherryFl2vaV1FinalTurbo_v10` (int8) | diffusion_models | 20.48 | Civitai 3326433 | Image/Rent/Sell/SellMerge | s3_arms |
| `10erosMaxInt8Ref2va_v11` (int8) — только если на томе нет `10Eros_Max_h3_TURBO-hybrid_beta4` | diffusion_models | 20.48 | Civitai 3255792 | Image/Rent/Sell/SellMerge | s3_arms |
| `qwen3-vl:8b-instruct` (Ollama) | ollama | 6.1 | ollama.com | Apache-2.0 | WS-04/серия 6, если останется время |
| Ноды: ComfyUI-Fizgig-H3-Still `10d5171` | custom_nodes | — | GitHub | MIT | генератор B |
| Ноды: ComfyUI-MM-1Frame `3e1b1f4` | custom_nodes | — | GitHub | ⚠ **нет LICENSE** | генератор C, только тест |

Уже на томе: базовый ref2va, video VAE, Qwen3-VL-кодировщик движков, Singularity ref2va v1.3 и LoRA Стейси. Проверяется командой `ws07_fetch.py --dry`.

## Сессия 2 (≈ 56 ГБ)

| Файл | ГБ | Источник | Лицензия |
|---|---|---|---|
| `seedvr2_ema_3b_fp8_e4m3fn` | 3.39 | HF numz/SeedVR2_comfyUI | Apache-2.0 |
| `seedvr2_ema_7b_fp8_e4m3fn` | 8.24 | там же | Apache-2.0 |
| `ema_vae_fp16` (SeedVR2) | 0.50 | там же | Apache-2.0 |
| `qwen3-vl:32b-instruct` q4_K_M (Ollama) | 21 | ollama.com | Apache-2.0 |
| `qwen3.6:27b` q4_K_M (Ollama) | 17 | ollama.com | Apache-2.0 (HF Qwen/Qwen3.6-27B) |
| RefMod: моды извлекаем сами из своих картинок (`ws07_run.py refmod`) | ~0.002 | — | — |
| Ноды: ComfyUI-MiniMaxH3Mod `f946208` — ставит `install/pod_setup.sh` (WS-10) | — | GitHub | MIT |
| Ноды: ComfyUI-SeedVR2_VideoUpscaler `4490bd1` | — | GitHub | Apache-2.0 (ставит diffusers, peft, gguf) |

## ⚠ Что важно для прода
1. **DaSiWa** — исключена пользователем 13.10. Для справки: на Civitai у неё ещё и нет права продажи (`Sell`), так что для прода она не годится в любом случае.
2. **10Eros Max `TURBO-hybrid_beta4`** (источник уточнён 15.10): [TenStrip/10Eros-Max](https://huggingface.co/TenStrip/10Eros-Max/blob/main/10Eros_Max_h3_TURBO-hybrid_beta4.safetensors); INT8 ConvRot-перепаковки есть у [cicalooo](https://huggingface.co/cicalooo/10Eros-Max-h3-int8-convrot). README TenStrip прямо помечает beta3/beta4 как **corrupted test versions**. Новые серии — beta5 (TURBO и non-TURBO) и beta6 8stepTURBO. Замена в графах/сериях пока не выбрана, файлы и параметры не менялись. Дополнительные условия hybrid непростые: автор перечисляет доноров LTX 2.3, Wan 2.2, Krea 2 и их лицензии; самостоятельного LICENSE в репо нет. Это не независимая проверка корректности переноса весов или лицензирования.
3. **t1 image VAE и MM-1Frame** — без лицензии. Только тест, после s1 удалить.
4. **Singularity** — файл тот же, что на HF `WarmBloodAban/Minimax-h3_Singularity` (Apache-2.0 в карточке). На Civitai лежит другая версия (не ref2va).
5. **Кодировщик движков** `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot` — сторонняя «abliterated» перепаковка, источник на томе не задокументирован. Официальная замена от Comfy-Org: `qwen3vl_32b_minimax_h3_int8_convrot` (27.14 ГБ). Для SFW-профиля стоит сравнить, но не в 1-й сессии.
6. **RefMod-набор `malcolmrey/minimaxh3`** — это моды реальных людей и знаменитостей. Для фермы не используем: моды Стейси извлекаем сами.
7. **RTX VSR** (Nvidia_RTX_Nodes) — код под Apache-2.0, но зависимость `nvidia-vfx` проприетарная NVIDIA. На Linux нужны драйвер 570.190+ и torch cu130+.
