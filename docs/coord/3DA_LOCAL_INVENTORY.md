# 3D-A: инвентаризация ПК пользователя и сервера студии (07.11)

## ПК (Windows 11, Git Bash)
| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti, 8192 MiB, драйвер 617.42 |
| RAM | 32 ГБ (34 222 137 344 B) |
| Диски | C: 477 ГБ, свободно **27 ГБ** (95 %); D: 1.9 ТБ, свободно 131 ГБ |
| Blender | `C:/Program Files/Blender Foundation/Blender 5.0/blender.exe` — 5.0.0 |
| ComfyUI пользователя | `D:/ComfyUi/ComfyUI` (Easy-Install, portable), **v0.29.0-7**; python_embeded 3.12.10, torch 2.9.1+cu130, CUDA ok; `.venv` 3.14.2 (не используется) |
| extra_model_paths.yaml | нет |
| custom_nodes | ComfyLiterals, Crystools, DepthAnythingV3, Easy-Use, GGUF, Helper-Nodes, Krea2T-Enhancer, Pixaroma, QwenVL, Seedance2-Direct-BytePlus, TiledDiffusion, WanVideoWrapper, Comfyroll, RES4LYF, advancedliveportrait, custom-scripts, florence2, higgsfield-direct, impact-pack, inpaint-cropandstitch, itools, kjnodes, krea2-controlnet, krea2edit, manager, ollama-image-to-prompt, rmbg, seedance-nodes, various, videohelpersuite, controlnet_aux, essentials, ipadapter_plus, layerstyle, controlaltai-nodes, crt-nodes, gguf, joycaption, qwen3-vl-nsfw, rgthree, seedance2-byteplus, seedvr2_videoupscaler, teacache, vnccs-utils, was-ns, x-flux |
| H3-модели | только `models/diffusion_models/10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors` (+копия в `models/MInimaxH3`); ref2va, энкодер nvfp4, видео-VAE, Fun CN — **нет** |
| Данные Стейси | `D:/stacy-lora` (не `C:\stacy-lora`); `mods/` = README + characters/locations/objects/outfits (файлов stacy_face/stacy_body там нет) |
| Лицо-референс | `C:/Users/user/Downloads/0003.png` = `Downloads/3DA_2026-11-05_step1/source/reference/0003.png` (md5 7f8429a4…) |
| Прочее | `C:/work/_src/ComfyUI0346` — исходники ComfyUI 0.34.6 (для сравнения с сервером) |

Классы в ComfyUI ПК (статически, grep по comfy_extras/nodes.py/custom_nodes; сервер не запускался — в 0.29 H3 нет):
MiniMaxH3ReferenceToVideo, MiniMaxH3FunControlNetApply, MiniMaxH3SigmaShift, H3LatentFromImage, H3ImageDecode, MiniMaxH3RefModApply — **нет**;
ModelPatchLoader, LoadImagesFromFolderKJ, DiffusionModelLoaderKJ, VHS_*, SAM3 (comfyui-rmbg) — есть.

**Правило п. 0 сработало (8 ГБ < 24 ГБ):** H3 на ПК не запускается. На ПК — только Blender (п. 2а), сборка графов `--build-only`, проверка графов офлайн (`check_graph_oi.py`) и листы.

## Сервер студии `comfy.next-genium.com` (07.11, `/system_stats`, `/object_info` → `C:/work/scene3d_work/oi/server_object_info.json`)
- ComfyUI 0.34.6, RTX 5090 32 109 MiB, RAM 123 ГБ, флаги `--cache-none --lowvram --enable-manager`.
- Есть: MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift, H3LatentFromImage, H3ImageDecode, H3ImageFrameSelector, MiniMaxH3RefModsLoader/Apply, ModelPatchLoader, LoadImagesFromFolderKJ, DiffusionModelLoaderKJ, VHS_VideoCombine, SetLatentNoiseMask, DifferentialDiffusion, KSamplerAdvanced, **SAM3Segment** (класс из AIO V2.2), SAM3_Detect/VideoTrack, SAM3DBody_*.
- Нет: **MiniMaxH3FunControlNetApply**.
- RefMods: 21 шт. (`ss340/*`, `ss456/*`), `farm/stacy_*` нет → `FARM_MODS=none`.
- `input/0003.png` на сервере уже есть и совпадает по md5 с локальным — не перезаливается.
- Графы L_R0/L_R1 (после `farm_env.prepare()`) проверены `check_graph_oi.py`: все классы и модели есть, не хватало только загружаемых карт.

## Окружение запуска
`C:/work/scene3d_work/farm_env.sh` (без секретов): FARM_COMFY = `C:/work/scene3d_work/comfy_mirror` (локальное зеркало input/ для загрузки и output/ для скачанных результатов), FARM_URL = сервер, FARM_REMOTE=1, FARM_MODS=none; FARM_AUTH — из shell пользователя (fallback на уже заданные в окружении студийные креды, значение не печатается).
