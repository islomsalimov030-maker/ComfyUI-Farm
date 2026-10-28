#!/usr/bin/env bash
# pod_setup.sh — ставит ферму (ng-farm) на под поверх bootstrap.sh работодателя. WS-10.
# Идемпотентен: уже стоящее пропускает, повтор после обрыва дёшев. Всё — в /workspace (переживает Terminate).
#
# Первый раз (после «УСПЕХ» от /workspace/bootstrap.sh), лучше в tmux — модели ~21 ГБ:
#   git clone https://github.com/islomsalimov030-maker/ComfyUI-Farm /workspace/ComfyUI/custom_nodes/ComfyUI-Farm
#   tmux new -s setup 'bash /workspace/ComfyUI/custom_nodes/ComfyUI-Farm/install/pod_setup.sh'
# Дальше — та же вторая команда. Флаги:
#   --skip-models   не качать модели (Z-image, AuraFace, MediaPipe)
#   --skip-ollama   не трогать Ollama
#   --no-restart    не перезапускать ComfyUI, даже если ставились ноды
#   --no-bot        не запускать бота
#
# Что делает: 1) клон/обновление ng-farm в custom_nodes; 2) ноды движков фермы и ручных воркфлоу WS-09;
# 3) Python-зависимости фермы и QA в venv ComfyUI (torch/numpy зафиксированы констрейнтом);
# 4) venv MiVOLO v2 (/workspace/venv_mivolo, без ultralytics); 5) модели в /workspace/models (sha256);
# 6) Ollama-модели; 7) $FARM_HOME и проверка секретов (сами секреты не пишет и не печатает);
# 8) venv бота и запуск в tmux farmbot; 9) перезапуск ComfyUI, если ставились ноды/пакеты, и pod_doctor.
# Не делает: секреты, LoRA Стейси (лежат на ПК — подсказка scp в конце), профили (WS-07).
set -uo pipefail

WS=/workspace
COMFY=$WS/ComfyUI
CN=$COMFY/custom_nodes
CPY=$COMFY/venv/bin/python
MODELS=$WS/models
FARM_HOME=${FARM_HOME:-$WS/farm}
REPO_URL=https://github.com/islomsalimov030-maker/ComfyUI-Farm
REPO=$CN/ComfyUI-Farm
MIVOLO_VENV=$WS/venv_mivolo
BOT_VENV=$WS/venv_farmbot
OLLAMA_TAGS=${FARM_OLLAMA_MODELS:-qwen3-vl:8b-instruct}   # prompt_model и judge_model профиля sfw_base
LOG=$WS/logs/pod_setup.log

SKIP_MODELS=0; SKIP_OLLAMA=0; NO_RESTART=0; NO_BOT=0
for a in "$@"; do
  case "$a" in
    --skip-models) SKIP_MODELS=1 ;;
    --skip-ollama) SKIP_OLLAMA=1 ;;
    --no-restart)  NO_RESTART=1 ;;
    --no-bot)      NO_BOT=1 ;;
    *) echo "неизвестный флаг: $a"; exit 2 ;;
  esac
done

FAIL=0        # что-то не встало — итог «ЕСТЬ ОШИБКИ»
CHANGED=0     # ставились ноды/пакеты — нужен перезапуск ComfyUI
TODO=()       # что сделать руками (секреты, LoRA)
log()  { echo -e "\n=== $* ==="; }
warn() { echo "  ! $*"; FAIL=1; }
todo() { echo "  → $*"; TODO+=("$*"); }

# ---------------------------------------------------------------- 0. окружение
check_env() {
  log "Окружение"
  set +u                                   # nvm из env.sh не переносит set -u
  [ -f /etc/rp_environment ] && . /etc/rp_environment
  [ -f $WS/env.sh ] && . $WS/env.sh        # ollama в PATH, OLLAMA_MODELS=/workspace/ollama
  set -u
  # образ RunPod кладёт кэши на том; если переменной нет — туда же, чтобы веса не пропали с подом
  export HF_HOME=${HF_HOME:-$WS/.cache/huggingface}
  export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$WS/.cache/pip}
  if [ -z "${RUNPOD_VOLUME_ID:-}" ] || ! mountpoint -q $WS; then
    echo "ОШИБКА: network volume не подключён к $WS — всё поставленное пропадёт. Стоп."; exit 1
  fi
  [ -x "$CPY" ] || { echo "ОШИБКА: нет $CPY — сначала /workspace/bootstrap.sh до УСПЕХ."; exit 1; }
  PY312=$(command -v python3.12 || command -v python3)
  echo "  ComfyUI venv: $("$CPY" -V 2>&1), torch $("$CPY" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
  echo "  HF_HOME=$HF_HOME"
  local free_gb; free_gb=$(df -BG --output=avail $WS | tail -1 | tr -dc 0-9)
  echo "  свободно на томе: ${free_gb} ГБ"
  if [ "$SKIP_MODELS" = 0 ] && [ "${free_gb:-0}" -lt 40 ]; then
    warn "на томе < 40 ГБ, моделям нужно до ~21 ГБ — проверьте место (кэши: $WS/.cache)"
  fi
  command -v ffmpeg >/dev/null || warn "нет ffmpeg (ставит bootstrap.sh) — бот не сможет пережимать видео"
  mkdir -p $WS/logs $MODELS
  # torch/torchvision/torchaudio/numpy не даём обновить ни одной ноде и ни одному пакету
  CONS=$(mktemp)
  "$CPY" -m pip freeze 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|numpy)==' > "$CONS"
  echo "  зафиксировано: $(tr '\n' ' ' < "$CONS")"
}

cpip() { "$CPY" -m pip install -q -c "$CONS" "$@"; }

# ---------------------------------------------------------------- 1. ng-farm
install_repo() {
  log "ng-farm (пакет нод фермы)"
  local found
  found=$(find_node "$REPO_URL" ComfyUI-Farm)
  if [ -n "$found" ]; then
    REPO=$found
    if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
      echo "  = $REPO: есть локальные правки — не обновляю"
    else
      local before; before=$(git -C "$REPO" rev-parse HEAD)
      git -C "$REPO" pull -q --ff-only || warn "git pull ng-farm не прошёл"
      [ "$before" != "$(git -C "$REPO" rev-parse HEAD)" ] && CHANGED=1
      echo "  = $REPO @ $(git -C "$REPO" rev-parse --short HEAD)"
    fi
  else
    git clone -q "$REPO_URL" "$REPO" || { echo "ОШИБКА: не склонировался $REPO_URL"; exit 1; }
    CHANGED=1
    echo "  + $REPO @ $(git -C "$REPO" rev-parse --short HEAD)"
  fi
  cpip -r "$REPO/requirements.txt" || warn "зависимости ng-farm"
}

# ---------------------------------------------------------------- 2. ноды
# имя каталога | git-URL | минимальный коммит (пусто — любой) | зачем
NODES=(
  "ComfyUI-KJNodes|https://github.com/kijai/ComfyUI-KJNodes||все движки (GPL-3.0)"
  "rgthree-comfy|https://github.com/rgthree/rgthree-comfy||Power Lora Loader (MIT)"
  "ComfyUI-Easy-Use|https://github.com/yolain/ComfyUI-Easy-Use||isMaskEmpty (GPL-3.0)"
  "ComfyUI-VideoHelperSuite|https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite||видео (GPL-3.0)"
  "ComfyUI-MiniMax-H3-Image-Studio|https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio||кадр, ref-edit (Unlicense)"
  "ComfyUI-Inpaint-CropAndStitch|https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch||manual_frame, frame_h3 (GPL-3.0)"
  "h3_img2img_latent|https://github.com/islomsalimov030-maker/h3_img2img_latent|512b1b3fe3d35a71ec69b55a9d59b7511370fd62|manual_frame движок 1 (свой, MIT)"
  "ComfyUI-StacyLoop|https://github.com/islomsalimov030-maker/ComfyUI-StacyLoop|095b77da4f4dc1a7817d0b263106166cd3b7b619|видео-петля (свой, MIT)"
  "ComfyUI-H3-FaceRefine|https://github.com/Carasibana/ComfyUI-H3-FaceRefine||видео face pass (MIT)"
  "Nvidia_RTX_Nodes_ComfyUI|https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI||RTX VSR (Apache-2.0 + NVIDIA SDK)"
  "ComfyUI-Impact-Pack|https://github.com/ltdrdata/ComfyUI-Impact-Pack||frame_h3 лицо Z (GPL-3.0)"
  "ComfyUI-Impact-Subpack|https://github.com/ltdrdata/ComfyUI-Impact-Subpack||frame_h3 YOLO (AGPL-3.0)"
  "ComfyUI-RMBG|https://github.com/1038lab/ComfyUI-RMBG||frame_h3 SAM3 (GPL-3.0; веса SAM — своя лицензия)"
  "ComfyUI-MiniMaxH3Mod|https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod||RefMod ручных воркфлоу (MIT; решение 13.10)"
)

norm_url() { echo "$1" | tr 'A-Z' 'a-z' | sed -E 's#\.git$##; s#/$##'; }

# каталог ноды: по origin (Manager называет каталоги по-своему) или по имени без учёта регистра
find_node() {
  local url; url=$(norm_url "$1")
  local d
  for d in "$CN"/*/; do
    d=${d%/}
    if [ -d "$d/.git" ] && [ "$(norm_url "$(git -C "$d" remote get-url origin 2>/dev/null)")" = "$url" ]; then
      echo "$d"; return
    fi
  done
  for d in "$CN"/*/; do
    d=${d%/}
    [ "$(basename "$d" | tr 'A-Z' 'a-z')" = "$(echo "$2" | tr 'A-Z' 'a-z')" ] && { echo "$d"; return; }
  done
}

install_node() {
  local name url min why dir
  IFS='|' read -r name url min why <<< "$1"
  dir=$(find_node "$url" "$name")
  if [ -n "$dir" ] && [ ! -d "$dir/.git" ] && [ -n "$min" ]; then
    # старая копия без git (так было со StacyLoop 04.10) — убираем в бэкап, ставим клон
    local bak=$WS/_backup/$(basename "$dir")_$(date +%F_%H%M)
    mkdir -p $WS/_backup && mv "$dir" "$bak" && echo "  ~ $name: копия без git → $bak"
    dir=""
  fi
  if [ -z "$dir" ]; then
    dir=$CN/$name
    git clone -q "$url" "$dir" || { warn "не склонировался $name"; return; }
    CHANGED=1
    echo "  + $name @ $(git -C "$dir" rev-parse --short HEAD) — $why"
  elif [ -n "$min" ] && ! git -C "$dir" merge-base --is-ancestor "$min" HEAD 2>/dev/null; then
    git -C "$dir" fetch -q origin && git -C "$dir" merge -q --ff-only "origin/$(git -C "$dir" rev-parse --abbrev-ref HEAD)" \
      || { warn "$name старее ${min:0:7}, обновить не вышло (локальные правки?)"; return; }
    CHANGED=1
    echo "  ↑ $name → $(git -C "$dir" rev-parse --short HEAD) (нужен ≥ ${min:0:7})"
  else
    echo "  = $name ($(basename "$dir"))"
    return       # стоящие ноды не трогаем: их зависимости уже в venv
  fi
  if [ -f "$dir/requirements.txt" ]; then
    cpip -r "$dir/requirements.txt" || warn "зависимости $name"
  fi
}

install_nodes() {
  log "Кастом-ноды"
  local n
  for n in "${NODES[@]}"; do install_node "$n"; done
  mkdir -p $MODELS/refmods       # моды RefMod (WS-07); видит ли их нода через extra_model_paths — проверить на поде
}

# ---------------------------------------------------------------- 3. Python-пакеты QA
# mediapipe и scenedetect объявляют opencv-contrib-python / opencv-python (не headless): с ними в venv
# окажутся две сборки cv2, перезаписывающие друг друга. Поэтому ставим их --no-deps, а зависимости — явно.
install_qa_deps() {
  log "Пакеты проверок качества (venv ComfyUI)"
  # MediaPipe Tasks грузит libEGL.so.1 — в образе RunPod её нет (apt вне /workspace: ставим на каждом новом поде)
  if ! ldconfig -p | grep -q "libEGL.so.1"; then
    (apt-get install -y -q libegl1 libgles2 >>"$LOG" 2>&1 || { apt-get update -q >>"$LOG" 2>&1 && \
      apt-get install -y -q libegl1 libgles2 >>"$LOG" 2>&1; }) && echo "  + libegl1 libgles2 (apt)" || warn "libEGL для MediaPipe"
  fi
  local before; before=$("$CPY" -m pip freeze 2>/dev/null | md5sum)
  "$CPY" -c "import cv2" 2>/dev/null || cpip opencv-python-headless || warn "opencv"
  # onnxruntime-gpu тоже даёт модуль onnxruntime — второй пакет ставить нельзя
  "$CPY" -c "import onnxruntime" 2>/dev/null || cpip onnxruntime || warn "onnxruntime"
  "$CPY" -c "import mediapipe" 2>/dev/null || {
    cpip --no-deps "mediapipe==1.1.0" && cpip "absl-py~=2.3" "flatbuffers~=25.9" "sounddevice~=0.5" matplotlib certifi
  } || warn "mediapipe"
  "$CPY" -c "import scenedetect" 2>/dev/null || {
    cpip --no-deps "scenedetect==0.7.2" && cpip "click~=8.0,!=8.3.0" platformdirs tqdm
  } || warn "scenedetect"
  "$CPY" -c "import huggingface_hub" 2>/dev/null || cpip huggingface_hub || warn "huggingface_hub"
  [ "$before" != "$("$CPY" -m pip freeze 2>/dev/null | md5sum)" ] && CHANGED=1
  "$CPY" - <<'EOF' || warn "импорт пакетов QA"
import importlib.metadata as md, cv2, mediapipe, onnxruntime, scenedetect, numpy, yaml, httpx, jinja2, gspread
cv = [d for d in ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless")
      if next(iter(md.distributions(name=d)), None)]
print(f"  cv2 {cv2.__version__} ({', '.join(cv)}), mediapipe {mediapipe.__version__}, onnxruntime "
      f"{onnxruntime.__version__} {onnxruntime.get_available_providers()[:2]}, numpy {numpy.__version__}")
if len(cv) > 1:
    print("  ! в venv несколько сборок OpenCV — работают, но кто поставлен последним, тот и cv2")
EOF
}

# ---------------------------------------------------------------- 4. MiVOLO v2 (возраст)
# Рецепт — docs/qa/frames_baseline.md «Возраст: MiVOLO v2»: torch CPU, без ultralytics (AGPL), --no-deps.
install_mivolo() {
  log "venv MiVOLO v2 ($MIVOLO_VENV)"
  local MPY=$MIVOLO_VENV/bin/python
  if [ -x "$MPY" ] && "$MPY" -c "import mivolo, torch, transformers, timm" 2>/dev/null; then
    echo "  = уже стоит"
  else
    [ -x "$MPY" ] || "$PY312" -m venv "$MIVOLO_VENV" || { warn "venv MiVOLO"; return; }
    "$MPY" -m pip install -q --upgrade pip
    "$MPY" -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 \
      && "$MPY" -m pip install -q "setuptools<70" wheel \
      && "$MPY" -m pip install -q transformers==4.51.0 accelerate==1.8.1 "timm==0.8.13.dev0" "lapx>=0.5.2" \
           scipy opencv-python-headless "numpy<2" huggingface_hub \
      && "$MPY" -m pip install -q --no-build-isolation --no-deps \
           "git+https://github.com/WildChlamydia/MiVOLO.git@37475e3f8818b5f22448003feec3e64b01bfb188" \
      || { warn "пакеты MiVOLO"; return; }
  fi
  "$MPY" -c "import importlib.util as u; assert u.find_spec('ultralytics') is None" \
    || warn "в venv MiVOLO есть ultralytics (AGPL) — не должно быть"
  # первый запуск качает веса iitolstykh/mivolo_v2 (ревизия зафиксирована в воркере) в HF_HOME
  local ready
  ready=$(echo | timeout 600 "$MPY" -I "$REPO/farm/qa/mivolo_worker.py" 2>>"$LOG" | head -1)
  case "$ready" in
    *'"ready": true'*) echo "  воркер MiVOLO: готов ($ready)" ;;
    *) warn "воркер MiVOLO не поднялся (лог: $LOG)" ;;
  esac
}

# ---------------------------------------------------------------- 5. модели
# путь в /workspace/models | URL | размер | sha256 | лицензия
MODEL_FILES=(
  "diffusion_models/z_image_turbo_bf16.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/diffusion_models/z_image_turbo_bf16.safetensors|12309866400|2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6|Apache-2.0 (Tongyi Z-Image-Turbo)"
  "text_encoders/qwen_3_4b.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/text_encoders/qwen_3_4b.safetensors|8044982048|6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a|Apache-2.0 (Qwen3-4B)"
  "vae/ae.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/vae/ae.safetensors|335304388|afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38|Apache-2.0 (автоэнкодер FLUX)"
  "face/auraface/glintr100.onnx|https://huggingface.co/fal/AuraFace-v1/resolve/af6d057c9b0ec4071d4c49c80e3539258798b609/glintr100.onnx|260694151|a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60|Apache-2.0 (AuraFace-v1)"
  "mediapipe/blaze_face_full_range.tflite|https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite|1083786||Apache-2.0"
  "mediapipe/face_landmarker.task|https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task|3758596||Apache-2.0"
  "mediapipe/hand_landmarker.task|https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task|7819105||Apache-2.0"
  "mediapipe/pose_landmarker_heavy.task|https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task|30664242||Apache-2.0"
)
# свои файлы: только проверка наличия (весов в интернете нет)
OWN_LORAS=(Stacy_v1_z-image-base-prose_3500.safetensors stacy_h3_face_v1.safetensors
           stacy_h3_face_v1_000001800.safetensors stacy_h3_A_stills_000001750.safetensors)

fetch_model() {
  local rel url size sha lic dst have
  IFS='|' read -r rel url size sha lic <<< "$1"
  dst=$MODELS/$rel
  # файл уже есть где-то в моделях (другая подпапка, models ComfyUI) — не дублируем
  have=$(find "$MODELS" "$COMFY/models" -name "$(basename "$rel")" -size "${size}c" 2>/dev/null | head -1)
  if [ -n "$have" ]; then echo "  = $(basename "$rel") ($have)"; return; fi
  mkdir -p "$(dirname "$dst")"
  echo "  ↓ $rel ($((size / 1048576)) МБ, $lic)"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$dst.part" "$url" || { warn "не скачался $rel"; return; }
  if [ "$(stat -c %s "$dst.part")" != "$size" ]; then warn "$rel: размер не совпал"; return; fi
  if [ -n "$sha" ] && [ "$(sha256sum "$dst.part" | cut -d' ' -f1)" != "$sha" ]; then
    warn "$rel: sha256 не совпал — файл удалён"; rm -f "$dst.part"; return
  fi
  mv "$dst.part" "$dst"
}

install_models() {
  log "Модели ($MODELS)"
  local m f
  for m in "${MODEL_FILES[@]}"; do fetch_model "$m"; done
  local aitk
  for f in "${OWN_LORAS[@]}"; do
    aitk=$(find "$WS/ai-toolkit/output" -name "$f" 2>/dev/null | head -1)
    if [ -n "$(find "$MODELS" "$COMFY/models" -name "$f" 2>/dev/null | head -1)" ]; then
      echo "  = $f"
    elif [ -n "$aitk" ]; then
      mkdir -p $MODELS/loras && cp -n "$aitk" $MODELS/loras/ && echo "  + $f (из $aitk)"
    else
      todo "нет LoRA $f — с ПК: scp -P <порт> -i <ключ> C:\\stacy-lora\\$f root@<IP>:$MODELS/loras/"
    fi
  done
}

# ---------------------------------------------------------------- 6. Ollama
install_ollama_models() {
  log "Ollama"
  if ! command -v ollama >/dev/null; then warn "ollama не в PATH (ставит bootstrap.sh)"; return; fi
  echo "  $(ollama -v 2>&1 | tail -1), OLLAMA_MODELS=${OLLAMA_MODELS:-?}"
  if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    warn "Ollama не отвечает на 11434 — перезапустите /workspace/bootstrap.sh"; return
  fi
  local t
  for t in $OLLAMA_TAGS; do
    if ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "$t"; then
      echo "  = $t"
    else
      echo "  ↓ $t"
      ollama pull "$t" || warn "ollama pull $t"
    fi
  done
}

# ---------------------------------------------------------------- 7. FARM_HOME и секреты
prepare_farm_home() {
  log "FARM_HOME ($FARM_HOME)"
  mkdir -p "$FARM_HOME"/{db,refs,results,cache,logs,state,secrets}
  chmod 700 "$FARM_HOME/secrets"
  local s=$FARM_HOME/secrets f
  for f in .env bot_access.yaml google_sa.json; do
    if [ -f "$s/$f" ]; then
      chmod 600 "$s/$f"; echo "  = secrets/$f"
    else
      case $f in
        .env)            todo "secrets/.env с FARM_BOT_TOKEN=… (bot/README.md §2)" ;;
        bot_access.yaml) todo "secrets/bot_access.yaml — id ревьюеров (bot/README.md §2)" ;;
        google_sa.json)  todo "secrets/google_sa.json — сервисный аккаунт таблицы (docs/SHEETS_SETUP.md)" ;;
      esac
    fi
  done
  [ -d "$REPO/profiles" ] || todo "в репо нет profiles/ — профиль sfw_base делает WS-07 (CONTRACTS §5)"
}

# ---------------------------------------------------------------- 8. бот
install_bot() {
  log "Бот ($BOT_VENV)"
  local BPY=$BOT_VENV/bin/python
  [ -x "$BPY" ] || "$PY312" -m venv "$BOT_VENV" || { warn "venv бота"; return; }
  "$BPY" -m pip install -q -r "$REPO/bot/requirements.txt" || { warn "зависимости бота"; return; }
  [ "$NO_BOT" = 1 ] && return
  if ! grep -qs '^FARM_BOT_TOKEN=' "$FARM_HOME/secrets/.env" || [ ! -f "$FARM_HOME/secrets/bot_access.yaml" ]; then
    echo "  бот не запущен: нет токена или списка доступа (см. → ниже)"; return
  fi
  if tmux has-session -t farmbot 2>/dev/null; then echo "  = farmbot уже запущен"; return; fi
  (cd "$REPO" && FARM_HOME=$FARM_HOME "$BPY" -m bot --check) || { warn "bot --check"; return; }
  tmux new-session -d -s farmbot -c "$REPO" bash -c \
    "FARM_HOME=$FARM_HOME $BPY -m bot 2>&1 | tee -a $FARM_HOME/logs/bot_main.log"
  sleep 3
  tmux has-session -t farmbot 2>/dev/null && echo "  + farmbot запущен (tmux attach -t farmbot)" \
    || warn "бот упал сразу — $FARM_HOME/logs/bot_main.log"
}

# ---------------------------------------------------------------- 9. ComfyUI
restart_comfy() {
  log "ComfyUI"
  local busy code start
  busy=$(curl -sf http://127.0.0.1:8188/queue | "$CPY" -c \
    "import json,sys;q=json.load(sys.stdin);print(len(q.get('queue_running',[]))+len(q.get('queue_pending',[])))" 2>/dev/null)
  if [ "$CHANGED" = 0 ]; then
    echo "  новых нод и пакетов нет — без перезапуска"
  elif [ "$NO_RESTART" = 1 ]; then
    todo "перезапустить ComfyUI (ставились ноды/пакеты): tmux kill-session -t comfy; /workspace/bootstrap.sh"
  elif [ -n "$busy" ] && [ "$busy" != 0 ]; then
    todo "ComfyUI занят ($busy в очереди) — перезапустить позже: tmux kill-session -t comfy; /workspace/bootstrap.sh"
  else
    start=$(( $(wc -l < $WS/logs/comfy.log 2>/dev/null || echo 0) + 1 ))
    tmux kill-session -t comfy 2>/dev/null || true
    # та же команда, что в bootstrap.sh работодателя; при лимите ОЗУ контейнера < 200 ГБ — без pinned memory и с
    # LRU-кэшем: кэш ComfyUI меряет ОЗУ хоста, а не лимит контейнера (WS-10b 23.10: OOM-kill на поде с лимитом 94 ГБ).
    # Видео 10 с FullHD (video_h3_loop@2) при лимите 94 ГБ — только с COMFY_ARGS="--cache-none": LRU держит энкодер
    # 25 ГБ, без этого OOM на стабилизаторе (WS-10, 25.10; запас с --cache-none 7–9 ГБ). Кадры — как по умолчанию.
    # WS-14 (29.10): LRU-кэш тоже копит энкодер 25 ГБ + клип FullHD 5 ГБ сверх лимита — при лимите < 200 ГБ
    # для фермы (кадры+видео) нужен именно --cache-none, а не LRU. Ставим его по умолчанию; свой COMFY_ARGS
    # пользователь может переопределить.
    local extra="${COMFY_ARGS:-}" lim
    lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
    if [ "$lim" != max ] && [ "$lim" -lt 200000000000 ]; then
      if [ -z "$extra" ]; then
        # --cache-none: без RAM-кэша ComfyUI (WS-14). --disable-pinned-memory: pinned-память
        # не считается cgroup-ом вовремя (WS-10b 23.10).
        extra="--disable-pinned-memory --cache-none"
        echo "  лимит ОЗУ контейнера $((lim / 1000000000)) ГБ → $extra (WS-14: без RAM-кэша)"
      else
        echo "  лимит ОЗУ контейнера $((lim / 1000000000)) ГБ → свой COMFY_ARGS: $extra"
      fi
    fi
    tmux new-session -d -s comfy -c "$COMFY" bash -c \
      "source $WS/env.sh; $CPY main.py --listen 127.0.0.1 --port 8188 --enable-manager $extra 2>&1 | tee -a $WS/logs/comfy.log"
    for _ in $(seq 1 240); do   # на сетевом томе (MooseFS) старт ComfyUI 10–13 мин (WS-10, 22.10)
      code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8188/system_stats || true)
      [ "$code" = 200 ] && break
      sleep 5
    done
    echo "  перезапущен: HTTP ${code:-000}"
    [ "${code:-}" = 200 ] || warn "ComfyUI не поднялся за 20 мин — $WS/logs/comfy.log"
    local errs
    errs=$(tail -n +"$start" $WS/logs/comfy.log | grep -E "IMPORT FAILED|Cannot import" | tail -10)
    [ -n "$errs" ] && { echo "$errs" | sed 's/^/  ! /'; FAIL=1; }
    tail -n +"$start" $WS/logs/comfy.log | grep -m1 "воркер фермы запущен" | sed 's/^/  /'
  fi
  echo "  pod_doctor: ноды и модели движков"
  "$CPY" "$REPO/install/pod_doctor.py" | sed 's/^/  /'
  [ "${PIPESTATUS[0]}" = 0 ] || warn "pod_doctor нашёл недостающее (см. выше)"
}

main() {
  echo "=== pod_setup $(date -u +%FT%TZ) ==="
  check_env
  install_repo
  install_nodes
  install_qa_deps
  install_mivolo
  [ "$SKIP_MODELS" = 1 ] || install_models
  [ "$SKIP_OLLAMA" = 1 ] || install_ollama_models
  prepare_farm_home
  install_bot
  restart_comfy
  rm -f "$CONS"
  if [ ${#TODO[@]} -gt 0 ]; then
    log "Сделать руками"
    printf '  → %s\n' "${TODO[@]}"
  fi
  if [ "$FAIL" = 0 ]; then echo "======== ФЕРМА: ГОТОВО ========"; else echo "======== ФЕРМА: ЕСТЬ ОШИБКИ (см. ! выше, лог $LOG) ========"; fi
  echo "Ферма после установки на паузе: пульт — нода Farm Control или POST /farm/control {\"action\":\"start\"}."
  return "$FAIL"
}

# Вся логика — в функциях, вызов в конце: bash читает файл по ходу выполнения, а шаг 1 обновляет репо
# (и этот файл) через git pull — так правка скрипта посреди прогона ничего не сломает.
mkdir -p $WS/logs
main "$@" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
