#!/usr/bin/env bash
# Ставит скилы и агентов фермы для Claude Code на поде.
# Конфиг Claude Code на поде живёт в /workspace/.claude (переживает Terminate).
# Запуск:  bash /workspace/farm/kit/install/install_claude_kit_on_pod.sh
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${CLAUDE_CONFIG_DIR:-/workspace/.claude}"
mkdir -p "$CFG/skills" "$CFG/agents"

for s in h3-prompt-writer comfyui-workflow-json stacy-farm-ops; do
  rm -rf "$CFG/skills/$s"; cp -r "$KIT/skills/$s" "$CFG/skills/$s"
done
for s in "$KIT"/skills/vendor-comfyui-mcp/*/; do
  n="$(basename "$s")"; rm -rf "$CFG/skills/$n"; cp -r "$s" "$CFG/skills/$n"
done
cp "$KIT/skills/vendor-comfyui-mcp/LICENSE" "$CFG/skills/LICENSE.comfyui-mcp"
cp "$KIT"/agents/*.md "$CFG/agents/"

echo "Скилы:";  ls "$CFG/skills"
echo "Агенты:"; ls "$CFG/agents"
echo "Готово. Перезапусти claude, спроси «какие скилы и агенты доступны?»."
