// Пульты ФОТО / ВИДЕО (NGFarmPanelPhoto / NGFarmPanelVideo): миграция сохранённых воркфлоу.
// Фронтенд раскладывает widgets_values по позиции. 27.10 use_Singularity встал первым включателем,
// поэтому старые раскладки переводим в новую:
//   A — до c25864e: use_Singularity нет (на одно значение меньше)          → [false, ...старые]
//   B — c25864e:    use_Singularity в конце, после model_override (булево) → переносим в начало
//   новая: последнее значение — строка model_override                      → без изменений
import { app } from "../../scripts/app.js";

const PANELS = new Set(["NGFarmPanelPhoto", "NGFarmPanelVideo"]);

function migrate(node, info) {
    const v = info?.widgets_values;
    const ws = node.widgets ?? [];
    const n = ws.length;
    if (!Array.isArray(v) || !n || ws[0]?.name !== "use_Singularity") return;
    let nv = null, kind = "";
    if (v.length === n - 1 && typeof v[0] === "boolean" && typeof v[n - 2] === "string") {
        nv = [false, ...v]; kind = "A (без Singularity)";
    } else if (v.length === n && typeof v[n - 1] === "boolean" && typeof v[n - 2] === "string") {
        nv = [v[n - 1], ...v.slice(0, n - 1)]; kind = "B (Singularity в конце)";
    }
    if (!nv) return;
    ws.forEach((w, i) => { w.value = nv[i]; });
    node.setDirtyCanvas?.(true, true);
    console.info(`[Farm.Panel] ${node.comfyClass} #${node.id}: раскладка ${kind} переведена в новую`);
}

app.registerExtension({
    name: "Farm.Panels",
    nodeCreated(node) {
        if (!PANELS.has(node.comfyClass)) return;
        const oc = node.onConfigure;
        node.onConfigure = function (...args) {
            const r = oc?.apply(this, args);
            migrate(node, args[0]);
            return r;
        };
    },
});
