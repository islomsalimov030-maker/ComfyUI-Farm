// «Таблица → промпты» (NGFarmTablePrompts): загрузка xlsx в input/farm_tables, профилей и системных промптов
// (yaml/md) в input/farm_profiles и скачивание итога.
// Штатная кнопка загрузки ComfyUI принимает только картинки/видео/аудио — xlsx грузим тем же /upload/image.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "NGFarmTablePrompts";
const SUBDIR = "farm_tables";
const PROFILES = "farm_profiles";

function pickFiles(accept, multiple = false) {
  return new Promise((resolve) => {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = accept;
    inp.multiple = multiple;
    inp.style.display = "none";
    inp.onchange = () => { resolve(Array.from(inp.files || [])); inp.remove(); };
    document.body.appendChild(inp);
    inp.click();
  });
}

function upload(file, subfolder) {
  const body = new FormData();
  body.append("image", file, file.name);
  body.append("subfolder", subfolder);
  body.append("type", "input");
  body.append("overwrite", "true");
  return api.fetchApi("/upload/image", { method: "POST", body });
}

async function uploadProfiles() {
  const files = await pickFiles(".yaml,.yml,.md", true);
  if (!files.length) return;
  const ok = [], bad = [];
  for (const f of files) {
    const r = await upload(f, PROFILES);
    (r.status === 200 ? ok : bad).push(r.status === 200 ? f.name : `${f.name} (${r.status})`);
  }
  alert((ok.length ? `В input/${PROFILES}/: ${ok.join(", ")}` : "") + (bad.length ? `\nНе загрузились: ${bad.join(", ")}` : ""));
}

async function uploadTable(node) {
  const [file] = await pickFiles(".xlsx");
  if (!file) return;
  const r = await upload(file, SUBDIR);
  if (r.status !== 200) { alert(`Таблица не загрузилась: ${r.status} ${r.statusText}`); return; }
  const d = await r.json();
  const value = d.subfolder ? `${d.subfolder}/${d.name}` : d.name;
  const w = node.widgets?.find((x) => x.name === "table");
  if (w) {
    const vals = w.options?.values;
    if (Array.isArray(vals) && !vals.includes(value)) vals.unshift(value);
    w.value = value;
    w.callback?.(value);
  }
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "NextGenium.Farm.TablePrompts",
  async nodeCreated(node) {
    if (node.comfyClass !== NODE) return;
    node.addWidget("button", "Загрузить таблицу (xlsx)", null, () => uploadTable(node), { serialize: false });
    node.addWidget("button", "Загрузить профиль / промпт (yaml, md)", null, () => uploadProfiles(), { serialize: false });
    let dl = null;
    const onExecuted = node.onExecuted;
    node.onExecuted = function (msg) {
      onExecuted?.apply(this, arguments);
      const url = msg?.farm_xlsx?.[0];
      if (!url) return;
      if (!dl) dl = node.addWidget("button", "Скачать таблицу с промптами", null, () => {
        window.open(api.apiURL(node.__farmXlsx), "_blank");
      }, { serialize: false });
      node.__farmXlsx = url;
      node.setDirtyCanvas?.(true, true);
    };
  },
});
