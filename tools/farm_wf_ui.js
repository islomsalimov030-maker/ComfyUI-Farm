// Сборка UI farm_auto@1 / farm_manual@1 в ComfyUI (23.10): API с ролями → новая вкладка → сабграфы движков →
// группы по блокам и моделям → переключатель групп (rgthree) → сохранение в Workflows.
// Запуск в консоли страницы ComfyUI:  await farmBuild("farm_manual@1")   (или "farm_auto@1")
window.farmBuild = async (name, opts = {}) => {
  const RAW = `https://raw.githubusercontent.com/islomsalimov030-maker/ComfyUI-Farm/${opts.branch || "ws10-pod-1022"}/workflows/${name}.api.json`;
  const app = window.app || window.comfyAPI.app.app;
  const LG = window.LiteGraph || window.comfyAPI?.litegraph?.LiteGraph;
  const Group = window.LGraphGroup || window.comfyAPI?.litegraph?.LGraphGroup || LG.LGraphGroup;
  const api0 = await (await fetch(RAW, { cache: "no-store" })).json();
  const MODELS = ["ref2v", "10Eros", "PinkCherry"];

  // новая вкладка, чтобы не трогать открытый воркфлоу
  try { await app.extensionManager.command.execute("Comfy.NewBlankWorkflow"); } catch (e) { /* старый фронтенд */ }

  const keys = Object.keys(api0), idmap = {}, role = {};
  keys.forEach((k, i) => (idmap[k] = String(i + 1)));
  const api = {};
  for (const k of keys) {
    const n = structuredClone(api0[k]);
    for (const [a, v] of Object.entries(n.inputs))
      if (Array.isArray(v) && v.length === 2 && typeof v[0] === "string" && idmap[v[0]] !== undefined) n.inputs[a] = [idmap[v[0]], v[1]];
    api[idmap[k]] = n; role[idmap[k]] = n._meta?.role || "?";
  }
  await app.loadApiJson(api, name);
  const N = keys.length;
  for (let i = 0; i < 100 && (app.graph._nodes?.length || 0) < N; i++) await new Promise((r) => setTimeout(r, 200));
  const g = app.graph;
  // роли по порядку создания нод (id после загрузки не совпадают с ключами API), сверка по заголовкам
  const byNode = new Map();
  const titles = keys.map((k) => api[idmap[k]]._meta?.title || "");
  const ordered = g._nodes.slice(-N);
  ordered.forEach((n, i) => { if ((n.title || "") === titles[i] || !titles[i]) byNode.set(n, role[idmap[keys[i]]]); });
  const R = (n) => byNode.get(n) || role["sub:" + n.id] || "?";
  const nodes = (pred) => g._nodes.filter((n) => pred(R(n), n));

  // сабграфы движков
  const sg = {};
  const engineRoles = [...new Set(Object.values(role).filter((r) => r.endsWith(":engine") && !r.startsWith("mask")))];
  const nice = (r) => {
    const m = r.match(/^(photo|video)(\d)/);
    if (m) return `${m[1] === "photo" ? "ДВИЖОК ФОТО" : "ДВИЖОК ВИДЕО"} · ${MODELS[+m[2] - 1]} (двойной клик — внутрь)`;
    if (r.startsWith("facepass")) return "ДВИЖОК ПРОХОДА ЛИЦА готового видео (двойной клик — внутрь)";
    return r;
  };
  for (const r of engineRoles) {
    const set = nodes((x) => x === r);
    if (!set.length) continue;
    const res = g.convertToSubgraph(new Set(set));
    const node = res.node || res;
    node.title = nice(r);
    if (res.subgraph) res.subgraph.name = nice(r);
    role["sub:" + node.id] = r.replace(":engine", ":sub");
    sg[r] = node;
  }
  const mk = nodes((x) => x === "mask:engine");
  if (mk.length) { const res = g.convertToSubgraph(new Set(mk)); const nd = res.node || res; nd.title = "маска: подготовка"; role["sub:" + nd.id] = "mask:sub"; }

  // раскладка и группы
  let X = 0;
  const groups = [];
  const place = (list, x, y, gap = 16) => {
    for (const n of list) {
      if (/^имя:/.test(n.title)) n.collapse?.(true);
      n.pos = [x, y]; y += (n.flags?.collapsed ? 30 : (n.size?.[1] || 80)) + gap;
    }
    return y;
  };
  const group = (title, list, color, x, w = 480) => {
    if (!list.length) return;
    const y1 = place(list, x + 20, 80);
    const gr = new Group(title); gr.color = color; g.add(gr);
    gr.pos = [x, 0]; gr.size = [Math.max(w, ...list.map((n) => (n.size?.[0] || 300) + 40)), y1 + 20];
    groups.push(gr);
    return gr.size[0];
  };
  const col = (title, pred, color, w) => { const l = nodes(pred); const ww = group(title, l, color, X, w); if (ww) X += ww + 60; return l; };
  const isAuto = name.startsWith("farm_auto");
  if (isAuto) col("ТАБЛИЦА — row + increment, Queue ×N = N строк", (r) => r === "table", "#3f789e");
  col("ФОТО · входы и пульт (общие для всех моделей)",
      (r) => r === "photo:in" || r === "photo1:in" || r === "photo1:knob" || r === "photo:names" || r === "mask" || r === "mask:sub", "#2f7d4a", 520);
  const muted = [];
  MODELS.forEach((m, i) => {
    const l = col(`ФОТО · модель ${m}`, (r) => new RegExp(`^photo${i + 1}:(arm|sub|out)$`).test(r) || (i > 0 && r === `photo${i + 1}:knob`), "#5b3a6e");
    if (i > 0) muted.push(...l);
  });
  if (isAuto) col("ZIP фото", (r) => r === "photo:zip", "#444");
  if (isAuto) col("ПЕРЕКЛЮЧАТЕЛЬ: видео из свежих фото (за один запуск) / из папки готовых кадров", (r) => r === "switch", "#8a5a00", 560);
  col("ВИДЕО · входы и пульт (общие для всех моделей)",
      (r) => r === "video:in" || r === "video1:in" || r === "video:panel" || r === "video1:panel" || r === "video1:knob" || r === "video:names", "#2f7d4a", 520);
  MODELS.forEach((m, i) => {
    const l = col(`ВИДЕО · модель ${m}`, (r) => new RegExp(`^video${i + 1}:(arm|sub|out)$`).test(r) || (i > 0 && r === `video${i + 1}:knob`), "#5b3a6e");
    if (i > 0) muted.push(...l);
  });
  if (isAuto) col("ZIP видео", (r) => r === "video:zip", "#444");
  if (!isAuto) muted.push(...col("ПРОХОД ЛИЦА ГОТОВОГО ВИДЕО — отдельно (включать, когда хватает памяти)", (r) => r.startsWith("facepass"), "#6e2e2e", 560));
  for (const n of muted) n.mode = 2;   // выключено (mute): не выполняется и память не занимает

  // переключатель групп rgthree
  try {
    const sw = LG.createNode("Fast Groups Muter (rgthree)");
    sw.title = "ВКЛ/ВЫКЛ блоков и моделей (выключенные не грузят модели)";
    sw.pos = [0, -420]; g.add(sw);
    sw.properties = { ...(sw.properties || {}), matchTitle: "^(ФОТО · модель|ВИДЕО · модель|ПРОХОД ЛИЦА|ZIP|ПЕРЕКЛЮЧАТЕЛЬ)", sort: "position" };
  } catch (e) { console.warn("rgthree muter", e); }
  g.setDirtyCanvas?.(true, true);

  const wf = g.serialize();
  wf.extra = { ...(wf.extra || {}), farm: { workflow: name, built: new Date().toISOString() } };
  const r = await fetch(`/api/userdata/workflows%2F${encodeURIComponent(name)}.json?overwrite=true`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(wf),
  });
  return { save: r.status, mapped: byNode.size, nodes: g._nodes.length, subgraphs: (wf.definitions?.subgraphs || []).length, groups: groups.length };
};
