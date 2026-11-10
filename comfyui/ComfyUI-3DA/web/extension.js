import { app } from '../../../scripts/app.js';
import { lockCamera } from './cameraLock.js';
import { sceneSelection } from './sceneSelection.js';

// The installed frontend has legacy extension hooks but no public Load3d
// service shim. Discover ONLY the service module actually loaded by core,
// then verify its capabilities. No pinned bundle hash or invented SDK.
let servicePromise;
async function coreService() {
  if (servicePromise) return servicePromise;
  const url = performance.getEntriesByType('resource').map(e => e.name)
    .find(u => new URL(u).origin === location.origin &&
      /\/assets\/load3dService-[\w-]+\.js$/.test(new URL(u).pathname));
  if (!url) return null;
  servicePromise = import(url).then(mod => {
    for (const candidate of Object.values(mod)) {
      if (typeof candidate !== 'function' || candidate.name !== 'useLoad3dService') continue;
      const service = candidate();
      if (typeof service?.getLoad3dAsync === 'function') return service;
    }
    throw new Error('3DA: core Load3D service API изменился');
  });
  return servicePromise;
}

function protectedState(node) {
  if (node.type !== 'Load3DAdvanced') return null;
  // Old safe presets are protected as well, but ordinary Load3D nodes retain
  // their navigation. Follow the real camera link, not a title substring.
  const linked = (node.outputs?.[2]?.links ?? []).some(id => {
    const l = node.graph?.links?.get?.(id) ?? node.graph?.links?.[id];
    return node.graph?.getNodeById(l?.target_id)?.type === '3DAProp';
  });
  let state = node.properties?.['3DA Camera Lock']?.state ?? node.properties?.['3DA Scene']?.camera;
  if (!state && linked && node.properties?.['Camera Config']?.state) {
    // Freeze a legacy safe preset's room camera once. Core's mutable Camera
    // Config is a snapshot destination, not the source of truth for a lock.
    state = JSON.parse(JSON.stringify(node.properties['Camera Config'].state));
    node.properties['3DA Camera Lock'] = { state };
  }
  return state ?? null;
}

const active = new Map();
const viewRecords = new WeakMap();
const clones = new Map();
let copyAdapter;
async function restoreRoom(record) {
  const setup = record.node.properties['3DA Scene'];
  if (!setup?.background || record.roomPending) return;
  const view = record.view;
  const scene = view.getSceneManager();
  const config = record.node.properties['Scene Config'] ??= {};
  // Core's node/fullscreen scene state can serialize a blank background.
  // The exported room plate is authoritative, independently of that state.
  config.backgroundImage = setup.background;
  config.backgroundRenderMode = 'tiled';
  if (record.roomReady === setup.background && scene.getCurrentBackgroundInfo().type === 'image') return;
  record.roomPending = true;
  try {
    view.toggleGrid(false);
    config.showGrid = false;
    view.setBackgroundRenderMode('tiled');
    await view.setBackgroundImage(setup.background);
    record.roomReady = setup.background;
    record.lock.restore();
    view.forceRender();
  } finally { record.roomPending = false; }
}
function dispose(record) {
  record.scene?.dispose();
  record.lock.dispose();
  const widget = record.node.widgets?.find(w => w.name === 'viewport_state');
  if (widget?.serializeValue === record.serialize) widget.serializeValue = record.originalSerialize;
  viewRecords.delete(record.view);
}
function adaptCopies(service) {
  if (copyAdapter) return;
  const original = service.copyLoad3dState;
  if (typeof original !== 'function') return;
  const adapted = async function (source, target, ...args) {
    const owner = viewRecords.get(source) ?? viewRecords.get(target);
    const transforms = viewRecords.get(source)?.scene?.capture();
    const result = await original.call(this, source, target, ...args);
    if (!owner) return result;
    let record = viewRecords.get(target);
    if (!record) {
      record = { node:owner.node, view:target, lock:lockCamera(target, protectedState(owner.node)) };
      clones.set(target, record);viewRecords.set(target, record);
      const remove = target.remove;
      target.remove = function (...args) {
        dispose(record);clones.delete(target);target.remove=remove;
        return remove.apply(this,args);
      };
    }
    record.scene?.dispose();
    record.scene = sceneSelection(target, owner.node, transforms);
    record.lock.restore();
    await restoreRoom(record);
    if (record.scene) owner.node.properties['3DA Scene Transforms'] = record.scene.capture();
    return result;
  };
  service.copyLoad3dState = adapted;
  copyAdapter = () => { if (service.copyLoad3dState === adapted) service.copyLoad3dState = original; };
}
let busy = false;
async function tick() {
  if (busy) return;
  busy = true;
  try {
    const graphs = [app.graph, ...Object.values(app.graph?.subgraphs ?? {})];
    const nodes = graphs.flatMap(g => g?._nodes ?? []);
    const wanted = new Set(nodes.filter(n => protectedState(n)));
    for (const [node, record] of active) {
      if (!wanted.has(node)) { dispose(record); active.delete(node); }
    }
    if (!wanted.size) return;
    const service = await coreService();
    if (!service) return;
    adaptCopies(service);
    for (const node of wanted) {
      const view = await service.getLoad3dAsync(node);
      if (!view) continue;
      const state = protectedState(node);
      const key = JSON.stringify(state);
      let record = active.get(node);
      if (record && (record.view !== view || record.key !== key)) {
        dispose(record); active.delete(node); record = null;
      }
      if (!record) {
        record = { node, view, key, lock: lockCamera(view, state) };
        active.set(node, record);
        viewRecords.set(view, record);
      }
      record.lock.restore();
      if (node.properties['3DA Scene']) {
        const root = view.getModelManager().currentModel;
        if (!record.scene || record.scene.root !== root) {
          record.scene?.dispose();
          record.scene = sceneSelection(view, node);
        }
        if (record.scene) {
          record.scene.syncMode();
          const transforms = record.scene.capture();
          if (JSON.stringify(node.properties['3DA Scene Transforms']) !== JSON.stringify(transforms))
            node.properties['3DA Scene Transforms'] = transforms;
        }
        await restoreRoom(record);
      }
      const widget = node.widgets?.find(w => w.name === 'viewport_state');
      if (widget?.serializeValue && widget.serializeValue !== record.serialize) {
        record.originalSerialize = widget.serializeValue;
        record.serialize = async function (...args) {
          record.lock.restore();
          const result = await record.originalSerialize.apply(this, args);
          if (node.properties['3DA Scene']) {
            if (!record.scene) throw new Error('3DA: объекты сцены ещё не загружены');
            const transforms = record.scene.capture();
            node.properties['3DA Scene Transforms'] = transforms;
            result.model_3d_info = transforms;
          }
          return result;
        };
        widget.serializeValue = record.serialize;
      }
    }
    for (const record of clones.values()) { record.lock.restore(); await restoreRoom(record); }
  } catch (e) { console.error('3DA camera protection:', e); }
  finally { busy = false; }
}

app.registerExtension({
  name: 'NextGenium.3DA.CameraLock',
  getNodeMenuItems(node) {
    const record = active.get(node);
    if (!record) return [];
    const restore = { content:'3DA · Восстановить только камеру комнаты', callback:() => record.lock.restore() };
    if (!record.scene) return [restore];
    return [restore, ...['translate','rotate','scale'].map((mode,i) => ({
      content:'3DA · '+['Перемещение','Вращение','Масштаб'][i],
      callback:() => record.scene.setMode(mode)
    })), ...record.scene.entities.map(e => ({
      content: (record.scene.selected() === e.id ? '✓ ' : '') + 'Выбрать · ' + e.label,
      callback: () => record.scene.select(e.id)
    }))];
  },
  nodeCreated(node) {
    // LiteGraph can assign node.type after the constructor lifecycle hook.
    if ((node.constructor?.comfyClass ?? node.type) !== '3DASceneExport') return;
    const original = node.onExecuted;
    node.onExecuted = function (message) {
      original?.call(this, message);
      const setup = JSON.parse(message.text[0]);
      const graph = node.graph;
      const viewport = graph._nodes.find(n => n.title === '3DA · Сцена: персонаж и предметы');
      const edits = graph._nodes.find(n => n.title === '3DA · Применить правки сцены');
      if (!viewport || !edits) return;
      viewport.properties['3DA Scene'] = setup;
      viewport.properties['3DA Camera Lock'] = { state: setup.camera };
      delete viewport.properties['3DA Scene Transforms'];
      viewport.properties['3DA Selected Entity'] = 'body';
      viewport.properties['Camera Config'] = { cameraType:'perspective', fov:setup.camera.fov, state:setup.camera, useCustomUp:true, hasCustomUp:true };
      viewport.properties['Scene Config'] = { showGrid:false, backgroundImage:setup.background, backgroundRenderMode:'tiled' };
      viewport.properties['Model Config'] = { upDirection:'original', materialMode:'original', gizmo:{ enabled:true, mode:'translate', position:{x:0,y:0,z:0},rotation:{x:0,y:0,z:0},scale:{x:1,y:1,z:1} } };
      const widget = viewport.widgets.find(w => w.name === 'model_file');
      widget.value = setup.model_file;
      widget.callback?.(setup.model_file);
      edits.widgets.find(w => w.name === 'manifest').value = setup.manifest;
      app.canvas.setDirty(true,true);
    };
  },
  setup() {
    const timer = setInterval(tick, 100);
    window.addEventListener('pagehide', () => {
      clearInterval(timer);
      for (const record of active.values()) dispose(record);
      for (const record of clones.values()) dispose(record);
      copyAdapter?.();clones.clear();
      active.clear();
    }, { once: true });
  }
});
