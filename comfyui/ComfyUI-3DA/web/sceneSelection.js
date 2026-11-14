// Named-object selection on the existing core gizmo, not another 3D editor.
import { parseObj, cameraModel, buildRoom, diagnose as roomDiagnose, occluderGeometry, worldVertices, bringToBody, centerOnAxis, centerOnFloor, padStep } from './roomGeometry.js';
export function sceneSelection(view, node, initialTransforms = null) {
  const setup = node.properties['3DA Scene'];
  if (!setup?.entities?.length) return null;
  const root = view.getModelManager().currentModel;
  if (!root) return null;
  const entities = setup.entities.map(e => ({ ...e, object: root.getObjectByName(e.name) }));
  if (entities.some(e => !e.object)) return null; // model load is still pending
  // Core stores one selected transform in Model Config. A scene has separate
  // entity transforms; never apply the selected body's TRS to the whole GLB.
  root.position.set(0, 0, 0); root.quaternion.set(0, 0, 0, 1); root.scale.set(1, 1, 1);
  const saved = initialTransforms ?? node.properties['3DA Scene Transforms'];
  if (saved) for (const e of entities) {
    const t = saved.find(t => t.entity_id === e.id);
    if (!t) continue;
    e.object.position.set(t.position.x, t.position.y, t.position.z);
    e.object.quaternion.set(t.quaternion.x, t.quaternion.y, t.quaternion.z, t.quaternion.w);
    e.object.scale.set(t.scale.x, t.scale.y, t.scale.z);
  }
  function capture() {
    root.updateMatrixWorld(true);
    return entities.map(e => {
      const p = e.object.getWorldPosition(e.object.position.clone());
      const q = e.object.getWorldQuaternion(e.object.quaternion.clone());
      const s = e.object.getWorldScale(e.object.scale.clone());
      return { entity_id: e.id, position: { x:p.x,y:p.y,z:p.z },
        quaternion: { x:q.x,y:q.y,z:q.z,w:q.w }, scale: { x:s.x,y:s.y,z:s.z } };
    });
  }
  let selected;
  const initialMode = node.properties['3DA Gizmo Mode'] ?? node.properties['Model Config']?.gizmo?.mode ?? 'translate';
  const toolbar = document.createElement('div');
  toolbar.className = 'threeda-scene-toolbar';
  Object.assign(toolbar.style, { position:'absolute', top:'8px', left:'8px', zIndex:'15',
    display:'flex', gap:'6px', flexWrap:'wrap', maxWidth:'95%' });
  const picker = document.createElement('select');
  picker.setAttribute('aria-label', 'Объект 3DA');
  picker.title = 'Выберите настоящий 3D-объект для гизмо';
  picker.className = 'threeda-entity-selector';
  Object.assign(picker.style, { color:'white', background:'#263744', padding:'5px', maxWidth:'240px' });
  for (const e of entities) picker.add(new Option(e.label, e.id));
  const modes = document.createElement('select');
  modes.className = 'threeda-mode-selector';
  modes.setAttribute('aria-label', 'Режим гизмо 3DA');
  Object.assign(modes.style, { color:'white', background:'#263744', padding:'5px' });
  for (const [id,label] of [['translate','Перемещение'],['rotate','Вращение'],['scale','Масштаб']])
    modes.add(new Option(label,id));
  // Live diagnostics against the room geometry of the location (any floor,
  // several levels or no floor): the same mesh the renderer occludes with.
  const status = document.createElement('span');
  status.className = 'threeda-scene-status';
  Object.assign(status.style, { color:'white', padding:'5px 8px', borderRadius:'4px',
    font:'12px sans-serif', background:'#2e7d32', whiteSpace:'pre-line' });
  // 0.3.5: предмет вне кадра не двигать вслепую — кнопка ставит выбранный предмет перед персонажем.
  const toFrame = document.createElement('button');
  toFrame.className = 'threeda-to-frame';
  toFrame.textContent = 'В кадр';
  toFrame.title = 'Предмет — перед персонажем; персонаж — в центр кадра (дальше двигайте гизмо)';
  Object.assign(toFrame.style, { color:'white', background:'#263744', padding:'5px 8px', border:'1px solid #4b6070', cursor:'pointer' });
  toolbar.append(picker,modes,toFrame,status);
  const parent = view.domElement.parentElement;
  parent?.appendChild(toolbar);
  for (const event of ['pointerdown','pointerup','click','dblclick','keydown'])
    toolbar.addEventListener(event, e => e.stopPropagation());
  picker.addEventListener('change', () => select(picker.value));
  toFrame.addEventListener('click', () => {
    const body = entities.find(e => e.kind === 'body');
    if (!selected || !body) return;
    // 0.3.17: персонаж — центр тела в центр кадра (его стрелки гизмо снова видны); предмет — перед персонажем.
    // 0.3.22: персонаж — по горизонтали в центр кадра, ноги на пол (без пола — высота прежняя)
    const delta = selected.kind === 'body' ? centerOnFloor(worldVertices(body.object), setup.camera, room?.floorY ?? null)
      : bringToBody(worldVertices(body.object), worldVertices(selected.object), setup.camera.position);
    if (!delta) return;
    selected.object.position.set(selected.object.position.x + delta[0], selected.object.position.y + delta[1],
                                 selected.object.position.z + delta[2]);
    root.updateMatrixWorld(true);
    view.forceRender();
    refresh();
  });
  let room = null, occluder = null, roomError = null, disposed = false;
  loadRoom(setup).then(loaded => {
    if (disposed || !loaded) return;
    room = loaded;
    occluder = roomOccluder(view, entities, loaded);
    refresh();
  }).catch(e => { roomError = e; console.error('3DA room:', e); refresh(); });
  const gizmo = view.getGizmoManager();
  const controls = gizmo.transformControls;
  let lastMode;
  let scaleStart;
  const beginScale = () => {
    scaleStart = selected && gizmo.getMode() === 'scale'
      ? selected.object.scale.clone() : null;
  };
  const uniformScale = () => {
    if (!scaleStart || !selected || gizmo.getMode() !== 'scale') return;
    // Use the real core handle, including X/Y/Z. Each handle changes size
    // uniformly for a character and for props: no stretching along one axis.
    const axis = (controls.axis ?? 'X')[0].toLowerCase();
    const ratio = selected.object.scale[axis] / scaleStart[axis];
    if (Number.isFinite(ratio) && ratio > 0) selected.object.scale.copy(scaleStart).multiplyScalar(ratio);
    else selected.object.scale.copy(scaleStart);
  };
  const endScale = () => { scaleStart = null; };
  let pending = false;
  const refresh = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; status.textContent = ''; diagnose(); });
  };
  function diagnose() {
    let lines;
    if (room) lines = roomDiagnose(room.analysis, room.camera, entities);
    else if (roomError) lines = [{ level:'warn', text:'Геометрия комнаты не загрузилась — проверки недоступны' }];
    else if (!setup.room_mesh) lines = [{ level:'warn', text:'Сцена без геометрии комнаты (старый экспорт) — выполните группу 1 заново' }];
    else lines = [{ level:'info', text:'Загружаю геометрию комнаты…' }];
    status.textContent = lines.map(l => l.text).join('\n');
    const worst = ['error','warn','info','ok'].find(level => lines.some(l => l.level === level));
    status.style.background = { error:'#b71c1c', warn:'#e65100', info:'#37474f', ok:'#2e7d32' }[worst];
  }
  controls?.addEventListener('mouseDown', beginScale);
  controls?.addEventListener('objectChange', uniformScale);
  controls?.addEventListener('mouseUp', endScale);
  controls?.addEventListener('objectChange', refresh);
  controls?.addEventListener('mouseUp', refresh);
  function syncMode() {
    const mode = gizmo.getMode();
    if (mode !== lastMode) {
      modes.value = mode;
      node.properties['3DA Gizmo Mode'] = mode;
      lastMode = mode;
    }
    // Body and props both scale uniformly (props since 0.1.9).
    modes.options[2].text = 'Масштаб (равномерно)';
  }
  function setMode(mode) {
    if (!['translate','rotate','scale'].includes(mode)) return;
    view.setGizmoMode(mode);
    syncMode();
  }
  modes.addEventListener('change', () => setMode(modes.value));
  function select(id) {
    const entity = entities.find(e => e.id === id);
    if (!entity) throw new Error('3DA: неизвестный объект');
    // setupForModel changes only the active TransformControls target.
    view.getGizmoManager().setupForModel(entity.object);
    view.setGizmoEnabled(true);
    selected = entity;
    picker.value = id;
    toFrame.disabled = !['prop','body'].includes(entity.kind);
    toFrame.style.opacity = toFrame.disabled ? '0.4' : '1';
    node.properties['3DA Selected Entity'] = id;
    syncMode();
    view.forceRender();
  }
  const pad = remotePad(view, setup, () => selected, () => gizmo.getMode(), () => {
    root.updateMatrixWorld(true); view.forceRender(); refresh(); });
  const savedSelection = node.properties['3DA Selected Entity'];
  select(entities.some(e => e.id === savedSelection) ? savedSelection : entities[0].id);
  setMode(initialMode);
  diagnose();
  return { root, entities, capture, select, setMode, syncMode, selected: () => selected.id,
    dispose: () => {
      controls?.removeEventListener('mouseDown', beginScale);
      controls?.removeEventListener('objectChange', uniformScale);
      controls?.removeEventListener('mouseUp', endScale);
      controls?.removeEventListener('objectChange', refresh);
      controls?.removeEventListener('mouseUp', refresh);
      disposed = true;
      occluder?.dispose();
      pad.dispose();
      toolbar.remove(); } };
}

// 0.3.18: пульт вместо помощника осей (нижний левый угол): тяните цветные кружки — выбранный объект
// двигается/вращается/масштабируется по режиму гизмо, даже если его стрелки ушли за кадр.
function remotePad(view, setup, getSelected, getMode, changed) {
  const helper = view.viewHelperManager;
  try { helper?.visibleViewHelper?.(false); } catch (e) { /* старое ядро — пульт просто накрывает угол */ }
  if (helper?.viewHelperContainer) helper.viewHelperContainer.style.display = 'none';
  const NS = 'http://www.w3.org/2000/svg';
  const pad = document.createElement('div');
  pad.className = 'threeda-remote-pad';
  pad.title = 'Пульт: тяните кружки — красный вправо/влево, зелёный вверх/вниз, синий ближе/дальше. ' +
    'Работает по режиму (перемещение, вращение, масштаб) с выбранным объектом, даже если его стрелки за кадром.';
  Object.assign(pad.style, { position:'absolute', left:'0', bottom:'0', width:'128px', height:'128px', zIndex:'15',
    background:'rgba(20,24,28,0.92)', borderTopRightRadius:'6px', touchAction:'none', cursor:'grab' });
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 128 128'); svg.setAttribute('width', '128'); svg.setAttribute('height', '128');
  const line = (x2, y2, color) => { const l = document.createElementNS(NS, 'line');
    Object.entries({ x1: 64, y1: 64, x2, y2, stroke: color, 'stroke-width': 4 }).forEach(([k, v]) => l.setAttribute(k, v));
    svg.appendChild(l); };
  const dot = (cx, cy, color, handle) => { const c = document.createElementNS(NS, 'circle');
    Object.entries({ cx, cy, r: 11, fill: color, stroke: '#111', 'stroke-width': 2 }).forEach(([k, v]) => c.setAttribute(k, v));
    c.dataset.handle = handle; c.style.cursor = 'pointer'; svg.appendChild(c); };
  line(110, 64, '#ff4d5e'); line(64, 18, '#7ee05b');
  dot(110, 64, '#ff4d5e', 'x'); dot(64, 18, '#7ee05b', 'y'); dot(64, 64, '#3f8cff', 'z');
  const label = document.createElementNS(NS, 'text');
  Object.entries({ x: 6, y: 120, fill: '#cfd8dc', 'font-size': 11, 'font-family': 'sans-serif' })
    .forEach(([k, v]) => label.setAttribute(k, v));
  svg.appendChild(label);
  pad.appendChild(svg);
  view.domElement.parentElement?.appendChild(pad);
  const names = { translate: 'перемещение', rotate: 'вращение', scale: 'масштаб' };
  const showMode = () => { label.textContent = 'пульт · ' + (names[getMode()] ?? ''); };
  showMode();
  const timer = setInterval(showMode, 500);
  let drag = null;
  const stop = e => e.stopPropagation();
  for (const ev of ['pointerdown','pointermove','pointerup','click','dblclick','wheel','contextmenu']) pad.addEventListener(ev, stop);
  pad.addEventListener('pointerdown', e => {
    const selected = getSelected();
    if (!selected) return;
    drag = { handle: e.target?.dataset?.handle ?? 'free', x: e.clientX, y: e.clientY };
    pad.setPointerCapture(e.pointerId);
    pad.style.cursor = 'grabbing';
  });
  pad.addEventListener('pointermove', e => {
    const selected = getSelected();
    if (!drag || !selected) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (!dx && !dy) return;
    const o = selected.object;
    const p = o.getWorldPosition(o.position.clone());
    const step = padStep(setup.camera, [p.x, p.y, p.z], getMode(), drag.handle, dx, dy,
                         view.domElement.clientHeight || 512);
    if (step.move) o.position.set(o.position.x + step.move[0], o.position.y + step.move[1], o.position.z + step.move[2]);
    if (step.axis && step.angle) {
      const axis = o.position.clone().set(step.axis[0], step.axis[1], step.axis[2]).normalize();
      o.quaternion.premultiply(o.quaternion.clone().setFromAxisAngle(axis, step.angle));
    }
    if (step.scale && Number.isFinite(step.scale)) o.scale.multiplyScalar(step.scale);   // равномерно
    changed();
  });
  const end = e => { drag = null; pad.style.cursor = 'grab'; try { pad.releasePointerCapture(e.pointerId); } catch (_) {} };
  pad.addEventListener('pointerup', end);
  pad.addEventListener('pointercancel', end);
  return { dispose() {
    clearInterval(timer); pad.remove();
    if (helper?.viewHelperContainer) helper.viewHelperContainer.style.display = '';
    try { helper?.visibleViewHelper?.(true); } catch (e) { /* ignore */ }
  } };
}

// One parsed room per mesh file: clones of the viewport share it.
const rooms = new Map();
function loadRoom(setup) {
  if (!setup.room_mesh) return Promise.resolve(null);
  const raster = setup.raster ?? [1344, 768];
  const key = JSON.stringify([setup.room_mesh, setup.camera, raster]);
  if (!rooms.has(key)) {
    const slash = setup.room_mesh.lastIndexOf('/');
    const url = new URL('api/view', document.baseURI);
    url.searchParams.set('filename', setup.room_mesh.slice(slash + 1));
    url.searchParams.set('subfolder', setup.room_mesh.slice(0, Math.max(slash, 0)));
    url.searchParams.set('type', 'input');
    rooms.set(key, fetch(url).then(r => {
      if (!r.ok) throw new Error('3DA: меш комнаты недоступен (HTTP ' + r.status + ')');
      return r.text();
    }).then(text => {
      const mesh = parseObj(text), camera = cameraModel(setup.camera, raster);
      return { mesh, camera, analysis: buildRoom(mesh, camera) };
    }));
    rooms.get(key).catch(() => rooms.delete(key));
  }
  return rooms.get(key);
}

// Depth-only room: hides whatever is behind the room surfaces, like the renderer,
// while the photo stays visible. Classes come from an already loaded mesh: no
// second three.js copy.
function roomOccluder(view, entities, room) {
  const scene = view.getSceneManager?.()?.scene;
  let sample;
  for (const e of entities) e.object.traverse(o => { if (!sample && o.isMesh) sample = o; });
  if (!scene || !sample) return null;
  const Geometry = sample.geometry.constructor;
  const Attribute = sample.geometry.attributes.position.constructor;
  const Material = Array.isArray(sample.material) ? sample.material[0].constructor : sample.material.constructor;
  const shape = occluderGeometry(room.mesh, room.camera);
  const geometry = new Geometry();
  geometry.setAttribute('position', new Attribute(shape.positions, 3));
  geometry.setIndex(new Attribute(shape.indices, 1));
  const material = new Material();
  material.colorWrite = false;
  material.depthWrite = true;
  material.side = 2; // DoubleSide
  const mesh = new sample.constructor(geometry, material);
  mesh.name = '3DA_room_occluder';
  mesh.renderOrder = -1;
  mesh.frustumCulled = false;
  mesh.raycast = () => {};
  scene.add(mesh);
  view.forceRender?.();
  return { mesh, dispose() { scene.remove(mesh); geometry.dispose(); material.dispose(); view.forceRender?.(); } };
}
