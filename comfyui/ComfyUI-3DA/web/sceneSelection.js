// Named-object selection on the existing core gizmo, not another 3D editor.
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
  // Live diagnostics: the renderer occludes the body by the room floor, the
  // viewport draws meshes over a flat photo. Show both honestly here.
  const status = document.createElement('span');
  status.className = 'threeda-scene-status';
  Object.assign(status.style, { color:'white', padding:'5px 8px', borderRadius:'4px',
    font:'12px sans-serif', background:'#2e7d32', whiteSpace:'pre-line' });
  toolbar.append(picker,modes,status);
  const parent = view.domElement.parentElement;
  parent?.appendChild(toolbar);
  for (const event of ['pointerdown','pointerup','click','dblclick','keydown'])
    toolbar.addEventListener(event, e => e.stopPropagation());
  picker.addEventListener('change', () => select(picker.value));
  const occluder = floorOccluder(view, entities);
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
    const lines = [];
    for (const e of entities) {
      const low = lowestY(e.object);
      if (Number.isFinite(low) && low < -FLOOR_TOLERANCE_M)
        lines.push(`${e.kind === 'body' ? 'Персонаж' : e.label}: ниже пола на ${Math.round(-low * 100)} см`);
      if (e.kind === 'body') {
        const s = e.object.getWorldScale(e.object.scale.clone()).x;
        if (Math.abs(s - 1) > SCALE_TOLERANCE)
          lines.push(`Персонаж: масштаб ×${s.toFixed(2)} — рост изменится так же`);
      }
    }
    status.textContent = lines.length ? lines.join('\n') : 'Пол и рост: ок';
    status.style.background = lines.length ? '#b71c1c' : '#2e7d32';
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
    node.properties['3DA Selected Entity'] = id;
    syncMode();
    view.forceRender();
  }
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
      occluder?.dispose();
      toolbar.remove(); } };
}

// Soft supports (mats, beds) may be pressed in up to placement.yaml pen_soft_m.
const FLOOR_TOLERANCE_M = 0.06;
const SCALE_TOLERANCE = 0.02;

// Lowest world-space vertex height of an entity (floor plane is y = 0: the
// room camera stands at cam_height above the fitted floor).
function lowestY(object) {
  let low = Infinity;
  object.updateMatrixWorld(true);
  object.traverse(mesh => {
    const position = mesh.isMesh && mesh.geometry?.attributes?.position;
    if (!position) return;
    const m = mesh.matrixWorld.elements, a = position.array, n = position.itemSize;
    for (let i = 0; i < a.length; i += n) {
      const y = m[1] * a[i] + m[5] * a[i + 1] + m[9] * a[i + 2] + m[13];
      if (y < low) low = y;
    }
  });
  return low;
}

// Depth-only floor: hides whatever is below the floor, like the renderer that
// occludes by the room plate. Drawn into depth only, so the photo stays visible.
// Classes come from an already loaded mesh: no second three.js copy.
function floorOccluder(view, entities) {
  const scene = view.getSceneManager?.()?.scene;
  let sample;
  for (const e of entities) e.object.traverse(o => { if (!sample && o.isMesh) sample = o; });
  if (!scene || !sample) return null;
  const Geometry = sample.geometry.constructor;
  const Attribute = sample.geometry.attributes.position.constructor;
  const Material = Array.isArray(sample.material) ? sample.material[0].constructor : sample.material.constructor;
  const geometry = new Geometry();
  // 60 x 60 m around the camera foot, slightly below 0 so resting contacts stay visible.
  const y = -0.01, r = 30;
  geometry.setAttribute('position', new Attribute(new Float32Array([-r,y,-r*2, r,y,-r*2, r,y,r, -r,y,r]), 3));
  geometry.setIndex([0, 2, 1, 0, 3, 2]);
  const material = new Material();
  material.colorWrite = false;
  material.depthWrite = true;
  material.side = 2; // DoubleSide
  const mesh = new sample.constructor(geometry, material);
  mesh.name = '3DA_floor_occluder';
  mesh.renderOrder = -1;
  mesh.raycast = () => {};
  scene.add(mesh);
  view.forceRender?.();
  return { mesh, dispose() { scene.remove(mesh); geometry.dispose(); material.dispose(); } };
}
