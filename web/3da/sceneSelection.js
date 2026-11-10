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
  toolbar.append(picker,modes);
  const parent = view.domElement.parentElement;
  parent?.appendChild(toolbar);
  for (const event of ['pointerdown','pointerup','click','dblclick','keydown'])
    toolbar.addEventListener(event, e => e.stopPropagation());
  picker.addEventListener('change', () => select(picker.value));
  const gizmo = view.getGizmoManager();
  const controls = gizmo.transformControls;
  let lastMode;
  let scaleStart;
  const beginScale = () => {
    scaleStart = selected?.kind === 'body' && gizmo.getMode() === 'scale'
      ? selected.object.scale.clone() : null;
  };
  const uniformScale = () => {
    if (!scaleStart || selected?.kind !== 'body' || gizmo.getMode() !== 'scale') return;
    // Use the real core handle, including X/Y/Z. Each handle changes size
    // uniformly for a character rather than stretching an individual limb.
    const axis = (controls.axis ?? 'X')[0].toLowerCase();
    const ratio = selected.object.scale[axis] / scaleStart[axis];
    if (Number.isFinite(ratio) && ratio > 0) selected.object.scale.copy(scaleStart).multiplyScalar(ratio);
    else selected.object.scale.copy(scaleStart);
  };
  const endScale = () => { scaleStart = null; };
  controls?.addEventListener('mouseDown', beginScale);
  controls?.addEventListener('objectChange', uniformScale);
  controls?.addEventListener('mouseUp', endScale);
  function syncMode() {
    const mode = gizmo.getMode();
    if (mode !== lastMode) {
      modes.value = mode;
      node.properties['3DA Gizmo Mode'] = mode;
      lastMode = mode;
    }
    modes.options[2].text = selected?.kind === 'body' ? 'Масштаб (равномерно)' : 'Масштаб';
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
  select(node.properties['3DA Selected Entity'] ?? 'body');
  setMode(initialMode);
  return { root, entities, capture, select, setMode, syncMode, selected: () => selected.id,
    dispose: () => {
      controls?.removeEventListener('mouseDown', beginScale);
      controls?.removeEventListener('objectChange', uniformScale);
      controls?.removeEventListener('mouseUp', endScale);
      toolbar.remove(); } };
}
