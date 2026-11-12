// Per-instance adapter for core Load3DAdvanced. No model/TRS writes.
export function lockCamera(view, state) {
  const controls = view.getControlsManager?.().controls;
  if (!controls || !view.setCameraState || !view.getCameraState || !view.setFOV)
    throw new Error('3DA: несовместимый core Load3D; камера не защищена');
  // Node properties can be Vue proxies, which structuredClone rejects.
  const wanted = JSON.parse(JSON.stringify(state));
  function matches(actual, expected) {
    if (typeof expected === 'number') return Number.isFinite(actual) && Math.abs(actual - expected) < 1e-10;
    if (expected && typeof expected === 'object')
      return actual && Object.keys(expected).every(k => matches(actual[k], expected[k]));
    return actual === expected;
  }
  const originalSet = view.setCameraState;
  const originalFOV = view.setFOV;
  const originalFit = view.fitToViewer;
  const originalCenter = view.centerCameraOnModel;
  const previous = Object.fromEntries(
    ['enableZoom', 'enablePan', 'enableRotate', 'enableDamping'].map(k => [k, controls[k]])
  );
  let restoring = false;
  const restore = () => {
    if (restoring) return;
    restoring = true;
    try {
      // dragging-changed re-enables OrbitControls on mouse-up. These flags
      // remain effective even when enabled becomes true again.
      controls.enableZoom = controls.enablePan = controls.enableRotate = false;
      controls.enableDamping = false;
      // Current core restores position/target/zoom here, but FOV has a separate
      // setter. The default 35° must not survive a valid room camera preset.
      if (Number.isFinite(wanted.fov) && Math.abs(view.getCameraState().fov - wanted.fov) > 1e-10)
        originalFOV.call(view, wanted.fov);
      if (!matches(view.getCameraState(), wanted))
        originalSet.call(view, structuredClone(wanted));
    } finally { restoring = false; }
  };
  view.setCameraState = () => originalSet.call(view, structuredClone(wanted));
  view.setFOV = () => originalFOV.call(view, wanted.fov);
  // Core Fit changes the MODEL to 5m, not just the camera. Block before it
  // runs; restoring a complete preset here would lose the user's transform.
  view.fitToViewer = () => restore();
  if (originalCenter) view.centerCameraOnModel = () => restore();
  controls.addEventListener('change', restore);
  restore();
  return {
    restore,
    dispose() {
      controls.removeEventListener('change', restore);
      Object.assign(controls, previous);
      view.setCameraState = originalSet;
      view.setFOV = originalFOV;
      view.fitToViewer = originalFit;
      if (originalCenter) view.centerCameraOnModel = originalCenter;
    }
  };
}
