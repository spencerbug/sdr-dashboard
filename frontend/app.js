const PATCH_SIZE_RATIO = 0.16;
const MAX_RETRY_DELAY_MS = 10000;

const state = {
  objects: [],
  currentIndex: 0,
  camera: { distance: 2.0, theta: 0.0, phi: 0.0 },
  defaultCamera: { distance: 2.0, theta: 0.0, phi: 0.0 },
  target: [0.0, 0.0, 0.0],
  resolution: { width: 0, height: 0 },
  lastFrameUrl: null,
  cameraSendHandle: null,
  ws: null,
  frameCanvas: null,
  frameCtx: null,
  frameReady: false,
  frameSequence: 0,
  patch: { x: 0.5, y: 0.5, size: 96, userPositioned: false },
};

state.frameCanvas = document.createElement("canvas");
state.frameCtx = state.frameCanvas.getContext("2d", { willReadFrequently: false });

const frameEl = document.getElementById("frame");
const statusEl = document.getElementById("status");
const objectLabelEl = document.getElementById("object-label");
const cameraReadoutEl = document.getElementById("camera-readout");
const patchEl = document.getElementById("patch");
const pipCanvas = document.getElementById("pip");
const pipCtx = pipCanvas?.getContext("2d");
if (!pipCtx) {
  throw new Error("Unable to initialise magnifier canvas context");
}

const prevObjectBtn = document.getElementById("prev-object");
const nextObjectBtn = document.getElementById("next-object");
const resetCameraBtn = document.getElementById("reset-camera");
const shutdownBtn = document.getElementById("shutdown-server");

pipCtx.imageSmoothingEnabled = false;

function formatCamera(camera) {
  return `dist: ${camera.distance.toFixed(2)} | theta: ${camera.theta.toFixed(2)} | phi: ${camera.phi.toFixed(2)}`;
}

function updateUI() {
  if (state.objects.length) {
    const name = state.objects[state.currentIndex] ?? "--";
    objectLabelEl.textContent = `Object: ${name}`;
  } else {
    objectLabelEl.textContent = "Object: --";
  }
  cameraReadoutEl.textContent = formatCamera(state.camera);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return null;
}

async function loadInitialState() {
  try {
    const data = await fetchJson("/api/state");
    state.objects = Array.isArray(data.objects)
      ? data.objects.map((name) => (typeof name === "string" ? name.replace(/^ycb\//, "") : ""))
      : [];
    state.currentIndex = typeof data.objectIndex === "number" ? data.objectIndex : 0;
    if (data.camera) {
      state.camera = { ...state.camera, ...data.camera };
    }
    if (data.defaultCamera) {
      state.defaultCamera = { ...data.defaultCamera };
    } else {
      state.defaultCamera = { ...state.camera };
    }
    if (Array.isArray(data.target)) {
      state.target = [...data.target];
    }
    if (data.resolution?.width && data.resolution?.height) {
      updateResolution(data.resolution.width, data.resolution.height);
    }
    state.patch.userPositioned = false;
    updateUI();
    updatePatchPosition();
  } catch (error) {
    console.error("Failed to load initial state", error);
    setStatus("Failed to load initial state", true);
  }
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("hidden", !message);
  statusEl.classList.toggle("error", isError);
}

function clearFrameUrl() {
  if (state.lastFrameUrl) {
    URL.revokeObjectURL(state.lastFrameUrl);
    state.lastFrameUrl = null;
  }
}

function updateResolution(width, height) {
  if (!width || !height) {
    return;
  }
  const changed = width !== state.resolution.width || height !== state.resolution.height;
  state.resolution = { width, height };
  const baseSize = Math.round(Math.min(width, height) * PATCH_SIZE_RATIO);
  state.patch.size = clamp(baseSize, 48, 240);
  if (changed && !state.patch.userPositioned) {
    state.patch.x = 0.5;
    state.patch.y = 0.5;
  }
  clampPatchWithinBounds();
}

function clampPatchWithinBounds() {
  const { width, height } = state.resolution;
  if (!width || !height) {
    return;
  }
  const half = state.patch.size / 2;
  const centerX = clamp(state.patch.x * width, half, width - half);
  const centerY = clamp(state.patch.y * height, half, height - half);
  state.patch.x = centerX / width;
  state.patch.y = centerY / height;
}

function updatePatchPosition() {
  const { width, height } = state.resolution;
  if (!width || !height) {
    return;
  }

  const displayWidth = frameEl.clientWidth;
  const displayHeight = frameEl.clientHeight;
  if (!displayWidth || !displayHeight) {
    return;
  }

  const patchWidthDisplay = (state.patch.size / width) * displayWidth;
  const patchHeightDisplay = (state.patch.size / height) * displayHeight;

  const centerDisplayX = state.patch.x * displayWidth;
  const centerDisplayY = state.patch.y * displayHeight;

  patchEl.style.width = `${patchWidthDisplay}px`;
  patchEl.style.height = `${patchHeightDisplay}px`;
  patchEl.style.transform = `translate(${centerDisplayX - patchWidthDisplay / 2}px, ${centerDisplayY - patchHeightDisplay / 2}px)`;

  updatePatchPreview();
}

function updatePatchPreview() {
  if (!state.frameReady) {
    return;
  }
  const { width, height } = state.resolution;
  if (!width || !height) {
    return;
  }

  const centerX = state.patch.x * width;
  const centerY = state.patch.y * height;
  const half = state.patch.size / 2;
  const sx = clamp(centerX - half, 0, width - state.patch.size);
  const sy = clamp(centerY - half, 0, height - state.patch.size);

  pipCtx.save();
  pipCtx.imageSmoothingEnabled = false;
  pipCtx.clearRect(0, 0, pipCanvas.width, pipCanvas.height);
  pipCtx.drawImage(
    state.frameCanvas,
    sx,
    sy,
    state.patch.size,
    state.patch.size,
    0,
    0,
    pipCanvas.width,
    pipCanvas.height,
  );
  pipCtx.restore();
  pipCtx.strokeStyle = "#ff4d4d";
  pipCtx.lineWidth = 2;
  pipCtx.strokeRect(1, 1, pipCanvas.width - 2, pipCanvas.height - 2);
}

function connectWebSocket(retryDelay = 1000) {
  clearFrameUrl();
  state.frameReady = false;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${window.location.host}/ws/frames`;
  const ws = new WebSocket(url);
  ws.binaryType = "blob";
  state.ws = ws;
  setStatus("Connecting to frame stream...");

  ws.onopen = () => {
    setStatus("");
  };

  ws.onmessage = async (event) => {
    if (typeof event.data === "string") {
      console.warn("Unexpected text frame", event.data);
      return;
    }
    const blob = event.data;
    clearFrameUrl();
    const blobUrl = URL.createObjectURL(blob);
    state.lastFrameUrl = blobUrl;
    frameEl.src = blobUrl;

    const frameId = ++state.frameSequence;
    try {
      const bitmap = await createImageBitmap(blob);
      if (frameId !== state.frameSequence) {
        bitmap.close();
        return;
      }
      updateFrameBuffer(bitmap);
    } catch (error) {
      console.error("Failed to process frame", error);
    }
  };

  ws.onerror = () => {
    setStatus("Frame stream error", true);
  };

  ws.onclose = () => {
    setStatus("Disconnected. Reconnecting...");
    state.ws = null;
    setTimeout(() => connectWebSocket(Math.min(retryDelay * 1.5, MAX_RETRY_DELAY_MS)), retryDelay);
  };
}

function updateFrameBuffer(bitmap) {
  updateResolution(bitmap.width, bitmap.height);
  state.frameCanvas.width = bitmap.width;
  state.frameCanvas.height = bitmap.height;
  state.frameCtx.drawImage(bitmap, 0, 0);
  bitmap.close();
  state.frameReady = true;
  updatePatchPosition();
}

function queueCameraUpdate() {
  updateUI();
  if (state.cameraSendHandle) {
    return;
  }
  state.cameraSendHandle = window.setTimeout(async () => {
    state.cameraSendHandle = null;
    try {
      await postJson("/api/camera", state.camera);
      setStatus("");
    } catch (error) {
      console.error("Failed to update camera", error);
      setStatus("Camera update failed", true);
    }
  }, 75);
}

async function changeObject(delta) {
  if (!state.objects.length) {
    return;
  }
  const nextIndex = (state.currentIndex + delta + state.objects.length) % state.objects.length;
  updateUI();
  try {
    const response = await postJson(`/api/object/${nextIndex}`, {});
    if (typeof response?.objectIndex === "number") {
      state.currentIndex = response.objectIndex;
    } else {
      state.currentIndex = nextIndex;
    }
    if (response?.camera) {
      state.camera = { ...state.camera, ...response.camera };
    }
    if (response?.defaultCamera) {
      state.defaultCamera = { ...response.defaultCamera };
    }
    if (Array.isArray(response?.target)) {
      state.target = [...response.target];
    }
    state.patch.userPositioned = false;
    state.frameReady = false;
    updateUI();
    updatePatchPosition();
    setStatus("");
  } catch (error) {
    console.error("Failed to change object", error);
    setStatus("Object change failed", true);
  }
}

async function resetCamera() {
  state.camera = { ...state.defaultCamera };
  updateUI();
  try {
    await postJson("/api/camera", state.camera);
    setStatus("");
  } catch (error) {
    console.error("Failed to reset camera", error);
    setStatus("Camera reset failed", true);
  }
}

async function shutdownServer() {
  if (!shutdownBtn) {
    return;
  }
  try {
    shutdownBtn.disabled = true;
    await postJson("/api/shutdown", {});
    setStatus("Server shutting down...");
    if (state.ws) {
      state.ws.close(1000, "Client requested shutdown");
    }
  } catch (error) {
    console.error("Failed to stop server", error);
    setStatus("Shutdown request failed", true);
    shutdownBtn.disabled = false;
  }
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function clampCameraAngle(camera) {
  const limit = Math.PI / 2 - 0.05;
  camera.phi = clamp(camera.phi, -limit, limit);
}

let dragging = false;
let lastPointer = { x: 0, y: 0 };

function handlePointerDown(event) {
  dragging = true;
  lastPointer = { x: event.clientX, y: event.clientY };
  event.preventDefault();
}

function handlePointerMove(event) {
  if (!dragging) {
    return;
  }
  const deltaX = event.clientX - lastPointer.x;
  const deltaY = event.clientY - lastPointer.y;
  lastPointer = { x: event.clientX, y: event.clientY };

  state.camera.theta += deltaX * 0.01;
  state.camera.phi -= deltaY * 0.01;
  clampCameraAngle(state.camera);
  queueCameraUpdate();
  event.preventDefault();
}

function handlePointerUp() {
  dragging = false;
}

function handleWheel(event) {
  event.preventDefault();
  const scale = Math.exp(-event.deltaY * 0.001);
  state.camera.distance = clamp(state.camera.distance * scale, 0.2, 20.0);
  queueCameraUpdate();
}

const patchDragState = {
  active: false,
  pointerId: null,
};

function patchPointerDown(event) {
  patchDragState.active = true;
  patchDragState.pointerId = event.pointerId;
  patchEl.setPointerCapture(event.pointerId);
  movePatchToEvent(event, true);
}

function patchPointerMove(event) {
  if (!patchDragState.active || event.pointerId !== patchDragState.pointerId) {
    return;
  }
  movePatchToEvent(event, true);
}

function patchPointerUp(event) {
  if (event.pointerId === patchDragState.pointerId) {
    patchEl.releasePointerCapture(event.pointerId);
    patchDragState.active = false;
    patchDragState.pointerId = null;
  }
}

function movePatchToEvent(event, markUser) {
  const coords = eventToImagePixels(event.clientX, event.clientY);
  if (!coords) {
    return;
  }
  setPatchCenterFromPixels(coords.x, coords.y, markUser);
  event.preventDefault();
}

function eventToImagePixels(clientX, clientY) {
  const rect = frameEl.getBoundingClientRect();
  const { width, height } = state.resolution;
  if (!rect.width || !rect.height || !width || !height) {
    return null;
  }
  const ratioX = clamp((clientX - rect.left) / rect.width, 0, 1);
  const ratioY = clamp((clientY - rect.top) / rect.height, 0, 1);
  return {
    x: ratioX * width,
    y: ratioY * height,
  };
}

function setPatchCenterFromPixels(x, y, markUser = false) {
  const { width, height } = state.resolution;
  if (!width || !height) {
    return;
  }
  const half = state.patch.size / 2;
  const clampedX = clamp(x, half, width - half);
  const clampedY = clamp(y, half, height - half);
  state.patch.x = clampedX / width;
  state.patch.y = clampedY / height;
  if (markUser) {
    state.patch.userPositioned = true;
  }
  updatePatchPosition();
}

function bindEvents() {
  prevObjectBtn.addEventListener("click", () => changeObject(-1));
  nextObjectBtn.addEventListener("click", () => changeObject(1));
  resetCameraBtn.addEventListener("click", () => resetCamera());
  if (shutdownBtn) {
    shutdownBtn.addEventListener("click", () => shutdownServer());
  }

  frameEl.addEventListener("mousedown", handlePointerDown);
  window.addEventListener("mousemove", handlePointerMove);
  window.addEventListener("mouseup", handlePointerUp);

  frameEl.addEventListener("touchstart", (event) => {
    if (event.touches.length !== 1) {
      return;
    }
    const touch = event.touches[0];
    handlePointerDown({ clientX: touch.clientX, clientY: touch.clientY, preventDefault: event.preventDefault.bind(event) });
  }, { passive: false });

  frameEl.addEventListener("touchmove", (event) => {
    if (event.touches.length !== 1) {
      return;
    }
    const touch = event.touches[0];
    handlePointerMove({ clientX: touch.clientX, clientY: touch.clientY, preventDefault: event.preventDefault.bind(event) });
  }, { passive: false });

  window.addEventListener("touchend", handlePointerUp);
  frameEl.addEventListener("wheel", handleWheel, { passive: false });

  patchEl.addEventListener("pointerdown", patchPointerDown);
  patchEl.addEventListener("pointermove", patchPointerMove);
  patchEl.addEventListener("pointerup", patchPointerUp);
  patchEl.addEventListener("pointercancel", patchPointerUp);

  frameEl.addEventListener("load", () => updatePatchPosition());
  window.addEventListener("resize", () => updatePatchPosition());
}

window.addEventListener("beforeunload", () => {
  clearFrameUrl();
  if (state.ws) {
    state.ws.close(1000);
  }
});

(async function init() {
  bindEvents();
  await loadInitialState();
  connectWebSocket();
})();
