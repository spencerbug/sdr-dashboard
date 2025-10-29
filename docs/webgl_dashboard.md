# Web-Based SDR Object Viewer Dashboard

Design notes for migrating the current Habitat-Sim Inspector into a browser-centric experience powered by WebGL and modern web tooling.

---

## 1. High-Level Goals
- **Platform-Portability:** Replace native DearPyGUI/UI dependencies with a browser dashboard that works on Windows, macOS, Linux, and WSL without GPU driver friction.
- **Hybrid Rendering:** Retain Habitat-Sim for physics, scene management, and render-to-texture, while enabling optional WebGL-first rendering for lighter-weight previews or fallback when headless CPU rendering is too slow.
- **Interactive Analysis:** Offer camera controls, patch/magnifier view, object cycling, and telemetry in an ergonomic web UX.
- **Extensibility:** Provide a modular architecture to add metrics, dataset filtering, logging, and automation without coupling tightly to UI code.

---

## 2. Proposed System Architecture

### 2.1 Core Components
- **Python Backend (FastAPI / Starlette):** 
  - Orchestrates Habitat-Sim simulations.
  - Exposes REST & WebSocket endpoints for control, data streaming, and telemetry.
  - Manages session state (loaded object, camera poses, patch location).
  - Provides static asset serving for the front-end bundle.

- **Simulation Engine (Habitat-Sim Headless):**
  - Runs in off-screen headless mode (OSMesa) to avoid EGL issues.
  - Supports rendering to RGBA buffers on demand.
  - Streams observation frames (RGB, depth, segmentation) back to the backend.

- **Web Front-End (TypeScript + React / Svelte / Vue):**
  - Handles visualization, control panels, and overlays.
  - Uses WebGL/Three.js (or Babylon.js) to render meshes or streamed textures.
  - Implements a magnifier patch view and object navigation UI.
  - Establishes a WebSocket connection for low-latency updates.

- **Asset Manager:**
  - Catalogs YCB / custom objects.
  - Handles lazy loading of mesh metadata, bounding boxes, dimensions, and preview thumbnails.

### 2.2 Data Flow
1. **Initialization:** Browser requests `/api/session` to create/view current sim session. Backend launches Habitat-Sim context if needed.
2. **Rendering Requests:** Front-end emits camera pose updates via WebSocket; backend renders frame (RGB, depth) and responds with encoded image (PNG/JPEG) or raw pixel buffer (ArrayBuffer) for WebGL textures.
3. **Patch Extraction:** Front-end can request patch coordinates; backend returns cropped pixel data for magnified view, or front-end extracts from texture.
4. **Object Cycling:** API call triggers object swap; backend reloads asset, recalculates framing, pushes updated metadata to clients.
5. **Telemetry:** Backend broadcasts frame timing, bounding boxes, metrics; front-end displays charts/indicators.

---

## 3. Front-End Implementation Notes

| Concern | Recommendation |
| --- | --- |
| **Framework** | React (Next.js or Vite) for rich component ecosystem; SvelteKit for lighter bundle and easier reactivity. |
| **Rendering** | Three.js for WebGL scene. Optionally, Babylon.js (comes with PBR materials, GUI). |
| **State Management** | Zustand / Redux Toolkit in React; built-in stores for Svelte. |
| **Communication** | native WebSocket API; use `socket.io` if multiplexing events is preferred. |
| **Styling** | Tailwind CSS or Chakra UI for rapid UI layout. |
| **Testing** | Playwright/Cypress for integration; Vitest/Jest for unit tests. |
| **Build & Deployment** | Static assets served by Python backend or via Docker multi-stage build bundling both UI and backend. |

### 3.1 WebGL Rendering Options
1. **Texture Streaming:** Use Habitat-Sim to produce RGBA frames, send to front-end via WebSocket, update a textured plane in Three.js. Simple, works with existing rendering.
2. **Mesh Streaming:** Pre-load mesh geometry into the browser directly (GLTF/GLB); use WebGL to render client-side with physically based shaders. Python backend handles camera computation, lighting metadata.
3. **Hybrid:** Default to texture streaming, but allow optional “client-side mesh rendering” mode to offload GPU rendering to browser (requires transferring object meshes + materials).

---

## 4. Backend Implementation Plan

### 4.1 Stack
- **FastAPI**: async REST & WebSocket endpoints, automatic OpenAPI docs.
- **uvicorn / Hypercorn**: ASGI server for production; use aiofiles for static serving.
- **Pydantic**: request/response schema validation (camera pose, patch requests, runtime settings).
- **Redis (optional)**: cache assets, maintain session state for multi-client sync.

### 4.2 API Sketch
- `POST /api/session`: create session, return session_id and initial state.
- `GET /api/session/{id}`: fetch session metadata.
- `POST /api/session/{id}/object`: load new object by asset key or index.
- `POST /api/session/{id}/camera`: update spherical/Cartesian camera parameters.
- `POST /api/session/{id}/patch`: request patch extraction; returns cropped image data.
- `WS /ws/session/{id}`: send incremental updates (camera pose, patch move, UI events) and receive frames, metrics.

### 4.3 Habitat-Sim Integration
- Maintain a persistent simulator instance per session.
- Use OSMesa headless configuration to avoid GPU driver issues.
- Render on demand in an async worker (ThreadPool) to avoid blocking event loop.
- Encode frames using `Pillow` or `pywebp` for faster transfer; base64 or binary.
- Provide additional outputs (depth, normal maps) as layered channels for advanced overlays.

### 4.4 Patch View Strategy
- Option A: Backend extracts patch from RGBA frame before encoding.
- Option B: Front-end samples patch from `ImageBitmap` using WebGL framebuffer blit; allows smoother magnification.
- Provide optional “picture-in-picture” overlay computed client-side for responsiveness.

---

## 5. Deployment & Operations

| Topic | Notes |
| --- | --- |
| **Dockerization** | Multi-stage: build front-end bundle → COPY into Python image → run uvicorn. Ensure OSMesa dependencies included. |
| **Environment** | Provide `.env` template for HABITAT_SIM_LOG_LEVEL, asset paths, optional GPU flags. |
| **Static Assets** | Serve web bundle via `/static`; configure cache headers. |
| **TLS** | Terminate at reverse proxy (nginx, Traefik) for secure WebSocket connections. |
| **Monitoring** | Instrument FastAPI with Prometheus metrics; log frame render time, queue depth. |
| **Testing** | CI pipeline: `pytest` for backend, `npm test` for front-end, `pytest --run-slow` for integration with headless Habitat. |

---

## 6. Stretch Features
- **Multi-User Sessions:** Broadcast updates to all viewers; add collaborative annotations.
- **Recording & Replay:** Save frame streams with timestamps; provide playback controls in UI.
- **Analytics Panels:** Display histograms for pixel values, bounding box overlays, object metadata.
- **Advanced Camera Controls:** Joystick/gamepad inputs via browser Gamepad API.
- **GPU Offload (Optional):** When running on native Linux/macOS with GPU drivers, offer toggle to use GPU-based Habitat render streaming (web app remains unchanged).
- **VR Mode:** Integrate WebXR to inspect objects in an immersive environment.

---

## 7. Next Steps
1. Scaffold FastAPI service with REST + WS endpoints.
2. Prototype Habitat-Sim headless rendering to base64 PNG streaming.
3. Bootstrapped front-end (React or Svelte) with WebSocket client, show raw frame texture.
4. Implement camera control UI and patch overlay (2D canvas first, upgrade to Three.js).
5. Integrate object catalog navigation and metadata panels.
6. Add packaging and Docker setup for reproducible deployment.
