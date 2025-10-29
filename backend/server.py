import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from habitat_bridge import HabitatRenderBridge

_LOG = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"


class CameraUpdate(BaseModel):
    distance: Optional[float] = Field(None, gt=0.0)
    theta: Optional[float] = None
    phi: Optional[float] = None
    target: Optional[List[float]] = Field(None, min_items=3, max_items=3)


_shutdown_handler: Optional[Callable[[], None]] = None


def register_shutdown_handler(handler: Callable[[], None]) -> None:
    global _shutdown_handler
    _shutdown_handler = handler


async def request_shutdown() -> None:
    if _shutdown_handler is None:
        raise HTTPException(status_code=503, detail="Shutdown handler not registered")
    loop = asyncio.get_running_loop()
    loop.call_soon(_shutdown_handler)


class ViewerService:
    def __init__(self, width: int = 1200, height: int = 800, fps: float = 20.0) -> None:
        self.width = width
        self.height = height
        self.frame_interval = 1.0 / max(fps, 1.0)
        self.object_configs = self._load_object_configs()
        if not self.object_configs:
            raise RuntimeError("No YCB object configs found under assets/ycb/configs")

        self.bridge = HabitatRenderBridge(width, height, self.object_configs)
        self.camera: Dict[str, float] = {"distance": 2.0, "theta": 0.0, "phi": 0.0}
        self.default_camera: Dict[str, float] = dict(self.camera)
        self.target: List[float] = [0.0, 0.0, 0.0]
        self.object_idx: int = 0

        self._clients: Set[WebSocket] = set()
        self._frame_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._running = asyncio.Event()

    async def start(self) -> None:
        await self.set_object(self.object_idx)
        self._running.set()
        self._frame_task = asyncio.create_task(self._frame_loop(), name="frame_loop")

    async def stop(self) -> None:
        self._running.clear()
        if self._frame_task is not None:
            self._frame_task.cancel()
            try:
                await self._frame_task
            except asyncio.CancelledError:
                pass
        await asyncio.to_thread(self.bridge.shutdown)

    async def register_client(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        _LOG.info("Client connected (%s active)", len(self._clients))

    async def unregister_client(self, websocket: WebSocket) -> None:
        if websocket in self._clients:
            self._clients.remove(websocket)
            _LOG.info("Client disconnected (%s active)", len(self._clients))

    def get_state(self) -> Dict[str, Any]:
        return {
            "camera": {**self.camera},
            "defaultCamera": {**self.default_camera},
            "target": list(self.target),
            "objectIndex": self.object_idx,
            "objects": [self._format_object_label(cfg) for cfg in self.object_configs],
            "resolution": {"width": self.width, "height": self.height},
        }

    async def set_object(self, index: int) -> Dict[str, Any]:
        if not self.object_configs:
            raise HTTPException(status_code=400, detail="No objects available")
        index %= len(self.object_configs)
        async with self._lock:
            result = await asyncio.to_thread(self.bridge.load_object, index)
            self.object_idx = index

            bounds = result.get("bounds", {}) if isinstance(result, dict) else {}
            camera_info = result.get("camera", {}) if isinstance(result, dict) else {}
            default_camera = result.get("default_camera", {}) if isinstance(result, dict) else {}

            radius = float(bounds.get("radius", 0.5))
            diameter = float(bounds.get("diameter", radius * 2.0))
            center = bounds.get("center")

            if isinstance(center, (list, tuple)) and len(center) == 3:
                self.target = [float(c) for c in center]
            else:
                self.target = [0.0, 0.0, 0.0]

            if default_camera:
                distance = float(default_camera.get("distance", 2.0))
                theta = float(default_camera.get("theta", 0.0))
                phi = float(default_camera.get("phi", 0.0))
            else:
                hfov = float(camera_info.get("hfov", 90.0))
                vfov = float(camera_info.get("vfov", hfov))
                fill_ratio = 0.8

                hfov_rad = math.radians(max(hfov, 1e-3))
                vfov_rad = math.radians(max(vfov, 1e-3))

                min_distance_h = radius / max(math.sin(hfov_rad / 2.0), 1e-3)
                min_distance_v = radius / max(math.sin(vfov_rad / 2.0), 1e-3)
                min_distance = max(min_distance_h, min_distance_v, radius * 1.05, 0.3)

                distance = max(min_distance / max(fill_ratio, 1e-3), diameter * 0.4, 0.3)
                theta = 0.0
                phi = 0.0

            self.camera = {"distance": distance, "theta": theta, "phi": phi}
            self.default_camera = dict(self.camera)

            return {
                "objectIndex": self.object_idx,
                "camera": {**self.camera},
                "defaultCamera": {**self.default_camera},
                "target": list(self.target),
                "bounds": bounds,
            }

    async def update_camera(self, update: CameraUpdate) -> Dict[str, Any]:
        async with self._lock:
            if update.distance is not None:
                self.camera["distance"] = max(0.1, float(update.distance))
            if update.theta is not None:
                self.camera["theta"] = float(update.theta)
            if update.phi is not None:
                # Clamp to avoid gimbal lock
                limit = math.pi / 2 - 0.05
                self.camera["phi"] = max(-limit, min(limit, float(update.phi)))
            if update.target is not None:
                self.target = [float(v) for v in update.target[:3]]
        return self.get_state()["camera"]

    async def _frame_loop(self) -> None:
        try:
            while self._running.is_set():
                if not self._clients:
                    await asyncio.sleep(self.frame_interval)
                    continue

                payload = await self._snapshot_camera()
                frame = await asyncio.to_thread(self.bridge.render, payload)
                success, buffer = cv2.imencode(".jpg", frame)
                if not success:
                    await asyncio.sleep(self.frame_interval)
                    continue
                data = buffer.tobytes()

                dead_clients: List[WebSocket] = []
                for client in list(self._clients):
                    try:
                        await client.send_bytes(data)
                    except WebSocketDisconnect:
                        dead_clients.append(client)
                    except RuntimeError:
                        dead_clients.append(client)

                for client in dead_clients:
                    await self.unregister_client(client)

                await asyncio.sleep(self.frame_interval)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # pragma: no cover
            _LOG.exception("Frame loop encountered an error: %s", exc)

    async def _snapshot_camera(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "distance": self.camera["distance"],
                "theta": self.camera["theta"],
                "phi": self.camera["phi"],
                "target": list(self.target),
            }

    @staticmethod
    def _format_object_label(config: Dict[str, Any]) -> str:
        raw_name = config.get("name")
        if not isinstance(raw_name, str) or not raw_name:
            raw_name = Path(config.get("config_file", "")).stem
        name = str(raw_name)
        if name.startswith("ycb/"):
            name = name[4:]
        if name.endswith(".object_config"):
            name = name[: -len(".object_config")]
        return name or "unknown"

    @staticmethod
    def _load_object_configs() -> List[Dict[str, Any]]:
        config_path = Path("assets/ycb/configs")
        object_configs: List[Dict[str, Any]] = []
        for path in sorted(config_path.glob("*.object_config.json")):
            with open(path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            config["config_file"] = str(path)
            config.setdefault("name", path.stem)
            object_configs.append(config)
        return object_configs


viewer = ViewerService()

app = FastAPI(title="YCB Viewer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    await viewer.start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await viewer.stop()


@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    return viewer.get_state()


@app.get("/api/objects")
async def get_objects() -> List[Dict[str, Any]]:
    return [
        {
            "index": idx,
            "name": ViewerService._format_object_label(cfg),
        }
        for idx, cfg in enumerate(viewer.object_configs)
    ]


@app.post("/api/object/{index}")
async def set_object(index: int) -> Dict[str, Any]:
    return await viewer.set_object(index)


@app.post("/api/camera")
async def update_camera(update: CameraUpdate) -> Dict[str, Any]:
    camera = await viewer.update_camera(update)
    return {"camera": camera}


@app.websocket("/ws/frames")
async def websocket_frames(websocket: WebSocket) -> None:
    await viewer.register_client(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await viewer.unregister_client(websocket)
    except Exception:
        await viewer.unregister_client(websocket)
        raise


if FRONTEND_INDEX.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root() -> FileResponse:
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(FRONTEND_INDEX)


@app.post("/api/shutdown")
async def shutdown_endpoint() -> Dict[str, str]:
    await request_shutdown()
    return {"status": "shutting_down"}