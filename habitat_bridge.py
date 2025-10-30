import json
import math
import os
import platform
import traceback
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Habitat-Sim and Magnum imports are intentionally inside the worker
# initialization to keep the main process lightweight. The worker
# function will import them when it starts.


def habitat_worker_main(
    command_queue,
    response_queue,
    shm_name: str,
    width: int,
    height: int,
    object_configs: List[Dict[str, Any]],
) -> None:
    """Entry point for the Habitat rendering worker process."""

    try:
        import habitat_sim
        from habitat_sim.agent import AgentConfiguration
        from habitat_sim import Configuration, SimulatorConfiguration

        import magnum as mn

        try:  # pragma: no cover - depends on habitat build
            from habitat_sim.gfx import DEFAULT_LIGHTING_KEY  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover - fallback for older builds
            DEFAULT_LIGHTING_KEY = "default"  # type: ignore[assignment]

    except Exception as exc:  # pragma: no cover - handled by parent
        response_queue.put({
            "type": "ready",
            "success": False,
            "message": f"Failed to import Habitat-Sim dependencies: {exc}",
        })
        return

    class WorkerState:
        def __init__(self) -> None:
            self.width = width
            self.height = height
            self.frame = shared_memory.SharedMemory(name=shm_name)
            self.frame_view = np.ndarray(
                (self.height, self.width, 3), dtype=np.uint8, buffer=self.frame.buf
            )

            self.object_configs = object_configs
            self.sim: Optional[habitat_sim.Simulator] = None
            self.agent = None
            self.object_handle: str | None = None
            self.object_ref = None

        def initialize(self) -> tuple[bool, str]:
            try:
                self._setup_habitat()
                return True, ""
            except Exception as exc:  # pragma: no cover - initialization failure
                traceback.print_exc()
                return False, str(exc)

        def shutdown(self) -> None:
            if self.sim is not None:
                self.sim.close()
                self.sim = None
            self.frame.close()

        def _setup_habitat(self) -> None:
            assert self.sim is None

            def build_config(gpu_device_id: int) -> habitat_sim.Configuration:
                backend_cfg = SimulatorConfiguration()
                backend_cfg.scene_id = "assets/scenes/examiner/void_black.glb"
                backend_cfg.enable_physics = False
                backend_cfg.allow_sliding = False
                backend_cfg.gpu_device_id = gpu_device_id
                backend_cfg.load_semantic_mesh = False
                backend_cfg.scene_light_setup = DEFAULT_LIGHTING_KEY

                camera_cfg = habitat_sim.CameraSensorSpec()
                camera_cfg.uuid = "color_sensor"
                camera_cfg.sensor_type = habitat_sim.SensorType.COLOR
                camera_cfg.resolution = [self.height, self.width]
                camera_cfg.position = [0.0, 0.0, 0.0]
                camera_cfg.orientation = [0.0, 0.0, 0.0]

                agent_cfg = AgentConfiguration()
                agent_cfg.sensor_specifications = [camera_cfg]

                return Configuration(backend_cfg, [agent_cfg])

            prefer_cpu = os.environ.get("YCB_VIEWER_FORCE_CPU", "0") == "1"
            custom_gpu = os.environ.get("YCB_VIEWER_GPU_DEVICE")
            candidate_devices: list[int] = []

            if custom_gpu is not None:
                try:
                    candidate_devices.append(int(custom_gpu))
                except ValueError:
                    pass

            running_under_wsl = (
                os.environ.get("WSL_DISTRO_NAME") is not None
                or os.environ.get("WSL_INTEROP") is not None
                or "microsoft" in platform.release().lower()
            )

            if prefer_cpu:
                candidate_devices.append(-1)
            else:
                if not running_under_wsl:
                    candidate_devices.append(0)
                else:
                    candidate_devices.extend([-1, 0])

            if -1 not in candidate_devices:
                candidate_devices.append(-1)

            last_error: Optional[Exception] = None

            for device in candidate_devices:
                cfg = build_config(device)
                if device == -1:
                    os.environ.setdefault("MAGNUM_GPU_DRIVER", "none")
                try:
                    self.sim = habitat_sim.Simulator(cfg)
                    self.agent = self.sim.get_agent(0)
                    return
                except Exception as exc:  # pragma: no cover - backend dependent
                    last_error = exc
                    self.sim = None

            raise RuntimeError(
                f"Unable to initialize Habitat-Sim simulator: {last_error}"
            )

        def _matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            if trace > 0:
                s = math.sqrt(trace + 1.0) * 2.0
                w = 0.25 * s
                x = (R[2, 1] - R[1, 2]) / s
                y = (R[0, 2] - R[2, 0]) / s
                z = (R[1, 0] - R[0, 1]) / s
            else:
                if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                    w = (R[2, 1] - R[1, 2]) / s
                    x = 0.25 * s
                    y = (R[0, 1] + R[1, 0]) / s
                    z = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                    w = (R[0, 2] - R[2, 0]) / s
                    x = (R[0, 1] + R[1, 0]) / s
                    y = 0.25 * s
                    z = (R[1, 2] + R[2, 1]) / s
                else:
                    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                    w = (R[1, 0] - R[0, 1]) / s
                    x = (R[0, 2] + R[2, 0]) / s
                    y = (R[1, 2] + R[2, 1]) / s
                    z = 0.25 * s
            return np.array([x, y, z, w])

        def load_object(self, index: int) -> Dict[str, Any]:
            assert self.sim is not None
            obj_mgr = self.sim.get_rigid_object_manager()
            obj_templates_mgr = self.sim.get_object_template_manager()

            if self.object_handle is not None:
                obj_mgr.remove_object_by_handle(self.object_handle)
                self.object_handle = None
                self.object_ref = None

            config = self.object_configs[index]
            render_asset = config.get("render_asset", "")
            if render_asset.startswith("../"):
                render_asset = str(Path("assets/ycb") / render_asset[3:])

            if not Path(render_asset).exists():
                # Continue anyway but warn caller.
                print(f"[HabitatWorker] Warning: render asset not found: {render_asset}")

            obj_template = obj_templates_mgr.create_new_template(render_asset)
            if obj_template is None:
                return {
                    "type": "error",
                    "message": f"Failed to create template for {render_asset}",
                }

            obj_template.scale = np.array([1.0, 1.0, 1.0])

            force_flat = config.get("force_flat_shading")
            if force_flat is None:
                requires_lighting = config.get("requires_lighting")
                if requires_lighting is not None:
                    force_flat = not bool(requires_lighting)

            if force_flat is not None:
                try:
                    obj_template.force_flat_shading = bool(force_flat)
                except AttributeError:
                    pass

            if "up" in config:
                try:
                    obj_template.orient_up = mn.Vector3(config["up"])  # type: ignore[assignment]
                except AttributeError:
                    pass
            if "front" in config:
                try:
                    obj_template.orient_front = mn.Vector3(config["front"])  # type: ignore[assignment]
                except AttributeError:
                    pass

            template_id = obj_templates_mgr.register_template(obj_template)
            obj = obj_mgr.add_object_by_template_id(template_id)
            if obj is None:
                return {
                    "type": "error",
                    "message": "Failed to add object to scene",
                }

            self.object_handle = obj.handle
            self.object_ref = obj

            obj.translation = np.array([0.0, 0.0, 0.0])

            bb = obj.aabb
            try:
                size_vec = np.array(bb.size(), dtype=np.float32)
            except TypeError:
                size_vec = np.array([1.0, 1.0, 1.0], dtype=np.float32)

            max_dimension = float(np.max(size_vec))
            radius = float(np.linalg.norm(size_vec) * 0.5)
            diameter = float(max(radius * 2.0, max_dimension))
            major_axis = float(max_dimension)

            try:
                center_vec = np.array(bb.center(), dtype=np.float32)
            except AttributeError:
                try:
                    center_vec = (np.array(bb.max, dtype=np.float32) + np.array(bb.min, dtype=np.float32)) * 0.5
                except AttributeError:
                    center_vec = np.zeros(3, dtype=np.float32)

            sensor = None
            if hasattr(self.agent, "_sensors"):
                sensor = self.agent._sensors.get("color_sensor")  # type: ignore[attr-defined]

            hfov_deg = float(getattr(sensor, "hfov", 90.0)) if sensor is not None else 90.0
            vfov_deg = float(getattr(sensor, "vfov", hfov_deg)) if sensor is not None else hfov_deg

            hfov_rad = math.radians(max(hfov_deg, 1e-3))
            vfov_rad = math.radians(max(vfov_deg, 1e-3))

            target_ratio = 0.8
            effective_fov = min(hfov_rad, vfov_rad)
            half_extent = max(major_axis * 0.5, 1e-3)
            ratio = max(min(target_ratio, 0.98), 1e-3)
            angle = max(effective_fov * ratio * 0.5, 1e-4)
            distance_for_ratio = half_extent / math.tan(angle)

            min_distance_h = radius / max(math.sin(hfov_rad / 2.0), 1e-3)
            min_distance_v = radius / max(math.sin(vfov_rad / 2.0), 1e-3)
            min_distance = max(min_distance_h, min_distance_v, radius * 1.05, 0.3)

            desired_distance = max(distance_for_ratio * 1.02, min_distance)

            default_theta = math.pi / 6.0
            default_phi = math.radians(12.0)

            return {
                "type": "object_loaded",
                "success": True,
                "index": index,
                "bounds": {
                    "size": size_vec.astype(float).tolist(),
                    "radius": float(radius),
                    "diameter": float(diameter),
                    "major_axis": float(major_axis),
                    "center": center_vec.astype(float).tolist(),
                },
                "camera": {
                    "hfov": hfov_deg,
                    "vfov": vfov_deg,
                },
                "default_camera": {
                    "distance": float(desired_distance),
                    "theta": float(default_theta),
                    "phi": float(default_phi),
                },
                "target": center_vec.astype(float).tolist(),
            }

        def render(self, camera: Dict[str, Any]) -> Dict[str, Any]:
            assert self.sim is not None
            assert self.agent is not None

            distance = float(camera.get("distance", 2.0))
            theta = float(camera.get("theta", 0.0))
            phi = float(camera.get("phi", 0.0))
            target = np.array(camera.get("target", [0.0, 0.0, 0.0]), dtype=np.float32)

            x = distance * math.cos(phi) * math.cos(theta)
            y = distance * math.cos(phi) * math.sin(theta)
            z = distance * math.sin(phi)
            camera_pos = target + np.array([x, y, z])

            forward = target - camera_pos
            forward_norm = np.linalg.norm(forward)
            if forward_norm > 0:
                forward /= forward_norm
            else:
                forward = np.array([0.0, 0.0, -1.0])

            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
            if right_norm > 0:
                right /= right_norm
                up = np.cross(right, forward)
            else:
                right = np.array([1.0, 0.0, 0.0])
                up = np.array([0.0, 1.0, 0.0])

            rotation_matrix = np.array(
                [
                    [right[0], up[0], -forward[0]],
                    [right[1], up[1], -forward[1]],
                    [right[2], up[2], -forward[2]],
                ]
            )

            state = self.agent.get_state()
            state.position = camera_pos
            state.rotation = self._matrix_to_quaternion(rotation_matrix)
            self.agent.set_state(state)

            if hasattr(self.sim, "get_light_setup") and self.object_ref is not None:
                try:
                    light_direction = target - camera_pos
                    norm = np.linalg.norm(light_direction)
                    if norm > 0:
                        light_direction /= norm
                    try:
                        light_setup = self.sim.get_light_setup(DEFAULT_LIGHTING_KEY)
                    except TypeError:  # pragma: no cover - older API
                        light_setup = self.sim.get_light_setup()

                    if hasattr(light_setup, "clear") and hasattr(
                        light_setup, "add_directional_light"
                    ):
                        light_setup.clear()
                        light_setup.add_directional_light(
                            direction=light_direction,
                            color=np.array([1.0, 1.0, 1.0]),
                            intensity=1.0,
                        )
                        try:
                            self.sim.set_light_setup(DEFAULT_LIGHTING_KEY, light_setup)
                        except TypeError:  # pragma: no cover - older API
                            self.sim.set_light_setup(light_setup)
                except AttributeError:
                    pass

            obs = self.sim.get_sensor_observations()
            image = obs.get("color_sensor")
            if image is None:
                self.frame_view.fill(0)
                return {"type": "frame"}

            if image.shape[2] == 4:
                image = image[:, :, :3]

            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

            self.frame_view[:, :, :] = image
            return {"type": "frame"}

    worker = WorkerState()
    ok, message = worker.initialize()
    response_queue.put({"type": "ready", "success": ok, "message": message})

    if not ok:
        worker.shutdown()
        return

    try:
        while True:
            cmd = command_queue.get()
            if cmd is None:
                continue
            ctype = cmd.get("type")

            if ctype == "shutdown":
                break

            if ctype == "load_object":
                index = int(cmd.get("index", 0))
                index %= max(len(worker.object_configs), 1)
                result = worker.load_object(index)
                response_queue.put(result)
                continue

            if ctype == "render":
                camera = cmd.get("camera", {})
                result = worker.render(camera)
                response_queue.put(result)
                continue

            response_queue.put({
                "type": "error",
                "message": f"Unknown command type: {ctype}",
            })

    except Exception as exc:  # pragma: no cover - runtime failure
        traceback.print_exc()
        response_queue.put({"type": "error", "message": str(exc)})
    finally:
        worker.shutdown()


class HabitatRenderBridge:
    """Thin IPC wrapper around the Habitat rendering worker."""

    def __init__(self, width: int, height: int, object_configs: List[Dict[str, Any]]):
        import multiprocessing as mp

        self.width = width
        self.height = height
        self.frame_size = width * height * 3

        ctx = mp.get_context("spawn")
        self.command_queue = ctx.Queue()
        self.response_queue = ctx.Queue()
        self.shared_frame = shared_memory.SharedMemory(create=True, size=self.frame_size)

        self.process = ctx.Process(
            target=habitat_worker_main,
            args=(
                self.command_queue,
                self.response_queue,
                self.shared_frame.name,
                self.width,
                self.height,
                object_configs,
            ),
            daemon=True,
        )
        self.process.start()

        ready_msg = self.response_queue.get()
        if not ready_msg.get("success", False):
            self.shutdown()
            raise RuntimeError(ready_msg.get("message", "Failed to start Habitat worker"))

    def load_object(self, index: int) -> Dict[str, Any]:
        self.command_queue.put({"type": "load_object", "index": index})
        return self._await_response()

    def render(self, camera: Dict[str, Any]) -> np.ndarray:
        self.command_queue.put({"type": "render", "camera": camera})
        msg = self._await_response()
        if msg.get("type") != "frame":
            raise RuntimeError(msg.get("message", "Failed to render frame"))
        frame = np.ndarray(
            (self.height, self.width, 3), dtype=np.uint8, buffer=self.shared_frame.buf
        ).copy()
        return frame

    def shutdown(self) -> None:
        try:
            if self.process is not None and self.process.is_alive():
                self.command_queue.put({"type": "shutdown"})
                self.process.join(timeout=5)
        finally:
            if self.process is not None and self.process.is_alive():  # pragma: no cover
                self.process.kill()
            try:
                self.shared_frame.close()
            except (FileNotFoundError, ValueError):
                pass
            try:
                self.shared_frame.unlink()
            except FileNotFoundError:
                pass
            # Close queues to release OS resources and silence resource tracker warnings.
            try:
                self.command_queue.close()
                self.command_queue.cancel_join_thread()
            except Exception:
                pass
            try:
                self.response_queue.close()
                self.response_queue.cancel_join_thread()
            except Exception:
                pass
            self.process = None

    def _await_response(self) -> Dict[str, Any]:
        msg = self.response_queue.get()
        if msg.get("type") == "error":
            raise RuntimeError(msg.get("message", "Habitat worker error"))
        return msg
