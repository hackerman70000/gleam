from dataclasses import dataclass

import numpy as np

from gleam.config import CameraConfig, SceneConfig


@dataclass(frozen=True)
class SceneParams:
    object_pos: np.ndarray  # (3,) float32
    light_pos: np.ndarray  # (3,) float32
    kd_255: np.ndarray  # (3,) uint8
    shininess: float

    def as_record(self, idx: int) -> dict:
        return {
            "idx": idx,
            "object_pos": self.object_pos.tolist(),
            "light_pos": self.light_pos.tolist(),
            "kd": [int(v) for v in self.kd_255],
            "n": float(self.shininess),
        }


class ScenesSampler:
    """Rejection sampler for Phong scenes that fit the given camera.

    Discards configurations in which the sphere would be behind the camera,
    outside the frustum, inside the light, or so dark that only ambient light
    matters. These produce degenerate or uninformative training examples.
    """

    def __init__(self, camera: CameraConfig, scene: SceneConfig, seed: int = 42) -> None:
        self.camera = camera
        self.scene = scene
        self._rng = np.random.default_rng(seed)

        eye = np.asarray(camera.eye, dtype=np.float64)
        target = np.asarray(camera.target, dtype=np.float64)
        view_dir = target - eye
        self._eye = eye
        self._view_dir = view_dir / np.linalg.norm(view_dir)
        self._half_fov_tan = float(np.tan(np.radians(camera.fovy_deg) / 2.0))

    def _is_valid(self, obj_pos: np.ndarray, light_pos: np.ndarray, kd: np.ndarray) -> bool:
        to_obj = obj_pos.astype(np.float64) - self._eye
        forward = float(np.dot(to_obj, self._view_dir))
        if not (self.scene.min_cam_distance <= forward <= self.scene.max_cam_distance):
            return False

        perp_vec = to_obj - forward * self._view_dir
        perp = float(np.linalg.norm(perp_vec))
        half_h = forward * self._half_fov_tan
        if perp > self.scene.frustum_margin * half_h:
            return False

        lp = float(np.linalg.norm(light_pos - obj_pos))
        if lp < self.scene.min_light_object_distance:
            return False

        if float(kd.max()) / 255.0 < self.scene.min_kd_component:
            return False

        return True

    def sample(self, max_attempts: int = 2000) -> SceneParams:
        lo_o, hi_o = self.scene.obj_pos_range
        lo_l, hi_l = self.scene.light_pos_range
        lo_n, hi_n = self.scene.shininess_range
        for _ in range(max_attempts):
            obj_pos = self._rng.uniform(lo_o, hi_o, 3).astype(np.float32)
            light_pos = self._rng.uniform(lo_l, hi_l, 3).astype(np.float32)
            kd = self._rng.integers(0, 256, 3).astype(np.uint8)
            if not self._is_valid(obj_pos, light_pos, kd):
                continue
            shininess = float(self._rng.uniform(lo_n, hi_n))
            return SceneParams(
                object_pos=obj_pos, light_pos=light_pos, kd_255=kd, shininess=shininess
            )
        raise RuntimeError(
            f"rejection sampling exceeded {max_attempts} attempts; loosen SceneConfig thresholds"
        )
