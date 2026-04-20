from abc import ABC, abstractmethod

import numpy as np

Vec3 = tuple[float, float, float]
Color3 = tuple[int, int, int]


class Renderer(ABC):
    """Uniform interface for any Phong renderer (GLSL ground truth or neural)."""

    @abstractmethod
    def render(
        self,
        object_pos: Vec3,
        light_pos: Vec3,
        kd_255: Color3,
        shininess: float,
    ) -> np.ndarray:
        """Return an ``(H, W, 3)`` uint8 image in RGB order."""

    def close(self) -> None:
        """Release any underlying resources. Default is a no-op."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
