from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywavefront


@dataclass(frozen=True)
class Mesh:
    """Triangle mesh with per-vertex positions and normals, pre-expanded (no indices)."""

    positions: np.ndarray  # (N, 3) float32
    normals: np.ndarray  # (N, 3) float32

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])


_VERTEX_FORMAT = "T2F_N3F_V3F"
_FLOATS_PER_VERTEX = 8  # 2 texcoord + 3 normal + 3 position


def load_obj(path: Path) -> Mesh:
    scene = pywavefront.Wavefront(
        str(path), collect_faces=True, create_materials=True, cache=False
    )
    mesh = next(iter(scene.meshes.values()))
    if not mesh.materials:
        raise ValueError(f"OBJ file has no materials with vertex data: {path}")

    material = mesh.materials[0]
    if material.vertex_format != _VERTEX_FORMAT:
        raise ValueError(
            f"Unsupported vertex format {material.vertex_format!r}, expected {_VERTEX_FORMAT!r}"
        )

    interleaved = np.asarray(material.vertices, dtype=np.float32).reshape(-1, _FLOATS_PER_VERTEX)
    normals = np.ascontiguousarray(interleaved[:, 2:5])
    positions = np.ascontiguousarray(interleaved[:, 5:8])
    return Mesh(positions=positions, normals=normals)
