"""Transformation of raw scene parameters into a feature vector for the network.

The research recommends 11 scalars:

    object_pos / 20                         # 3
    (light_pos - object_pos) / 40           # 3
    log(|light - object| + 1) / log(70)     # 1
    kd / 255                                # 3
    (shininess - 3) / 17                    # 1

All components lie in roughly [-1, 1] so the MLP has a well-conditioned input.
"""

from __future__ import annotations

import numpy as np
import torch

FEATURE_DIM = 11

POS_SCALE = 20.0
LIGHT_OFFSET_SCALE = 40.0
LOG_DIST_SCALE = float(np.log(70.0))
SHININESS_MIN = 3.0
SHININESS_SCALE = 17.0


def raw_to_features(
    object_pos: np.ndarray | torch.Tensor,
    light_pos: np.ndarray | torch.Tensor,
    kd_255: np.ndarray | torch.Tensor,
    shininess: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Convert raw scene params (any batched shape) into a feature tensor.

    Each input should broadcast to shape ``(..., 3)`` or ``(...,)`` for scalars.
    Returns a tensor with last dim == ``FEATURE_DIM``.
    """
    object_pos = torch.as_tensor(object_pos, dtype=torch.float32)
    light_pos = torch.as_tensor(light_pos, dtype=torch.float32)
    kd_255 = torch.as_tensor(kd_255, dtype=torch.float32)
    shininess = torch.as_tensor(shininess, dtype=torch.float32)

    delta = light_pos - object_pos
    distance = delta.norm(dim=-1, keepdim=True)
    log_dist = torch.log1p(distance) / LOG_DIST_SCALE

    pos_norm = object_pos / POS_SCALE
    delta_norm = delta / LIGHT_OFFSET_SCALE
    kd_norm = kd_255 / 255.0
    shininess_norm = ((shininess - SHININESS_MIN) / SHININESS_SCALE).unsqueeze(-1)

    return torch.cat([pos_norm, delta_norm, log_dist, kd_norm, shininess_norm], dim=-1)
