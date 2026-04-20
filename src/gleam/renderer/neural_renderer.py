from pathlib import Path

import numpy as np
import torch

from gleam.data.features import FEATURE_DIM, raw_to_features
from gleam.models.generator import CondGenerator
from gleam.renderer.base import Color3, Renderer, Vec3


class NeuralRenderer(Renderer):
    """Inference-only Renderer that calls the trained EMA generator.

    Loads ``ema_generator.pt`` (or any checkpoint with an ``ema_generator``
    state-dict), runs a single forward pass per call, and returns a uint8 image
    matching the :class:`GLSLRenderer` output.
    """

    def __init__(
        self,
        ckpt_path: Path,
        device: str | None = None,
        state_key: str | None = None,
    ) -> None:
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state_key is not None:
            state = ckpt[state_key]
        else:
            state = ckpt.get("ema_generator", ckpt)
        self.generator = CondGenerator(feature_dim=FEATURE_DIM).to(device)
        self.generator.load_state_dict(state)
        self.generator.eval()

    @torch.no_grad()
    def render(
        self,
        object_pos: Vec3,
        light_pos: Vec3,
        kd_255: Color3,
        shininess: float,
    ) -> np.ndarray:
        feats = raw_to_features(
            np.asarray(object_pos, dtype=np.float32),
            np.asarray(light_pos, dtype=np.float32),
            np.asarray(kd_255, dtype=np.float32),
            np.asarray(shininess, dtype=np.float32),
        )
        feats = feats.to(self.device).unsqueeze(0)
        out = self.generator(feats).squeeze(0)
        arr = out.clamp(-1.0, 1.0).add(1.0).mul(127.5).byte()
        return arr.permute(1, 2, 0).cpu().numpy()
