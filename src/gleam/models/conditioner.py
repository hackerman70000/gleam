import math
from collections.abc import Sequence

import torch
from torch import nn


class FourierFeatures(nn.Module):
    """NeRF-style positional encoding.

    ``γ(p) = [p, sin(2^k π p), cos(2^k π p)]`` for ``k = 0..L-1``. Expands the
    low-dimensional conditioning vector so the MLP downstream can fit
    high-frequency dependencies (sharp speculars, backface transitions).
    """

    def __init__(self, in_dim: int, num_bands: int = 6, include_input: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_bands = num_bands
        self.include_input = include_input
        freqs = (2.0 ** torch.arange(num_bands, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return self.in_dim * (2 * self.num_bands) + (self.in_dim if self.include_input else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = x.unsqueeze(-1) * self.freqs  # (B, in_dim, num_bands)
        encoded = torch.cat([projected.sin(), projected.cos()], dim=-1).flatten(1)
        if self.include_input:
            return torch.cat([x, encoded], dim=-1)
        return encoded


class FiLM(nn.Module):
    """Feature-wise linear modulation: ``out = (1 + γ) ⊙ x + β``.

    The ``1 + γ`` reparameterization lets us zero-initialize the projection so
    modulation starts as identity and the generator can train stably before
    the conditioner contributes.
    """

    def __init__(self, cond_dim: int, num_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * num_features)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.num_features = num_features

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(cond).chunk(2, dim=-1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return feat * (1.0 + gamma) + beta


class SceneConditioner(nn.Module):
    """Encodes raw 11-dim scene features with Fourier features + a small MLP.

    Produces a single conditioning vector that is then consumed by per-layer
    FiLM modules in the generator.
    """

    def __init__(
        self,
        raw_dim: int = 11,
        num_bands: int = 6,
        hidden: int = 256,
        cond_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = FourierFeatures(raw_dim, num_bands=num_bands, include_input=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cond_dim),
        )
        self.cond_dim = cond_dim

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(raw_features))
