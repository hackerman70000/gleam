import torch
from torch import nn
from torch.nn import functional as F

from gleam.models.conditioner import FiLM, SceneConditioner


class _UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.film1 = FiLM(cond_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.film2 = FiLM(cond_dim, out_ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.silu(self.film1(self.norm1(self.conv1(x)), cond))
        x = F.silu(self.film2(self.norm2(self.conv2(x)), cond))
        return x


class CondGenerator(nn.Module):
    """Param-to-image generator: ``scene features -> 128 × 128 × 3``.

    Architecture: MLP lifts the conditioning to a 4×4 feature map, then five
    upsample-conv-FiLM blocks take us to 128×128. Channel counts taper
    geometrically so the parameter count stays near 8M.
    """

    def __init__(
        self,
        feature_dim: int = 11,
        cond_hidden: int = 256,
        cond_dim: int = 128,
        base_ch: int = 256,
    ) -> None:
        super().__init__()
        self.conditioner = SceneConditioner(
            raw_dim=feature_dim, hidden=cond_hidden, cond_dim=cond_dim
        )
        self.base_ch = base_ch

        self.init_fc = nn.Linear(cond_dim, base_ch * 4 * 4)
        channels = [base_ch, base_ch, base_ch // 2, base_ch // 4, base_ch // 8, base_ch // 16]
        self.blocks = nn.ModuleList(
            [_UpsampleBlock(channels[i], channels[i + 1], cond_dim) for i in range(5)]
        )
        self.to_rgb = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        cond = self.conditioner(raw_features)
        x = self.init_fc(cond).view(-1, self.base_ch, 4, 4)
        for block in self.blocks:
            x = block(x, cond)
        return torch.tanh(self.to_rgb(x))
