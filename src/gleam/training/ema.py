import copy

import torch
from torch import nn


class EMA:
    """Exponential moving average of model parameters.

    Keeps a shadow copy of the generator that converges to the long-run mean of
    parameter trajectories — in practice EMA weights produce noticeably nicer
    samples than the raw optimizer weights, particularly near the end of
    training. Decay ~0.999 is a standard choice for GANs with bs≈32.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)
