import torch
from torch import nn
from torch.nn import functional as F


def non_saturating_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Standard non-saturating BCE discriminator loss (Goodfellow 2014)."""
    real_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    fake_loss = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    return 0.5 * (real_loss + fake_loss)


def non_saturating_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


def r1_penalty(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    conditioning: torch.Tensor,
) -> torch.Tensor:
    """R1 gradient penalty (Mescheder 2018): ``(γ/2) · ||∇ D(x_real)||²``.

    Applied lazily (every N steps) to amortize the second backward pass. Caller
    multiplies by ``γ`` and optionally a ``r1_every`` factor.
    """
    real_images = real_images.detach().requires_grad_(True)
    d_real = discriminator(real_images, conditioning)
    grads = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
    )[0]
    return 0.5 * grads.pow(2).flatten(1).sum(dim=1).mean()
