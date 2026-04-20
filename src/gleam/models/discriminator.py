import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (70×70 receptive field for 128² inputs).

    Conditioning is injected as extra channels via spatial broadcast: the raw
    feature vector is tiled to match the image grid and concatenated with the
    RGB input. InstanceNorm is preferred over BatchNorm because real/fake
    statistics differ and the GAN recipe tolerates it well.
    """

    def __init__(self, feature_dim: int = 11, in_channels: int = 3, ndf: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        in_ch = in_channels + feature_dim
        layers: list[nn.Module] = []
        channels = [ndf, ndf * 2, ndf * 4, ndf * 8]
        for i, out_ch in enumerate(channels):
            stride = 2 if i < 3 else 1
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1),
                nn.InstanceNorm2d(out_ch) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        layers += [nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, raw_features: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        cond_map = raw_features.view(b, self.feature_dim, 1, 1).expand(b, self.feature_dim, h, w)
        return self.net(torch.cat([image, cond_map], dim=1))
