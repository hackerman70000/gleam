"""Visualise the *current* (non-EMA) generator on a handful of validation samples.

Useful to disambiguate "EMA is cold" from "the generator has actually collapsed".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gleam.data.dataset import PhongDataset
from gleam.data.features import FEATURE_DIM
from gleam.models.generator import CondGenerator


def main(
    ckpt: Path = Path("outputs/checkpoints/latest.pt"),
    dataset_dir: Path = Path("dataset"),
    out_path: Path = Path("outputs/raw_G_current.png"),
    n_samples: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    G = CondGenerator(feature_dim=FEATURE_DIM).to(device)
    G.load_state_dict(ck["generator"])
    G.eval()

    ds = PhongDataset(dataset_dir, split="val")
    feats = torch.stack([ds[i][0] for i in range(n_samples)]).to(device)
    reals = torch.stack([ds[i][1] for i in range(n_samples)]).to(device)

    with torch.no_grad():
        fakes = G(feats)

    def denorm(t: torch.Tensor) -> np.ndarray:
        return (
            t.detach().clamp(-1, 1).add(1).mul(127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        )

    ra = denorm(reals)
    fa = denorm(fakes)
    pairs = [np.concatenate([ra[i], fa[i]], axis=1) for i in range(n_samples)]
    cols = 4
    rows = [np.concatenate(pairs[r * cols : (r + 1) * cols], axis=1) for r in range(n_samples // cols)]
    grid = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    print(f"saved {out_path}  (shape={grid.shape}, epoch={ck.get('epoch', '?')})")


if __name__ == "__main__":
    main()
