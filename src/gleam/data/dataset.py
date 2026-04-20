import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from gleam.data.features import raw_to_features

Split = Literal["train", "val", "test", "all"]


class PhongDataset(Dataset):
    """Dataset of pre-rendered Phong images + scene parameters.

    Returns tuples ``(features, image)`` where:
    * ``features`` is a float32 tensor shaped ``(FEATURE_DIM,)``
    * ``image``    is a float32 tensor shaped ``(3, 128, 128)`` in ``[-1, 1]``
    """

    def __init__(self, dataset_dir: Path, split: Split = "train") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        params = np.load(self.dataset_dir / "params.npz")

        with open(self.dataset_dir / "params.json") as f:
            meta = json.load(f)
        if split == "all":
            indices = np.arange(meta["num_samples"], dtype=np.int64)
        else:
            indices = np.asarray(meta["splits"][split], dtype=np.int64)

        self.indices = indices
        self.object_pos = torch.from_numpy(params["object_pos"][indices].astype(np.float32))
        self.light_pos = torch.from_numpy(params["light_pos"][indices].astype(np.float32))
        self.kd = torch.from_numpy(params["kd"][indices].astype(np.float32))
        self.shininess = torch.from_numpy(params["n"][indices].astype(np.float32))
        self.features = raw_to_features(
            self.object_pos, self.light_pos, self.kd, self.shininess
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i])
        with Image.open(self.image_dir / f"{idx:06d}.png") as im:
            arr = np.array(im.convert("RGB"), dtype=np.uint8)
        img = torch.from_numpy(arr).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        return self.features[i], img
