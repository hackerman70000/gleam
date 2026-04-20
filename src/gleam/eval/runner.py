import json
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from gleam.data.dataset import Split
from gleam.eval.metrics import (
    compute_flip,
    compute_hausdorff_canny,
    compute_lpips,
    compute_ssim,
)
from gleam.renderer.neural_renderer import NeuralRenderer
from gleam.utils.logging import get_device


def _load_indices(dataset_dir: Path, split: Split) -> list[int]:
    with open(dataset_dir / "params.json") as f:
        meta = json.load(f)
    if split == "all":
        return list(range(meta["num_samples"]))
    return list(meta["splits"][split])


def _load_sample(dataset_dir: Path, params_npz: np.lib.npyio.NpzFile, idx: int) -> dict:
    img_path = dataset_dir / "images" / f"{idx:06d}.png"
    with Image.open(img_path) as im:
        gt = np.array(im.convert("RGB"), dtype=np.uint8)
    return {
        "idx": idx,
        "object_pos": params_npz["object_pos"][idx].tolist(),
        "light_pos": params_npz["light_pos"][idx].tolist(),
        "kd": params_npz["kd"][idx].tolist(),
        "n": float(params_npz["n"][idx]),
        "gt": gt,
    }


def run_evaluation(
    ckpt: Path,
    dataset_dir: Path,
    output_dir: Path,
    split: str = "test",
    lpips_net: str = "alex",
    canny_sigma: float = 1.0,
    device: str | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(device)
    logger.info(f"device={device} | ckpt={ckpt} | split={split}")

    renderer = NeuralRenderer(ckpt_path=ckpt, device=device)
    lpips_model = lpips.LPIPS(net=lpips_net).to(device).eval()

    dataset_dir = Path(dataset_dir)
    params = np.load(dataset_dir / "params.npz")
    indices = _load_indices(dataset_dir, split)  # type: ignore[arg-type]
    logger.info(f"evaluating {len(indices)} samples")

    rows: list[dict] = []
    for idx in tqdm(indices, desc=f"eval/{split}"):
        sample = _load_sample(dataset_dir, params, idx)
        pred = renderer.render(
            object_pos=tuple(sample["object_pos"]),
            light_pos=tuple(sample["light_pos"]),
            kd_255=tuple(sample["kd"]),
            shininess=sample["n"],
        )
        rows.append(
            {
                "idx": sample["idx"],
                "FLIP": compute_flip(pred, sample["gt"]),
                "LPIPS": compute_lpips(pred, sample["gt"], lpips_model, device),
                "SSIM": compute_ssim(pred, sample["gt"], device),
                "Hausdorff_Canny": compute_hausdorff_canny(pred, sample["gt"], sigma=canny_sigma),
            }
        )

    df = pd.DataFrame(rows)
    per_sample_path = output_dir / f"{split}_per_sample.csv"
    df.to_csv(per_sample_path, index=False)

    metric_cols = ["FLIP", "LPIPS", "SSIM", "Hausdorff_Canny"]
    agg = df[metric_cols].agg(["mean", "std", "median"]).T
    agg.index.name = "metric"
    agg_path = output_dir / f"{split}_aggregates.csv"
    agg.to_csv(agg_path)

    logger.info(f"\n{agg.to_string()}")
    logger.info(f"per-sample -> {per_sample_path}")
    logger.info(f"aggregates -> {agg_path}")
