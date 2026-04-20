"""One-shot builder for every artefact needed in the project report.

Produces, under ``outputs/report/``:

* ``aggregates.csv / .md / .tex`` — 4-metric summary (mean/std/median)
* ``<split>_full.csv``            — per-sample metrics + scene parameters
* ``visual_best.png``              — 8 easiest test cases (lowest FLIP)
* ``visual_worst.png``             — 8 hardest test cases (highest FLIP)
* ``visual_random.png``            — 8 random test cases
* ``scenario_{easy,medium,hard}.png`` — three hand-picked configurations rendered
                                        side-by-side with the GLSL ground truth
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from gleam.eval.metrics import (
    compute_flip,
    compute_hausdorff_canny,
    compute_lpips,
    compute_ssim,
)
from gleam.renderer.gl_renderer import GLSLRenderer
from gleam.renderer.neural_renderer import NeuralRenderer
from gleam.utils.logging import get_device

METRIC_COLS: list[str] = ["FLIP", "LPIPS", "SSIM", "Hausdorff_Canny"]

# Hand-picked reference scenes covering an easy / medium / hard case.
HANDPICKED_SCENARIOS: list[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[int, int, int], float]] = [
    ("easy", (0.0, 0.0, 15.0), (5.0, 5.0, 20.0), (200, 100, 100), 5.0),
    ("medium", (3.0, 2.0, 10.0), (10.0, 8.0, 10.0), (50, 200, 80), 10.0),
    ("hard", (-4.0, 3.0, 12.0), (15.0, 0.0, -5.0), (40, 200, 220), 19.0),
]


def _pair_grid(records: list[dict[str, Any]], indices: list[int], cols: int) -> np.ndarray:
    by_idx = {r["idx"]: r for r in records}
    pairs = [np.concatenate([by_idx[i]["gt"], by_idx[i]["pred"]], axis=1) for i in indices]
    rows = [
        np.concatenate(pairs[r * cols : (r + 1) * cols], axis=1)
        for r in range(len(indices) // cols)
    ]
    return np.concatenate(rows, axis=0)


def _write_markdown_table(agg: pd.DataFrame, path: Path) -> None:
    lines = ["| Metric | Mean | Std | Median |", "| --- | ---: | ---: | ---: |"]
    for name, row in agg.iterrows():
        lines.append(
            f"| {name} | {row['mean']:.4f} | {row['std']:.4f} | {row['median']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_latex_table(agg: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Metric & Mean & Std & Median \\",
        r"\midrule",
    ]
    for name, row in agg.iterrows():
        safe = name.replace("_", r"\_")
        lines.append(f"{safe} & {row['mean']:.4f} & {row['std']:.4f} & {row['median']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def build_report(
    ckpt: Path,
    dataset_dir: Path,
    output_dir: Path,
    split: str = "test",
    n_best: int = 8,
    n_worst: int = 8,
    n_random: int = 8,
    use_ema: bool = True,
    device: str | None = None,
    seed: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device)
    variant = "ema" if use_ema else "raw"
    logger.info(f"device={device} | ckpt={ckpt} | split={split} | generator={variant}")

    state_key = None if use_ema else "generator"
    renderer = NeuralRenderer(ckpt_path=ckpt, device=device, state_key=state_key)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "params.json") as f:
        meta = json.load(f)
    params = np.load(dataset_dir / "params.npz")
    indices: list[int] = list(meta["splits"][split])

    records: list[dict[str, Any]] = []
    for idx in tqdm(indices, desc=f"report/{split}"):
        with Image.open(dataset_dir / "images" / f"{idx:06d}.png") as im:
            gt = np.array(im.convert("RGB"), dtype=np.uint8)
        object_pos = tuple(float(v) for v in params["object_pos"][idx])
        light_pos = tuple(float(v) for v in params["light_pos"][idx])
        kd = tuple(int(v) for v in params["kd"][idx])
        shininess = float(params["n"][idx])
        pred = renderer.render(object_pos, light_pos, kd, shininess)
        records.append(
            {
                "idx": idx,
                "gt": gt,
                "pred": pred,
                "object_pos": object_pos,
                "light_pos": light_pos,
                "kd": kd,
                "shininess": shininess,
                "FLIP": compute_flip(pred, gt),
                "LPIPS": compute_lpips(pred, gt, lpips_model, device),
                "SSIM": compute_ssim(pred, gt, device),
                "Hausdorff_Canny": compute_hausdorff_canny(pred, gt),
            }
        )

    df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("gt", "pred")} for r in records]
    )
    agg = df[METRIC_COLS].agg(["mean", "std", "median"]).T
    agg.index.name = "metric"

    agg.to_csv(output_dir / "aggregates.csv")
    df.to_csv(output_dir / f"{split}_full.csv", index=False)
    _write_markdown_table(agg, output_dir / "aggregates.md")
    _write_latex_table(agg, output_dir / "aggregates.tex")
    logger.info(f"\n{agg.to_string()}")

    df_by_flip = df.sort_values("FLIP").reset_index(drop=True)
    best_idx = df_by_flip.head(n_best)["idx"].astype(int).tolist()
    worst_idx = df_by_flip.tail(n_worst)["idx"].astype(int).tolist()[::-1]
    rng = np.random.default_rng(seed)
    random_idx = rng.choice(df["idx"].to_numpy(dtype=int), size=n_random, replace=False).tolist()

    for name, idxs, cols in (
        ("best", best_idx, 4),
        ("worst", worst_idx, 4),
        ("random", random_idx, 4),
    ):
        grid = _pair_grid(records, idxs, cols=cols)
        out_path = output_dir / f"visual_{name}.png"
        Image.fromarray(grid).save(out_path)
        logger.info(f"  {name}: {out_path}")

    with GLSLRenderer() as gl:
        for name, obj_pos, light, kd, n in HANDPICKED_SCENARIOS:
            gt = gl.render(obj_pos, light, kd, n)
            pred = renderer.render(obj_pos, light, kd, n)
            pair = np.concatenate([gt, pred], axis=1)
            out_path = output_dir / f"scenario_{name}.png"
            Image.fromarray(pair).save(out_path)
            logger.info(f"  scenario '{name}': {out_path}")

    logger.info(f"report artefacts -> {output_dir}")
