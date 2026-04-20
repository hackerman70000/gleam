import contextlib
from dataclasses import asdict
from pathlib import Path
from time import time

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gleam.config import GleamConfig
from gleam.data.dataset import PhongDataset
from gleam.data.features import FEATURE_DIM
from gleam.models.discriminator import PatchDiscriminator
from gleam.models.generator import CondGenerator
from gleam.training.ema import EMA
from gleam.training.losses import (
    non_saturating_d_loss,
    non_saturating_g_loss,
    r1_penalty,
)
from gleam.utils.logging import get_device, log_environment


def _autocast_ctx(device: str, use_amp: bool):
    if not use_amp:
        return contextlib.nullcontext()
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    # MPS bf16 autocast is experimental; skip it quietly.
    return contextlib.nullcontext()


def _maybe_compile(module: nn.Module, use_compile: bool, device: str) -> nn.Module:
    if not use_compile:
        return module
    if device != "cuda":
        logger.warning(f"--compile requested but device={device}; skipping torch.compile")
        return module
    return torch.compile(module, mode="reduce-overhead", fullgraph=False)


def _denorm(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().clamp(-1.0, 1.0).add(1.0).mul(127.5).byte()
    return arr.permute(0, 2, 3, 1).cpu().numpy()


def _save_sample_grid(
    generator: nn.Module,
    features: torch.Tensor,
    real: torch.Tensor,
    out_path: Path,
    cols: int = 4,
) -> None:
    generator.eval()
    with torch.no_grad():
        fake = generator(features)
    generator.train()
    fake_arr = _denorm(fake)
    real_arr = _denorm(real)
    pairs = np.stack([real_arr, fake_arr], axis=1)  # (N, 2, H, W, 3)
    n = pairs.shape[0]
    rows = (n + cols - 1) // cols
    h, w = pairs.shape[2], pairs.shape[3]
    canvas = np.zeros((rows * h, cols * 2 * w, 3), dtype=np.uint8)
    for idx in range(n):
        r, c = divmod(idx, cols)
        canvas[r * h : (r + 1) * h, (c * 2) * w : (c * 2 + 1) * w] = pairs[idx, 0]
        canvas[r * h : (r + 1) * h, (c * 2 + 1) * w : (c * 2 + 2) * w] = pairs[idx, 1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


@torch.no_grad()
def _validate(generator: nn.Module, loader: DataLoader, device: str) -> float:
    generator.eval()
    total, count = 0.0, 0
    for feats, real in loader:
        feats = feats.to(device)
        real = real.to(device)
        fake = generator(feats)
        total += F.l1_loss(fake, real, reduction="sum").item()
        count += real.numel()
    generator.train()
    return total / max(count, 1)


def run_training(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    resume: Path | None = None,
    config: GleamConfig | None = None,
    device_override: str | None = None,
    num_workers: int | None = None,
    use_amp: bool = False,
    use_compile: bool = False,
) -> None:
    cfg = config or GleamConfig()
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(device_override)
    log_environment(device)
    workers = cfg.train.num_workers if num_workers is None else num_workers
    logger.info(
        f"batch_size={batch_size}, epochs={epochs}, workers={workers}, "
        f"amp={use_amp}, compile={use_compile}"
    )

    train_ds = PhongDataset(dataset_dir, split="train")
    val_ds = PhongDataset(dataset_dir, split="val")
    logger.info(f"train={len(train_ds)} val={len(val_ds)}")

    pin = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )

    sample_feats, sample_real = next(iter(val_loader))
    sample_feats = sample_feats[:8].to(device)
    sample_real = sample_real[:8].to(device)

    generator = CondGenerator(feature_dim=FEATURE_DIM).to(device)
    discriminator = PatchDiscriminator(feature_dim=FEATURE_DIM).to(device)
    ema = EMA(generator, decay=cfg.train.ema_decay)
    ema.shadow.to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.train.lr_g, betas=cfg.train.adam_betas)
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.train.lr_d, betas=cfg.train.adam_betas
    )

    start_epoch = 0
    if resume is not None:
        logger.info(f"resuming from {resume}")
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        ema.load_state_dict(ckpt["ema_generator"])
        ema.shadow.to(device)
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1

    global_step = 0
    for epoch in range(start_epoch, epochs):
        t0 = time()
        running = {"d": 0.0, "g": 0.0, "l1": 0.0, "r1": 0.0}
        steps = 0

        for batch_idx, (features, real_images) in enumerate(train_loader):
            features = features.to(device, non_blocking=pin)
            real_images = real_images.to(device, non_blocking=pin)

            # --- Discriminator step ---
            with _autocast_ctx(device, use_amp):
                with torch.no_grad():
                    fake_images = generator(features)
                d_real = discriminator(real_images, features)
                d_fake = discriminator(fake_images, features)
                d_loss = non_saturating_d_loss(d_real, d_fake)

            r1_value = torch.zeros((), device=device)
            do_r1 = (batch_idx % cfg.train.r1_every) == 0
            if do_r1:
                # R1 gradient penalty in full precision — small grad norms in
                # bf16 collapse to zero and destabilise the regulariser.
                r1_value = r1_penalty(discriminator, real_images.float(), features.float())
                d_loss = d_loss + cfg.train.r1_gamma * cfg.train.r1_every * r1_value

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # --- Generator step ---
            with _autocast_ctx(device, use_amp):
                fake_images = generator(features)
                d_fake_for_g = discriminator(fake_images, features)
                g_adv = non_saturating_g_loss(d_fake_for_g)
                # Separate per-region means: sphere is ~0.4% of pixels, so a
                # single global L1 is dominated by the background reconstruction
                # ("output all black" solution). Computing foreground and
                # background means independently and summing with a weight
                # makes `foreground_weight` actually mean "sphere counts Nx
                # relative to background".
                fg_mask = (
                    (real_images > -0.95).any(dim=1, keepdim=True).float().expand_as(real_images)
                )
                pixel_l1 = (fake_images - real_images).abs()
                fg_pixels = fg_mask.sum().clamp_min(1.0)
                bg_pixels = (1.0 - fg_mask).sum().clamp_min(1.0)
                fg_l1 = (pixel_l1 * fg_mask).sum() / fg_pixels
                bg_l1 = (pixel_l1 * (1.0 - fg_mask)).sum() / bg_pixels
                g_l1 = bg_l1 + cfg.train.foreground_weight * fg_l1
                g_loss = g_adv + cfg.train.l1_lambda * g_l1

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            # EMA update on the underlying (non-compiled) module.
            ema_src = generator._orig_mod if hasattr(generator, "_orig_mod") else generator
            ema.update(ema_src)

            running["d"] += float(d_loss.detach())
            running["g"] += float(g_adv.detach())
            running["l1"] += float(g_l1.detach())
            running["r1"] += float(r1_value.detach())
            steps += 1
            global_step += 1

        elapsed = time() - t0
        avg = {k: v / max(steps, 1) for k, v in running.items()}
        logger.info(
            f"epoch {epoch + 1:3d}/{epochs} "
            f"| D {avg['d']:.3f} | G_adv {avg['g']:.3f} | L1 {avg['l1']:.4f} "
            f"| R1 {avg['r1']:.4f} | {elapsed:.1f}s"
        )

        if (epoch + 1) % cfg.train.val_every == 0 or epoch == epochs - 1:
            val_l1 = _validate(ema.shadow, val_loader, device)
            logger.info(f"  val L1 (EMA): {val_l1:.4f}")
            _save_sample_grid(
                ema.shadow, sample_feats, sample_real, sample_dir / f"epoch_{epoch + 1:04d}.png"
            )

        if (epoch + 1) % cfg.train.ckpt_every == 0 or epoch == epochs - 1:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:04d}.pt"
            gen_src = generator._orig_mod if hasattr(generator, "_orig_mod") else generator
            dis_src = (
                discriminator._orig_mod
                if hasattr(discriminator, "_orig_mod")
                else discriminator
            )
            torch.save(
                {
                    "epoch": epoch,
                    "generator": gen_src.state_dict(),
                    "discriminator": dis_src.state_dict(),
                    "ema_generator": ema.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "config": asdict(cfg),
                },
                ckpt_path,
            )
            latest = ckpt_dir / "latest.pt"
            ema_only = ckpt_dir / "ema_generator.pt"
            torch.save(
                {"ema_generator": ema.state_dict(), "feature_dim": FEATURE_DIM, "epoch": epoch},
                ema_only,
            )
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(ckpt_path.name)
            logger.info(f"  saved {ckpt_path.name}")

    logger.info("training complete")
