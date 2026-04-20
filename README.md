# gleam

Neural Phong shader trained as a [Pix2Pix-style](https://arxiv.org/abs/1611.07004)
conditional GAN. Scene parameters `(object_pos, light_pos, kd, shininess)` in,
128×128 RGB out. Evaluated against a GLSL reference with FLIP, LPIPS, SSIM, and
Hausdorff on Canny edges.

![Eight random test samples: ground truth on the left of each pair, neural renderer on the right](outputs/report/visual_random.png)
_Eight random test samples: ground truth on the left of each pair, neural renderer on the right_

## Setup

Requires [uv](https://docs.astral.sh/uv/). macOS (MPS) and Linux (CUDA) are
both supported; on Linux CUDA wheels (`cu128`) are selected automatically.

```bash
uv sync
```

## Usage

```bash
uv run gleam generate-data --n 15000
uv run gleam train --epochs 120 --batch-size 128 --amp --compile
uv run gleam eval   --ckpt outputs/checkpoints/ema_generator.pt
uv run gleam report --ckpt outputs/checkpoints/ema_generator.pt
uv run gleam render --ckpt outputs/checkpoints/ema_generator.pt \
    --object-pos 3 2 5 --light-pos 10 8 10 \
    --kd 200 50 50 --shininess 10 --compare --out render.png
```
