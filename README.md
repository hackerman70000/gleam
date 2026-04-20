# gleam

Neural Phong shader trained as a conditional GAN. SIGK Projekt 3.

Given scene parameters `(object_pos, light_pos, kd, shininess)` the network
predicts a 128×128 RGB image that mirrors the output of the ground-truth
Phong shader. The pipeline is split into three stages — data synthesis with a
headless ModernGL renderer, training of the cGAN generator, and quantitative
evaluation against the reference renderer (FLIP, LPIPS, SSIM, Hausdorff on
Canny edges).

## Requirements

- macOS (tested on Apple Silicon with PyTorch MPS) or Linux/CUDA
- Python 3.12 via [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Commands

### Dataset generation

Renders `n` Phong images with randomized parameters and writes them together
with a `params.json` manifest and a `params.npz` for fast loading.

```bash
uv run gleam generate-data --n 3000 --out dataset
```

### Training

Trains the cGAN (Pix2Pix-style: `L1(λ=100) + non-saturating BCE + R1(γ=1,
lazy every 16 steps)`, Adam(β=0.5, 0.999), EMA decay 0.999). Checkpoints land
in `outputs/checkpoints/`; visual samples in `outputs/samples/epoch_XXXX.png`.

```bash
uv run gleam train --epochs 300 --batch-size 32
uv run gleam train --resume outputs/checkpoints/latest.pt   # resume
```

On Apple Silicon MPS expect ~100 s per epoch at `batch_size=32`, so a full
300-epoch run takes roughly 8 h.

### Evaluation

Evaluates the trained generator against the ground-truth renderer on the test
split. Writes `test_per_sample.csv` (per-image metrics) and
`test_aggregates.csv` (mean / std / median).

```bash
uv run gleam eval --ckpt outputs/checkpoints/ema_generator.pt --split test
```

Metrics: FLIP (↓), LPIPS (↓), SSIM (↑), Hausdorff on Canny edges (↓, pixels).

### Single render / side-by-side comparison

```bash
uv run gleam render \
  --ckpt outputs/checkpoints/ema_generator.pt \
  --object-pos 3 2 5 --light-pos 10 8 10 \
  --kd 200 50 50 --shininess 10 \
  --out render.png --compare
```

## Layout

```
gleam/
├── pyproject.toml              # uv-managed dependencies
├── resources/
│   ├── shaders/phong/          # vertex + fragment shaders
│   └── models/                 # sphere.obj
├── src/gleam/
│   ├── cli.py                  # typer entry point (`gleam ...`)
│   ├── config.py               # CameraConfig, SceneConfig, PhongConstants, ...
│   ├── renderer/
│   │   ├── base.py             # Renderer ABC
│   │   ├── mesh.py             # OBJ loader
│   │   ├── gl_renderer.py      # headless ModernGL (ground truth)
│   │   └── neural_renderer.py  # NeuralRenderer(Renderer)
│   ├── data/
│   │   ├── sampling.py         # rejection sampler (frustum + light checks)
│   │   ├── splits.py           # train/val/test split helper
│   │   ├── features.py         # raw params → 11-dim feature vector
│   │   ├── generate.py         # batch rendering
│   │   └── dataset.py          # PyTorch Dataset
│   ├── models/
│   │   ├── conditioner.py      # FourierFeatures, FiLM, SceneConditioner
│   │   ├── generator.py        # CondGenerator (MLP → 5× upsample)
│   │   └── discriminator.py    # PatchGAN
│   ├── training/
│   │   ├── losses.py           # BCE + R1 penalty
│   │   ├── ema.py              # shadow-copy EMA
│   │   └── trainer.py          # main training loop
│   ├── eval/
│   │   ├── metrics.py          # FLIP, LPIPS, SSIM, Hausdorff-Canny
│   │   ├── runner.py           # batch evaluation
│   │   └── render_single.py    # one-shot render + side-by-side
│   └── utils/logging.py        # loguru setup, device detection
└── dataset/, outputs/          # generated artefacts (gitignored)
```

## Design notes

- **Feature encoding.** Raw scene parameters are rescaled and expanded:
  `obj_pos/20`, `(light−obj)/40`, `log(1+‖light−obj‖)/log 70`, `kd/255`,
  `(n−3)/17` — 11 dimensions, each in roughly `[−1, 1]`. Inside
  `SceneConditioner` these are expanded with Fourier features (L=6) so the
  downstream MLP can fit sharp speculars.
- **Generator.** `conditioner → 4×4 feature map → 5× {upsample + conv +
  GroupNorm + FiLM + SiLU} → conv → tanh`. FiLM layers start as identity
  (zero-initialised `γ, β`), so training starts from an unconditional state.
- **Discriminator.** PatchGAN with the raw 11-dim feature broadcast as extra
  channels; InstanceNorm (mandatory because real/fake statistics differ).
- **Rejection sampling.** Keeps the nominal ranges from the assignment
  (±20 on all axes, kd ∈ [0, 255], n ∈ [3, 20]) but discards configurations
  that place the sphere behind the camera, outside the frustum, inside the
  light, or almost black.

## Reproducibility

- Deterministic RNG seed for dataset (`--seed`, default 42).
- ModernGL on the same GPU produces bit-identical renders across runs; OpenGL
  makes no cross-device promise, so expect slight differences between machines
  (ARB invariance rules).

## Running on NVIDIA DGX Spark (Blackwell, Linux ARM64)

The project is configured to pick up **CUDA 12.8** PyTorch wheels automatically
when installed on Linux (see `[tool.uv.sources]` in `pyproject.toml`); macOS
continues to use MPS. ModernGL falls back to **EGL** for headless rendering
— no X display needed.

### One-off setup on the DGX

```bash
# 1. Copy the project (without the local dataset / venv) to the DGX.
rsync -av --exclude .venv --exclude dataset --exclude outputs \
      gleam/ dgx:~/gleam/

# 2. SSH in and install uv if it is not already present.
ssh dgx
curl -LsSf https://astral.sh/uv/install.sh | sh   # one-off

# 3. Sync dependencies — uv picks cu128 wheels on linux.
cd ~/gleam
uv sync
```

Verify CUDA and EGL are wired up:

```bash
uv run python -c "
import torch, moderngl
print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
ctx = moderngl.create_context(standalone=True, backend='egl')
print('gl:', ctx.info['GL_VERSION'])
"
```

### Generate the dataset on the DGX

Regenerate rather than transferring PNGs — OpenGL is not guaranteed to be
bit-identical across GPU vendors, and the DGX renders 3 000 images in under
two seconds.

```bash
uv run gleam generate-data --n 3000
```

### Training with CUDA-optimised flags

```bash
./scripts/train_dgx.sh         # batch 128, 16 workers, BF16 autocast, torch.compile
# or fully explicit
uv run gleam train --device cuda --batch-size 128 --num-workers 16 --amp --compile
```

BF16 autocast runs the forward passes of G and D in `bfloat16` while keeping
the R1 gradient penalty in `float32` (small gradient norms collapse to zero in
bf16). `--compile` uses `torch.compile(mode='reduce-overhead')` for a steady
speed-up on Blackwell.

### Pull the trained checkpoint back

```bash
rsync -av dgx:~/gleam/outputs/checkpoints/ema_generator.pt outputs/checkpoints/
uv run gleam eval --ckpt outputs/checkpoints/ema_generator.pt
```
