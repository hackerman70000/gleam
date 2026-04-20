#!/usr/bin/env bash
# Recommended training settings for NVIDIA DGX Spark (Blackwell, Linux ARM).
# BF16 autocast is safe on Blackwell; torch.compile gives a steady ~20% win.

set -euo pipefail
cd "$(dirname "$0")/.."

uv run gleam train \
    --epochs 300 \
    --batch-size 128 \
    --num-workers 16 \
    --device cuda \
    --amp \
    --compile \
    "$@"
