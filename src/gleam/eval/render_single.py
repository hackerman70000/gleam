from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from gleam.renderer.gl_renderer import GLSLRenderer
from gleam.renderer.neural_renderer import NeuralRenderer


def render_single_image(
    ckpt: Path,
    object_pos: tuple[float, float, float],
    light_pos: tuple[float, float, float],
    kd: tuple[int, int, int],
    shininess: float,
    out_path: Path,
    compare_with_gt: bool,
) -> None:
    neural = NeuralRenderer(ckpt_path=ckpt)
    pred = neural.render(object_pos, light_pos, kd, shininess)

    if compare_with_gt:
        with GLSLRenderer() as gl:
            gt = gl.render(object_pos, light_pos, kd, shininess)
        tile = np.concatenate([gt, pred], axis=1)
        Image.fromarray(tile, "RGB").save(out_path)
        logger.info(f"saved side-by-side GT | neural -> {out_path}")
    else:
        Image.fromarray(pred, "RGB").save(out_path)
        logger.info(f"saved neural render -> {out_path}")
