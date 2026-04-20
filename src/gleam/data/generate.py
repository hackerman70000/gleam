import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from gleam.config import IMG_SIZE, CameraConfig, GleamConfig, PhongConstants, SceneConfig
from gleam.data.sampling import SceneParams, ScenesSampler
from gleam.data.splits import make_splits
from gleam.renderer.gl_renderer import GLSLRenderer


def _write_params_json(
    out_path: Path,
    num_samples: int,
    seed: int,
    camera: CameraConfig,
    scene: SceneConfig,
    phong: PhongConstants,
    samples: list[dict],
    splits: dict,
) -> None:
    payload = {
        "num_samples": num_samples,
        "image_size": IMG_SIZE,
        "seed": seed,
        "camera": asdict(camera),
        "scene": asdict(scene),
        "phong_constants_0_255": {
            "ka": list(phong.ka_255),
            "ks": list(phong.ks_255),
            "ia": list(phong.ia_255),
            "id": list(phong.id_255),
            "is": list(phong.is_255),
        },
        "splits": splits,
        "samples": samples,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_params_npz(out_path: Path, samples: list[SceneParams]) -> None:
    np.savez(
        out_path,
        object_pos=np.stack([s.object_pos for s in samples]).astype(np.float32),
        light_pos=np.stack([s.light_pos for s in samples]).astype(np.float32),
        kd=np.stack([s.kd_255 for s in samples]).astype(np.uint8),
        n=np.array([s.shininess for s in samples], dtype=np.float32),
    )


def generate_dataset(
    out_dir: Path,
    num_samples: int,
    seed: int,
    config: GleamConfig | None = None,
) -> None:
    cfg = config or GleamConfig()
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    sampler = ScenesSampler(cfg.camera, cfg.scene, seed=seed)
    samples: list[SceneParams] = []
    records: list[dict] = []

    logger.info(f"Rendering {num_samples} samples -> {out_dir}")
    with GLSLRenderer(camera=cfg.camera) as renderer:
        for idx in tqdm(range(num_samples), desc="rendering", unit="img"):
            params = sampler.sample()
            img = renderer.render(
                object_pos=tuple(float(v) for v in params.object_pos),
                light_pos=tuple(float(v) for v in params.light_pos),
                kd_255=tuple(int(v) for v in params.kd_255),
                shininess=params.shininess,
            )
            Image.fromarray(img, "RGB").save(img_dir / f"{idx:06d}.png", compress_level=1)
            samples.append(params)
            records.append(params.as_record(idx))

    split = make_splits(
        num_samples=num_samples,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=seed,
    )
    _write_params_json(
        out_dir / "params.json",
        num_samples=num_samples,
        seed=seed,
        camera=cfg.camera,
        scene=cfg.scene,
        phong=cfg.phong,
        samples=records,
        splits=split.to_dict(),
    )
    _write_params_npz(out_dir / "params.npz", samples)
    logger.info(
        f"Done. train={len(split.train)}, val={len(split.val)}, test={len(split.test)}"
    )
