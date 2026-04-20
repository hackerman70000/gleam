from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RESOURCES_DIR: Path = PROJECT_ROOT / "resources"
SHADERS_DIR: Path = RESOURCES_DIR / "shaders"
MODELS_DIR: Path = RESOURCES_DIR / "models"
DEFAULT_DATASET_DIR: Path = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

IMG_SIZE: int = 128
DEFAULT_SEED: int = 42


@dataclass(frozen=True)
class CameraConfig:
    eye: tuple[float, float, float] = (0.0, 0.0, 25.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fovy_deg: float = 45.0
    near: float = 0.1
    far: float = 500.0


@dataclass(frozen=True)
class SceneConfig:
    sphere_radius: float = 1.0  # matches the provided sphere.obj
    # Raw ranges from the assignment (PDF).
    obj_pos_range: tuple[float, float] = (-20.0, 20.0)
    light_pos_range: tuple[float, float] = (-20.0, 20.0)
    shininess_range: tuple[float, float] = (3.0, 20.0)
    # Rejection-sampling thresholds tuned to the default camera (eye_z=25, fovy=45).
    # Raw position ranges still match the PDF, but we discard samples whose sphere
    # would be too tiny (< ~5 px) or tangent to the camera near plane.
    min_cam_distance: float = 5.0
    max_cam_distance: float = 30.0
    min_light_object_distance: float = 1.5
    frustum_margin: float = 0.9
    min_kd_component: float = 5.0 / 255.0


@dataclass(frozen=True)
class PhongConstants:
    """Static Phong coefficients from the assignment (0-255 and normalized)."""

    ka_255: tuple[int, int, int] = (76, 76, 76)
    ks_255: tuple[int, int, int] = (255, 255, 255)
    ia_255: tuple[int, int, int] = (25, 25, 25)
    id_255: tuple[int, int, int] = (255, 255, 255)
    is_255: tuple[int, int, int] = (255, 255, 255)

    @property
    def ka(self) -> tuple[float, float, float]:
        return tuple(v / 255.0 for v in self.ka_255)  # type: ignore[return-value]

    @property
    def ia(self) -> tuple[float, float, float]:
        return tuple(v / 255.0 for v in self.ia_255)  # type: ignore[return-value]


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 3000
    val_fraction: float = 0.1
    test_fraction: float = 0.2
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 300
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    adam_betas: tuple[float, float] = (0.5, 0.999)
    l1_lambda: float = 100.0
    # Foreground (non-black GT pixels) gets `1 + foreground_weight` multiplier
    # in the L1 loss. Compensates for the 99.6% / 0.4% background / sphere
    # imbalance that otherwise lets the model win by outputting "all black".
    foreground_weight: float = 10.0
    r1_gamma: float = 0.1
    r1_every: int = 16
    ema_decay: float = 0.995
    val_every: int = 10
    ckpt_every: int = 25
    num_workers: int = 4


@dataclass(frozen=True)
class GleamConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    phong: PhongConstants = field(default_factory=PhongConstants)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
