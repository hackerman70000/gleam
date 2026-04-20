from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    train: np.ndarray  # int64 indices
    val: np.ndarray
    test: np.ndarray

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "train": self.train.tolist(),
            "val": self.val.tolist(),
            "test": self.test.tolist(),
        }


def make_splits(
    num_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> DatasetSplit:
    if not (0.0 <= val_fraction < 1.0) or not (0.0 < test_fraction < 1.0):
        raise ValueError("fractions must lie in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val + test fractions must be < 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)

    n_test = int(round(num_samples * test_fraction))
    n_val = int(round(num_samples * val_fraction))
    test = indices[:n_test]
    val = indices[n_test : n_test + n_val]
    train = indices[n_test + n_val :]
    return DatasetSplit(
        train=np.sort(train), val=np.sort(val), test=np.sort(test)
    )
