from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class DatasetScalerPolicy:
    dataset_dir: Path
    data_is_standardized: bool
    stats_path: Path | None

    @property
    def requires_forward_transform(self) -> bool:
        return not self.data_is_standardized


def resolve_data_is_standardized(meta) -> bool:
    if "data_is_standardized" not in meta:
        return False
    data_is_standardized = meta["data_is_standardized"]
    if not isinstance(data_is_standardized, bool):
        raise ValueError("meta field 'data_is_standardized' must be a boolean")
    return data_is_standardized


def resolve_dataset_scaler_policy(dataset_dir, data_is_standardized) -> DatasetScalerPolicy:
    dataset_dir = Path(dataset_dir).expanduser()
    if not isinstance(data_is_standardized, bool):
        raise ValueError("runtime field 'data_is_standardized' must be a boolean")
    stats_path = dataset_dir / "stats.npz"
    if data_is_standardized and not stats_path.exists():
        raise ValueError(
            "dataset '{}' sets data_is_standardized=true but is missing '{}'".format(
                dataset_dir,
                stats_path.name,
            )
        )

    return DatasetScalerPolicy(
        dataset_dir=dataset_dir,
        data_is_standardized=data_is_standardized,
        stats_path=stats_path if data_is_standardized else None,
    )


def load_standard_scaler_stats(stats_path) -> tuple[np.ndarray, np.ndarray]:
    stats_path = Path(stats_path).expanduser()
    with np.load(stats_path, allow_pickle=False) as stats:
        if "mean" not in stats or "std" not in stats:
            raise ValueError("stats file '{}' must define 'mean' and 'std' arrays".format(stats_path))
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != std.shape:
        raise ValueError(
            "stats file '{}' must define 'mean' and 'std' with the same shape, got {} and {}".format(
                stats_path,
                tuple(mean.shape),
                tuple(std.shape),
            )
        )
    std = np.where(std == 0.0, 1.0, std).astype(np.float32, copy=False)
    return mean, std


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = None
        self.std = None
        self.set_stats(mean, std)

    @classmethod
    def fit(cls, data):
        tensor = torch.as_tensor(data, dtype=torch.float32)

        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True, unbiased=False)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return cls(mean, std)

    def set_stats(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).detach()
        self.std = torch.as_tensor(std, dtype=torch.float32).detach()
        return self

    def transform(self, input_data, mask=None):
        output = (input_data - self.mean) / self.std
        if mask is None:
            return output
        return torch.where(mask, output, input_data)

    def inverse_transform(self, input_data, mask=None):
        output = input_data * self.std + self.mean
        if mask is None:
            return output
        return torch.where(mask, output, input_data)
