from __future__ import annotations

import json
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

from .scaler import resolve_data_is_standardized


REMOVED_GRID3D_ARGS = ("train_patch_shape", "eval_tile_shape", "eval_tile_overlap")


def _reject_removed_grid3d_args(kwargs) -> None:
    for arg_name in REMOVED_GRID3D_ARGS:
        if arg_name in kwargs:
            raise ValueError(
                "{} is no longer supported; Grid3D now always uses full-volume windows".format(arg_name)
            )


class Grid3DStepDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        split,
        hist_len,
        pred_len,
        use_mmap=False,
        use_coords=True,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.split = str(split)
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.use_mmap = bool(use_mmap)
        self.use_coords = bool(use_coords)

        if self.hist_len <= 0 or self.pred_len <= 0:
            raise ValueError("hist_len and pred_len must be > 0")

        with (self.dataset_dir / "meta.json").open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.grid_shape = tuple(int(size) for size in self.meta["grid_shape"])
        self.channel_names = list(self.meta["channel_names"])

        self.variable = np.load(
            self.dataset_dir / "{}_data.npy".format(self.split),
            mmap_mode="r" if self.use_mmap else None,
            allow_pickle=False,
        )
        self.timestamps = np.load(
            self.dataset_dir / "{}_timestamps.npy".format(self.split),
            mmap_mode="r" if self.use_mmap else None,
            allow_pickle=False,
        )
        self.timestamps = np.array(self.timestamps, dtype=np.float64, copy=True, order="C")

        self.coord = None
        if self.use_coords:
            self.coord = np.load(
                self.dataset_dir / "coord.npy",
                mmap_mode="r" if self.use_mmap else None,
                allow_pickle=False,
            )
            self.coord = np.array(self.coord, dtype=np.float32, copy=True, order="C")

        self.total_windows = int(self.variable.shape[0]) - (self.hist_len + self.pred_len) + 1
        if self.total_windows <= 0:
            raise ValueError("invalid dataset split for sliding window")

    def __len__(self):
        return self.total_windows

    def _to_c_contiguous(self, array: np.ndarray, dtype) -> np.ndarray:
        return np.array(array, dtype=dtype, copy=True, order="C")

    def _load_window(self, start_index: int, stop_index: int) -> np.ndarray:
        return self._to_c_contiguous(self.variable[start_index:stop_index], np.float32)

    def _build_coords(self) -> np.ndarray:
        return self.coord

    def _load_timestamps(self, start_index: int, stop_index: int) -> np.ndarray:
        return self.timestamps[start_index:stop_index]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index < 0 or index >= self.total_windows:
            raise IndexError("sample index {} is out of range".format(index))
        window_id = int(index)

        hist_start = window_id
        hist_stop = hist_start + self.hist_len
        pred_stop = hist_stop + self.pred_len

        item = {
            "inputs": self._load_window(hist_start, hist_stop),
            "targets": self._load_window(hist_stop, pred_stop),
            "inputs_timestamps": self._load_timestamps(hist_start, hist_stop),
            "targets_timestamps": self._load_timestamps(hist_stop, pred_stop),
            "window_id": np.asarray(window_id, dtype=np.int64),
        }
        if self.use_coords:
            item["coords"] = self._build_coords()
        return item


class Grid3DDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        _reject_removed_grid3d_args(kwargs)
        self.config = dict(kwargs)
        self.dataset = str(self.config["dataset"])
        self.num_workers = int(kwargs["num_workers"])
        self.batch_size = int(kwargs["batch_size"])
        self.hist_len = int(kwargs["hist_len"])
        self.pred_len = int(kwargs["pred_len"])
        self.use_mmap = bool(kwargs.get("use_mmap", True))
        self.use_coords = bool(kwargs.get("use_coords", True))
        self.pin_memory = kwargs.get("pin_memory")
        if self.pin_memory is None:
            self.pin_memory = kwargs.get("accelerator", "auto") in {"gpu", "cuda"}
        self.persistent_workers = kwargs.get("persistent_workers")
        if self.persistent_workers is None:
            self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = kwargs.get("prefetch_factor", 2)

        dataset_root = Path(kwargs["data_root"]).expanduser()
        self.dataset_dir = dataset_root / self.dataset
        with (self.dataset_dir / "meta.json").open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)

    def _create_loader(self, dataset, batch_size, shuffle, drop_last):
        loader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=self.pin_memory,
        )
        if self.num_workers > 0:
            loader_args["persistent_workers"] = self.persistent_workers
            loader_args["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**loader_args)

    def _build_split_dataset(self, split_name):
        return Grid3DStepDataset(
            dataset_dir=self.dataset_dir,
            split=split_name,
            hist_len=self.hist_len,
            pred_len=self.pred_len,
            use_mmap=self.use_mmap,
            use_coords=self.use_coords,
        )

    def export_task_hparams(self) -> dict:
        channel_names = list(self.meta["channel_names"])
        exported = {
            "grid_shape": [int(size) for size in self.meta["grid_shape"]],
            "channel_names": channel_names,
            "in_channels": len(channel_names),
            "coord_channels": 3,
            "history_len": self.hist_len,
            "data_is_standardized": resolve_data_is_standardized(self.meta),
        }
        if "grid_spacing_m" in self.meta:
            exported["grid_spacing_m"] = [float(value) for value in self.meta["grid_spacing_m"]]
        return exported

    def train_dataloader(self):
        return self._create_loader(
            dataset=self._build_split_dataset("train"),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return self._create_loader(
            dataset=self._build_split_dataset("val"),
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return self._create_loader(
            dataset=self._build_split_dataset("test"),
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
