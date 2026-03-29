import json
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_dataset_meta(dataset_dir):
    meta_path = Path(dataset_dir) / "meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        return meta_path, json.load(handle)


def load_npy_array(path, use_mmap=False, dtype=None):
    array = np.load(path, mmap_mode="r" if use_mmap else None, allow_pickle=False)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _restore_basic_ts_timestamp_column(column, description, freq):
    desc = str(description).strip().lower()
    steps_per_day = int((24 * 60) / int(freq))
    feature_sizes = {
        "time of day": steps_per_day,
        "day of week": 7,
        "day of month": 31,
        "day of year": 366,
    }
    size = feature_sizes[desc]
    column = np.asarray(column, dtype=np.float32)
    min_value = float(np.nanmin(column))
    max_value = float(np.nanmax(column))

    if min_value >= -1e-6 and max_value <= 1.0 + 1e-6:
        restored = np.floor(column * size + 1e-6)
    elif min_value >= -0.5 - 1e-6 and max_value <= 0.5 + 1e-6:
        restored = np.floor((column + 0.5) * size + 1e-6)
    else:
        restored = np.rint(column)

    return np.clip(restored, 0, size - 1).astype(np.float32, copy=False)


def restore_basic_ts_timestamps(raw_timestamps, descriptions, freq):
    timestamps = np.asarray(raw_timestamps, dtype=np.float32)
    if timestamps.ndim == 1:
        timestamps = timestamps[:, None]
    columns = [
        _restore_basic_ts_timestamp_column(column, description, freq)
        for description, column in zip(descriptions, timestamps.T, strict=True)
    ]
    return np.stack(columns, axis=-1).astype(np.float32, copy=False)


class BasicTSSequenceDataset(Dataset):
    def __init__(self, hist_len, pred_len, variable, timestamps, precompute_window_index=False):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.variable = variable
        self.timestamps = timestamps
        self.total_windows = len(self.variable) - (self.hist_len + self.pred_len) + 1
        if self.total_windows <= 0:
            raise ValueError("invalid dataset split for sliding window")
        self.window_start_index = None
        if precompute_window_index:
            self.window_start_index = np.arange(self.total_windows, dtype=np.int32)

    def __getitem__(self, index):
        hist_start = int(self.window_start_index[index]) if self.window_start_index is not None else index
        hist_end = hist_start + self.hist_len
        pred_end = hist_end + self.pred_len

        item = {
            "inputs": self.variable[hist_start:hist_end, ...],
            "inputs_timestamps": self.timestamps[hist_start:hist_end, ...],
            "targets": self.variable[hist_end:pred_end, ...],
            "targets_timestamps": self.timestamps[hist_end:pred_end, ...],
        }
        return item

    def __len__(self):
        return self.total_windows


class MTSDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self.dataset_name = str(kwargs["dataset_name"])
        self.num_workers = int(kwargs["num_workers"])
        self.batch_size = int(kwargs["batch_size"])
        self.hist_len = int(kwargs["hist_len"])
        self.pred_len = int(kwargs["pred_len"])
        self.pin_memory = kwargs.get("pin_memory")
        if self.pin_memory is None:
            self.pin_memory = kwargs.get("accelerator", "auto") in {"gpu", "cuda"}
        self.persistent_workers = kwargs.get("persistent_workers")
        if self.persistent_workers is None:
            self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = kwargs.get("prefetch_factor", 2)
        self.use_mmap = bool(kwargs.get("use_mmap", False))
        self.precompute_window_index = bool(kwargs.get("precompute_window_index", False))

        dataset_root = Path(kwargs["data_root"]).expanduser()
        self.dataset_dir = dataset_root / self.dataset_name
        self.meta_path, self.meta = load_dataset_meta(self.dataset_dir)
        self._setup_dataset()

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

    def _setup_dataset(self):
        self.split_variable = {}
        self.split_time_feature = {}

        for split_name in ("train", "val", "test"):
            data_path = self.dataset_dir / "{}_data.npy".format(split_name)
            variable = load_npy_array(data_path, use_mmap=self.use_mmap, dtype=np.float32)
            self.split_variable[split_name] = variable

        timestamp_descriptions = self.meta["timestamps_description"]

        configured_time_feature_descriptions = self.config.get("time_feature_descriptions")
        if configured_time_feature_descriptions is not None:
            if tuple(timestamp_descriptions) != tuple(configured_time_feature_descriptions):
                raise ValueError(
                    "dataset meta timestamps_description {} does not match config time_feature_descriptions {}".format(
                        list(timestamp_descriptions),
                        list(configured_time_feature_descriptions),
                    )
                )
        freq = int(self.meta["frequency (minutes)"])

        for split_name in ("train", "val", "test"):
            timestamp_path = self.dataset_dir / "{}_timestamps.npy".format(split_name)
            raw_timestamps = load_npy_array(timestamp_path, use_mmap=self.use_mmap, dtype=np.float32)
            time_feature = restore_basic_ts_timestamps(raw_timestamps, timestamp_descriptions, freq)
            self.split_time_feature[split_name] = time_feature

    def _build_split_dataset(self, split_name):
        return BasicTSSequenceDataset(
            hist_len=self.hist_len,
            pred_len=self.pred_len,
            variable=self.split_variable[split_name],
            timestamps=self.split_time_feature[split_name],
            precompute_window_index=self.precompute_window_index,
        )

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
            batch_size=self.batch_size,
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
