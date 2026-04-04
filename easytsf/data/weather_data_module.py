import json
from collections import OrderedDict
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset


class WeatherShardDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        split,
        hist_len,
        pred_len,
        input_channel_names=None,
        target_channel_names=None,
        use_mmap=False,
        shard_cache_size=2,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.split = str(split)
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.use_mmap = bool(use_mmap)
        self.shard_cache_size = max(int(shard_cache_size), 0)

        if self.hist_len <= 0 or self.pred_len <= 0:
            raise ValueError("hist_len and pred_len must be > 0")

        with (self.dataset_dir / "meta.json").open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        with (self.dataset_dir / "channels.json").open("r", encoding="utf-8") as handle:
            self.channels = json.load(handle)
        with (self.dataset_dir / "static_channels.json").open("r", encoding="utf-8") as handle:
            self.static_channels = json.load(handle)

        self.split_dir = self.dataset_dir / self.split
        with (self.split_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        self.timestamps = np.load(
            self.split_dir / "timestamps.npy",
            mmap_mode="r" if self.use_mmap else None,
            allow_pickle=False,
        )
        self.timestamp_ns = np.asarray(self.timestamps, dtype="datetime64[ns]").view(np.int64)
        self.static = np.load(
            self.dataset_dir / "static.npy",
            mmap_mode="r" if self.use_mmap else None,
            allow_pickle=False,
        )

        default_input_channel_names = list(self.meta["input_channels"])
        default_target_channel_names = list(self.meta["target_channels"])
        if input_channel_names is None:
            input_channel_names = default_input_channel_names
        if target_channel_names is None:
            target_channel_names = default_target_channel_names

        self.input_channel_names = [str(item) for item in input_channel_names]
        self.target_channel_names = [str(item) for item in target_channel_names]
        if not set(self.target_channel_names).issubset(set(self.input_channel_names)):
            raise ValueError("target_channel_names must be a subset of input_channel_names")

        name_to_index = {channel["name"]: channel["index"] for channel in self.channels}
        self.input_channel_indices = []
        for name in self.input_channel_names:
            try:
                self.input_channel_indices.append(name_to_index[name])
            except KeyError as exc:
                raise ValueError(
                    "unknown channel '{}'; available channels are {}".format(
                        name,
                        sorted(name_to_index),
                    )
                ) from exc

        self.target_channel_indices = []
        for name in self.target_channel_names:
            try:
                self.target_channel_indices.append(name_to_index[name])
            except KeyError as exc:
                raise ValueError(
                    "unknown channel '{}'; available channels are {}".format(
                        name,
                        sorted(name_to_index),
                    )
                ) from exc

        self.files = list(self.manifest["files"])
        self.file_stops = np.asarray([int(item["stop"]) for item in self.files], dtype=np.int64)
        self.total_steps = int(self.manifest["num_steps"])
        self.total_windows = self.total_steps - (self.hist_len + self.pred_len) + 1
        if self.total_windows <= 0:
            raise ValueError("invalid weather split for sliding window")

        self._shard_cache = OrderedDict()

    def __len__(self):
        return self.total_windows

    def _load_shard(self, shard_idx):
        if shard_idx in self._shard_cache:
            shard = self._shard_cache.pop(shard_idx)
            self._shard_cache[shard_idx] = shard
            return shard

        shard_path = self.split_dir / self.files[shard_idx]["path"]
        shard = np.load(shard_path, mmap_mode="r" if self.use_mmap else None, allow_pickle=False)
        if self.shard_cache_size > 0:
            self._shard_cache[shard_idx] = shard
            while len(self._shard_cache) > self.shard_cache_size:
                self._shard_cache.popitem(last=False)
        return shard

    def _slice_steps(self, start_idx, stop_idx):
        pieces = []
        cursor = int(start_idx)
        while cursor < int(stop_idx):
            shard_idx = int(np.searchsorted(self.file_stops, cursor, side="right"))
            shard_meta = self.files[shard_idx]
            shard = self._load_shard(shard_idx)
            shard_local_start = cursor - int(shard_meta["start"])
            shard_local_stop = min(int(shard_meta["length"]), shard_local_start + (int(stop_idx) - cursor))
            pieces.append(np.asarray(shard[shard_local_start:shard_local_stop]))
            cursor += shard_local_stop - shard_local_start
        return np.concatenate(pieces, axis=0)

    def __getitem__(self, index):
        if index < 0 or index >= self.total_windows:
            raise IndexError("weather sample index {} is out of range for split '{}'".format(index, self.split))

        hist_start = int(index)
        hist_stop = hist_start + self.hist_len
        pred_stop = hist_stop + self.pred_len

        input_steps = self._slice_steps(hist_start, hist_stop)
        target_steps = self._slice_steps(hist_stop, pred_stop)

        item = {
            "inputs": np.ascontiguousarray(input_steps[:, self.input_channel_indices, :, :], dtype=np.float32),
            "targets": np.ascontiguousarray(target_steps[:, self.target_channel_indices, :, :], dtype=np.float32),
            "inputs_timestamps": np.ascontiguousarray(self.timestamp_ns[hist_start:hist_stop], dtype=np.int64),
            "targets_timestamps": np.ascontiguousarray(self.timestamp_ns[hist_stop:pred_stop], dtype=np.int64),
        }
        if self.static.shape[0] > 0:
            item["static_inputs"] = np.ascontiguousarray(self.static, dtype=np.float32)
        return item


class WeatherDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = dict(kwargs)
        self.dataset = str(self.config["dataset"])
        self.num_workers = int(kwargs["num_workers"])
        self.batch_size = int(kwargs["batch_size"])
        self.hist_len = int(kwargs["hist_len"])
        self.pred_len = int(kwargs["pred_len"])
        self.use_mmap = bool(kwargs.get("use_mmap", True))
        self.shard_cache_size = int(kwargs.get("shard_cache_size", 2))
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
        with (self.dataset_dir / "channels.json").open("r", encoding="utf-8") as handle:
            self.channels = json.load(handle)
        with (self.dataset_dir / "static_channels.json").open("r", encoding="utf-8") as handle:
            self.static_channels = json.load(handle)

        self.latitude = np.load(
            self.dataset_dir / "latitude.npy",
            mmap_mode="r" if self.use_mmap else None,
            allow_pickle=False,
        )
        self.longitude = np.load(
            self.dataset_dir / "longitude.npy",
            mmap_mode="r" if self.use_mmap else None,
            allow_pickle=False,
        )
        self.static = np.load(
            self.dataset_dir / "static.npy",
            mmap_mode="r" if self.use_mmap else None,
            allow_pickle=False,
        )
        self.input_channel_names = list(kwargs.get("input_channel_names") or self.meta["input_channels"])
        self.target_channel_names = list(kwargs.get("target_channel_names") or self.meta["target_channels"])
        self.static_channel_names = list(self.meta.get("static_channels") or [item["name"] for item in self.static_channels])

        self.split_climatology = {}
        for split_name in ("train", "val", "test"):
            climatology_path = self.dataset_dir / split_name / "climatology.npz"
            climatology_npz = np.load(climatology_path)
            self.split_climatology[split_name] = np.concatenate(
                [np.asarray(climatology_npz[name], dtype=np.float32) for name in self.target_channel_names],
                axis=0,
            )

    def _build_split_dataset(self, split_name):
        return WeatherShardDataset(
            dataset_dir=self.dataset_dir,
            split=split_name,
            hist_len=self.hist_len,
            pred_len=self.pred_len,
            input_channel_names=self.input_channel_names,
            target_channel_names=self.target_channel_names,
            use_mmap=self.use_mmap,
            shard_cache_size=self.shard_cache_size,
        )

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
