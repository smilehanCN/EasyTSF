import json
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .base import (
    BaseDataInterface,
    DATA_FILE_NAME,
    GeneralTSFDataset,
    SPLIT_NAMES,
    build_time_feature,
    load_dataset_arrays,
    load_meta_split_lengths,
    load_npy_array,
    require_dataset_file,
)
from .spec import DataSpec


GRID_MASK_FILE_NAME = "grid_mask.npy"
COORD_FILE_NAME = "coord.npy"
GRID3D_SHARDED_STORAGE_FORMAT = "grid3d_split_sharded_npy_v1"
GRID3D_SPLIT_TIMESTAMP_FILE_NAME = "timestamps.npy"
GRID3D_SPLIT_MANIFEST_FILE_NAME = "manifest.json"
GRID3D_SHARD_LEN = 5


@dataclass(frozen=True)
class _ShardFileSpec:
    path: Path
    length: int
    start: int
    stop: int


def _load_json_mapping(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("json file must contain a mapping: {}".format(path))
    return data


def _require_split_dir(dataset_dir, split_name, dataset_name):
    split_dir = Path(dataset_dir) / split_name
    if not split_dir.exists():
        raise FileNotFoundError(
            "3D grid dataset '{}' must include split directory '{}' under '{}'".format(
                dataset_name,
                split_name,
                dataset_dir,
            )
        )
    if not split_dir.is_dir():
        raise NotADirectoryError("expected split directory for '{}' at {}".format(split_name, split_dir))
    return split_dir


class _Grid3DShardReader:
    def __init__(self, split_dir, use_mmap=False, max_open_shards=None):
        self.split_dir = Path(split_dir)
        self.use_mmap = bool(use_mmap)
        self.max_open_shards = int(max_open_shards if max_open_shards is not None else (8 if self.use_mmap else 1))
        self._open_shards = OrderedDict()
        self._expected_item_shape = None
        self._dtype = None

        manifest_path = self.split_dir / GRID3D_SPLIT_MANIFEST_FILE_NAME
        manifest = _load_json_mapping(manifest_path)
        self.split_name = str(manifest.get("split", self.split_dir.name))
        self.shard_len = int(manifest.get("shard_len", 0))
        if self.shard_len != GRID3D_SHARD_LEN:
            raise ValueError(
                "3D grid split '{}' must use shard_len={} but manifest declares {}".format(
                    self.split_name,
                    GRID3D_SHARD_LEN,
                    self.shard_len,
                )
            )

        raw_files = manifest.get("files")
        if not isinstance(raw_files, list) or len(raw_files) == 0:
            raise ValueError("3D grid split '{}' manifest must define a non-empty 'files' list".format(self.split_name))

        num_steps = int(manifest.get("num_steps", -1))
        num_shards = int(manifest.get("num_shards", -1))
        if num_shards != len(raw_files):
            raise ValueError(
                "3D grid split '{}' manifest num_shards {} does not match files length {}".format(
                    self.split_name,
                    num_shards,
                    len(raw_files),
                )
            )

        self.shards = []
        running_start = 0
        for file_idx, entry in enumerate(raw_files):
            if not isinstance(entry, dict):
                raise ValueError(
                    "3D grid split '{}' manifest file entry {} must be a mapping".format(
                        self.split_name,
                        file_idx,
                    )
                )
            rel_path = entry.get("path")
            if not rel_path:
                raise ValueError(
                    "3D grid split '{}' manifest file entry {} is missing 'path'".format(
                        self.split_name,
                        file_idx,
                    )
                )
            shard_path = (self.split_dir / str(rel_path)).resolve()
            try:
                shard_path.relative_to(self.split_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    "3D grid split '{}' shard path escapes split directory: {}".format(
                        self.split_name,
                        shard_path,
                    )
                ) from exc
            if not shard_path.exists():
                raise FileNotFoundError("3D grid split shard file not found: {}".format(shard_path))

            shard_length = int(entry.get("length", 0))
            if shard_length <= 0:
                raise ValueError(
                    "3D grid split '{}' manifest file entry {} must define a positive 'length'".format(
                        self.split_name,
                        file_idx,
                    )
                )
            self.shards.append(
                _ShardFileSpec(
                    path=shard_path,
                    length=shard_length,
                    start=running_start,
                    stop=running_start + shard_length,
                )
            )
            running_start += shard_length

        if num_steps != running_start:
            raise ValueError(
                "3D grid split '{}' manifest num_steps {} does not match shard lengths {}".format(
                    self.split_name,
                    num_steps,
                    running_start,
                )
            )

        self.total_steps = running_start
        self._shard_starts = tuple(spec.start for spec in self.shards)
        self._get_shard_array(0)

    @property
    def item_shape(self):
        return self._expected_item_shape

    @property
    def dtype(self):
        return self._dtype

    def _validate_shard_array(self, shard_index, array):
        spec = self.shards[shard_index]
        if array.ndim != 5:
            raise ValueError(
                "3D grid shard must store arrays as [T, C, X, Y, Z], but '{}' has shape {}".format(
                    spec.path,
                    tuple(array.shape),
                )
            )
        if int(array.shape[0]) != spec.length:
            raise ValueError(
                "3D grid shard '{}' length {} does not match manifest length {}".format(
                    spec.path,
                    int(array.shape[0]),
                    spec.length,
                )
            )
        if array.dtype != np.float32:
            raise ValueError("3D grid shard '{}' must be float32, but found {}".format(spec.path, array.dtype))

        item_shape = tuple(int(size) for size in array.shape[1:])
        if self._expected_item_shape is None:
            self._expected_item_shape = item_shape
            self._dtype = array.dtype
        elif item_shape != self._expected_item_shape:
            raise ValueError(
                "3D grid shard '{}' shape {} does not match expected {}".format(
                    spec.path,
                    item_shape,
                    self._expected_item_shape,
                )
            )

    def _load_shard_array(self, path):
        return np.load(path, mmap_mode="r" if self.use_mmap else None, allow_pickle=False)

    def _get_shard_array(self, shard_index):
        if shard_index in self._open_shards:
            array = self._open_shards.pop(shard_index)
            self._open_shards[shard_index] = array
            return array

        array = self._load_shard_array(self.shards[shard_index].path)
        self._validate_shard_array(shard_index, array)
        self._open_shards[shard_index] = array
        if len(self._open_shards) > self.max_open_shards:
            self._open_shards.popitem(last=False)
        return array

    def read_window(self, start, stop):
        if start < 0 or stop > self.total_steps or start >= stop:
            raise IndexError(
                "invalid 3D grid window [{}, {}) for split '{}' of length {}".format(
                    start,
                    stop,
                    self.split_name,
                    self.total_steps,
                )
            )

        shard_index = bisect_right(self._shard_starts, start) - 1
        current = int(start)
        pieces = []
        while current < stop:
            spec = self.shards[shard_index]
            shard_array = self._get_shard_array(shard_index)
            local_start = current - spec.start
            take_stop = min(stop, spec.stop)
            local_stop = take_stop - spec.start
            pieces.append(shard_array[local_start:local_stop, ...])
            current = take_stop
            shard_index += 1

        if len(pieces) == 1:
            return np.array(pieces[0], copy=True)
        return np.concatenate([np.asarray(piece) for piece in pieces], axis=0)


class ShardedGrid3DWindowDataset(Dataset):
    def __init__(self, hist_len, pred_len, reader, time_feature, precompute_window_index=False):
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.reader = reader
        self.time_feature = np.asarray(time_feature, dtype=np.float32)
        if len(self.time_feature) != self.reader.total_steps:
            raise ValueError(
                "3D grid split '{}' timestamp length {} does not match data length {}".format(
                    self.reader.split_name,
                    len(self.time_feature),
                    self.reader.total_steps,
                )
            )

        self.total_windows = self.reader.total_steps - (self.hist_len + self.pred_len) + 1
        if self.total_windows <= 0:
            raise ValueError(
                "3D grid split '{}' is too short for hist_len {} and pred_len {}".format(
                    self.reader.split_name,
                    self.hist_len,
                    self.pred_len,
                )
            )
        self.window_start_index = None
        if precompute_window_index:
            self.window_start_index = np.arange(self.total_windows, dtype=np.int32)

    def __getitem__(self, index):
        hist_start = int(self.window_start_index[index]) if self.window_start_index is not None else int(index)
        hist_end = hist_start + self.hist_len
        pred_end = hist_end + self.pred_len
        window = self.reader.read_window(hist_start, pred_end)
        return (
            window[:self.hist_len, ...],
            self.time_feature[hist_start:hist_end, ...],
            window[self.hist_len:, ...],
            self.time_feature[hist_end:pred_end, ...],
        )

    def __len__(self):
        return self.total_windows


def _load_optional_side_array(path, use_mmap=False):
    if path is None or not path.exists():
        return None
    return load_npy_array(path, use_mmap=use_mmap, dtype=np.float32)


def load_grid_dataset_arrays(npz_path, grid_mask_path=None, coord_path=None, use_mmap=False, cache_npz_as_npy=None):
    variable, timestamp = load_dataset_arrays(
        npz_path,
        use_mmap=use_mmap,
        cache_npz_as_npy=cache_npz_as_npy,
    )
    grid_mask = _load_optional_side_array(grid_mask_path, use_mmap=use_mmap)
    coord = _load_optional_side_array(coord_path, use_mmap=use_mmap)
    return variable, timestamp, grid_mask, coord


class GridDataInterface(BaseDataInterface):
    def __init__(self, **kwargs):
        self.grid_mask = None
        self.coord = None
        self.channel_num = None
        self.spatial_shape = None
        self.spatial_ndim = None
        self.variable = None
        self.time_feature = None
        self.time_feature_descriptions = ()
        self.storage_format = None
        self.split_datasets = {}
        super().__init__(**kwargs)

    def _setup_dataset(self):
        self.storage_format = self.meta.get("storage_format")
        if self.storage_format is not None:
            if str(self.storage_format) != GRID3D_SHARDED_STORAGE_FORMAT:
                raise ValueError(
                    "unsupported grid storage_format '{}' in {}".format(
                        self.storage_format,
                        self.meta_path,
                    )
                )
            self._setup_sharded_grid3d_dataset()
            return

        self._setup_legacy_npz_grid_dataset()

    def _setup_legacy_npz_grid_dataset(self):
        self.data_path = require_dataset_file(self.dataset_dir, DATA_FILE_NAME, self.dataset_name)
        split_lengths = load_meta_split_lengths(self.meta, self.meta_path)

        variable, raw_timestamp, grid_mask, coord = load_grid_dataset_arrays(
            self.data_path,
            grid_mask_path=self.dataset_dir / GRID_MASK_FILE_NAME,
            coord_path=self.dataset_dir / COORD_FILE_NAME,
            use_mmap=self.use_mmap,
            cache_npz_as_npy=self.cache_npz_as_npy,
        )
        variable = variable.astype(np.float32, copy=False)
        if variable.ndim not in {4, 5}:
            raise ValueError(
                "grid dataset must store scaled_variable as [L, C, H, W] or [L, C, X, Y, Z]: {}".format(self.data_path)
            )
        if variable.ndim == 5:
            raise ValueError(
                "3D grid datasets must use storage_format '{}' with split shard directories under '{}'".format(
                    GRID3D_SHARDED_STORAGE_FORMAT,
                    self.dataset_dir,
                )
            )
        if len(raw_timestamp) != len(variable):
            raise ValueError(
                "timestamp length {} does not match data length {} for {}".format(
                    len(raw_timestamp),
                    len(variable),
                    self.data_path,
                )
            )
        if sum(split_lengths) != int(len(variable)):
            raise ValueError(
                "grid dataset split_lengths {} do not sum to total length {} for {}".format(
                    split_lengths,
                    int(len(variable)),
                    self.meta_path,
                )
            )

        self.channel_num = int(variable.shape[1])
        self.spatial_ndim = int(variable.ndim - 2)
        self.spatial_shape = tuple(int(size) for size in variable.shape[2:])

        if grid_mask is not None:
            grid_mask = grid_mask.astype(np.float32, copy=False)
            if tuple(grid_mask.shape) != self.spatial_shape:
                raise ValueError(
                    "grid_mask shape {} does not match dataset spatial shape {}".format(
                        tuple(grid_mask.shape),
                        self.spatial_shape,
                    )
                )

        if coord is not None:
            coord = coord.astype(np.float32, copy=False)
            expected_coord_shape = (self.spatial_ndim,) + self.spatial_shape
            if tuple(coord.shape) != expected_coord_shape:
                raise ValueError(
                    "coord shape {} does not match expected grid coord shape {}".format(
                        tuple(coord.shape),
                        expected_coord_shape,
                    )
                )

        self.variable = variable
        self.grid_mask = grid_mask
        self.coord = coord
        if bool(self.meta.get("has_graph")):
            raise ValueError("grid dataset meta must not declare has_graph=true: {}".format(self.meta_path))
        self.time_feature = build_time_feature(
            raw_timestamp,
            self.time_feature_cls,
            self.norm_time_feature,
            self.freq,
        )
        self.data_spec = self._build_data_spec()

        self._record_resolved_conf("split_lengths", split_lengths, "grid dataset meta", validate_keys=("data_split", "split_lengths"))
        self._record_resolved_conf("time_feature_dim", int(self.time_feature.shape[-1]), "grid dataset timestamps")
        self._record_resolved_conf("time_feature_descriptions", tuple(self.time_feature_descriptions), "grid dataset layout")
        self._record_resolved_conf("has_graph", False, "grid dataset layout")

    def _setup_sharded_grid3d_dataset(self):
        split_lengths = load_meta_split_lengths(self.meta, self.meta_path)
        grid_mask = _load_optional_side_array(self.dataset_dir / GRID_MASK_FILE_NAME, use_mmap=self.use_mmap)
        coord = _load_optional_side_array(self.dataset_dir / COORD_FILE_NAME, use_mmap=self.use_mmap)

        split_time_feature = {}
        split_datasets = {}
        expected_item_shape = None
        time_feature_dim = None

        for split_name, split_length in zip(SPLIT_NAMES, split_lengths):
            split_dir = _require_split_dir(self.dataset_dir, split_name, self.dataset_name)
            split_reader = _Grid3DShardReader(split_dir, use_mmap=self.use_mmap)
            if split_reader.total_steps != int(split_length):
                raise ValueError(
                    "3D grid split '{}' manifest length {} does not match meta split_lengths {} for {}".format(
                        split_name,
                        split_reader.total_steps,
                        split_lengths,
                        self.meta_path,
                    )
                )

            if expected_item_shape is None:
                expected_item_shape = tuple(split_reader.item_shape)
            elif tuple(split_reader.item_shape) != expected_item_shape:
                raise ValueError(
                    "3D grid split '{}' shape {} does not match expected {}".format(
                        split_name,
                        tuple(split_reader.item_shape),
                        expected_item_shape,
                    )
                )

            timestamp_path = split_dir / GRID3D_SPLIT_TIMESTAMP_FILE_NAME
            if not timestamp_path.exists():
                raise FileNotFoundError("3D grid split '{}' must include '{}'".format(split_dir, GRID3D_SPLIT_TIMESTAMP_FILE_NAME))
            raw_timestamp = load_npy_array(timestamp_path, use_mmap=self.use_mmap)
            if len(raw_timestamp) != int(split_length):
                raise ValueError(
                    "3D grid split '{}' timestamp length {} does not match split length {}".format(
                        split_name,
                        len(raw_timestamp),
                        split_length,
                    )
                )

            split_time_feature[split_name] = build_time_feature(
                raw_timestamp,
                self.time_feature_cls,
                self.norm_time_feature,
                self.freq,
            )
            if time_feature_dim is None:
                time_feature_dim = int(split_time_feature[split_name].shape[-1])
            elif int(split_time_feature[split_name].shape[-1]) != time_feature_dim:
                raise ValueError(
                    "3D grid split '{}' time feature width {} does not match expected {}".format(
                        split_name,
                        int(split_time_feature[split_name].shape[-1]),
                        time_feature_dim,
                    )
                )

            split_datasets[split_name] = ShardedGrid3DWindowDataset(
                self.hist_len,
                self.pred_len,
                split_reader,
                split_time_feature[split_name],
                precompute_window_index=self.precompute_window_index,
            )

        if expected_item_shape is None or len(expected_item_shape) != 4:
            raise ValueError("3D grid shard reader failed to resolve shape under {}".format(self.dataset_dir))

        self.channel_num = int(expected_item_shape[0])
        self.spatial_shape = tuple(int(size) for size in expected_item_shape[1:])
        self.spatial_ndim = 3

        if grid_mask is not None:
            grid_mask = grid_mask.astype(np.float32, copy=False)
            if tuple(grid_mask.shape) != self.spatial_shape:
                raise ValueError(
                    "grid_mask shape {} does not match dataset spatial shape {}".format(
                        tuple(grid_mask.shape),
                        self.spatial_shape,
                    )
                )

        if coord is not None:
            coord = coord.astype(np.float32, copy=False)
            expected_coord_shape = (self.spatial_ndim,) + self.spatial_shape
            if tuple(coord.shape) != expected_coord_shape:
                raise ValueError(
                    "coord shape {} does not match expected grid coord shape {}".format(
                        tuple(coord.shape),
                        expected_coord_shape,
                    )
                )

        self.grid_mask = grid_mask
        self.coord = coord
        self.split_datasets = split_datasets
        self.time_feature = split_time_feature
        if bool(self.meta.get("has_graph")):
            raise ValueError("grid dataset meta must not declare has_graph=true: {}".format(self.meta_path))
        self.data_spec = self._build_data_spec()

        self._record_resolved_conf(
            "split_lengths",
            [int(item) for item in split_lengths],
            "grid dataset meta",
            validate_keys=("data_split", "split_lengths"),
        )
        self._record_resolved_conf("time_feature_dim", int(time_feature_dim or 0), "grid dataset timestamps")
        self._record_resolved_conf("time_feature_descriptions", tuple(self.time_feature_descriptions), "grid dataset layout")
        self._record_resolved_conf("has_graph", False, "grid dataset layout")

    def _build_data_spec(self):
        if isinstance(self.time_feature, dict):
            first_time_feature = next(iter(self.time_feature.values()), None)
            time_feature_dim = int(first_time_feature.shape[-1]) if first_time_feature is not None and first_time_feature.ndim == 2 else 0
        else:
            time_feature_dim = int(self.time_feature.shape[-1]) if self.time_feature.ndim == 2 else 0
        return DataSpec(
            layout_kind="grid",
            spatial_ndim=int(self.spatial_ndim),
            spatial_shape=tuple(self.spatial_shape),
            channel_num=int(self.channel_num),
            has_graph=False,
            has_grid_mask=self.grid_mask is not None,
            has_coord=self.coord is not None,
            time_feature_dim=time_feature_dim,
            time_feature_descriptions=tuple(self.time_feature_descriptions),
        )

    def _split_bounds(self):
        split_lengths = self.get_resolved_conf_updates()["split_lengths"]
        train_len, val_len, test_len = [int(item) for item in split_lengths]
        return train_len, val_len, test_len

    def train_dataloader(self):
        if self._train_loader is None:
            if self.storage_format == GRID3D_SHARDED_STORAGE_FORMAT:
                self._train_loader = self._create_loader(
                    dataset=self.split_datasets["train"],
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                return self._train_loader
            train_len, _, _ = self._split_bounds()
            train_dataset = GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                self.variable[:train_len],
                self.time_feature[:train_len],
                precompute_window_index=self.precompute_window_index,
            )
            self._train_loader = self._create_loader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            if self.storage_format == GRID3D_SHARDED_STORAGE_FORMAT:
                self._val_loader = self._create_loader(
                    dataset=self.split_datasets["val"],
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                )
                return self._val_loader
            train_len, val_len, _ = self._split_bounds()
            val_dataset = GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                self.variable[train_len - self.hist_len:train_len + val_len],
                self.time_feature[train_len - self.hist_len:train_len + val_len],
                precompute_window_index=self.precompute_window_index,
            )
            self._val_loader = self._create_loader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )
        return self._val_loader

    def test_dataloader(self):
        if self._test_loader is None:
            if self.storage_format == GRID3D_SHARDED_STORAGE_FORMAT:
                self._test_loader = self._create_loader(
                    dataset=self.split_datasets["test"],
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )
                return self._test_loader
            train_len, val_len, _ = self._split_bounds()
            test_dataset = GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                self.variable[train_len + val_len - self.hist_len:],
                self.time_feature[train_len + val_len - self.hist_len:],
                precompute_window_index=self.precompute_window_index,
            )
            self._test_loader = self._create_loader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )
        return self._test_loader
