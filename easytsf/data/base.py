import json
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset


DATA_ARRAY_KEY = "scaled_variable"
TIMESTAMP_ARRAY_KEY = "timestamp"
DATA_FILE_NAME = "data.npz"
META_FILE_NAME = "meta.json"
GRAPH_FILE_NAME = "adj_mx.pkl"
SPLIT_NAMES = ("train", "val", "test")


def resolve_dataset_dir(data_root, dataset_name):
    dataset_root = Path(data_root).expanduser()
    dataset_dir = dataset_root / str(dataset_name)
    legacy_path = dataset_root / "{}.npz".format(dataset_name)

    if dataset_dir.exists():
        if not dataset_dir.is_dir():
            raise NotADirectoryError("dataset '{}' path is not a directory: {}".format(dataset_name, dataset_dir))
        return dataset_dir

    if legacy_path.exists():
        raise FileNotFoundError(
            "legacy flat dataset layout is no longer supported for '{}': found {}; expected directory layout at {}".format(
                dataset_name,
                legacy_path,
                dataset_dir,
            )
        )

    raise FileNotFoundError(
        "dataset '{}' must be stored under directory '{}'".format(
            dataset_name,
            dataset_dir,
        )
    )


def require_dataset_file(dataset_dir, filename, dataset_name):
    path = Path(dataset_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            "dataset '{}' must include '{}' under directory '{}'".format(
                dataset_name,
                filename,
                dataset_dir,
            )
        )
    return path


def load_dataset_meta(dataset_dir, dataset_name):
    meta_path = require_dataset_file(dataset_dir, META_FILE_NAME, dataset_name)
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, Mapping):
        raise ValueError("dataset meta must be a mapping: {}".format(meta_path))

    meta_name = meta.get("name")
    if not meta_name:
        raise ValueError("dataset meta must define 'name': {}".format(meta_path))
    if str(meta_name) != str(dataset_name):
        raise ValueError(
            "dataset meta name '{}' does not match dataset directory/config '{}': {}".format(
                meta_name,
                dataset_name,
                meta_path,
            )
        )
    return meta_path, dict(meta)


def _coerce_int_list(value, name, expected_length=None):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("{} must be a sequence of integers".format(name))
    items = [int(item) for item in value]
    if expected_length is not None and len(items) != expected_length:
        raise ValueError("{} must contain exactly {} items".format(name, expected_length))
    if any(item <= 0 for item in items):
        raise ValueError("{} must contain positive integers".format(name))
    return items


def load_dataset_frequency(meta, meta_path):
    for key in ("frequency (minutes)", "frequency_minutes", "freq"):
        if key in meta and meta[key] is not None:
            freq = int(meta[key])
            if freq <= 0:
                raise ValueError("dataset frequency must be positive: {} ({})".format(freq, meta_path))
            return freq
    raise ValueError("dataset meta must define frequency in minutes: {}".format(meta_path))


def load_meta_split_lengths(meta, meta_path):
    raw_value = meta.get("split_lengths")
    if raw_value is None:
        raise ValueError("dataset meta must define 'split_lengths': {}".format(meta_path))
    return _coerce_int_list(raw_value, "split_lengths", expected_length=3)


def load_timestamp_descriptions(meta):
    raw_value = meta.get("timestamps_description")
    if raw_value is None:
        return ()
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise ValueError("timestamps_description must be a sequence of strings")
    return tuple(str(item) for item in raw_value)


def _cache_dir_for_npz(npz_path):
    npz_path = Path(npz_path)
    return npz_path.parent / ".easytsf_cache" / npz_path.stem


def _cache_path_for_key(npz_path, key):
    return _cache_dir_for_npz(npz_path) / "{}.npy".format(key)


def _atomic_save_array(target_path, array):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name("{}.{}.tmp.npy".format(target_path.stem, os.getpid()))
    np.save(temp_path, array)
    temp_path.replace(target_path)


def _cache_needs_refresh(npz_path, keys):
    npz_path = Path(npz_path)
    source_mtime = npz_path.stat().st_mtime
    for key in keys:
        cache_path = _cache_path_for_key(npz_path, key)
        if not cache_path.exists() or cache_path.stat().st_mtime < source_mtime:
            return True
    return False


def _materialize_npy_cache(npz_path, keys):
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        for key in keys:
            if key not in data:
                continue
            array = data[key]
            if key == DATA_ARRAY_KEY:
                array = array.astype(np.float32, copy=False)
            _atomic_save_array(_cache_path_for_key(npz_path, key), array)


def _ensure_npy_cache(npz_path, keys):
    try:
        if _cache_needs_refresh(npz_path, keys):
            _materialize_npy_cache(npz_path, keys)
        return True
    except OSError:
        return False


def load_dataset_arrays(npz_path, use_mmap=False, cache_npz_as_npy=None):
    if cache_npz_as_npy is None:
        cache_npz_as_npy = use_mmap

    required_keys = (DATA_ARRAY_KEY, TIMESTAMP_ARRAY_KEY)
    if use_mmap and cache_npz_as_npy and _ensure_npy_cache(npz_path, required_keys):
        variable = np.load(_cache_path_for_key(npz_path, DATA_ARRAY_KEY), mmap_mode="r")
        timestamp = np.load(_cache_path_for_key(npz_path, TIMESTAMP_ARRAY_KEY), mmap_mode="r")
        return variable, timestamp

    with np.load(npz_path) as data:
        variable = data[DATA_ARRAY_KEY].astype(np.float32, copy=False)
        timestamp = data[TIMESTAMP_ARRAY_KEY]
    return variable, timestamp


def load_npy_array(path, use_mmap=False, dtype=None):
    array = np.load(path, mmap_mode="r" if use_mmap else None, allow_pickle=False)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _load_pickle(path):
    try:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
    except UnicodeDecodeError:
        with Path(path).open("rb") as handle:
            return pickle.load(handle, encoding="latin1")


def load_graph_array(graph_path):
    graph_path = Path(graph_path)
    suffix = graph_path.suffix.lower()
    if suffix not in {".pkl", ".pickle"}:
        raise ValueError("graph file must be a BasicTS adjacency pickle (.pkl): {}".format(graph_path))

    graph = _load_pickle(graph_path)
    if isinstance(graph, (tuple, list)) and len(graph) >= 3:
        graph = graph[2]
    return np.asarray(graph, dtype=np.float32)


def build_time_feature(raw_timestamp, time_feature_cls, norm_time_feature, freq):
    timestamp = np.asarray(raw_timestamp).astype("datetime64[m]")
    if len(time_feature_cls) == 0:
        return np.empty((len(timestamp), 0), dtype=np.float32)

    day_values = timestamp.astype("datetime64[D]")
    day_indices = day_values.astype(np.int64)
    minute_values = (timestamp - day_values).astype("timedelta64[m]").astype(np.int64)
    month_start = timestamp.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    year_start = timestamp.astype("datetime64[Y]").astype("datetime64[D]").astype(np.int64)

    time_feature = np.empty((len(timestamp), len(time_feature_cls)), dtype=np.float32)
    for feature_idx, tf_cls in enumerate(time_feature_cls):
        if tf_cls == "tod":
            tod_size = int((24 * 60) / freq) - 1
            tod = minute_values / int(freq)
            time_feature[:, feature_idx] = tod / tod_size - 0.5 if norm_time_feature else tod
        elif tf_cls == "dow":
            dow_size = 7 - 1
            dow = (day_indices + 3) % 7
            time_feature[:, feature_idx] = dow / dow_size - 0.5 if norm_time_feature else dow
        elif tf_cls == "dom":
            dom_size = 31 - 1
            dom = day_indices - month_start
            time_feature[:, feature_idx] = dom / dom_size - 0.5 if norm_time_feature else dom
        elif tf_cls == "doy":
            doy_size = 366 - 1
            doy = day_indices - year_start
            time_feature[:, feature_idx] = doy / doy_size - 0.5 if norm_time_feature else doy
        else:
            raise NotImplementedError("unsupported time feature: {}".format(tf_cls))
    return time_feature


def _restore_basic_ts_timestamp_column(column, description, freq):
    desc = str(description).strip().lower()
    steps_per_day = int((24 * 60) / int(freq))
    feature_sizes = {
        "time of day": steps_per_day,
        "day of week": 7,
        "day of month": 31,
        "day of year": 366,
    }
    if desc not in feature_sizes:
        raise NotImplementedError("unsupported BasicTS timestamp description: {}".format(description))

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
    if timestamps.ndim != 2:
        raise ValueError("BasicTS timestamps must be a 2D array, but received shape {}".format(tuple(timestamps.shape)))
    if len(descriptions) != timestamps.shape[1]:
        raise ValueError(
            "timestamps_description length {} does not match timestamp width {}".format(
                len(descriptions),
                int(timestamps.shape[1]),
            )
        )

    restored = np.empty_like(timestamps, dtype=np.float32)
    for feature_idx, description in enumerate(descriptions):
        restored[:, feature_idx] = _restore_basic_ts_timestamp_column(
            timestamps[:, feature_idx],
            description,
            freq,
        )
    return restored


def _ensure_writable_array(array):
    if hasattr(array, "flags") and not array.flags.writeable:
        return np.array(array, copy=True)
    return array


class GeneralTSFDataset(Dataset):
    def __init__(self, hist_len, pred_len, variable, time_feature, precompute_window_index=False):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.variable = variable
        self.time_feature = time_feature
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

        var_x = _ensure_writable_array(self.variable[hist_start:hist_end, ...])
        tf_x = _ensure_writable_array(self.time_feature[hist_start:hist_end, ...])
        var_y = _ensure_writable_array(self.variable[hist_end:pred_end, ...])
        tf_y = _ensure_writable_array(self.time_feature[hist_end:pred_end, ...])
        return var_x, tf_x, var_y, tf_y

    def __len__(self):
        return self.total_windows


class BasicTSSequenceDataset(Dataset):
    def __init__(self, hist_len, pred_len, variable, timestamps=None, precompute_window_index=False):
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
            "inputs": _ensure_writable_array(self.variable[hist_start:hist_end, ...]),
            "targets": _ensure_writable_array(self.variable[hist_end:pred_end, ...]),
        }
        if self.timestamps is not None and self.timestamps.shape[1] > 0:
            item["inputs_timestamps"] = _ensure_writable_array(self.timestamps[hist_start:hist_end, ...])
            item["targets_timestamps"] = _ensure_writable_array(self.timestamps[hist_end:pred_end, ...])
        return item

    def __len__(self):
        return self.total_windows


def _normalize_compare_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_normalize_compare_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_compare_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class BaseDataInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self.dataset_name = str(kwargs["dataset_name"])
        self.num_workers = int(kwargs["num_workers"])
        self.batch_size = int(kwargs["batch_size"])
        self.hist_len = int(kwargs["hist_len"])
        self.pred_len = int(kwargs["pred_len"])
        self.time_feature_cls = tuple(kwargs.get("time_feature_cls", []))
        self.norm_time_feature = bool(kwargs.get("norm_time_feature", False))
        self.pin_memory = kwargs.get("pin_memory")
        if self.pin_memory is None:
            self.pin_memory = kwargs.get("accelerator", "auto") in {"gpu", "cuda"}
        self.persistent_workers = kwargs.get("persistent_workers")
        if self.persistent_workers is None:
            self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = kwargs.get("prefetch_factor", 2)
        self.use_mmap = kwargs.get("use_mmap", False)
        self.cache_npz_as_npy = kwargs.get("cache_npz_as_npy")
        self.precompute_window_index = kwargs.get("precompute_window_index", False)

        self.dataset_dir = resolve_dataset_dir(kwargs["data_root"], self.dataset_name)
        self.meta_path, self.meta = load_dataset_meta(self.dataset_dir, self.dataset_name)
        self.freq = load_dataset_frequency(self.meta, self.meta_path)
        self.graph_path = self._resolve_graph_path(kwargs.get("graph_path"))

        self.data_spec = None
        self._resolved_conf_updates = {}
        self._record_resolved_conf("freq", self.freq, "dataset meta")
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

        self._setup_dataset()

    def _resolve_graph_path(self, graph_path):
        if graph_path:
            resolved_path = Path(graph_path).expanduser()
            if not resolved_path.is_absolute():
                resolved_path = self.dataset_dir / resolved_path
            if not resolved_path.exists():
                raise FileNotFoundError("graph file not found: {}".format(resolved_path))
            return resolved_path

        default_graph_path = self.dataset_dir / GRAPH_FILE_NAME
        if default_graph_path.exists():
            return default_graph_path

        if bool(self.meta.get("has_graph")):
            raise FileNotFoundError(
                "dataset '{}' declares has_graph=true but '{}' is missing under '{}'".format(
                    self.dataset_name,
                    GRAPH_FILE_NAME,
                    self.dataset_dir,
                )
            )
        return None

    def _validate_config_match(self, config_key, resolved_value, source_name):
        if config_key not in self.config or self.config[config_key] is None:
            return
        configured_value = self.config[config_key]
        if _normalize_compare_value(configured_value) != _normalize_compare_value(resolved_value):
            raise ValueError(
                "config key '{}'={} conflicts with {} {}".format(
                    config_key,
                    configured_value,
                    source_name,
                    resolved_value,
                )
            )

    def _record_resolved_conf(self, conf_key, resolved_value, source_name, validate_keys=None):
        validate_keys = tuple(validate_keys or (conf_key,))
        for validate_key in validate_keys:
            self._validate_config_match(validate_key, resolved_value, source_name)
        self._resolved_conf_updates[conf_key] = resolved_value

    def get_resolved_conf_updates(self):
        return dict(self._resolved_conf_updates)

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
        raise NotImplementedError
