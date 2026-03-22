import os
from functools import lru_cache
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


DATA_ARRAY_KEY = "scaled_variable"
TIMESTAMP_ARRAY_KEY = "timestamp"
STAT_KEYS = ("mean", "std")


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
    os.replace(temp_path, target_path)


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
            if key in (DATA_ARRAY_KEY, *STAT_KEYS):
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


@lru_cache(maxsize=None)
def _load_dataset_stats_cached(npz_path, use_mmap=False, cache_npz_as_npy=None):
    if cache_npz_as_npy is None:
        cache_npz_as_npy = use_mmap

    if use_mmap and cache_npz_as_npy and _ensure_npy_cache(npz_path, STAT_KEYS):
        mean = np.load(_cache_path_for_key(npz_path, "mean"), mmap_mode="r")
        std = np.load(_cache_path_for_key(npz_path, "std"), mmap_mode="r")
        return mean, std

    with np.load(npz_path) as data:
        mean = data["mean"].astype(np.float32, copy=False)
        std = data["std"].astype(np.float32, copy=False)
    return mean, std


def load_dataset_stats(npz_path, use_mmap=False, cache_npz_as_npy=None):
    return _load_dataset_stats_cached(str(npz_path), bool(use_mmap), cache_npz_as_npy)


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

        var_x = self.variable[hist_start:hist_end, ...]
        tf_x = self.time_feature[hist_start:hist_end, ...]

        var_y = self.variable[hist_end:pred_end, ...]
        tf_y = self.time_feature[hist_end:pred_end, ...]

        return var_x, tf_x, var_y, tf_y

    def __len__(self):
        return self.total_windows


class DataInterface(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_workers = kwargs['num_workers']
        self.batch_size = kwargs['batch_size']
        self.hist_len = kwargs['hist_len']
        self.pred_len = kwargs['pred_len']
        self.norm_time_feature = kwargs['norm_time_feature']
        self.train_len, self.val_len, self.test_len = kwargs['data_split']
        self.time_feature_cls = kwargs['time_feature_cls']
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

        self.data_path = os.path.join(kwargs['data_root'], "{}.npz".format(kwargs['dataset_name']))
        self.config = kwargs

        self.variable, self.time_feature = self.__read_data__()
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    def __read_data__(self):
        variable, raw_timestamp = load_dataset_arrays(
            self.data_path,
            use_mmap=self.use_mmap,
            cache_npz_as_npy=self.cache_npz_as_npy,
        )
        variable = variable.astype(np.float32, copy=False)
        timestamp = pd.DatetimeIndex(raw_timestamp)

        # time_feature
        if len(self.time_feature_cls) == 0:
            return variable, np.empty((len(variable), 0), dtype=np.float32)

        time_feature = np.empty((len(variable), len(self.time_feature_cls)), dtype=np.float32)
        for feature_idx, tf_cls in enumerate(self.time_feature_cls):
            if tf_cls == "tod":
                tod_size = int((24 * 60) / self.config['freq']) - 1
                tod = (timestamp.hour.to_numpy() * 60 + timestamp.minute.to_numpy()) / self.config['freq']
                if self.norm_time_feature:
                    time_feature[:, feature_idx] = tod / tod_size - 0.5
                else:
                    time_feature[:, feature_idx] = tod
            elif tf_cls == "dow":
                dow_size = 7 - 1
                dow = timestamp.dayofweek.to_numpy()  # 0 ~ 6
                if self.norm_time_feature:
                    time_feature[:, feature_idx] = dow / dow_size - 0.5
                else:
                    time_feature[:, feature_idx] = dow
            elif tf_cls == "dom":
                dom_size = 31 - 1
                dom = timestamp.day.to_numpy() - 1  # 0 ~ 30
                if self.norm_time_feature:
                    time_feature[:, feature_idx] = dom / dom_size - 0.5
                else:
                    time_feature[:, feature_idx] = dom
            elif tf_cls == "doy":
                doy_size = 366 - 1
                doy = timestamp.dayofyear.to_numpy() - 1  # 0 ~ 181
                if self.norm_time_feature:
                    time_feature[:, feature_idx] = doy / doy_size - 0.5
                else:
                    time_feature[:, feature_idx] = doy
            else:
                raise NotImplementedError

        return variable, time_feature

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
        if self._train_loader is None:
            self._train_dataset = GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                self.variable[:self.train_len],
                self.time_feature[:self.train_len],
                precompute_window_index=self.precompute_window_index
            )
            self._train_loader = self._create_loader(
                dataset=self._train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self._val_dataset = GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                self.variable[self.train_len - self.hist_len:self.train_len + self.val_len],
                self.time_feature[self.train_len - self.hist_len:self.train_len + self.val_len],
                precompute_window_index=self.precompute_window_index
            )
            self._val_loader = self._create_loader(
                dataset=self._val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )
        return self._val_loader

    def test_dataloader(self):
        if self._test_loader is None:
            self._test_dataset = GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                self.variable[self.train_len + self.val_len - self.hist_len:],
                self.time_feature[self.train_len + self.val_len - self.hist_len:],
                precompute_window_index=self.precompute_window_index
            )
            self._test_loader = self._create_loader(
                dataset=self._test_dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False
            )
        return self._test_loader
