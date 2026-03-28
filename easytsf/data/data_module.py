import numpy as np

from easytsf.scaler import fit_zscore_stats

from .base import (
    BaseDataInterface,
    BasicTSSequenceDataset,
    SPLIT_NAMES,
    load_graph_array,
    load_npy_array,
    load_timestamp_descriptions,
    require_dataset_file,
    restore_basic_ts_timestamps,
)
from .spec import DataSpec


def _split_data_file_name(split_name):
    return "{}_data.npy".format(split_name)


def _split_timestamp_file_name(split_name):
    return "{}_timestamps.npy".format(split_name)


def _serialize_stat_value(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class DataInterface(BaseDataInterface):
    def _load_regular_settings(self):
        regular_settings = self.meta.get("regular_settings")
        if not isinstance(regular_settings, dict):
            raise ValueError("sequence dataset meta must define regular_settings as a mapping: {}".format(self.meta_path))

        required_keys = ("norm_each_channel", "rescale", "null_val")
        missing_keys = [key for key in required_keys if key not in regular_settings]
        if missing_keys:
            raise ValueError(
                "sequence dataset regular_settings must define {} for {}".format(
                    missing_keys,
                    self.meta_path,
                )
            )

        self.norm_each_channel = bool(regular_settings["norm_each_channel"])
        self.rescale = bool(regular_settings["rescale"])
        self.null_val = regular_settings["null_val"]

    def _build_scaler_stats(self):
        train_variable = self.split_variable["train"]
        mean, std = fit_zscore_stats(
            train_variable,
            null_val=self.null_val,
            norm_each_channel=self.norm_each_channel,
        )
        self.scaler_stats = {
            "mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
        }

    def _setup_dataset(self):
        self.split_variable = {}
        self.split_time_feature = {}
        self.time_feature_descriptions = ()
        self.scaler_stats = None
        self.norm_each_channel = None
        self.rescale = False
        self.null_val = None
        split_lengths = []
        timestamp_presence = []
        timestamp_descriptions = load_timestamp_descriptions(self.meta)

        expected_var_num = None
        expected_timestamp_dim = None
        meta_split_lengths = self.meta.get("split_lengths")
        if meta_split_lengths is not None:
            meta_split_lengths = [int(item) for item in meta_split_lengths]
            if len(meta_split_lengths) != len(SPLIT_NAMES):
                raise ValueError("dataset meta split_lengths must contain three items: {}".format(self.meta_path))

        for split_idx, split_name in enumerate(SPLIT_NAMES):
            data_path = require_dataset_file(self.dataset_dir, _split_data_file_name(split_name), self.dataset_name)
            variable = load_npy_array(data_path, use_mmap=self.use_mmap, dtype=np.float32)
            if variable.ndim != 2:
                if variable.ndim == 1:
                    raise ValueError(
                        "mtsf/stf dataset '{}' must store {} as [L, N]; for univariate forecasting, store it as [L, 1]".format(
                            data_path,
                            _split_data_file_name(split_name),
                        )
                    )
                raise ValueError(
                    "mtsf/stf dataset '{}' must store {} as [L, N], but received shape {}".format(
                        data_path,
                        _split_data_file_name(split_name),
                        tuple(variable.shape),
                    )
                )

            split_length = int(len(variable))
            split_lengths.append(split_length)
            if meta_split_lengths is not None and split_length != int(meta_split_lengths[split_idx]):
                raise ValueError(
                    "dataset meta split_lengths {} does not match {} length {} for {}".format(
                        meta_split_lengths,
                        split_name,
                        split_length,
                        self.meta_path,
                    )
                )

            if expected_var_num is None:
                expected_var_num = int(variable.shape[1])
            elif int(variable.shape[1]) != expected_var_num:
                raise ValueError(
                    "dataset splits must have consistent width, but '{}' has {} and expected {}".format(
                        data_path,
                        int(variable.shape[1]),
                        expected_var_num,
                    )
                )

            timestamp_path = self.dataset_dir / _split_timestamp_file_name(split_name)
            has_timestamp = timestamp_path.exists()
            timestamp_presence.append(has_timestamp)
            if has_timestamp:
                raw_timestamps = load_npy_array(timestamp_path, use_mmap=self.use_mmap, dtype=np.float32)
                if len(timestamp_descriptions) == 0:
                    raise ValueError(
                        "dataset '{}' provides {} but meta.json is missing timestamps_description".format(
                            self.dataset_name,
                            timestamp_path.name,
                        )
                    )
                time_feature = restore_basic_ts_timestamps(raw_timestamps, timestamp_descriptions, self.freq)
                if len(time_feature) != split_length:
                    raise ValueError(
                        "timestamp length {} does not match data length {} for {}".format(
                            len(time_feature),
                            split_length,
                            timestamp_path,
                        )
                    )
                timestamp_dim = int(time_feature.shape[1])
                if expected_timestamp_dim is None:
                    expected_timestamp_dim = timestamp_dim
                elif timestamp_dim != expected_timestamp_dim:
                    raise ValueError(
                        "dataset timestamp width {} does not match expected {} for {}".format(
                            timestamp_dim,
                            expected_timestamp_dim,
                            timestamp_path,
                        )
                    )
            else:
                time_feature = None

            self.split_variable[split_name] = variable
            self.split_time_feature[split_name] = time_feature

        if any(timestamp_presence) and not all(timestamp_presence):
            raise FileNotFoundError(
                "dataset '{}' must provide BasicTS timestamps for all splits or none of them".format(self.dataset_name)
            )

        if not any(timestamp_presence):
            expected_timestamp_dim = 0
            self.time_feature_descriptions = ()
            for split_name in SPLIT_NAMES:
                split_length = int(len(self.split_variable[split_name]))
                self.split_time_feature[split_name] = np.empty((split_length, 0), dtype=np.float32)
        else:
            self.time_feature_descriptions = tuple(timestamp_descriptions)

        meta_num_vars = self.meta.get("num_vars")
        if meta_num_vars is not None and int(meta_num_vars) != expected_var_num:
            raise ValueError(
                "dataset meta num_vars {} does not match data width {} for {}".format(
                    int(meta_num_vars),
                    expected_var_num,
                    self.meta_path,
                )
            )

        self.var_num = int(expected_var_num)
        self.time_feature_dim = int(expected_timestamp_dim)
        self._load_regular_settings()
        self._build_scaler_stats()
        self.graph = self._read_graph()
        meta_has_graph = self.meta.get("has_graph")
        if meta_has_graph is not None and bool(meta_has_graph) != bool(self.graph is not None):
            raise ValueError(
                "dataset meta has_graph={} does not match graph side file presence for {}".format(
                    meta_has_graph,
                    self.meta_path,
                )
            )
        self.data_spec = self._build_data_spec()

        self._record_resolved_conf("var_num", self.var_num, "dataset files/meta")
        self._record_resolved_conf("split_lengths", split_lengths, "dataset split files", validate_keys=("data_split", "split_lengths"))
        self._record_resolved_conf("time_feature_dim", self.time_feature_dim, "dataset timestamps/meta")
        self._record_resolved_conf(
            "time_feature_descriptions",
            tuple(self.time_feature_descriptions),
            "dataset timestamps/meta",
        )
        self._record_resolved_conf("has_graph", self.graph is not None, "dataset graph side input")
        self._record_resolved_conf("norm_each_channel", self.norm_each_channel, "dataset regular_settings")
        self._record_resolved_conf("rescale", self.rescale, "dataset regular_settings")
        self._record_resolved_conf("null_val", self.null_val, "dataset regular_settings")
        self._record_resolved_conf(
            "scaler_stats",
            {
                "mean": _serialize_stat_value(self.scaler_stats["mean"]),
                "std": _serialize_stat_value(self.scaler_stats["std"]),
            },
            "dataset train split stats",
        )

    def _read_graph(self):
        if self.graph_path is None:
            return None
        graph = load_graph_array(self.graph_path)
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("graph adjacency must be a square matrix: {}".format(self.graph_path))
        if graph.shape[0] != self.var_num:
            raise ValueError(
                "graph node count {} does not match dataset width {} for {}".format(
                    int(graph.shape[0]),
                    self.var_num,
                    self.graph_path,
                )
            )
        return graph

    def _build_data_spec(self):
        return DataSpec(
            layout_kind="sequence",
            spatial_ndim=0,
            spatial_shape=(),
            channel_num=None,
            has_graph=self.graph is not None,
            has_grid_mask=False,
            has_coord=False,
            time_feature_dim=int(self.time_feature_dim),
            time_feature_descriptions=tuple(self.time_feature_descriptions),
        )

    def _build_split_dataset(self, split_name):
        return BasicTSSequenceDataset(
            hist_len=self.hist_len,
            pred_len=self.pred_len,
            variable=self.split_variable[split_name],
            timestamps=self.split_time_feature[split_name],
            precompute_window_index=self.precompute_window_index,
        )

    def train_dataloader(self):
        if self._train_loader is None:
            self._train_loader = self._create_loader(
                dataset=self._build_split_dataset("train"),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self._val_loader = self._create_loader(
                dataset=self._build_split_dataset("val"),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )
        return self._val_loader

    def test_dataloader(self):
        if self._test_loader is None:
            self._test_loader = self._create_loader(
                dataset=self._build_split_dataset("test"),
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )
        return self._test_loader
