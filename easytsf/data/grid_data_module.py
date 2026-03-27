import numpy as np

from .base import (
    BaseDataInterface,
    DATA_FILE_NAME,
    GeneralTSFDataset,
    build_time_feature,
    load_dataset_arrays,
    load_meta_split_lengths,
    load_npy_array,
    require_dataset_file,
)
from .spec import DataSpec


GRID_MASK_FILE_NAME = "grid_mask.npy"
COORD_FILE_NAME = "coord.npy"


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
        super().__init__(**kwargs)

    def _setup_dataset(self):
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

    def _build_data_spec(self):
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
