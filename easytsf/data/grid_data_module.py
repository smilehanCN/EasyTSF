import numpy as np

from .data_module import (
    DATA_ARRAY_KEY,
    TIMESTAMP_ARRAY_KEY,
    DataInterface,
    _cache_path_for_key,
    _ensure_npy_cache,
    build_time_feature,
)
from .spec import DataSpec


GRID_MASK_ARRAY_KEY = "grid_mask"
COORD_ARRAY_KEY = "coord"
LEGACY_SPATIAL_MASK_ARRAY_KEY = "spatial_mask"


def load_grid_dataset_arrays(npz_path, use_mmap=False, cache_npz_as_npy=None):
    if cache_npz_as_npy is None:
        cache_npz_as_npy = use_mmap

    cache_keys = [DATA_ARRAY_KEY, TIMESTAMP_ARRAY_KEY]
    with np.load(npz_path) as data:
        has_grid_mask = GRID_MASK_ARRAY_KEY in data
        has_legacy_mask = LEGACY_SPATIAL_MASK_ARRAY_KEY in data
        has_coord = COORD_ARRAY_KEY in data

    mask_cache_key = None
    if has_grid_mask:
        mask_cache_key = GRID_MASK_ARRAY_KEY
    elif has_legacy_mask:
        mask_cache_key = LEGACY_SPATIAL_MASK_ARRAY_KEY

    if mask_cache_key is not None:
        cache_keys.append(mask_cache_key)
    if has_coord:
        cache_keys.append(COORD_ARRAY_KEY)

    if use_mmap and cache_npz_as_npy and _ensure_npy_cache(npz_path, cache_keys):
        variable = np.load(_cache_path_for_key(npz_path, DATA_ARRAY_KEY), mmap_mode="r")
        timestamp = np.load(_cache_path_for_key(npz_path, TIMESTAMP_ARRAY_KEY), mmap_mode="r")
        grid_mask = None
        coord = None
        if mask_cache_key is not None:
            grid_mask = np.load(_cache_path_for_key(npz_path, mask_cache_key), mmap_mode="r")
        if has_coord:
            coord = np.load(_cache_path_for_key(npz_path, COORD_ARRAY_KEY), mmap_mode="r")
        return variable, timestamp, grid_mask, coord

    with np.load(npz_path) as data:
        variable = data[DATA_ARRAY_KEY].astype(np.float32, copy=False)
        timestamp = data[TIMESTAMP_ARRAY_KEY]
        if has_grid_mask:
            grid_mask = data[GRID_MASK_ARRAY_KEY].astype(np.float32, copy=False)
        elif has_legacy_mask:
            grid_mask = data[LEGACY_SPATIAL_MASK_ARRAY_KEY].astype(np.float32, copy=False)
        else:
            grid_mask = None
        coord = data[COORD_ARRAY_KEY].astype(np.float32, copy=False) if has_coord else None
    return variable, timestamp, grid_mask, coord


class GridDataInterface(DataInterface):
    def __init__(self, **kwargs):
        self.grid_mask = None
        self.coord = None
        self.channel_num = None
        self.spatial_shape = None
        self.spatial_ndim = None
        super().__init__(**kwargs)

    def _read_data(self):
        variable, raw_timestamp, grid_mask, coord = load_grid_dataset_arrays(
            self.data_path,
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

        self.grid_mask = grid_mask
        self.coord = coord
        time_feature = build_time_feature(
            raw_timestamp,
            self.time_feature_cls,
            self.norm_time_feature,
            self.config["freq"],
        )
        return variable, time_feature

    def _read_graph(self):
        return None

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
        )
