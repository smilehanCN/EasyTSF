import numpy as np

from .data_module import (
    DataInterface,
    build_time_feature,
    load_dataset_arrays,
)
from .spec import DataSpec


GRID_MASK_FILE_NAME = "grid_mask.npy"
COORD_FILE_NAME = "coord.npy"


def _load_optional_side_array(path, use_mmap=False):
    if path is None or not path.exists():
        return None

    load_kwargs = {"allow_pickle": False}
    if use_mmap:
        load_kwargs["mmap_mode"] = "r"
    array = np.load(path, **load_kwargs)
    return array.astype(np.float32, copy=False)


def load_grid_dataset_arrays(npz_path, grid_mask_path=None, coord_path=None, use_mmap=False, cache_npz_as_npy=None):
    variable, timestamp = load_dataset_arrays(
        npz_path,
        use_mmap=use_mmap,
        cache_npz_as_npy=cache_npz_as_npy,
    )
    grid_mask = _load_optional_side_array(grid_mask_path, use_mmap=use_mmap)
    coord = _load_optional_side_array(coord_path, use_mmap=use_mmap)
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
