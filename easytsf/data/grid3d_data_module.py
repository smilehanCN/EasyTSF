from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset


def normalize_spatial_shape(raw_shape, grid_shape, field_name: str) -> tuple[int, int, int]:
    if raw_shape is None:
        return tuple(int(size) for size in grid_shape)
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 3:
        raise ValueError("{} must be a length-3 list/tuple".format(field_name))
    shape = tuple(int(size) for size in raw_shape)
    if any(size <= 0 for size in shape):
        raise ValueError("{} values must be > 0".format(field_name))
    if any(size > limit for size, limit in zip(shape, grid_shape, strict=True)):
        raise ValueError("{} {} exceeds grid_shape {}".format(field_name, shape, grid_shape))
    return shape


def normalize_overlap(raw_overlap, tile_shape) -> tuple[int, int, int]:
    if raw_overlap is None:
        return (0, 0, 0)
    if not isinstance(raw_overlap, (list, tuple)) or len(raw_overlap) != 3:
        raise ValueError("tile_overlap must be a length-3 list/tuple")
    overlap = tuple(int(size) for size in raw_overlap)
    if any(size < 0 for size in overlap):
        raise ValueError("tile_overlap values must be >= 0")
    if any(size >= limit for size, limit in zip(overlap, tile_shape, strict=True)):
        raise ValueError("tile_overlap {} must be smaller than tile_shape {}".format(overlap, tile_shape))
    return overlap


def compute_axis_starts(full_size: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size >= full_size:
        return [0]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(
            "tile_size {} and overlap {} produce invalid non-positive stride".format(tile_size, overlap)
        )

    starts = [0]
    last_start = full_size - tile_size
    while starts[-1] < last_start:
        next_start = min(starts[-1] + stride, last_start)
        if next_start == starts[-1]:
            break
        starts.append(next_start)
    return starts


def build_tile_bboxes(
    grid_shape: tuple[int, int, int],
    tile_shape: tuple[int, int, int],
    tile_overlap: tuple[int, int, int],
) -> list[tuple[int, int, int, int, int, int]]:
    axis_starts = [
        compute_axis_starts(full_size=full_size, tile_size=tile_size, overlap=overlap)
        for full_size, tile_size, overlap in zip(grid_shape, tile_shape, tile_overlap, strict=True)
    ]
    bboxes = []
    for y_start, x_start, z_start in product(*axis_starts):
        y_stop = min(y_start + tile_shape[0], grid_shape[0])
        x_stop = min(x_start + tile_shape[1], grid_shape[1])
        z_stop = min(z_start + tile_shape[2], grid_shape[2])
        bboxes.append((y_start, y_stop, x_start, x_stop, z_start, z_stop))
    return bboxes


def build_valid_crop_slices(
    tile_bbox: tuple[int, int, int, int, int, int],
    grid_shape: tuple[int, int, int],
    tile_overlap: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    y_start, y_stop, x_start, x_stop, z_start, z_stop = tile_bbox
    tile_shape = (y_stop - y_start, x_stop - x_start, z_stop - z_start)

    left_crops = []
    right_crops = []
    for axis_start, axis_stop, full_size, overlap, axis_tile_size in zip(
        (y_start, x_start, z_start),
        (y_stop, x_stop, z_stop),
        grid_shape,
        tile_overlap,
        tile_shape,
        strict=True,
    ):
        left_crop = 0 if axis_start == 0 else overlap // 2
        right_crop = 0 if axis_stop == full_size else overlap - overlap // 2
        left_crops.append(left_crop)
        right_crops.append(axis_tile_size - right_crop)

    return (
        slice(left_crops[0], right_crops[0]),
        slice(left_crops[1], right_crops[1]),
        slice(left_crops[2], right_crops[2]),
    )


class Grid3DStepDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        split,
        hist_len,
        pred_len,
        *,
        mode,
        use_mmap=False,
        use_coords=True,
        patch_shape=None,
        tile_shape=None,
        tile_overlap=None,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.split = str(split)
        self.mode = str(mode)
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.use_mmap = bool(use_mmap)
        self.use_coords = bool(use_coords)

        if self.hist_len <= 0 or self.pred_len <= 0:
            raise ValueError("hist_len and pred_len must be > 0")
        if self.mode not in {"train", "eval"}:
            raise ValueError("mode must be 'train' or 'eval'")

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

        self.coord = None
        if self.use_coords:
            self.coord = np.load(
                self.dataset_dir / "coord.npy",
                mmap_mode="r" if self.use_mmap else None,
                allow_pickle=False,
            )

        self.total_windows = int(self.variable.shape[0]) - (self.hist_len + self.pred_len) + 1
        if self.total_windows <= 0:
            raise ValueError("invalid dataset split for sliding window")

        self.patch_shape = None
        self.tile_shape = None
        self.tile_overlap = None
        self.tile_bboxes = None
        if self.mode == "train":
            self.patch_shape = normalize_spatial_shape(patch_shape, self.grid_shape, "patch_shape")
        else:
            self.tile_shape = normalize_spatial_shape(tile_shape, self.grid_shape, "tile_shape")
            self.tile_overlap = normalize_overlap(tile_overlap, self.tile_shape)
            self.tile_bboxes = build_tile_bboxes(self.grid_shape, self.tile_shape, self.tile_overlap)

    def __len__(self):
        if self.mode == "train":
            return self.total_windows
        return self.total_windows * len(self.tile_bboxes)

    def _sample_patch_bbox(self) -> tuple[int, int, int, int, int, int]:
        starts = []
        for patch_size, full_size in zip(self.patch_shape, self.grid_shape, strict=True):
            max_start = full_size - patch_size
            starts.append(0 if max_start <= 0 else int(np.random.randint(0, max_start + 1)))
        return (
            starts[0],
            starts[0] + self.patch_shape[0],
            starts[1],
            starts[1] + self.patch_shape[1],
            starts[2],
            starts[2] + self.patch_shape[2],
        )

    def _load_window(self, start_index: int, stop_index: int, bbox: tuple[int, int, int, int, int, int]) -> np.ndarray:
        y_start, y_stop, x_start, x_stop, z_start, z_stop = bbox
        return np.asarray(
            self.variable[start_index:stop_index, :, y_start:y_stop, x_start:x_stop, z_start:z_stop],
            dtype=np.float32,
        )

    def _build_coords(self, bbox: tuple[int, int, int, int, int, int]) -> np.ndarray:
        y_start, y_stop, x_start, x_stop, z_start, z_stop = bbox
        return np.asarray(
            self.coord[:, y_start:y_stop, x_start:x_stop, z_start:z_stop],
            dtype=np.float32,
        )

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if self.mode == "train":
            if index < 0 or index >= self.total_windows:
                raise IndexError("train sample index {} is out of range".format(index))
            window_id = int(index)
            bbox = self._sample_patch_bbox()
        else:
            if index < 0 or index >= len(self):
                raise IndexError("eval sample index {} is out of range".format(index))
            window_id = int(index // len(self.tile_bboxes))
            bbox = self.tile_bboxes[int(index % len(self.tile_bboxes))]

        hist_start = window_id
        hist_stop = hist_start + self.hist_len
        pred_stop = hist_stop + self.pred_len

        item = {
            "inputs": np.ascontiguousarray(self._load_window(hist_start, hist_stop, bbox), dtype=np.float32),
            "targets": np.ascontiguousarray(self._load_window(hist_stop, pred_stop, bbox), dtype=np.float32),
            "inputs_timestamps": np.ascontiguousarray(self.timestamps[hist_start:hist_stop], dtype=np.float64),
            "targets_timestamps": np.ascontiguousarray(self.timestamps[hist_stop:pred_stop], dtype=np.float64),
            "window_id": np.asarray(window_id, dtype=np.int64),
            "tile_bbox": np.asarray(bbox, dtype=np.int64),
        }
        if self.use_coords:
            item["coords"] = np.ascontiguousarray(self._build_coords(bbox), dtype=np.float32)
        return item


class Grid3DDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = dict(kwargs)
        self.dataset = str(self.config["dataset"])
        self.num_workers = int(kwargs["num_workers"])
        self.batch_size = int(kwargs["batch_size"])
        self.hist_len = int(kwargs["hist_len"])
        self.pred_len = int(kwargs["pred_len"])
        self.use_mmap = bool(kwargs.get("use_mmap", True))
        self.use_coords = bool(kwargs.get("use_coords", True))
        self.train_patch_shape = kwargs.get("train_patch_shape")
        self.eval_tile_shape = kwargs.get("eval_tile_shape")
        self.eval_tile_overlap = kwargs.get("eval_tile_overlap", (0, 0, 0))
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
        if split_name == "train":
            return Grid3DStepDataset(
                dataset_dir=self.dataset_dir,
                split=split_name,
                hist_len=self.hist_len,
                pred_len=self.pred_len,
                mode="train",
                use_mmap=self.use_mmap,
                use_coords=self.use_coords,
                patch_shape=self.train_patch_shape,
            )
        return Grid3DStepDataset(
            dataset_dir=self.dataset_dir,
            split=split_name,
            hist_len=self.hist_len,
            pred_len=self.pred_len,
            mode="eval",
            use_mmap=self.use_mmap,
            use_coords=self.use_coords,
            tile_shape=self.eval_tile_shape,
            tile_overlap=self.eval_tile_overlap,
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
