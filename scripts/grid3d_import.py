from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from numpy.lib.format import open_memmap


TIME_PATTERN = re.compile(r"t(\d+)\.nc$")
SUPPORTED_SOURCE_FORMATS = {"wf4cast_hdf5_netcdf_like"}
TARGET_DATASET_PRESET = "windshear_v1_0416"
DEFAULT_Z_SLICE_START = 0
DEFAULT_Z_SLICE_END = 30
DEFAULT_CENTER_CROP_SIZE_Y = 400
DEFAULT_CENTER_CROP_SIZE_X = 400
DEFAULT_STORAGE_FORMAT = "grid3d_split_npy_v1"


@dataclass(frozen=True)
class Grid3DRecord:
    path: Path
    time_index: int
    time_s: float


def dump_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=False)
        handle.write("\n")


def parse_time_index(path: Path) -> int:
    match = TIME_PATTERN.search(path.name)
    if match is None:
        raise ValueError("cannot parse time index from '{}'".format(path.name))
    return int(match.group(1))


def load_grid3d_step(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with h5py.File(Path(path), "r") as handle:
        u = np.asarray(handle["U"][:], dtype=np.float32)
        v = np.asarray(handle["V"][:], dtype=np.float32)
        w = np.asarray(handle["W"][:], dtype=np.float32)
        time_s = float(np.asarray(handle["time_s"][()], dtype=np.float64))
    return u, v, w, time_s


def load_grid3d_tensor(
    path: str | Path,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, float]:
    u, v, w, time_s = load_grid3d_step(path)
    path = Path(path)

    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError(
            "inconsistent channel shapes in '{}': U={}, V={}, W={}".format(
                path,
                tuple(int(size) for size in u.shape),
                tuple(int(size) for size in v.shape),
                tuple(int(size) for size in w.shape),
            )
        )
    if expected_shape is not None and u.shape != expected_shape:
        raise ValueError(
            "unexpected grid shape in '{}': expected {}, got {}".format(
                path,
                tuple(int(size) for size in expected_shape),
                tuple(int(size) for size in u.shape),
            )
        )

    return np.stack([u, v, w], axis=0).astype(np.float32, copy=False), time_s


def load_grid3d_coords(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(Path(path), "r") as handle:
        x = np.asarray(handle["x"][:], dtype=np.float32)
        y = np.asarray(handle["y"][:], dtype=np.float32)
        z = np.asarray(handle["z"][:], dtype=np.float32)
    return x, y, z


def infer_axis_spacing(axis_values: np.ndarray, axis_name: str) -> float:
    values = np.asarray(axis_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("axis '{}' must be 1D, got shape {}".format(axis_name, tuple(int(size) for size in values.shape)))
    if values.size <= 1:
        return 1.0

    deltas = np.diff(values)
    if np.any(deltas <= 0.0):
        raise ValueError("axis '{}' must be strictly increasing".format(axis_name))
    if not np.allclose(deltas, deltas[0], rtol=1e-5, atol=1e-6):
        raise ValueError(
            "axis '{}' must be evenly spaced, got min_delta={} and max_delta={}".format(
                axis_name,
                float(deltas.min()),
                float(deltas.max()),
            )
        )
    return float(deltas[0])


def build_records(input_dir: str | Path, pattern: str) -> list[Grid3DRecord]:
    dataset_dir = Path(input_dir).expanduser().resolve()
    records = []
    for path in sorted(dataset_dir.glob(pattern), key=parse_time_index):
        time_index = parse_time_index(path)
        with h5py.File(path, "r") as handle:
            time_s = float(np.asarray(handle["time_s"][()], dtype=np.float64))
        records.append(Grid3DRecord(path=path, time_index=time_index, time_s=time_s))
    if len(records) == 0:
        raise ValueError("no files matched pattern '{}' under '{}'".format(pattern, dataset_dir))
    return records


def _parse_structured_arg(raw_value: str | None, *, default: Any) -> Any:
    if raw_value is None:
        return default
    parsed = yaml.safe_load(raw_value)
    return default if parsed is None else parsed


def _normalize_split_spec(raw_split_spec: dict[str, Any] | None, total_steps: int) -> dict[str, tuple[int, int]] | None:
    if raw_split_spec is None:
        return None
    if not isinstance(raw_split_spec, dict):
        raise ValueError("split_spec must be a mapping")

    normalized = {}
    for split_name in ("train", "val", "test"):
        if split_name not in raw_split_spec:
            raise ValueError("split_spec must define '{}'".format(split_name))
        split_value = raw_split_spec[split_name]
        if isinstance(split_value, dict):
            start = split_value.get("start")
            end = split_value.get("end")
        elif isinstance(split_value, (list, tuple)) and len(split_value) == 2:
            start, end = split_value
        else:
            raise ValueError(
                "split_spec['{}'] must be a dict with start/end or a 2-item list".format(split_name)
            )

        start_index = 0 if start is None else int(start)
        end_index = total_steps if end is None else int(end)
        if start_index < 0 or end_index < 0 or start_index > end_index or end_index > total_steps:
            raise ValueError(
                "split_spec['{}'] is out of bounds for total_steps={}".format(split_name, total_steps)
            )
        normalized[split_name] = (start_index, end_index)

    occupied = []
    for split_name in ("train", "val", "test"):
        start_index, end_index = normalized[split_name]
        if start_index == end_index:
            raise ValueError("split '{}' is empty".format(split_name))
        occupied.append((start_index, end_index, split_name))
    occupied.sort()
    for previous, current in zip(occupied[:-1], occupied[1:], strict=True):
        if previous[1] > current[0]:
            raise ValueError(
                "split '{}' overlaps with split '{}'".format(previous[2], current[2])
            )
    return normalized


def build_default_split_spec(total_steps: int, train_fraction: float = 0.6, val_fraction: float = 0.2) -> dict[str, tuple[int, int]]:
    if total_steps < 3:
        raise ValueError("at least 3 time steps are required to build train/val/test splits")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    train_end = max(1, int(total_steps * train_fraction))
    val_end = max(train_end + 1, int(total_steps * (train_fraction + val_fraction)))
    val_end = min(val_end, total_steps - 1)
    if val_end <= train_end:
        raise ValueError("validation split would be empty for total_steps={}".format(total_steps))

    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_steps),
    }


def infer_frequency_seconds(records: list[Grid3DRecord]) -> float | None:
    if len(records) < 2:
        return None
    deltas = np.diff([record.time_s for record in records])
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        return None
    return float(positive_deltas[0])


def normalize_slice_bounds(axis_size: int, slice_start: int, slice_end: int | None, axis_name: str) -> tuple[int, int]:
    start_index = int(slice_start)
    end_index = axis_size if slice_end is None else int(slice_end)
    if start_index < 0 or end_index < 0 or start_index >= end_index or end_index > axis_size:
        raise ValueError(
            "invalid {} slice [{}, {}) for axis size {}".format(
                axis_name,
                start_index,
                end_index,
                axis_size,
            )
        )
    return start_index, end_index


def resolve_center_crop_bounds(axis_size: int, crop_size: int, axis_name: str) -> tuple[int, int]:
    target_size = int(crop_size)
    if target_size <= 0:
        raise ValueError("{} center crop size must be > 0, got {}".format(axis_name, target_size))
    if axis_size <= target_size:
        return 0, int(axis_size)
    start_index = (int(axis_size) - target_size) // 2
    return start_index, start_index + target_size


def compute_axis_shear(flow: np.ndarray, spatial_dim: int, spacing_m: float) -> np.ndarray:
    if flow.ndim != 4 or int(flow.shape[0]) != 3:
        raise ValueError("expected flow tensor [3,Y,X,Z], got shape {}".format(tuple(int(size) for size in flow.shape)))
    axis_length = int(flow.shape[spatial_dim])
    out = np.zeros(flow.shape[1:], dtype=np.float32)
    if axis_length <= 1:
        return out

    lhs_index = [slice(None)] * 4
    rhs_index = [slice(None)] * 4
    lhs_index[spatial_dim] = slice(1, None)
    rhs_index[spatial_dim] = slice(0, -1)
    vector_delta = flow[tuple(lhs_index)] - flow[tuple(rhs_index)]
    magnitude = np.sqrt(np.maximum(np.sum(vector_delta * vector_delta, axis=0), 0.0)) / float(spacing_m)

    out_spatial_dim = spatial_dim - 1
    fill_index = [slice(None)] * 3
    fill_index[out_spatial_dim] = slice(0, -1)
    out[tuple(fill_index)] = magnitude

    last_index = [slice(None)] * 3
    last_index[out_spatial_dim] = -1
    out[tuple(last_index)] = magnitude[tuple(last_index)]
    return out


def build_feature_tensor_from_flow(
    flow: np.ndarray,
    *,
    grid_spacing_m: tuple[float, float, float],
) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    dy_m, dx_m, dz_m = grid_spacing_m
    shear_x = compute_axis_shear(flow, spatial_dim=2, spacing_m=dx_m)
    shear_y = compute_axis_shear(flow, spatial_dim=1, spacing_m=dy_m)
    shear_z = compute_axis_shear(flow, spatial_dim=3, spacing_m=dz_m)
    return np.concatenate(
        [
            flow,
            shear_x[None, ...],
            shear_y[None, ...],
            shear_z[None, ...],
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def load_grid3d_feature_tensor(
    path: str | Path,
    *,
    y_slice_start: int,
    y_slice_end: int | None,
    x_slice_start: int,
    x_slice_end: int | None,
    z_slice_start: int,
    z_slice_end: int | None,
    grid_spacing_m: tuple[float, float, float],
    expected_feature_shape: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, float]:
    flow, time_s = load_grid3d_tensor(path)
    y_start_index, y_end_index = normalize_slice_bounds(int(flow.shape[1]), y_slice_start, y_slice_end, "y")
    x_start_index, x_end_index = normalize_slice_bounds(int(flow.shape[2]), x_slice_start, x_slice_end, "x")
    z_start_index, z_end_index = normalize_slice_bounds(int(flow.shape[3]), z_slice_start, z_slice_end, "z")
    flow = flow[:, y_start_index:y_end_index, x_start_index:x_end_index, z_start_index:z_end_index]
    feature_tensor = build_feature_tensor_from_flow(
        flow,
        grid_spacing_m=grid_spacing_m,
    )
    if expected_feature_shape is not None and tuple(int(size) for size in feature_tensor.shape) != expected_feature_shape:
        raise ValueError(
            "unexpected feature shape in '{}': expected {}, got {}".format(
                path,
                expected_feature_shape,
                tuple(int(size) for size in feature_tensor.shape),
            )
        )
    return feature_tensor.astype(np.float32, copy=False), time_s


def _compute_channel_stats(
    records: list[Grid3DRecord],
    *,
    expected_feature_shape: tuple[int, int, int, int],
    y_slice_start: int,
    y_slice_end: int | None,
    x_slice_start: int,
    x_slice_end: int | None,
    z_slice_start: int,
    z_slice_end: int | None,
    grid_spacing_m: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    channel_count = int(expected_feature_shape[0])
    channel_sums = np.zeros(channel_count, dtype=np.float64)
    channel_sum_squares = np.zeros(channel_count, dtype=np.float64)
    total_value_count = 0

    for record in records:
        step, _ = load_grid3d_feature_tensor(
            record.path,
            y_slice_start=y_slice_start,
            y_slice_end=y_slice_end,
            x_slice_start=x_slice_start,
            x_slice_end=x_slice_end,
            z_slice_start=z_slice_start,
            z_slice_end=z_slice_end,
            grid_spacing_m=grid_spacing_m,
            expected_feature_shape=expected_feature_shape,
        )
        step = step.astype(np.float64, copy=False)
        channel_sums += step.sum(axis=(1, 2, 3))
        channel_sum_squares += np.square(step).sum(axis=(1, 2, 3))
        total_value_count += step.shape[1] * step.shape[2] * step.shape[3]

    if total_value_count <= 0:
        raise ValueError("training split is empty; cannot compute channel statistics")

    mean = channel_sums / float(total_value_count)
    variance = channel_sum_squares / float(total_value_count) - np.square(mean)
    variance = np.maximum(variance, 1e-12)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


def import_grid3d_dataset(
    *,
    input_dir: str | Path,
    out_dir: str | Path,
    pattern: str = "wind_grid_t*.nc",
    split_spec: dict[str, Any] | None = None,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    center_crop_size_y: int = DEFAULT_CENTER_CROP_SIZE_Y,
    center_crop_size_x: int = DEFAULT_CENTER_CROP_SIZE_X,
    z_slice_start: int = DEFAULT_Z_SLICE_START,
    z_slice_end: int | None = DEFAULT_Z_SLICE_END,
    source_format: str = "wf4cast_hdf5_netcdf_like",
) -> dict[str, Any]:
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(
            "unsupported source_format '{}'; supported formats are {}".format(
                source_format,
                sorted(SUPPORTED_SOURCE_FORMATS),
            )
        )
    resolved_z_slice_start = int(z_slice_start)
    resolved_z_slice_end = None if z_slice_end is None else int(z_slice_end)
    resolved_center_crop_size_y = int(center_crop_size_y)
    resolved_center_crop_size_x = int(center_crop_size_x)

    records = build_records(input_dir=input_dir, pattern=pattern)
    normalized_split_spec = _normalize_split_spec(split_spec, len(records))
    if normalized_split_spec is None:
        normalized_split_spec = build_default_split_spec(
            total_steps=len(records),
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )

    x, y, z = load_grid3d_coords(records[0].path)
    source_grid_shape = (int(y.size), int(x.size), int(z.size))
    y_slice_start_index, y_slice_end_index = resolve_center_crop_bounds(
        int(y.size),
        resolved_center_crop_size_y,
        "y",
    )
    x_slice_start_index, x_slice_end_index = resolve_center_crop_bounds(
        int(x.size),
        resolved_center_crop_size_x,
        "x",
    )
    y = np.asarray(y[y_slice_start_index:y_slice_end_index], dtype=np.float32)
    x = np.asarray(x[x_slice_start_index:x_slice_end_index], dtype=np.float32)
    resolved_z_slice_start, resolved_z_slice_end = normalize_slice_bounds(
        int(z.size),
        resolved_z_slice_start,
        resolved_z_slice_end,
        "z",
    )
    z = np.asarray(z[resolved_z_slice_start:resolved_z_slice_end], dtype=np.float32)
    dy_m = infer_axis_spacing(y, "y")
    dx_m = infer_axis_spacing(x, "x")
    dz_m = infer_axis_spacing(z, "z")
    sample_step, _ = load_grid3d_feature_tensor(
        records[0].path,
        y_slice_start=y_slice_start_index,
        y_slice_end=y_slice_end_index,
        x_slice_start=x_slice_start_index,
        x_slice_end=x_slice_end_index,
        z_slice_start=resolved_z_slice_start,
        z_slice_end=resolved_z_slice_end,
        grid_spacing_m=(dy_m, dx_m, dz_m),
    )
    expected_feature_shape = tuple(int(size) for size in sample_step.shape)
    expected_shape = expected_feature_shape[1:]
    channel_names = ["U", "V", "W", "shear_x", "shear_y", "shear_z"]
    derived_channel_names = ["shear_x", "shear_y", "shear_z"]

    train_start, train_end = normalized_split_spec["train"]
    mean, std = _compute_channel_stats(
        records[train_start:train_end],
        expected_feature_shape=expected_feature_shape,
        y_slice_start=y_slice_start_index,
        y_slice_end=y_slice_end_index,
        x_slice_start=x_slice_start_index,
        x_slice_end=x_slice_end_index,
        z_slice_start=resolved_z_slice_start,
        z_slice_end=resolved_z_slice_end,
        grid_spacing_m=(dy_m, dx_m, dz_m),
    )
    mean_view = mean[:, None, None, None]
    std_view = std[:, None, None, None]

    dataset_dir = Path(out_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    def normalize_axis(values):
        values = np.asarray(values, dtype=np.float32)
        span = float(values[-1] - values[0])
        if span == 0.0:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - values[0]) / span) * 2.0 - 1.0

    yy, xx, zz = np.meshgrid(
        normalize_axis(y),
        normalize_axis(x),
        normalize_axis(z),
        indexing="ij",
    )
    np.save(dataset_dir / "coord.npy", np.stack([yy, xx, zz], axis=0).astype(np.float32, copy=False))
    np.savez(
        dataset_dir / "axes.npz",
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        z=np.asarray(z, dtype=np.float32),
    )
    np.savez(dataset_dir / "stats.npz", mean=mean, std=std)

    split_lengths = {}
    for split_name in ("train", "val", "test"):
        split_start, split_end = normalized_split_spec[split_name]
        split_records = records[split_start:split_end]
        timestamps = np.asarray([record.time_s for record in split_records], dtype=np.float64)
        np.save(dataset_dir / "{}_timestamps.npy".format(split_name), timestamps)
        split_lengths[split_name] = int(len(split_records))
        split_data = open_memmap(
            dataset_dir / "{}_data.npy".format(split_name),
            mode="w+",
            dtype=np.float32,
            shape=(len(split_records), *expected_feature_shape),
        )

        for step_index, record in enumerate(split_records):
            step, _ = load_grid3d_feature_tensor(
                record.path,
                y_slice_start=y_slice_start_index,
                y_slice_end=y_slice_end_index,
                x_slice_start=x_slice_start_index,
                x_slice_end=x_slice_end_index,
                z_slice_start=resolved_z_slice_start,
                z_slice_end=resolved_z_slice_end,
                grid_spacing_m=(dy_m, dx_m, dz_m),
                expected_feature_shape=expected_feature_shape,
            )
            normalized_step = (step - mean_view) / std_view
            split_data[step_index] = normalized_step.astype(np.float32, copy=False)
        split_data.flush()
        del split_data

    meta = {
        "task_type": "grid_prediction",
        "dataset_preset": TARGET_DATASET_PRESET,
        "storage_format": DEFAULT_STORAGE_FORMAT,
        "data_layout": "T,C,Y,X,Z",
        "data_is_standardized": True,
        "source_grid_shape": [int(size) for size in source_grid_shape],
        "grid_shape": [int(expected_shape[0]), int(expected_shape[1]), int(expected_shape[2])],
        "xy_slice_indices": {
            "y": [int(y_slice_start_index), int(y_slice_end_index)],
            "x": [int(x_slice_start_index), int(x_slice_end_index)],
        },
        "z_slice_indices": [int(resolved_z_slice_start), int(resolved_z_slice_end)],
        "channel_names": channel_names,
        "velocity_channel_names": ["U", "V", "W"],
        "derived_channel_names": derived_channel_names,
        "storage_dtype": "float32",
        "frequency_seconds": infer_frequency_seconds(records),
        "grid_spacing_m": [float(dy_m), float(dx_m), float(dz_m)],
        "axis_layout": ["y", "x", "z"],
        "coord_min": [float(y[0]), float(x[0]), float(z[0])],
        "coord_max": [float(y[-1]), float(x[-1]), float(z[-1])],
        "split_lengths": split_lengths,
        "max_supported_hist_len": 10,
        "max_supported_pred_len": 10,
        "source_format": str(source_format),
    }
    dump_json(dataset_dir / "meta.json", meta)
    return meta


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import raw 3D wind .nc files into the WindShearV1_0416 cache format."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with WindField4Cast-style single-step .nc files.")
    parser.add_argument("--out-dir", required=True, help="Output directory for the EasyTSF Grid3D dataset.")
    parser.add_argument(
        "--source-format",
        default="wf4cast_hdf5_netcdf_like",
        choices=sorted(SUPPORTED_SOURCE_FORMATS),
        help="Source file layout.",
    )
    parser.add_argument(
        "--pattern",
        default="wind_grid_t*.nc",
        help="Glob pattern for single-step raw files.",
    )
    parser.add_argument(
        "--center-crop-size-y",
        default=DEFAULT_CENTER_CROP_SIZE_Y,
        type=int,
        help="Center crop size for Y axis before feature derivation.",
    )
    parser.add_argument(
        "--center-crop-size-x",
        default=DEFAULT_CENTER_CROP_SIZE_X,
        type=int,
        help="Center crop size for X axis before feature derivation.",
    )
    parser.add_argument(
        "--z-slice-start",
        default=DEFAULT_Z_SLICE_START,
        type=int,
        help="Inclusive start index for the Z axis before writing the cache.",
    )
    parser.add_argument(
        "--z-slice-end",
        default=DEFAULT_Z_SLICE_END,
        type=int,
        help="Exclusive end index for the Z axis before writing the cache. WindShearV1_0416 uses 30 by default.",
    )
    parser.add_argument(
        "--split-spec",
        default=None,
        help="Optional YAML/JSON mapping for train/val/test using [start, end) index ranges.",
    )
    parser.add_argument(
        "--train-fraction",
        default=0.6,
        type=float,
        help="Train fraction used when split-spec is omitted.",
    )
    parser.add_argument(
        "--val-fraction",
        default=0.2,
        type=float,
        help="Validation fraction used when split-spec is omitted.",
    )
    return parser


if __name__ == "__main__":
    args = build_cli_parser().parse_args()
    import_grid3d_dataset(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        pattern=args.pattern,
        split_spec=_parse_structured_arg(args.split_spec, default=None),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        center_crop_size_y=int(args.center_crop_size_y),
        center_crop_size_x=int(args.center_crop_size_x),
        z_slice_start=args.z_slice_start,
        z_slice_end=args.z_slice_end,
        source_format=args.source_format,
    )
