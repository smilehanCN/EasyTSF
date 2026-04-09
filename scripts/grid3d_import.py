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


def load_grid3d_coords(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(Path(path), "r") as handle:
        x = np.asarray(handle["x"][:], dtype=np.float32)
        y = np.asarray(handle["y"][:], dtype=np.float32)
        z = np.asarray(handle["z"][:], dtype=np.float32)
    return x, y, z


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


def _compute_channel_stats(records: list[Grid3DRecord]) -> tuple[np.ndarray, np.ndarray]:
    channel_sums = np.zeros(3, dtype=np.float64)
    channel_sum_squares = np.zeros(3, dtype=np.float64)
    total_value_count = 0

    for record in records:
        u, v, w, _ = load_grid3d_step(record.path)
        if u.shape != expected_shape or v.shape != expected_shape or w.shape != expected_shape:
            raise ValueError("unexpected grid shape in '{}'".format(record.path))
        step = np.stack([u, v, w], axis=0).astype(np.float64, copy=False)
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
    source_format: str = "wf4cast_hdf5_netcdf_like",
) -> dict[str, Any]:
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(
            "unsupported source_format '{}'; supported formats are {}".format(
                source_format,
                sorted(SUPPORTED_SOURCE_FORMATS),
            )
        )

    records = build_records(input_dir=input_dir, pattern=pattern)
    normalized_split_spec = _normalize_split_spec(split_spec, len(records))
    if normalized_split_spec is None:
        normalized_split_spec = build_default_split_spec(
            total_steps=len(records),
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )

    x, y, z = load_grid3d_coords(records[0].path)
    sample_u, sample_v, sample_w, _ = load_grid3d_step(records[0].path)
    expected_shape = sample_u.shape

    train_start, train_end = normalized_split_spec["train"]
    mean, std = _compute_channel_stats(records[train_start:train_end])
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
            shape=(len(split_records), 3, *expected_shape),
        )

        for step_index, record in enumerate(split_records):
            u, v, w, _ = load_grid3d_step(record.path)
            step = np.stack([u, v, w], axis=0).astype(np.float32, copy=False)
            normalized_step = (step - mean_view) / std_view
            split_data[step_index] = normalized_step.astype(np.float32, copy=False)
        split_data.flush()
        del split_data

    meta = {
        "task_type": "grid_prediction",
        "storage_format": DEFAULT_STORAGE_FORMAT,
        "data_layout": "T,C,Y,X,Z",
        "grid_shape": [int(expected_shape[0]), int(expected_shape[1]), int(expected_shape[2])],
        "channel_names": ["U", "V", "W"],
        "storage_dtype": "float32",
        "frequency_seconds": infer_frequency_seconds(records),
        "split_lengths": split_lengths,
        "max_supported_hist_len": 10,
        "max_supported_pred_len": 10,
        "source_format": str(source_format),
    }
    dump_json(dataset_dir / "meta.json", meta)
    return meta


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import 3D grid forecasting data into split-level .npy arrays.")
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
        source_format=args.source_format,
    )
