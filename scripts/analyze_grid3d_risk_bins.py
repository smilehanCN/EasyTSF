#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


HEAD_ORDER = ("shear_x", "shear_y", "shear_z", "speed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Grid3D risk label bin ratios for one data split.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML containing dataset and risk_bins.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--data-root", type=Path, default=None, help="Override data_root from config.")
    parser.add_argument("--y-chunk", type=int, default=32, help="Number of Y rows processed per chunk.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write JSON summary.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("config must be a YAML mapping")
    return loaded


def validate_thresholds(config: dict) -> dict[str, tuple[float, ...]]:
    raw_bins = config.get("risk_bins")
    if not isinstance(raw_bins, dict):
        raise ValueError("config must contain risk_bins mapping")

    expected_bin_count = int(config.get("risk_num_classes", 3)) - 1

    def parse_thresholds(key: str, values) -> tuple[float, ...]:
        if not isinstance(values, (list, tuple)):
            raise ValueError("risk_bins.{} must be a list".format(key))
        if len(values) != expected_bin_count:
            raise ValueError(
                "risk_bins.{} must contain {} thresholds for risk_num_classes={}".format(
                    key,
                    expected_bin_count,
                    expected_bin_count + 1,
                )
            )
        parsed = tuple(float(value) for value in values)
        if any(left >= right for left, right in zip(parsed, parsed[1:])):
            raise ValueError("risk_bins.{} must be strictly increasing".format(key))
        return parsed

    thresholds = {}
    speed_values = raw_bins.get("speed")
    if speed_values is None:
        raise ValueError("risk_bins.speed must be defined")
    thresholds["speed"] = parse_thresholds("speed", speed_values)

    shear_values = raw_bins.get("shear")
    if shear_values is None:
        raise ValueError("risk_bins.shear must be defined")
    parsed_shear = parse_thresholds("shear", shear_values)
    for key in ("shear_x", "shear_y", "shear_z"):
        thresholds[key] = parsed_shear
    return thresholds


def count_bins(values: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    labels = np.searchsorted(np.asarray(thresholds, dtype=np.float32), values.ravel(), side="right")
    return np.bincount(labels, minlength=len(thresholds) + 1).astype(np.int64, copy=False)


def load_dataset_meta(dataset_dir: Path) -> dict:
    with (dataset_dir / "meta.json").open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("meta.json must be a JSON object")
    return loaded


def load_grid_spacing_m(dataset_dir: Path) -> tuple[float, float, float]:
    meta = load_dataset_meta(dataset_dir)
    raw_spacing = meta.get("grid_spacing_m")
    if not isinstance(raw_spacing, list) or len(raw_spacing) != 3:
        raise ValueError("dataset meta must define grid_spacing_m as [dy, dx, dz]")
    spacing = tuple(float(value) for value in raw_spacing)
    if any(value <= 0.0 for value in spacing):
        raise ValueError("dataset meta grid_spacing_m values must be > 0")
    return spacing


def build_recommended_class_weights(counts: dict[str, np.ndarray]) -> list[float]:
    aggregate_count = None
    for head in HEAD_ORDER:
        head_count = np.asarray(counts[head], dtype=np.int64)
        aggregate_count = head_count if aggregate_count is None else aggregate_count + head_count
    assert aggregate_count is not None
    if np.any(aggregate_count <= 0):
        raise ValueError(
            "cannot build shared risk_class_weights because aggregate class counts contain zeros: {}".format(
                aggregate_count.tolist()
            )
        )
    max_count = float(aggregate_count.max())
    return [round(max_count / float(count), 2) for count in aggregate_count.tolist()]


def class_ranges(thresholds: tuple[float, ...], unit: str) -> list[str]:
    ranges = []
    for class_index in range(len(thresholds) + 1):
        if class_index == 0:
            ranges.append("< {:.6g} {}".format(thresholds[0], unit))
        elif class_index == len(thresholds):
            ranges.append(">= {:.6g} {}".format(thresholds[-1], unit))
        else:
            ranges.append(
                "[{:.6g}, {:.6g}) {}".format(
                    thresholds[class_index - 1],
                    thresholds[class_index],
                    unit,
                )
            )
    return ranges


def target_time_occurrences(num_frames: int, hist_len: int, pred_len: int) -> list[int]:
    total_windows = num_frames - (hist_len + pred_len) + 1
    if total_windows <= 0:
        raise ValueError(
            "split has {} frames, but hist_len={} and pred_len={} leave no windows".format(
                num_frames,
                hist_len,
                pred_len,
            )
        )
    counter: Counter[int] = Counter()
    for window_start in range(total_windows):
        for offset in range(pred_len):
            counter[window_start + hist_len + offset] += 1

    expanded = []
    for time_index in sorted(counter):
        expanded.extend([time_index] * counter[time_index])
    return expanded


def physical_chunk(
    data: np.ndarray,
    time_index: int,
    y_start: int,
    y_stop: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    chunk = np.array(data[time_index, :, y_start:y_stop, :, :], dtype=np.float32, copy=True)
    chunk *= std[:, None, None, None]
    chunk += mean[:, None, None, None]
    return chunk


def analyze(config: dict, split: str, data_root: Path | None, y_chunk: int) -> dict:
    dataset = str(config["dataset"])
    root = Path(config.get("data_root", "dataset")) if data_root is None else data_root
    dataset_dir = root.expanduser() / dataset
    hist_len = int(config["hist_len"])
    pred_len = int(config["pred_len"])
    thresholds = validate_thresholds(config)
    grid_spacing_m = load_grid_spacing_m(dataset_dir)
    dy_m, dx_m, dz_m = grid_spacing_m

    if y_chunk <= 0:
        raise ValueError("--y-chunk must be > 0")

    data = np.load(dataset_dir / "{}_data.npy".format(split), mmap_mode="r", allow_pickle=False)
    with np.load(dataset_dir / "stats.npz") as stats:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)

    if data.ndim != 5 or data.shape[1] != 3:
        raise ValueError("expected data layout [T,3,Y,X,Z], got {}".format(data.shape))

    num_frames, _, height, width, depth = data.shape
    target_indices = target_time_occurrences(num_frames, hist_len, pred_len)
    counts = {head: np.zeros(len(thresholds[head]) + 1, dtype=np.int64) for head in HEAD_ORDER}

    for occurrence_index, time_index in enumerate(target_indices, start=1):
        for y_start in range(0, height, y_chunk):
            y_stop = min(y_start + y_chunk, height)
            extra_stop = min(y_stop + 1, height)
            chunk = physical_chunk(data, time_index, y_start, extra_stop, mean, std)
            y_count = y_stop - y_start
            main = chunk[:, :y_count, :, :]

            speed = np.sqrt(np.maximum(np.sum(main * main, axis=0), 0.0))
            counts["speed"] += count_bins(speed, thresholds["speed"])

            delta_x = main[:, :, 1:, :] - main[:, :, :-1, :]
            shear_x = np.sqrt(np.maximum(np.sum(delta_x * delta_x, axis=0), 0.0)) / dx_m
            counts["shear_x"] += count_bins(shear_x, thresholds["shear_x"])
            counts["shear_x"] += count_bins(shear_x[:, -1:, :], thresholds["shear_x"])

            delta_z = main[:, :, :, 1:] - main[:, :, :, :-1]
            shear_z = np.sqrt(np.maximum(np.sum(delta_z * delta_z, axis=0), 0.0)) / dz_m
            counts["shear_z"] += count_bins(shear_z, thresholds["shear_z"])
            counts["shear_z"] += count_bins(shear_z[:, :, -1:], thresholds["shear_z"])

            if y_stop < height:
                delta_y = chunk[:, 1 : y_count + 1, :, :] - chunk[:, :y_count, :, :]
                shear_y = np.sqrt(np.maximum(np.sum(delta_y * delta_y, axis=0), 0.0)) / dy_m
                counts["shear_y"] += count_bins(shear_y, thresholds["shear_y"])
            elif y_count > 1:
                delta_y = main[:, 1:, :, :] - main[:, :-1, :, :]
                shear_y = np.sqrt(np.maximum(np.sum(delta_y * delta_y, axis=0), 0.0)) / dy_m
                counts["shear_y"] += count_bins(shear_y, thresholds["shear_y"])
                counts["shear_y"] += count_bins(shear_y[-1:, :, :], thresholds["shear_y"])
            else:
                prev = physical_chunk(data, time_index, height - 2, height, mean, std)
                delta_y = prev[:, 1:, :, :] - prev[:, :-1, :, :]
                shear_y = np.sqrt(np.maximum(np.sum(delta_y * delta_y, axis=0), 0.0)) / dy_m
                counts["shear_y"] += count_bins(shear_y, thresholds["shear_y"])

        if occurrence_index % 10 == 0 or occurrence_index == len(target_indices):
            print(
                "processed {}/{} target occurrences".format(occurrence_index, len(target_indices)),
                flush=True,
            )

    units = {
        "speed": "m/s",
        "shear_x": "m/s per 1m",
        "shear_y": "m/s per 1m",
        "shear_z": "m/s per 1m",
    }
    summary = {
        "dataset": dataset,
        "split": split,
        "dataset_dir": str(dataset_dir),
        "data_shape": list(data.shape),
        "hist_len": hist_len,
        "pred_len": pred_len,
        "target_occurrences": len(target_indices),
        "grid_points_per_target": int(height * width * depth),
        "grid_spacing_m": list(grid_spacing_m),
        "risk_bins": {key: list(values) for key, values in thresholds.items()},
        "heads": {},
    }
    for head in HEAD_ORDER:
        head_counts = counts[head]
        total = int(head_counts.sum())
        summary["heads"][head] = {
            "unit": units[head],
            "class_ranges": class_ranges(thresholds[head], units[head]),
            "counts": [int(value) for value in head_counts.tolist()],
            "ratios": [float(value / total) if total else 0.0 for value in head_counts.tolist()],
        }
    summary["recommended_risk_class_weights"] = build_recommended_class_weights(counts)
    return summary


def print_summary(summary: dict) -> None:
    print(
        json.dumps(
            {
                "dataset": summary["dataset"],
                "split": summary["split"],
                "data_shape": summary["data_shape"],
                "hist_len": summary["hist_len"],
                "pred_len": summary["pred_len"],
                "target_occurrences": summary["target_occurrences"],
                "grid_points_per_target": summary["grid_points_per_target"],
                "grid_spacing_m": summary["grid_spacing_m"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for head, item in summary["heads"].items():
        print("\n{}".format(head))
        print("| Class | Range | Count | Ratio |")
        print("| --- | --- | ---: | ---: |")
        for class_index, (range_text, count, ratio) in enumerate(
            zip(item["class_ranges"], item["counts"], item["ratios"])
        ):
            print("| {} | {} | {} | {:.6%} |".format(class_index, range_text, count, ratio))
    print("\nrecommended_risk_class_weights")
    print(summary["recommended_risk_class_weights"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = analyze(config, args.split, args.data_root, args.y_chunk)
    print_summary(summary)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
