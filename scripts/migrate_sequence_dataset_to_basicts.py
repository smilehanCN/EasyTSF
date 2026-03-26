import argparse
import json
import pickle
from pathlib import Path

import numpy as np


FEATURE_NAME_MAP = {
    "tod": "time of day",
    "dow": "day of week",
    "dom": "day of month",
    "doy": "day of year",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate a legacy sequence dataset directory to BasicTS-style files.")
    parser.add_argument("--data_root", default="dataset", type=str, help="Dataset root directory.")
    parser.add_argument("--dataset_name", required=True, type=str, help="Dataset directory name under data_root.")
    parser.add_argument("--split_lengths", required=True, type=str, help="Comma-separated train,val,test lengths.")
    parser.add_argument("--freq", required=True, type=int, help="Dataset frequency in minutes.")
    parser.add_argument(
        "--timestamp_features",
        default="tod,dow",
        type=str,
        help="Comma-separated timestamp features from {tod,dow,dom,doy}. Use an empty string to skip timestamp files.",
    )
    parser.add_argument(
        "--graph_source",
        default=None,
        type=str,
        help="Optional graph source path. Defaults to <dataset_dir>/graph.npy if it exists.",
    )
    parser.add_argument("--output_name", default=None, type=str, help="Optional dataset name written into meta.json.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files and meta.json.")
    return parser.parse_args()


def parse_split_lengths(raw_value):
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--split_lengths must contain exactly three comma-separated integers")
    split_lengths = [int(part) for part in parts]
    if any(item <= 0 for item in split_lengths):
        raise ValueError("--split_lengths must contain positive integers")
    return split_lengths


def parse_timestamp_features(raw_value):
    if raw_value is None:
        return []
    parts = [part.strip().lower() for part in raw_value.split(",") if part.strip()]
    for part in parts:
        if part not in FEATURE_NAME_MAP:
            raise ValueError("unsupported timestamp feature: {}".format(part))
    return parts


def load_legacy_dataset(dataset_dir):
    data_path = Path(dataset_dir) / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError("legacy dataset must contain data.npz: {}".format(data_path))
    with np.load(data_path) as data:
        if "scaled_variable" not in data or "timestamp" not in data:
            raise KeyError("legacy data.npz must contain 'scaled_variable' and 'timestamp': {}".format(data_path))
        variable = data["scaled_variable"].astype(np.float32, copy=False)
        timestamp = data["timestamp"]
    if variable.ndim != 2:
        raise ValueError("legacy sequence migration expects scaled_variable with shape [L, N], got {}".format(variable.shape))
    if len(variable) != len(timestamp):
        raise ValueError("timestamp length {} does not match data length {}".format(len(timestamp), len(variable)))
    return variable, timestamp


def _day_indices(timestamp_minutes):
    day_values = timestamp_minutes.astype("datetime64[D]")
    return day_values.astype(np.int64)


def build_basic_ts_timestamps(raw_timestamp, feature_names, freq):
    if len(feature_names) == 0:
        return None, []

    timestamp = np.asarray(raw_timestamp).astype("datetime64[m]")
    day_values = timestamp.astype("datetime64[D]")
    day_indices = _day_indices(timestamp)
    minute_values = (timestamp - day_values).astype("timedelta64[m]").astype(np.int64)
    month_start = timestamp.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    year_start = timestamp.astype("datetime64[Y]").astype("datetime64[D]").astype(np.int64)
    steps_per_day = int((24 * 60) / int(freq))

    features = []
    descriptions = []
    for feature_name in feature_names:
        descriptions.append(FEATURE_NAME_MAP[feature_name])
        if feature_name == "tod":
            values = (minute_values / int(freq)) / steps_per_day
        elif feature_name == "dow":
            values = ((day_indices + 3) % 7) / 7.0
        elif feature_name == "dom":
            values = (day_indices - month_start) / 31.0
        else:
            values = (day_indices - year_start) / 366.0
        features.append(values.astype(np.float32))

    return np.stack(features, axis=-1).astype(np.float32), descriptions


def load_graph(graph_source):
    graph_source = Path(graph_source)
    loaded = np.load(graph_source, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "graph" not in loaded:
                raise KeyError("graph npz must contain 'graph': {}".format(graph_source))
            graph = loaded["graph"]
        finally:
            loaded.close()
    else:
        graph = loaded
    return np.asarray(graph, dtype=np.float32)


def ensure_writable(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError("refusing to overwrite existing file without --overwrite: {}".format(path))


def save_split_files(dataset_dir, variable, timestamps, split_lengths, overwrite):
    offsets = np.cumsum([0] + split_lengths)
    for split_name, start, end in zip(("train", "val", "test"), offsets[:-1], offsets[1:]):
        data_path = dataset_dir / "{}_data.npy".format(split_name)
        ensure_writable(data_path, overwrite)
        np.save(data_path, variable[start:end].astype(np.float32))

        timestamp_path = dataset_dir / "{}_timestamps.npy".format(split_name)
        if timestamps is not None:
            ensure_writable(timestamp_path, overwrite)
            np.save(timestamp_path, timestamps[start:end].astype(np.float32))
        elif overwrite and timestamp_path.exists():
            timestamp_path.unlink()


def maybe_save_graph(dataset_dir, graph_source, overwrite):
    if graph_source is None:
        default_source = dataset_dir / "graph.npy"
        if not default_source.exists():
            graph_path = dataset_dir / "adj_mx.pkl"
            if overwrite and graph_path.exists():
                graph_path.unlink()
            return False
        graph_source = default_source

    graph = load_graph(graph_source)
    graph_path = dataset_dir / "adj_mx.pkl"
    ensure_writable(graph_path, overwrite)
    with graph_path.open("wb") as handle:
        pickle.dump(graph, handle)
    return True


def save_meta(dataset_dir, dataset_name, freq, variable, split_lengths, timestamps, descriptions, has_graph, overwrite):
    meta_path = dataset_dir / "meta.json"
    ensure_writable(meta_path, overwrite)
    meta = {
        "name": dataset_name,
        "frequency (minutes)": int(freq),
        "shape": [int(variable.shape[0]), int(variable.shape[1])],
        "num_time_steps": int(variable.shape[0]),
        "num_vars": int(variable.shape[1]),
        "split_lengths": [int(item) for item in split_lengths],
        "has_graph": bool(has_graph),
        "timestamps_description": list(descriptions),
        "regular_settings": {},
    }
    if timestamps is not None:
        meta["timestamps_shape"] = [int(timestamps.shape[0]), int(timestamps.shape[1])]
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    dataset_dir = data_root / args.dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError("dataset directory not found: {}".format(dataset_dir))

    split_lengths = parse_split_lengths(args.split_lengths)
    feature_names = parse_timestamp_features(args.timestamp_features)
    dataset_name = args.output_name or args.dataset_name

    variable, raw_timestamp = load_legacy_dataset(dataset_dir)
    if sum(split_lengths) != int(len(variable)):
        raise ValueError(
            "split_lengths {} do not sum to dataset length {} for {}".format(
                split_lengths,
                int(len(variable)),
                dataset_dir,
            )
        )

    timestamps, descriptions = build_basic_ts_timestamps(raw_timestamp, feature_names, args.freq)
    save_split_files(dataset_dir, variable, timestamps, split_lengths, overwrite=args.overwrite)
    has_graph = maybe_save_graph(dataset_dir, args.graph_source, overwrite=args.overwrite)
    save_meta(
        dataset_dir,
        dataset_name=dataset_name,
        freq=args.freq,
        variable=variable,
        split_lengths=split_lengths,
        timestamps=timestamps,
        descriptions=descriptions,
        has_graph=has_graph,
        overwrite=args.overwrite,
    )
    print("Migrated dataset '{}' under {}".format(dataset_name, dataset_dir))


if __name__ == "__main__":
    main()
