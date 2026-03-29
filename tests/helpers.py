import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_EXPERIMENT_PATH = REPO_ROOT / "tests" / "fixtures" / "experiments" / "itransformer_smoke.yaml"
SMOKE_BENCHMARK_PATH = REPO_ROOT / "tests" / "fixtures" / "benchmarks" / "itransformer_smoke.py"


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True, allow_unicode=True)


def create_synthetic_dataset(dataset_root, dataset_name="SmokeSet", var_num=2):
    dataset_root = Path(dataset_root)
    dataset_dir = dataset_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    def make_split(length, offset):
        index = np.arange(length, dtype=np.float32) + np.float32(offset)
        columns = []
        for feature_idx in range(var_num):
            columns.append(np.sin(index / (feature_idx + 2.0)) + feature_idx * 0.1)
        data = np.stack(columns, axis=-1).astype(np.float32)
        timestamps = (index % 24).astype(np.float32)[:, None]
        return data, timestamps

    for split_name, length, offset in (("train", 16, 0), ("val", 14, 16), ("test", 14, 30)):
        data, timestamps = make_split(length, offset)
        np.save(dataset_dir / "{}_data.npy".format(split_name), data, allow_pickle=False)
        np.save(dataset_dir / "{}_timestamps.npy".format(split_name), timestamps, allow_pickle=False)

    meta = {
        "frequency (minutes)": 60,
        "timestamps_description": ["time of day"],
    }
    with (dataset_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True, ensure_ascii=False)
    return dataset_root


def run_command(args, cwd=None, timeout=300):
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd or REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def load_last_json_line(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise AssertionError("command produced no output")
    return json.loads(lines[-1])
