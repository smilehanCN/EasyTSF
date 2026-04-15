from pathlib import Path

import pytest
import yaml

from easytsf.workflow.benchmark import load_benchmark


def test_load_benchmark_runs_preflight_before_any_tune_execution(tmp_path):
    experiment_path = tmp_path / "invalid_experiment.yaml"
    benchmark_path = tmp_path / "invalid_benchmark.py"

    experiment_conf = {
        "model": "unet3d",
        "task": "mtsf",
        "val_metric": "val/loss",
        "dataset": "dummy_dataset",
        "save_root": str(tmp_path / "save"),
        "seed": 42,
    }
    experiment_path.write_text(yaml.safe_dump(experiment_conf, sort_keys=True), encoding="utf-8")

    benchmark_path.write_text(
        "\n".join(
            [
                "benchmark_config = {",
                "    'name': 'invalid_pair',",
                "    'search_save_dir': {!r},".format(str(tmp_path / "search")),
                "    'search_config': {",
                "        'num_samples': 1,",
                "        'cpus_per_trial': 1,",
                "        'gpus_per_trial': 0.0,",
                "        'num_gpus': 0,",
                "    },",
                "    'experiment': {!r},".format(str(experiment_path)),
                "    'param_space': {},",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not supported for task 'mtsf'"):
        load_benchmark(str(benchmark_path))
