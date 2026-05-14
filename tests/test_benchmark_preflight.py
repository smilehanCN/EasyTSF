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


def _write_valid_experiment(path: Path) -> None:
    experiment_conf = {
        "model": "PCMLP",
        "task": "mtsf",
        "val_metric": "val/loss",
        "dataset": "dummy_dataset",
        "save_root": str(path.parent / "save"),
        "seed": 42,
    }
    path.write_text(yaml.safe_dump(experiment_conf, sort_keys=True), encoding="utf-8")


def test_load_benchmark_supports_defaults_and_benchmark_relative_experiment(tmp_path):
    config_dir = tmp_path / "benchmarks"
    config_dir.mkdir()
    experiment_path = tmp_path / "experiment.yaml"
    benchmark_path = config_dir / "ray_benchmark.py"
    _write_valid_experiment(experiment_path)

    benchmark_path.write_text(
        "\n".join(
            [
                "benchmark_config = {",
                "    'name': 'ray_smoke',",
                "    'experiment': '../experiment.yaml',",
                "    'param_space': {},",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    benchmark_conf = load_benchmark(str(benchmark_path))

    assert benchmark_conf["name"] == "ray_smoke"
    assert benchmark_conf["experiment_path"] == str(experiment_path.resolve())
    assert benchmark_conf["search_config"]["backend"] == "ray"
    assert benchmark_conf["search_config"]["num_samples"] == 1
    assert Path(benchmark_conf["search_save_dir"]).name == "ray_smoke"
