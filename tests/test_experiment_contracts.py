from pathlib import Path

import pytest

from experiment import load_experiment_config
from tests.helpers import REPO_ROOT, SMOKE_EXPERIMENT_PATH, load_yaml, write_yaml


def test_experiment_config_requires_all_sections(tmp_path):
    config = load_yaml(SMOKE_EXPERIMENT_PATH)
    config.pop("runtime")
    config_path = tmp_path / "missing_runtime.yaml"
    write_yaml(config_path, config)

    with pytest.raises(ValueError) as exc_info:
        load_experiment_config(str(config_path))

    assert "runtime.task_name" in str(exc_info.value)


def test_experiment_config_requires_mandatory_keys(tmp_path):
    config = load_yaml(SMOKE_EXPERIMENT_PATH)
    config["train"].pop("batch_size")
    config_path = tmp_path / "missing_batch_size.yaml"
    write_yaml(config_path, config)

    with pytest.raises(ValueError) as exc_info:
        load_experiment_config(str(config_path))

    assert "train.batch_size" in str(exc_info.value)


def test_experiment_config_rejects_invalid_task_name(tmp_path):
    config = load_yaml(SMOKE_EXPERIMENT_PATH)
    config["runtime"]["task_name"] = "grid"
    config_path = tmp_path / "invalid_task.yaml"
    write_yaml(config_path, config)

    with pytest.raises(ValueError, match="unsupported task_name"):
        load_experiment_config(str(config_path))


def test_tqnet_requires_time_feature_descriptions(tmp_path):
    config = load_yaml(REPO_ROOT / "config" / "experiments" / "tqnet" / "etth1.yaml")
    config["data"].pop("time_feature_descriptions")
    config_path = tmp_path / "invalid_tqnet.yaml"
    write_yaml(config_path, config)

    with pytest.raises(ValueError) as exc_info:
        load_experiment_config(str(config_path))

    assert "time_feature_descriptions" in str(exc_info.value)
