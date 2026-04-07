import pytest

from easytsf.task import get_task_spec, validate_task_runtime_conf


def test_validate_task_runtime_conf_accepts_valid_pair():
    runtime_conf = {
        "task": "weatherbench",
        "model": "WeatherBenchPersistence",
        "val_metric": "val/loss",
    }

    task_spec = validate_task_runtime_conf(runtime_conf)
    assert task_spec == get_task_spec("weatherbench")


def test_validate_task_runtime_conf_rejects_invalid_model_for_task():
    runtime_conf = {
        "task": "mtsf",
        "model": "UNet3D",
        "val_metric": "val/loss",
    }

    with pytest.raises(ValueError, match="not supported for task 'mtsf'"):
        validate_task_runtime_conf(runtime_conf)


def test_validate_task_runtime_conf_rejects_stgcn_without_graph_task_support():
    runtime_conf = {
        "task": "mtsf",
        "model": "STGCN",
        "val_metric": "val/loss",
    }

    with pytest.raises(ValueError, match="not supported for task 'mtsf'"):
        validate_task_runtime_conf(runtime_conf)


def test_validate_task_runtime_conf_rejects_invalid_val_metric():
    runtime_conf = {
        "task": "weatherbench",
        "model": "WeatherBenchPersistence",
        "val_metric": "val/mae",
    }

    with pytest.raises(ValueError, match="supported validation metrics"):
        validate_task_runtime_conf(runtime_conf)
