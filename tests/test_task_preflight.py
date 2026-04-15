import pytest
import torch.nn as nn

from easytsf.task import get_task_spec, validate_task_runtime_conf
from easytsf.task.base import BaseForecastTask


def test_validate_task_runtime_conf_accepts_valid_pair():
    runtime_conf = {
        "task": "weatherbench",
        "model": "WeatherBenchPersistence",
        "val_metric": "val/loss",
    }

    task_spec = validate_task_runtime_conf(runtime_conf)
    assert task_spec == get_task_spec("weatherbench")


def test_validate_task_runtime_conf_accepts_pcmlp_for_mtsf():
    runtime_conf = {
        "task": "mtsf",
        "model": "PCMLP",
        "val_metric": "val/loss",
    }

    task_spec = validate_task_runtime_conf(runtime_conf)
    assert task_spec == get_task_spec("mtsf")


def test_validate_task_runtime_conf_rejects_invalid_model_for_task():
    runtime_conf = {
        "task": "mtsf",
        "model": "unet3d",
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


def test_validate_task_runtime_conf_accepts_risk_macro_f1_and_max_mode():
    runtime_conf = {
        "task": "grid3d_risk_prediction",
        "model": "unet3d",
        "val_metric": "val/macro_f1",
        "val_metric_mode": "max",
    }

    task_spec = validate_task_runtime_conf(runtime_conf)
    assert task_spec == get_task_spec("grid3d_risk_prediction")


def test_validate_task_runtime_conf_rejects_invalid_val_metric_mode():
    runtime_conf = {
        "task": "grid3d_risk_prediction",
        "model": "unet3d",
        "val_metric": "val/macro_f1",
        "val_metric_mode": "largest",
    }

    with pytest.raises(ValueError, match="val_metric_mode"):
        validate_task_runtime_conf(runtime_conf)


def test_base_forecast_task_lets_model_constructor_report_missing_args(monkeypatch):
    class DummyModel(nn.Module):
        def __init__(self, required_arg):
            super().__init__()
            self.required_arg = required_arg

        def forward(self, *args, **kwargs):
            raise NotImplementedError

    class DummyTask(BaseForecastTask):
        def postprocess_outputs(self, prediction, label, targets_mask=None):
            return prediction, label

    monkeypatch.setattr("easytsf.task.base.get_model_class", lambda name: DummyModel)

    with pytest.raises(TypeError, match="required_arg"):
        DummyTask(model="DummyModel", optimizer="Adam", lr_scheduler="StepLR")
