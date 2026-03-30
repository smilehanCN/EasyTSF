from pathlib import Path

import pytest

import easytsf.workflow.config as workflow_config


def test_load_experiment_config_supports_absolute_path(tmp_path):
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("model_name: demo_model\n", encoding="utf-8")

    conf = workflow_config.load_experiment_config(str(config_path))

    assert conf["model_name"] == "demo_model"
    assert "task_name" not in conf


def test_load_experiment_config_supports_relative_path(tmp_path, monkeypatch):
    config_path = tmp_path / "configs" / "demo.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("dataset_name: demo_dataset\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    conf = workflow_config.load_experiment_config("configs/demo.yaml")

    assert conf["dataset_name"] == "demo_dataset"
    assert "task_name" not in conf


def test_finalize_runtime_conf_applies_overrides_and_defaults():
    conf = workflow_config.finalize_runtime_conf(
        {
            "model_name": "base_model",
            "dataset_name": "demo_dataset",
            "save_root": "save",
            "seed": 0,
        },
        overrides={"model_name": "override_model"},
    )

    assert conf["model_name"] == "override_model"
    assert conf["task_name"] == "mtsf"


def test_finalize_runtime_conf_ignores_none_override_values():
    conf = workflow_config.finalize_runtime_conf(
        {
            "model_name": "base_model",
            "dataset_name": "demo_dataset",
            "save_root": "save",
            "seed": 0,
        },
        overrides={"task_name": None},
    )

    assert conf["task_name"] == "mtsf"


def test_parse_devices_rejects_zero_device_count():
    with pytest.raises(ValueError, match="devices=0 is invalid"):
        workflow_config.parse_devices(0)

    with pytest.raises(ValueError, match="devices=0 is invalid"):
        workflow_config.parse_devices("0")


def test_load_experiment_config_no_longer_applies_overrides(tmp_path):
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("model_name: base_model\n", encoding="utf-8")

    conf = workflow_config.load_experiment_config(str(config_path))

    assert conf["model_name"] == "base_model"
