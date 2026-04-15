from pathlib import Path

from easytsf.model.registry import get_model_class
from easytsf.workflow.config import load_experiment_config


def test_arrow_model_is_registered_without_importing_optional_runtime_dependencies():
    model_cls = get_model_class("ARROW")
    assert model_cls.__name__ == "Model"


def test_arrow_experiment_config_loads_expected_defaults():
    experiment_conf = load_experiment_config("config/experiments/arrow/weatherbench.yaml")
    assert experiment_conf["model"] == "ARROW"
    assert experiment_conf["task"] == "weatherbench"
    assert experiment_conf["hist_len"] == 1
    assert experiment_conf["pred_len"] == 1
    assert experiment_conf["weather_loss"] == "lat_weighted_mse"
    assert experiment_conf["arrow_train_intervals"] == [6, 12, 24]
    assert Path("config/benchmarks/arrow/weatherbench.py").exists()
