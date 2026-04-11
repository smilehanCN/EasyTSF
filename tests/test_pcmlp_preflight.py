from pathlib import Path

import pytest
import torch

from easytsf.model.pcmlp import EncoderMLP, Model
from easytsf.task import get_task_spec, validate_task_runtime_conf
from easytsf.workflow.benchmark import load_benchmark
from easytsf.workflow.config import load_experiment_config


EXPERIMENT_PATHS = (
    "config/experiments/pcmlp/etth1.yaml",
    "config/experiments/pcmlp/weather.yaml",
    "config/experiments/pcmlp/pems03.yaml",
    "config/experiments/pcmlp/traffic.yaml",
)

BENCHMARK_PATHS = (
    "config/benchmarks/pcmlp/etth1.py",
    "config/benchmarks/pcmlp/weather.py",
    "config/benchmarks/pcmlp/pems03.py",
    "config/benchmarks/pcmlp/traffic.py",
)


def test_pcmlp_constructor_accepts_valid_hparams():
    model = Model(
        hist_len=96,
        pred_len=96,
        var_num=7,
        freq=60,
        use_norm=True,
        patch_size=16,
        patch_step=8,
        init_dim=256,
        dim_assign_alg="step2",
        use_tod=True,
        use_dow=False,
        head_drop=0.1,
        encoder_drop=0.0,
        use_tokenizer_var_aware=True,
        use_encoder_var_aware=True,
        time_feature_descriptions=("time of day",),
    )
    assert isinstance(model, Model)
    assert isinstance(model.encoder, EncoderMLP)


@pytest.mark.parametrize(
    ("override", "error_match"),
    [
        ({"patch_size": 100}, "patch_size"),
        ({"patch_step": 7}, "compatible patch extraction"),
        ({"use_tod": True, "time_feature_descriptions": ()}, "time of day"),
    ],
)
def test_pcmlp_constructor_validates_shape_constraints(override, error_match):
    kwargs = {
        "hist_len": 96,
        "pred_len": 96,
        "var_num": 7,
        "freq": 60,
        "use_norm": True,
        "patch_size": 16,
        "patch_step": 8,
        "init_dim": 256,
        "dim_assign_alg": "step2",
        "use_tod": True,
        "use_dow": False,
        "head_drop": 0.1,
        "encoder_drop": 0.0,
        "use_tokenizer_var_aware": True,
        "use_encoder_var_aware": True,
        "time_feature_descriptions": ("time of day",),
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match=error_match):
        Model(**kwargs)


def test_pcmlp_runtime_shape_errors_surface_from_tensor_ops():
    model = Model(
        hist_len=96,
        pred_len=96,
        var_num=7,
        freq=60,
        use_norm=True,
        patch_size=16,
        patch_step=8,
        init_dim=256,
        dim_assign_alg="step2",
        use_tod=True,
        use_dow=False,
        head_drop=0.1,
        encoder_drop=0.0,
        use_tokenizer_var_aware=True,
        use_encoder_var_aware=True,
        time_feature_descriptions=("time of day",),
    )
    var_x = torch.randn(2, 95, 7)
    marker_x = torch.zeros(2, 95, 1)

    with pytest.raises((RuntimeError, IndexError)):
        model(var_x, marker_x, None)


@pytest.mark.parametrize("experiment_path", EXPERIMENT_PATHS)
def test_pcmlp_experiment_configs_load_and_validate(experiment_path):
    experiment_conf = load_experiment_config(experiment_path)
    assert experiment_conf["model"] == "PCMLP"
    assert experiment_conf["task"] == "mtsf"
    assert experiment_conf["hist_len"] == 96
    assert experiment_conf["pred_len"] == 96
    assert experiment_conf["max_epochs"] == 30
    assert experiment_conf["patch_size"] == 16
    assert experiment_conf["patch_step"] == 8
    assert "encoder_type" not in experiment_conf
    assert validate_task_runtime_conf(experiment_conf) == get_task_spec("mtsf")


@pytest.mark.parametrize("benchmark_path", BENCHMARK_PATHS)
def test_pcmlp_benchmarks_load_and_validate(benchmark_path):
    benchmark_conf = load_benchmark(benchmark_path)
    expected_experiment = str(Path(benchmark_path).with_suffix(".yaml")).replace("benchmarks", "experiments")

    assert benchmark_conf["task_name"] == "mtsf"
    assert benchmark_conf["base_conf"]["model"] == "PCMLP"
    assert benchmark_conf["base_conf"]["task"] == "mtsf"
    assert benchmark_conf["base_conf"]["hist_len"] == 96
    assert benchmark_conf["base_conf"]["pred_len"] == 96
    assert benchmark_conf["base_conf"]["max_epochs"] == 30
    assert Path(benchmark_conf["search_save_dir"]).name.startswith("pcmlp_")
    assert benchmark_conf["base_conf"]["dataset"].lower() in expected_experiment.lower()
