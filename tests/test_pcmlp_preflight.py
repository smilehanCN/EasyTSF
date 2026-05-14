from pathlib import Path

from easytsf.model.pcmlp import EncoderMLP, Model
from easytsf.task import get_task_spec, validate_task_runtime_conf
from easytsf.workflow.benchmark import load_benchmark
from easytsf.workflow.config import load_experiment_config


def test_pcmlp_constructor_smoke():
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

    assert isinstance(model.encoder, EncoderMLP)


def test_pcmlp_representative_experiment_and_benchmark_preflight():
    experiment_path = "config/experiments/pcmlp/etth1.yaml"
    benchmark_path = "config/benchmarks/pcmlp/etth1.py"

    experiment_conf = load_experiment_config(experiment_path)
    benchmark_conf = load_benchmark(benchmark_path)

    assert experiment_conf["model"] == "PCMLP"
    assert experiment_conf["task"] == "mtsf"
    assert validate_task_runtime_conf(experiment_conf) == get_task_spec("mtsf")
    assert benchmark_conf["task_name"] == "mtsf"
    assert benchmark_conf["base_conf"]["model"] == "PCMLP"
    assert Path(benchmark_conf["search_save_dir"]).name.startswith("pcmlp_")
