import os

from ray import tune


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _devices_per_trial() -> int:
    raw = os.environ.get("BENCH_DEVICES_PER_TRIAL")
    if raw is None:
        raw = os.environ.get("BENCH_GPUS_PER_TRIAL", "1")
    return max(1, int(float(raw)))


def _strategy_for_devices() -> str:
    return "auto" if _devices_per_trial() <= 1 else "ddp"


def _run_tag() -> str:
    return str(os.environ.get("WSH_RUN_TAG", "default"))


def _batch_size() -> int:
    return max(1, _env_int("WSH_BATCH_SIZE", 1))


def _precision() -> str:
    return str(os.environ.get("WSH_PRECISION", "bf16-mixed"))

def _max_epochs() -> int:
    return max(1, _env_int("WSH_MAX_EPOCHS", 50))


CASES = {
    "lr1e-05_e48": {"lr": 1e-5, "afno_embed_dim": 48},
    "lr2e-05_e48": {"lr": 2e-5, "afno_embed_dim": 48},
    "lr5e-05_e48": {"lr": 5e-5, "afno_embed_dim": 48},
    "lr1e-04_e48": {"lr": 1e-4, "afno_embed_dim": 48},
    "lr1e-05_e64": {"lr": 1e-5, "afno_embed_dim": 64},
    "lr2e-05_e64": {"lr": 2e-5, "afno_embed_dim": 64},
    "lr5e-05_e64": {"lr": 5e-5, "afno_embed_dim": 64},
    "lr1e-04_e64": {"lr": 1e-4, "afno_embed_dim": 64},
}


def _pick(case_key: str):
    return tune.sample_from(lambda cfg: CASES[cfg["case_id"]][case_key])


benchmark_config = {
    "name": "afno3d_windshear_v1_0417_h10_p10_tune_{}".format(_run_tag()),
    "search_save_dir": "save/benchmarks/afno3d_windshear_v1_0417_h10_p10_tune_{}".format(_run_tag()),
    "search_config": {
        "num_samples": 1,
        "cpus_per_trial": _env_int("BENCH_CPUS_PER_TRIAL", 4),
        "gpus_per_trial": float(_devices_per_trial()),
        "num_gpus": _env_int("BENCH_NUM_GPUS", _devices_per_trial()),
    },
    "experiment": "config/experiments/afno3d/windshear_v1_0417_h10_p10.yaml",
    "param_space": {
        "case_id": tune.grid_search(list(CASES.keys())),
        "tune_tag": tune.sample_from(lambda cfg: cfg["case_id"]),
        "seed": tune.grid_search([42]),
        "batch_size": _batch_size(),
        "devices": _devices_per_trial(),
        "strategy": _strategy_for_devices(),
        "precision": _precision(),
        "max_epochs": _max_epochs(),
        "lr": _pick("lr"),
        "afno_embed_dim": _pick("afno_embed_dim"),
    },
}
