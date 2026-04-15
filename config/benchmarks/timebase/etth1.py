from ray import tune


benchmark_config = {
    "name": "timebase_etth1",
    "search_save_dir": "save/benchmarks/timebase_etth1",
    "search_config": {
        "num_samples": 1,
        "cpus_per_trial": 2,
        "gpus_per_trial": 0.5,
        "num_gpus": 4,
    },
    "experiment": "config/experiments/timebase/etth1.yaml",
    "param_space": {
        "hist_len": 720,
        "pred_len": tune.grid_search([96, 192]),
        "period_len": 24,
        "basis_num": 6,
        "use_period_norm": True,
        "use_orthogonal": True,
        "individual": False,
        "aux_loss_weight": tune.grid_search([0.0, 0.04, 0.08, 0.12, 0.16, 0.2]),
        "batch_size": tune.grid_search([64, 256]),
        "lr": tune.grid_search([0.05, 0.1, 0.4]),
    },
}
