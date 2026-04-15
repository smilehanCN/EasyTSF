from ray import tune


benchmark_config = {
    "name": "mixlinear_etth1",
    "search_save_dir": "save/benchmarks/mixlinear_etth1",
    "search_config": {
        "num_samples": 1,
        "cpus_per_trial": 2,
        "gpus_per_trial": 0.5,
        "num_gpus": 4,
    },
    "experiment": "config/experiments/mixlinear/etth1.yaml",
    "param_space": {
        "hist_len": 720,
        "pred_len": tune.grid_search([96, 192]),
        "period_len": 24,
        "lpf": tune.grid_search([1, 5]),
        "alpha": 0.95,
        "lr": 0.03,
    },
}
