from ray import tune


benchmark_config = {
    "name": "arrow_weatherbench",
    "search_save_dir": "save/benchmarks/arrow_weatherbench",
    "search_config": {
        "num_samples": 1,
        "cpus_per_trial": 2,
        "gpus_per_trial": 0.5,
        "num_gpus": 4,
    },
    "experiment": "config/experiments/arrow/weatherbench.yaml",
    "param_space": {
        "arrow_hidden_size": tune.grid_search([768, 1024]),
        "arrow_depth": tune.grid_search([12, 16]),
        "batch_size": tune.grid_search([2, 4]),
        "lr": tune.grid_search([1e-4, 3e-4]),
    },
}
