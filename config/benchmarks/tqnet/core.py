from ray import tune


benchmark_config = {
    "name": "tqnet_electricity",
    "search_save_dir": "save/benchmarks/tqnet_electricity",
    "search_config": {
        "num_samples": 20,
        "cpus_per_trial": 2,
        "gpus_per_trial": 0.5,
        "num_gpus": 4,
    },
    "experiment": "config/experiments/tqnet/electricity.yaml",
    "param_space": {
        "hist_len": 96,
        "pred_len": tune.grid_search([96, 192]),
        "d_model": tune.choice([256, 512, 1024]),
        "dropout": tune.choice([0.0, 0.05, 0.1, 0.2, 0.3]),
        "lr": tune.loguniform(5e-4, 5e-2),
        "channel_aggre_heads": tune.choice([2, 4, 8]),
        "use_revin": tune.choice([False, True]),
    },
}
