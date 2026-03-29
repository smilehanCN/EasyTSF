from ray import tune


benchmark = {
    "name": "tqnet_electricity",
    "seeds": [0, 1, 2],
    "search_config": {
        "num_samples": 20,
        "cpus_per_trial": 2,
        "gpus_per_trial": 0.5,
        "num_gpus": 4,
    },
    "experiment": "tqnet/electricity",
    "param_space": {
        "hist_len": 96,
        "pred_len": tune.grid_search([12, 24]),
        "d_model": tune.choice([256, 384, 512, 768]),
        "dropout": tune.choice([0.0, 0.05, 0.1, 0.2, 0.3]),
        "lr": tune.loguniform(5e-4, 5e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "channel_aggre_heads": tune.choice([2, 4, 8]),
        "gradient_clip_val": tune.choice([0.0, 0.5, 1.0]),
        "use_revin": tune.choice([False, True]),
    },
}
