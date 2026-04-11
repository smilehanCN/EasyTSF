from ray import tune


benchmark_config = {
    "name": "pcmlp_traffic",
    "search_save_dir": "save/benchmarks/pcmlp_traffic",
    "search_config": {
        "num_samples": 1,
        "cpus_per_trial": 2,
        "gpus_per_trial": 0.5,
        "num_gpus": 4,
    },
    "experiment": "config/experiments/pcmlp/traffic.yaml",
    "param_space": {
        "hist_len": 96,
        "pred_len": 96,
        "patch_size": tune.choice([8, 16, 24]),
        "patch_step": 8,
        "init_dim": tune.choice([128, 256, 512]),
        "dim_assign_alg": tune.choice(["uniform", "step2", "step4"]),
        "head_drop": tune.choice([0.0, 0.1, 0.2]),
        "encoder_drop": tune.choice([0.0, 0.1, 0.2]),
        "lr": tune.loguniform(5e-5, 5e-4),
    },
}
