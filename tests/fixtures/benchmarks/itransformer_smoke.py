from ray import tune


benchmark = {
    "name": "itransformer_smoke",
    "experiment": "tests/fixtures/experiments/itransformer_smoke.yaml",
    "seeds": [0, 1],
    "search_config": {
        "num_samples": 1,
        "cpus_per_trial": 1,
        "gpus_per_trial": 0.0,
        "num_gpus": 0,
    },
    "param_space": {
        "d_model": tune.grid_search([8]),
        "dropout": tune.grid_search([0.0]),
    },
}
