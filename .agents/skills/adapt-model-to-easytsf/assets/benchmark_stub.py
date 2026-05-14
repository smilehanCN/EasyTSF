from ray import tune


benchmark_config = {
    "name": "your_model_your_dataset",
    "search_save_dir": "save/benchmarks/your_model_your_dataset",
    "search_config": {
        "num_samples": 20,
        "cpus_per_trial": 2,
        "gpus_per_trial": 0.5,
        "num_gpus": 4,
    },
    "experiment": "config/experiments/your_model/your_dataset.yaml",
    "param_space": {
        # Prefer real high-impact hyperparameters from source code or close baselines.
        # If the task is not yet runnable, keep this as a planning draft and replace placeholders with validated ranges.
        "hist_len": 96,
        "pred_len": tune.grid_search([96, 192]),
        "lr": tune.loguniform(5e-4, 5e-3),
        # Example task-specific placeholders:
        # "graph_topology": "replace_with_validated_topology_ref",
        # "grid_shape": "replace_with_validated_grid_shape",
    },
}
