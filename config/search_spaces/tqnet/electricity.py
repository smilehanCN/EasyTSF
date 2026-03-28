from ray import tune


# Search around the aligned upstream baseline while keeping the weekly cycle fixed.
param_space = {
    "d_model": tune.choice([256, 384, 512, 768]),
    "dropout": tune.choice([0.0, 0.05, 0.1, 0.2]),
    "lr": tune.loguniform(5e-4, 5e-3),
    "batch_size": tune.choice([16, 32, 64]),
    "channel_aggre_heads": tune.choice([2, 4, 8]),
    "gradient_clip_val": tune.choice([0.0, 0.5, 1.0]),
    "use_revin": tune.choice([True, False]),
}
