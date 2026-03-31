# Shape and Config Mapping

## Constructor Mapping Rule

EasyTSF does not pass a nested model config object into the model.

`MTSFTask._build_model()` inspects `Model.__init__` and passes flat config keys by exact parameter name. That means:

- every required constructor parameter must have a matching flat config key
- optional constructor parameters should use explicit Python defaults
- avoid relying on `**kwargs` as the primary contract

## Common Key Mapping

Typical external concepts map onto EasyTSF keys like this:

| External concept | EasyTSF key |
| --- | --- |
| model name | `model` |
| dataset name | `dataset` |
| history length / input length / seq len | `hist_len` |
| prediction length / horizon | `pred_len` |
| number of variables / channels / nodes | `var_num` |
| learning rate | `lr` |
| optimizer type | `optimizer` |
| scheduler type | `lr_scheduler` |
| batch size | `batch_size` |
| epoch count | `max_epochs` |

Model-specific hyperparameters should stay as flat keys under the `# model` block in the experiment preset.

## Tensor Shape Mapping

EasyTSF task-facing shapes are typically:

- `var_x`: `[B, hist_len, N]`
- `marker_x`: `[B, hist_len, T]`
- `marker_y`: `[B, pred_len, T]`
- label: `[B, pred_len, N]`

Migration rules:

- If the source model expects `[B, N, L]`, transpose inside the model.
- If the source model expects only history time features, use `marker_x`.
- If the source model derives a future seasonal or cycle index, use `marker_y`.
- If the source model does not use markers, delete them explicitly and keep the public signature unchanged.
- If the source model returns `[B, N, pred_len]`, transpose before returning.

## Config Section Mapping

EasyTSF experiment presets are flat YAML files that usually follow these comment blocks:

- `# model`
- `# data`
- `# train`
- `# runtime`

Use this split:

- `# model`: architecture hyperparameters and model-specific switches
- `# data`: dataset identifier, `hist_len`, `pred_len`, `var_num`, optional `time_feature_descriptions`
- `# train`: optimizer, scheduler, batch size, patience, metric space
- `# runtime`: `task`, dataloader settings, `data_root`, `save_root`, device settings

Do not move runtime-only concerns into model code.

## Boundary Examples

Compatible examples:

- pure sequence forecasting models
- models that only need history values and optional time features
- models that can normalize or transpose internally without changing the public task contract

Stop and report incompatibility for:

- models that need graph adjacency during initialization or forward when the current path does not inject it
- models that depend on spatial metadata not present in the maintained dataset contract
- models that require extra decoder tensors, caches, or custom rollout state
