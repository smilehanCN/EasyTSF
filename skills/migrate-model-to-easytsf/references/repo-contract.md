# EasyTSF Repo Contract

## Maintained Path

EasyTSF currently maintains one public task path: `mtsf`.

The practical chain is:

```text
dataset -> MTSDataModule -> MTSFTask -> Model -> experiment/benchmark workflow
```

Treat this path as the source of truth when migrating a model.

## Config Flow

- Experiment presets live under `config/experiments/<model_id>/`.
- Runtime merge order is fixed: `experiment preset < runtime overrides`.
- `MTSFTask._build_model()` inspects `Model.__init__` and passes flat config keys by parameter name.
- Required constructor parameters must exist as flat config keys.
- Do not hide constructor requirements inside nested config objects or opaque `**kwargs`.

## Forward Contract

The maintained model interface is:

```python
forward(var_x, marker_x, marker_y)
```

Practical expectations:

- `var_x`: history values, typically `[B, hist_len, N]`
- `marker_x`: history time features, typically `[B, hist_len, T]`
- `marker_y`: future time features, typically `[B, pred_len, T]`
- output: prediction tensor compatible with the task label, typically `[B, pred_len, N]`

If the source model uses `[B, N, L]`, transpose internally. Do not change the public EasyTSF contract to preserve the source layout.

If the model does not use markers, ignore them explicitly with `del marker_x, marker_y` or `del marker_y`.

## Task Semantics

`MTSFTask`:

- scales inputs and targets with `StandardScaler`
- builds labels from the last `pred_len` time steps of the target window
- instantiates the model from explicit constructor arguments
- computes train, validation, and test metrics

Migration consequence:

- The model should usually operate on scaled variables
- The model does not own label construction
- The model should not assume direct access to datamodule internals

## Dataset Contract

The maintained dataset layout is:

```text
dataset/<dataset>/
  train_data.npy
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  meta.json
```

Required metadata:

- `timestamps_description`
- `frequency (minutes)`

The maintained path is sequence-only. It does not automatically load or inject graph adjacency, coordinate tensors, masks, or other side inputs.

## Boundary Rule

If the source model requires unsupported inputs or a broader repository contract, stop and report the incompatibility before editing code.
