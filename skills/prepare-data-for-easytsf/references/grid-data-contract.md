# Grid Prediction Data Contract

This contract is a design target for repository expansion.

## Minimum artifacts

Recommended dataset layout:

```text
<data_root>/<dataset_name>/
  train_data.npy
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  meta.json
```

Optional but common:

- `stats.npz` with `mean` and `std` for explicitly standardized storage
- `grid_mask.npy`
- coordinate metadata
- static cell features

## Core expectations

- temporal arrays should preserve grid structure, for example `[L, H, W]` or `[L, H, W, C]`
- `meta.json` should describe:
  - temporal frequency
  - grid shape
  - channel semantics when present
  - timestamp feature descriptions when time markers exist
- `meta.json` may optionally define `data_is_standardized`
  - `true`: `*_data.npy` is already standardized and `stats.npz` is required
  - missing or `false`: `*_data.npy` is treated as raw storage and runtime fits scaling stats from `train_data.npy`
- `stats.npz` is ignored unless `data_is_standardized` is explicitly `true`

## Extension-plan outputs

When the current repository cannot run the task yet, the skill should specify:

- how the datamodule should emit grid tensors
- whether static masks or coordinates are task-defining
- candidate config fields such as:
  - `task: grid_prediction`
  - grid shape
  - channel count
  - mask or coordinate switches
- which current sequence-only assumptions must be removed
