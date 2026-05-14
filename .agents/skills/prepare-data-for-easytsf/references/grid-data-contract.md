# Grid Prediction Data Contract

Grid prediction data should preserve spatial structure instead of flattening grid cells into sequence features.

## Maintained Grid3D Layout

```text
<data_root>/<dataset_name>/
  train_data.npy
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  meta.json
  coord.npy        # optional, 3,Y,X,Z
  axes.npz         # optional physical axes
  stats.npz        # required only when meta.data_is_standardized=true
```

The maintained Grid3D tensor layout is:

```text
T,C,Y,X,Z
```

`Grid3DDataModule` emits:

```text
inputs:  B,hist_len,C,Y,X,Z
targets: B,pred_len,C,Y,X,Z
coords:  B,3,Y,X,Z        # when use_coords=true and coord.npy exists
```

## Required Metadata

`meta.json` should define:

- `task_type`
- `storage_format`
- `data_layout`
- `grid_shape`
- `channel_names`
- `split_lengths`
- `data_is_standardized`

For WindShear-style data, it may also define:

- `grid_spacing_m`
- `axis_layout`
- `velocity_channel_names`
- `derived_channel_names`

## Scaling Policy

- `data_is_standardized=true`: `*_data.npy` is already standardized and `stats.npz` must provide `mean` and `std`.
- Missing or false `data_is_standardized`: runtime treats `*_data.npy` as raw data and fits scaling stats from `train_data.npy`.
- `stats.npz` is ignored unless `data_is_standardized=true`.

## Extension Guidance

If a grid dataset needs masks, static features, topology, irregular coordinates, or non-3D layouts, define those artifacts explicitly and plan the matching task changes. Do not smuggle them through undocumented keys.
