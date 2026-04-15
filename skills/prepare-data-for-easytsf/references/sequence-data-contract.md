# Sequence Prediction Data Contract

This is the closest contract to the current runnable EasyTSF path.

## Directory layout

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

## Core expectations

- `*_data.npy` is usually `[L, N]`
- univariate data should still be stored as `[L, 1]`
- `*_timestamps.npy` shares the same leading length `L`
- `meta.json` should define:
  - `frequency (minutes)`
  - `timestamps_description`

## Config mapping

When the data fits the current repository path, map:

- `dataset`: dataset folder name
- `data_root`: parent folder
- `var_num`: second dimension `N`
- `time_feature_descriptions`: subset of `timestamps_description`
- task-aware recommendation: `sequence_prediction`
- current runtime note: the existing code path is still implemented through `task: mtsf`

## Fields not inferable from raw data

Do not guess these directly from the files:

- `hist_len`
- `pred_len`
- model architecture keys
- training hyperparameters

Those belong to experiment design, not raw data validation.
