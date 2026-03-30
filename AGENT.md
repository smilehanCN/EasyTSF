# AGENT.md

This repository is optimized for rapid multivariate time-series forecasting research. Keep it small, explicit, and `mtsf`-only.

## Mission

- Support fast model iteration on the maintained `mtsf` path.
- Keep experiment and benchmark workflow reproducible and easy to resume.
- Delete dead compatibility code instead of preserving old abstractions.

## Current Boundaries

- Maintain only `mtsf`.
- Do not reintroduce grid tasks, graph side inputs, `npz` dataset compatibility, or generic TSF abstractions.
- Do not rebuild old runner variants, plugin systems, or large registry trees.
- Prefer explicit config and direct code paths over framework-style indirection.

## Architecture Map

### Config

- `config/experiments/<model_id>/*.yaml`: runnable experiment presets
- `config/benchmarks/<model_id>/*.py`: benchmark configs

Config merge priority is fixed:

`experiment + runtime overrides`

Every experiment preset must be self-contained and explicitly set `task_name: mtsf`.

### Data

- `easytsf/data/mts_data_module.py` contains `MTSDataModule`.
- Dataset layout is directory-based:
  - `train_data.npy`
  - `val_data.npy`
  - `test_data.npy`
  - `train_timestamps.npy`
  - `val_timestamps.npy`
  - `test_timestamps.npy`
  - `meta.json` with frequency and `timestamps_description`

### Task

- `easytsf/task/mtsf.py` contains `MTSFTask`.
- Keep task semantics simple: model forward uses `forward(var_x, marker_x, marker_y)`.

### Model

- `easytsf/model/<model_id>.py` should define a top-level `Model` class.
- The maintained output shape should stay label-compatible: typically `[B, pred_len, N]`.
- Keep model-private helpers inside the model file unless there is real cross-model reuse.

### Workflow

- `easytsf/workflow/experiment.py` contains single-experiment orchestration.
- `easytsf/workflow/benchmark.py` contains benchmark orchestration.
- Keep workflow logic out of `data`, `task`, and `model`.

## Change Guidelines

- When adding a maintained model, change only the model file and matching experiment presets when possible.
- If a code path exists only for historical compatibility and the current `mtsf` path does not use it, delete it.
- For this research codebase, do not add explicit checks or defensive pre-validation for cases that Python/NumPy/PyTorch will naturally reject at runtime.
- Keep explicit checks limited to format or structure consistency issues that Python cannot naturally surface, especially when they prevent silent semantic misuse such as config meaning not matching dataset meaning.
- Documentation must describe the current repository state, not removed features.

## Validation Guidelines

Prefer lightweight validation:

1. `conda activate easytsf`
2. `python -m compileall easytsf`

Do not run `pytest`.
