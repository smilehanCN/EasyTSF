# AGENT.md

This repository is optimized for rapid multivariate time-series forecasting research. Keep it small, explicit, and `mtsf`-only.

## Hard Boundaries

- Maintain only `mtsf`.
- Do not reintroduce grid tasks, graph side inputs, `npz` dataset compatibility, or generic TSF abstractions.
- Do not expand the repository contract just to mimic an external codebase.
- Prefer explicit config, direct code paths, and flat constructor arguments over framework-style indirection.
- Delete dead compatibility code instead of preserving legacy abstractions.

If an external model needs inputs or runtime state that the current `mtsf` path does not provide, stop with an incompatibility report instead of hard-migrating it.

## Repo Contract

### Config

- `config/experiments/<model_id>/*.yaml`: runnable experiment presets
- `config/benchmarks/<model_id>/*.py`: benchmark configs
- Config merge priority is fixed: `experiment preset < runtime overrides`
- Every experiment preset must be self-contained and explicitly set `task: mtsf`
- Model constructor arguments are read from flat config keys by name via `MTSFTask._build_model()`

### Data

- `easytsf/data/mts_data_module.py` contains `MTSDataModule`
- Dataset layout is directory-based:
  - `train_data.npy`
  - `val_data.npy`
  - `test_data.npy`
  - `train_timestamps.npy`
  - `val_timestamps.npy`
  - `test_timestamps.npy`
  - `meta.json` with `frequency (minutes)` and `timestamps_description`
- The maintained path is sequence-only; it does not inject graph side inputs or other extra tensors

### Task

- `easytsf/task/mtsf.py` contains `MTSFTask`
- The task preprocesses batches, scales variables, builds labels, and instantiates the model from explicit constructor arguments
- Model forward must use `forward(var_x, marker_x, marker_y)`
- Prediction must stay label-compatible, typically `[B, pred_len, N]`

### Workflow

- `easytsf/workflow/experiment.py` contains single-experiment orchestration
- `easytsf/workflow/benchmark.py` contains benchmark orchestration
- `easytsf/workflow/report.py` summarizes benchmark outputs
- Keep workflow logic out of `data`, `task`, and `model`

## Migration Playbook

When the user wants to migrate a model from another repository:

1. Read the external model code first: constructor, `forward`, helper modules, and config fragments.
2. Read the EasyTSF contract before editing: current task, data path, and config flow.
3. Decide compatibility early. If the source model depends on graph side input, decoder caches, custom datamodule state, or other unsupported inputs, stop and write an incompatibility report.
4. If compatible, map external constructor arguments onto explicit `Model.__init__` parameters that can be passed from flat experiment config keys.
5. Adapt the forward path to `forward(var_x, marker_x, marker_y)`. Ignore unused markers explicitly with `del`.
6. Normalize tensor layout inside the model when needed, but keep the public output shape label-compatible, typically `[B, pred_len, N]`.
7. Wire the model into `easytsf/model/registry.py`, add one runnable experiment preset, and sync public docs if the maintained surface changed.

## Add a Maintained Model

Definition of done for a maintained model path:

1. Add `easytsf/model/<model_id>.py` with a top-level `Model` class.
2. Register it in `easytsf/model/registry.py`.
3. Add at least one runnable preset under `config/experiments/<model_id>/`.
4. Add benchmark wiring only if the model is ready for search on the maintained path.
5. Update `README.md` and `docs/readme_cn.md` when public onboarding or support status changed.
6. Run `python -m compileall easytsf`.

Keep model-private helpers inside the model file unless there is real cross-model reuse.

## Change Guidelines

- Change the minimum surface necessary for the current `mtsf` path.
- If a code path exists only for historical compatibility and the current path does not use it, delete it.
- Do not add defensive pre-validation for cases that Python, NumPy, or PyTorch already reject naturally.
- Keep explicit checks for structure or semantic mismatches that would otherwise fail silently.
- Documentation must describe the current repository state, not removed features.

## Validation Guidelines

Prefer lightweight validation:

1. `python -m compileall easytsf`
2. A small smoke experiment if local data is available

Do not run `pytest`.
