# EasyTSF Repository Convention

This document is the maintainer-facing convention for keeping EasyTSF small, explicit, and useful for forecasting research.

## Core Principle

Every maintained prediction path must define four layers:

1. data contract
2. task contract
3. model interface
4. workflow surface

Do not force a task into an existing path when its data or model interface needs different semantics.

## Task Taxonomy

- `sequence_prediction`: flat multivariate time-series windows.
- `grid_prediction`: dense grid tensors with optional coordinates or static grid metadata.
- `graph_prediction`: extension target only; not currently a runnable task.

## Maintained Runtime Surface

- `mtsf`: sequence forecasting with `MTSDataModule` and `MTSFTask`.
- `grid3d_forecasting`: full-volume 3D grid forecasting with `Grid3DDataModule`.
Grid3D shear input/output ablations are historical research recipes for now. They should not be registered as core tasks unless the task implementation is reintroduced deliberately.

The public workflow entrypoints are:

- `python -m easytsf.workflow.experiment <experiment.yaml>`
- `python -m easytsf.workflow.benchmark <benchmark.py>`
- `python -m easytsf.workflow.report <benchmark.py>`

## Config Rules

- Experiment presets live under `config/experiments/<model_id>/`.
- Benchmark configs live under `config/benchmarks/<model_id>/`.
- Presets should stay flat and explicit.
- Merge order is `experiment preset < runtime overrides < benchmark param_space`.
- Historical one-off research launches should live under `recipes/`, not become core workflow APIs.

## Data Rules

- A data module owns artifact loading and window construction.
- A task owns task-specific preprocessing, label construction, metrics, and model instantiation.
- A model should not read dataset files directly.
- If `meta.data_is_standardized=true`, `stats.npz` must exist and defines inverse-transform stats.
- If `meta.data_is_standardized` is missing or false, runtime scaling is fit from `train_data.npy`.

## Model Rules

- Model constructors should use explicit flat arguments.
- Shared helpers are acceptable only when they remove repeated behavior across multiple models.
- Keep model-private logic inside the model file unless there is real cross-model reuse.
- Do not preserve dead compatibility paths for removed experiments.

## Documentation Rules

- README and skill references must describe the actual runnable repository state.
- Graph prediction should be documented as an extension target until a real task/data/model/workflow path exists.
- Research recipes may be documented as recipes, but should not be presented as stable library APIs.
