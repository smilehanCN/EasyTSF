# AGENT.md

This file is for AI coding agents working in this repository. The goal is to keep the repo optimized for rapid multivariate, graph spatiotemporal, and regular-grid spatiotemporal forecasting research, not for heavy framework building.

## Mission

- Use shared experiment/study workflow to support fast model iteration across `mtsf`, `stf`, `grid2dtsf`, and `grid3dtsf`.
- Keep experiments comparable, reproducible, and easy to resume.
- Spend complexity on model ideas and benchmark workflow, not on scaffolding.

## Core Concepts

### `experiment`

An `experiment` is one runnable training/evaluation preset. It already contains:

- model hyper-parameters
- dataset selection
- dataset-specific training hyper-parameters
- one default `hist_len/pred_len`

The preferred layout is `config/experiments/<model_id>/<dataset_id>.yaml`, with lowercase config ids.

### `study`

A `study` is a thin batch benchmark specification. It only decides:

- which experiment presets to run
- which seeds to repeat
- which case-level overrides to apply
- how results are resumed and aggregated

`study` must not become a second experiment layer. Do not move dataset-specific training recipes into study overrides.

## Current Boundaries

- Maintain `mtsf`, static-graph `stf`, native 2D-grid `grid2dtsf`, and native 3D-grid `grid3dtsf`.
- Do not treat flattened grid data, future dynamic-graph variants, or irregular mesh inputs as already-native tasks.
- Do not reintroduce old runner variants, visualization branches, auxiliary losses, or task-dispatch trees.
- Do not rebuild `ray_tune.py`, Python `exp_conf`, or `exp_runner` style orchestration.
- Do not add plugin systems, registries, dataset-profile layers, or large test matrices unless the repository goal changes.

## Architecture Map

### CLI

- `train.py`: single-experiment training entry
- `evaluate.py`: single-experiment evaluation entry
- `study.py`: batch benchmark entry

### Config

- `config/tasks/mtsf.yaml`: default task config for the MTSF path
- `config/tasks/stf.yaml`: default task config for the static-graph STF path
- `config/tasks/grid2dtsf.yaml`: default task config for the native 2D-grid forecasting path
- `config/tasks/grid3dtsf.yaml`: default task config for the native 3D-grid forecasting path
- `config/datasets/catalog.yaml`: dataset metadata and data-loading hints
- `config/experiments/<model_id>/*.yaml`: experiment presets
- `config/studies/<model_id>/*.yaml`: study specs
- `config/search_spaces/<model_id>/*.py`: Ray Tune search spaces

Config merge priority is fixed:

`experiment > dataset > task`

Do not break this rule and do not scatter experiment-specific constants into scripts.
Use `runtime.task_name` to choose the task defaults; when omitted, default to `mtsf`.

### Data

- `easytsf/data/data_module.py` contains the shared `DataInterface`.
- `easytsf/data/grid_data_module.py` contains `GridDataInterface`.
- Default data format is `dataset/<dataset_name>.npz`.
- The current pipeline only requires `scaled_variable` and `timestamp`.
- Static graph datasets may also define `data.graph_path`, resolved relative to `data_root`.
- Grid datasets may additionally define `grid_mask` and `coord` inside the `.npz` file.
- Sliding-window split logic belongs in `DataInterface`, not in model-specific loaders.

### Task Layer

- `easytsf/task/mtsf.py` contains `MTSFTask`.
- `easytsf/task/stf.py` contains `STFTask`.
- `easytsf/task/gridstf.py` contains `Grid2DTSFTask` and `Grid3DTSFTask`.
- Keep task semantics separate: do not push graph-aware behavior into `MTSFTask`.

### Model Layer

- `easytsf/model/<model_id>.py` should define a top-level `Model` class.
- Prefer one main file per model. Keep model-private helpers inside that file unless at least two models already share the same logic.
- Model file names are lowercase. `model_name` may keep the paper-style spelling, but the main class name is always `Model`.
- Model interfaces are task-specific:
  - `mtsf`: `forward(var_x, marker_x)`
  - `stf`: `forward(var_x, marker_x, graph)`
  - `grid2dtsf`: `forward(var_x, marker_x, grid_mask=None, coord=None)`
  - `grid3dtsf`: `forward(var_x, marker_x, grid_mask=None, coord=None)`
- The default output shape should stay label-compatible:
  - `mtsf` / `stf`: typically `[B, pred_len, N]`
  - `grid2dtsf`: `[B, pred_len, C, H, W]`
  - `grid3dtsf`: `[B, pred_len, C, X, Y, Z]`

### Workflow Layer

- `easytsf/workflow/experiment.py` contains single-experiment orchestration and config loading.
- `easytsf/workflow/study.py` contains batch benchmark orchestration.
- Keep workflow code out of `data`, `model`, and `task`.

## Change Guidelines

- When adding a standard model, prefer changing only `easytsf/model/` and the corresponding experiment presets.
- Use `--set section.key=value` for temporary overrides instead of creating many one-off configs.
- If a dataset or horizon needs a long-lived special recipe, promote it to an experiment preset instead of growing study complexity.
- Keep batch benchmarking thin: explicit cases, seeds, resume, and aggregation.
- Do not reintroduce model-private code into a shared `layer/` package unless real reuse already exists.
- Documentation must match the actual repository state. Do not describe features, files, or entrypoints that do not exist.

## Validation Guidelines

Default validation should stay lightweight. Prefer:

1. `conda activate easytsf`
2. `python train.py -h`
3. `python evaluate.py -h`
4. `python study.py -h`
5. Experiment YAMLs can be loaded
6. At least one search-space module can be loaded
7. If local data exists, run the smallest possible smoke benchmark

The validation goal for research-oriented changes is: the main path still works. It is not to build a full CI discipline inside this repo.
