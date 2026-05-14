# Current Repo Surface

Inspect the repository before finalizing a workflow answer. This file captures the expected public surface, but the code remains the source of truth.

## Runnable Tasks

The task registry currently exposes:

- `mtsf`
  - family: `sequence_prediction`
  - data module: `MTSDataModule`
  - task class: `MTSFTask`
  - model call: `forward(var_x, marker_x, marker_y)`
- `grid3d_forecasting`
  - family: `grid_prediction`
  - data module: `Grid3DDataModule`
  - task class: `Grid3DForecastingTask`
  - model call: `forward(x, coords=None)`

Graph prediction and Grid3D shear input/output ablations are extension targets, not runnable tasks in the current core package.

## Workflow Entrypoints

- `python -m easytsf.workflow.experiment <experiment-yaml> [--set KEY=VALUE ...]`
- `python -m easytsf.workflow.benchmark <benchmark.py> [--no-resume] [--verbose]`
- `python -m easytsf.workflow.report <benchmark.py> [--results-dir DIR] [--out FILE]`

## Registered Models

Use `easytsf/model/registry.py` as the source of truth. The current maintained surface includes sequence models such as `MixLinear`, `PCMLP`, `TQNet`, and `TimeBase`, plus Grid3D models such as `unet3d`, `fno3d`, `afno3d`, `patchstg_flat3d`, `simvpv2_3d`, and `unet3d_patchcat`.

## Planning Rule

For sequence and maintained Grid3D forecasting tasks, prefer the current runnable workflow surface. For graph prediction, shear input/output ablations, or other new grid semantics, produce an explicit extension plan covering data, task, model, and workflow changes.
