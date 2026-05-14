# EasyTSF

EasyTSF is a lightweight forecasting research toolkit built around explicit task contracts and low-overhead experiment workflows. The goal is to make a new time-series forecasting idea easy to land as a runnable experiment without hiding task-specific assumptions in a large framework.

## Current Support

EasyTSF describes every prediction path in four layers:

1. data contract
2. task contract
3. model interface
4. workflow surface

The current runnable task surface is:

| Task | Family | Data module | Model interface | Status |
| --- | --- | --- | --- | --- |
| `mtsf` | `sequence_prediction` | `MTSDataModule` | `forward(var_x, marker_x, marker_y)` | maintained |
| `grid3d_forecasting` | `grid_prediction` | `Grid3DDataModule` | `forward(x, coords=None)` | maintained |

Graph prediction and Grid3D shear input/output ablations are explicit extension targets. They are not exposed as runnable tasks in the current core package.

## Install

Use Python `>=3.11`.

```bash
python -m pip install -e .
```

The package dependencies are declared in `pyproject.toml`. Local datasets, checkpoints, logs, and benchmark outputs are intentionally not part of the package.

## Workflows

Run one experiment:

```bash
python -m easytsf.workflow.experiment config/experiments/tqnet/etth1.yaml
```

Override flat config keys at runtime:

```bash
python -m easytsf.workflow.experiment config/experiments/unet3d/windfield4cast_demo.yaml --set devices=auto --set max_epochs=1
```

Run a benchmark:

```bash
python -m easytsf.workflow.benchmark config/benchmarks/mixlinear/etth1.py
```

Build a report from benchmark outputs:

```bash
python -m easytsf.workflow.report config/benchmarks/mixlinear/etth1.py --out reports/mixlinear_etth1.csv
```

Config merge priority is fixed:

```text
experiment preset < runtime overrides < benchmark param_space
```

## Data Contracts

Sequence datasets use:

```text
<data_root>/<dataset>/
  train_data.npy
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  meta.json
  stats.npz       # required only when meta.data_is_standardized=true
```

Grid3D datasets use the same split naming plus grid artifacts:

```text
<data_root>/<dataset>/
  train_data.npy          # T,C,Y,X,Z
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  coord.npy               # optional coordinates, 3,Y,X,Z
  axes.npz                # optional physical axes
  stats.npz               # required only when standardized
  meta.json
```

## Repository Layout

- `easytsf/data/`: dataset readers, split window datasets, and scaling utilities
- `easytsf/task/`: task-owned preprocessing, label construction, metrics, and model instantiation
- `easytsf/model/`: model adapters with explicit constructor arguments
- `easytsf/workflow/`: experiment, benchmark, and report entrypoints
- `config/experiments/`: runnable experiment presets
- `config/benchmarks/`: Ray Tune benchmark configs
- `scripts/`: stable utility scripts such as dataset import
- `recipes/`: research launchers and one-off experiment orchestration
- `skills/`: agent-facing repository contracts

## Validation

For lightweight validation after code changes:

```bash
python -m compileall easytsf scripts
```

For install validation:

```bash
python -m pip install -e .
```

The repository intentionally favors compile/install/smoke checks over heavyweight full experiment runs by default.
