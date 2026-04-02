# Workflow Contract

EasyTSF exposes three workflow surfaces:

- `experiment`
- `benchmark`
- `report`

## Experiment

Current entrypoint:

```bash
python -m easytsf.workflow.experiment <experiment-yaml> [--set KEY=VALUE ...]
```

Use this for the smallest runnable unit.

## Benchmark

Current entrypoint:

```bash
python -m easytsf.workflow.benchmark <benchmark.py>
```

Benchmark applies flat overrides from `param_space` on top of a base experiment preset.

## Report

Current entrypoint:

```bash
python -m easytsf.workflow.report <benchmark.py> [--results-dir <dir>] [--out <report.csv>]
```

Report summarizes benchmark outputs into grouped metrics.

## Task-aware guidance

- for `sequence_prediction`, the current repository can usually map directly onto the existing `mtsf` runtime
- for `graph_prediction` and `grid_prediction`, the workflow result should specify:
  - required new task module
  - required datamodule outputs
  - experiment config additions
  - benchmark and report integration points

Do not present unsupported tasks as immediate runtime errors if the user is asking for planning or contract design.
