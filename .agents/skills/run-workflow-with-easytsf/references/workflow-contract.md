# Workflow Contract

EasyTSF exposes three workflow surfaces.

## Experiment

```bash
python -m easytsf.workflow.experiment <experiment-yaml> [--set KEY=VALUE ...]
```

Use this for one runnable experiment. Runtime overrides are flat `KEY=VALUE` updates on top of the experiment preset.

## Benchmark

```bash
python -m easytsf.workflow.benchmark <benchmark.py> [--no-resume] [--verbose]
```

Benchmark applies `param_space` overrides on top of the base experiment preset referenced by the benchmark config.

## Report

```bash
python -m easytsf.workflow.report <benchmark.py> [--results-dir DIR] [--out FILE]
```

Report scans Lightning `metrics.csv` and `hparams.yaml` files, groups runs by fixed config and benchmark parameters, and emits mean/std metric columns.

## Task-Aware Guidance

- For `sequence_prediction`, use `task: mtsf` when the data matches the sequence contract.
- For maintained 3D grid forecasting, use `task: grid3d_forecasting`.
- For WindShear input/output representation ablations and graph prediction, produce an extension plan; do not claim the current workflow can run them directly.
