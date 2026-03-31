# EasyTSF

EasyTSF is a lightweight multivariate forecasting algorithm library with a reproducible experiment and benchmark workflow. The repository intentionally keeps the public surface small so researchers can move from an external implementation to a runnable `mtsf` experiment without inheriting a large framework.

For a Chinese companion guide, see [docs/readme_cn.md](docs/readme_cn.md).

## Overview

- Maintain one public task path: multivariate time-series forecasting (`mtsf`)
- Keep experiment presets explicit and reproducible
- Support benchmark search with a small workflow API
- Optimize for migrating forecasting models into a stable repository contract

## Repository Scope

EasyTSF currently maintains:

- `mtsf` as the only public task
- Runnable experiment presets under `config/experiments/`
- Benchmark search configs under `config/benchmarks/`
- Sequence-only data loading under `easytsf/data/`
- Small registries for task and model lookup

EasyTSF does not maintain:

- grid tasks
- generic TSF abstractions
- `npz` dataset compatibility
- plugin-style runners
- graph side inputs or other extra inputs on the maintained `mtsf` path

## Models in Repository

The following models are registered in `easytsf/model/registry.py`.

| Registered | Example preset | Benchmark example | Notes |
| --- | --- | --- | --- |
| `iTransformer` | No | No | Registered model file only; add your own preset before treating it as a maintained path. |
| `TQNet` | `config/experiments/tqnet/*.yaml` | `config/benchmarks/tqnet/core.py` | Current fully wired example path. |
| `STGCN` | No | No | Registered model file only; initialization currently requires `graph`, which the maintained `mtsf` path does not inject automatically. |
| `STID` | No | No | Registered model file only; no shipped preset yet. |
| `SparseTSF` | No | No | Registered model file only; no shipped preset yet. |

## Quick Start

Install the package into your active Python environment:

```bash
python -m pip install -e .
```

Run a lightweight repository check:

```bash
python -m compileall easytsf
```

Run a single experiment from the CLI:

```bash
python -m easytsf.workflow.experiment config/experiments/tqnet/etth1.yaml \
  --set data_root=dataset \
  --set save_root=checkpoint \
  --set accelerator=auto \
  --set devices=auto
```

Run a single experiment from Python:

```python
from easytsf.workflow import finalize_runtime_conf, load_experiment_config, run_experiment

base_conf = load_experiment_config("config/experiments/tqnet/etth1.yaml")
runtime_conf = finalize_runtime_conf(
    base_conf,
    overrides={
        "data_root": "dataset",
        "save_root": "checkpoint",
        "accelerator": "auto",
        "devices": "auto",
    },
)
run_experiment(runtime_conf)
```

## Dataset Contract

The maintained `mtsf` dataset layout is directory-based:

```text
dataset/<dataset>/
  train_data.npy
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  meta.json
```

Repository expectations:

- `*_data.npy` must be shaped `[L, N]`
- Univariate forecasting still uses `[L, 1]`, not bare `[L]`
- All splits must agree on the channel dimension
- All splits must provide timestamp arrays
- `meta.json` must contain `timestamps_description`
- `meta.json` must contain `frequency (minutes)` because timestamp restoration depends on it

If a model needs time features, `time_feature_descriptions` in the experiment config selects which timestamp columns the datamodule forwards into `marker_x` and `marker_y`.

## Experiment/Benchmark Workflow

EasyTSF keeps the runtime contract explicit:

```text
experiment preset < runtime overrides
```

An experiment preset is the static recipe. Runtime-specific values such as `data_root`, `save_root`, `seed`, `devices`, and `accelerator` should be overridden at run time instead of hard-coded into model logic.

Run benchmark search:

```bash
python -m easytsf.workflow.benchmark config/benchmarks/tqnet/core.py
```

Build a benchmark report from search outputs:

```bash
python -m easytsf.workflow.report config/benchmarks/tqnet/core.py \
  --out save/benchmarks/tqnet_electricity/report.csv
```

## Migrate a Model into EasyTSF

When bringing in a model from another project, keep the adaptation minimal and explicit:

1. Add `easytsf/model/<model_id>.py` with a top-level `Model` class.
2. Expose constructor arguments as explicit `Model.__init__` parameters. `MTSFTask` reads the signature and passes flat config keys by name.
3. Adapt the forward interface to `forward(var_x, marker_x, marker_y)`.
4. Return predictions that are label-compatible, typically `[B, pred_len, N]`.
5. Register the model in `easytsf/model/registry.py`.
6. Add at least one runnable preset under `config/experiments/<model_id>/`.
7. Add benchmark wiring only if the migrated model is ready for search.

If the external model depends on unsupported inputs such as graph side input, decoder caches, or a custom dataset contract that the current `mtsf` path does not provide, stop and redesign explicitly instead of silently expanding the repository scope.

The repository-level migration rules for Codex live in [AGENT.md](AGENT.md).

## Use Codex with EasyTSF

Install the repository version of the migration skill into your local Codex skills directory:

```bash
python scripts/install_codex_skill.py
```

Use the skill when you already have external model code and want to map it into EasyTSF:

```text
Use $migrate-model-to-easytsf to inspect this external forecasting model implementation and tell me whether it fits the current EasyTSF mtsf path. If it fits, generate an EasyTSF-ready model plan plus one experiment preset draft. If it does not fit, stop with an incompatibility report.
```

Use a normal prompt when you only want to read or compare papers:

```text
Summarize this forecasting paper, compare it with the models already registered in EasyTSF, and tell me whether it looks compatible with the current mtsf contract before we touch any code.
```

The skill source of truth lives in [`skills/migrate-model-to-easytsf/`](skills/migrate-model-to-easytsf/).

## Chinese Guide

- Chinese onboarding guide: [docs/readme_cn.md](docs/readme_cn.md)
- Codex repository contract: [AGENT.md](AGENT.md)
