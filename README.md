# EasyTSF

## Overview
EasyTSF (**E**xperiment-friendly **A**ssistant for **Y**our **T**ime-**S**eries **F**orecasting): easy for humans, easy for AI.

EasyTSF is a lightweight time-series forecasting algorithm library built on Lightning, with reproducible workflows and agent-friendly skills:
- Workflow design for researchers who want a clear, low-overhead path from model code to reproducible experiments and benchmarks.
- Skill design for AI agents that need explicit repository conventions to inspect, migrate, and use models correctly.

For a Chinese companion guide, see [docs/readme_cn.md](docs/readme_cn.md).

## Workflow Design

Workflow design is the human-facing contract in EasyTSF. It is built around three workflows that cover the core loop of time-series forecasting research and application:

### Experiment, Benchmark and Report

`experiment` is the base unit. The Quick Start example below runs one `experiment`; `benchmark` expands that unit into many runs for parameter tuning, and `report` aggregates those runs into a readable summary.

- `experiment`: run a single experiment. This is the smallest runnable unit, used to debug model code and training behavior, and it is also the foundation of `benchmark`.
- `benchmark`: run batches of experiments from a benchmark config. This is the main path for hyperparameter search and optimization.
- `report`: summarize batch experiment outputs into comparable results for inspection and analysis.

These workflows build on Lightning, keep the maintained path intentionally small, and separate static experiment presets from runtime overrides so researchers can move from single-run debugging to batch evaluation with less cognitive overhead.

### Quick Start

Use Python `>=3.11`.

Install the package into your active Python environment:

```bash
python -m pip install -e .
```

Run a single experiment from the CLI:

```bash
python -m easytsf.workflow.experiment config/experiments/tqnet/etth1.yaml
```

Run benchmark search:

```bash
python -m easytsf.workflow.benchmark config/benchmarks/tqnet/core.py
```

Build a benchmark report from search outputs:

```bash
python -m easytsf.workflow.report config/benchmarks/tqnet/core.py
```

## Skill Design

Skill design is the AI-facing contract in EasyTSF. Skills turn repository conventions into reusable interfaces so AI agents can understand repository workflows and use the project more reliably. The current repository ships `migrate-model-to-easytsf`, and more Skills can be added over time.

### Install the Skill

To use the repository version locally, place it into your Codex skills directory manually. Codex auto-discovers skills from `${CODEX_HOME}/skills` when `CODEX_HOME` is set, otherwise from `~/.codex/skills`.

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/migrate-model-to-easytsf "${CODEX_HOME:-$HOME/.codex}/skills/"
```

If you want the installed skill to stay synced with this repository while you edit it, use a symlink instead of copying:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$(pwd)/skills/migrate-model-to-easytsf" "${CODEX_HOME:-$HOME/.codex}/skills/migrate-model-to-easytsf"
```

### Migrate a Model into EasyTSF

When bringing in a model from another project, keep the adaptation minimal and explicit:

1. Add `easytsf/model/<model_id>.py` with a top-level `Model` class.
2. Expose constructor arguments as explicit `Model.__init__` parameters. `MTSFTask` reads the signature and passes flat config keys by name.
3. Adapt the forward interface to `forward(var_x, marker_x, marker_y)`.
4. Return predictions that are label-compatible, typically `[B, pred_len, N]`.
5. Register the model in `easytsf/model/registry.py`.
6. Add at least one runnable preset under `config/experiments/<model_id>/`.
7. Add benchmark wiring only if the migrated model is ready for search.

If the external model depends on unsupported inputs such as graph side input, decoder caches, or a custom dataset contract that the current EasyTSF workflow does not provide, stop and redesign explicitly instead of silently expanding the repository scope.

Use the Skill when you already have external model code and want to map it into EasyTSF:

```text
Use $migrate-model-to-easytsf to inspect this external forecasting model implementation and tell me whether it fits the current EasyTSF contract. If it fits, generate an EasyTSF-ready model plan plus one experiment preset draft. If it does not fit, stop with an incompatibility report.
```

Use a normal prompt instead when you only want to read or compare papers:

```text
Summarize this forecasting paper, compare it with the models already registered in EasyTSF, and tell me whether it looks compatible with the current EasyTSF contract before we touch any code.
```

The repository-level migration rules for Codex live in [AGENT.md](AGENT.md).

The Skill source of truth lives in [`skills/migrate-model-to-easytsf/`](skills/migrate-model-to-easytsf/).
