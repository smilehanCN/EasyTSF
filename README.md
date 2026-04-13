# EasyTSF

## Overview
EasyTSF (**E**xperiment-friendly **A**ssistant for **Y**our **T**ime-**S**eries **F**orecasting): easy for humans, easy for AI.

EasyTSF is a lightweight prediction-task library built on Lightning, with reproducible workflows and agent-friendly skills:
- Workflow design for researchers who want a clear path from model code to reproducible experiments and benchmarks.
- Skill design for AI agents that need explicit contracts for data, tasks, model interfaces, and workflow surfaces.

For a Chinese companion guide, see [docs/readme_cn.md](docs/readme_cn.md).

## Workflow Design

Workflow design is the human-facing contract in EasyTSF. It is built around three public workflow surfaces:

### Experiment, Benchmark and Report

`experiment` is the base unit. The Quick Start example below runs one `experiment`; `benchmark` expands that unit into many runs for parameter tuning, and `report` aggregates those runs into a readable summary.

- `experiment`: run a single experiment. This is the smallest runnable unit, used to debug model code and training behavior, and it is also the foundation of `benchmark`.
- `benchmark`: run batches of experiments from a benchmark config. This is the main path for hyperparameter search and optimization.
- `report`: summarize batch experiment outputs into comparable results for inspection and analysis.

These workflows build on Lightning and separate static experiment presets from runtime overrides so researchers can move from single-run debugging to batch evaluation with less cognitive overhead.

## Prediction Tasks

EasyTSF documents prediction work in four layers:

1. `data contract`
2. `task contract`
3. `model interface`
4. `workflow surface`

The taxonomy used by the Skills and docs is:

- `sequence_prediction`
- `graph_prediction`
- `grid_prediction`

The current runnable codebase still centers on one concrete sequence-oriented implementation through the existing `mtsf` path. It also includes a maintained `grid3d_forecasting` path for 3D grid forecasting based on per-step `.npy` caches imported from WindField4Cast-style raw files. Graph prediction remains an explicit extension target rather than a hidden edge case.

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

Run the maintained MixLinear ETTh1 preset:

```bash
python -m easytsf.workflow.experiment config/experiments/mixlinear/etth1.yaml
```

Run the adapted TimeBase ETTh1 preset:

```bash
python -m easytsf.workflow.experiment config/experiments/timebase/etth1.yaml
```

Run the ARROW WeatherBench preset after installing the optional ARROW dependencies and preparing interval diff stats:

```bash
python -m easytsf.workflow.experiment config/experiments/arrow/weatherbench.yaml
```

Import a WindField4Cast-style raw directory into the Grid3D per-step cache layout:

```bash
python scripts/grid3d_import.py --input-dir /path/to/raw_nc_dir --out-dir dataset/windfield4cast_demo
```

Run the maintained UNet3D Grid3D preset:

```bash
python -m easytsf.workflow.experiment config/experiments/unet3d/windfield4cast_demo.yaml
```

Run benchmark search:

```bash
python -m easytsf.workflow.benchmark config/benchmarks/mixlinear/etth1.py
```

Run the TimeBase benchmark search:

```bash
python -m easytsf.workflow.benchmark config/benchmarks/timebase/etth1.py
```

Build a benchmark report from search outputs:

```bash
python -m easytsf.workflow.report config/benchmarks/mixlinear/etth1.py
```

## Skill Design

Skill design is the AI-facing contract in EasyTSF. Skills turn repository conventions into reusable interfaces so AI agents can classify prediction tasks, map them onto the current repository surface, and plan explicit extensions when the current code does not yet support them. The current repository ships three maintained Skills:

- `prepare-data-for-easytsf`: inspect local data artifacts, classify the prediction task, and map the data contract onto the current repo or an extension plan
- `adapt-model-to-easytsf`: inspect external model code, classify the target prediction task, and map the required model/task/repo changes
- `run-workflow-with-easytsf`: plan or run task-aware experiment, benchmark, and report workflows from the current repo surface

### Install the Skill

To use the repository versions locally, place them into your Codex skills directory manually. Codex auto-discovers skills from `${CODEX_HOME}/skills` when `CODEX_HOME` is set, otherwise from `~/.codex/skills`.

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/* "${CODEX_HOME:-$HOME/.codex}/skills/"
```

If you want the installed Skills to stay synced with this repository while you edit them, symlink each skill directory:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
for skill_dir in skills/*; do
  ln -s "$(pwd)/${skill_dir}" "${CODEX_HOME:-$HOME/.codex}/$(basename "${skill_dir}")"
done
```

### Prepare Data for EasyTSF

Use this Skill when the user already has local data artifacts and wants an AI agent to classify the task before touching configs or runs:

```text
Use $prepare-data-for-easytsf to inspect this dataset directory, classify it as sequence, graph, or grid prediction data, and tell me whether the current EasyTSF repo can use it directly or needs a contract extension.
```

### Adapt a Model to EasyTSF

Use this Skill when the user already has external model code and wants an AI agent to map it onto the right prediction-task contract:

```text
Use $adapt-model-to-easytsf to inspect this model implementation, classify it as sequence, graph, or grid prediction, and tell me whether EasyTSF can adapt it directly or needs new task/data/workflow layers.
```

### Run an EasyTSF Workflow

Use this Skill when the user wants exact workflow guidance, from one experiment to a benchmark plus report:

```text
Use $run-workflow-with-easytsf to classify this prediction task, tell me whether the current EasyTSF repo can run it directly, and give me the exact experiment or benchmark workflow surface.
```

## Current Implementation Note

Today, the current runnable code path still uses the sequence-oriented `mtsf` implementation, and it also ships maintained `grid3d_forecasting` and `weatherbench` grid-forecasting paths:

- `easytsf/data/mts_data_module.py`
- `easytsf/task/mtsf.py`
- `easytsf/data/weather_data_module.py`
- `easytsf/task/weatherbench.py`
- `easytsf/workflow/experiment.py`
- `easytsf/workflow/benchmark.py`
- `easytsf/workflow/report.py`

The Skills and docs deliberately speak in task-aware prediction language beyond that concrete implementation. If a request targets graph or grid prediction, the expected response is an explicit extension plan rather than a forced downgrade into sequence-only assumptions.

The maintained sequence model surface now includes runnable `MixLinear` and `TimeBase` presets at `config/experiments/mixlinear/etth1.yaml` and `config/experiments/timebase/etth1.yaml`, plus benchmark examples at `config/benchmarks/mixlinear/etth1.py` and `config/benchmarks/timebase/etth1.py`.

The maintained WeatherBench surface now includes `WeatherBenchPersistence` and an ARROW stage-1 adaptation at `config/experiments/arrow/weatherbench.yaml`. ARROW keeps `timm` and `xformers` as optional dependencies and expects interval diff statistics such as `diff_stats_6.npz` or `normalize_diff_mean/std_<hours>.npz` in the dataset directory. The RL rollout scheduler from stage 2 is intentionally left as a documented extension boundary in [docs/arrow_rl_extension.md](docs/arrow_rl_extension.md).

The repository-level collaboration rules for Codex live in [AGENT.md](AGENT.md).

Skill sources of truth live under [`skills/`](skills/).
