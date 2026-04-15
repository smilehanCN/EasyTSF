# AGENT.md

This repository is optimized for explicit prediction-task contracts and low-overhead research workflows. Keep it small, explicit, and honest about which task layers already exist versus which ones still need extension.

## Hard Boundaries

- Do not pretend one existing runtime path is the universal contract for every prediction task.
- Do not expand the repository contract just to mimic an external codebase.
- Prefer explicit config, direct code paths, and flat constructor arguments over framework-style indirection.
- Delete dead compatibility code instead of preserving legacy abstractions.
- When adding a new prediction task, define its data contract, task contract, model interface, and workflow surface explicitly.

If an external model or dataset needs behavior beyond the current sequence-oriented runtime, surface the missing repository layers explicitly instead of forcing the task into sequence-only assumptions.

## Repo Contract

EasyTSF documents prediction work in four layers:

1. `data contract`
2. `task contract`
3. `model interface`
4. `workflow surface`

The documented taxonomy is:

- `sequence_prediction`
- `graph_prediction`
- `grid_prediction`

### Config

- `config/experiments/<model_id>/*.yaml`: runnable experiment presets
- `config/benchmarks/<model_id>/*.py`: benchmark configs
- Config merge priority is fixed: `experiment preset < runtime overrides`
- Current runnable sequence presets still use `task: mtsf`
- Current model constructor arguments are read from flat config keys by name via `MTSFTask._build_model()`

### Data

- Every prediction task must make its data artifacts explicit.
- The current concrete implementation lives in `easytsf/data/mts_data_module.py`.
- The current runnable sequence layout is directory-based with:
  - `train_data.npy`
  - `val_data.npy`
  - `test_data.npy`
  - `train_timestamps.npy`
  - `val_timestamps.npy`
  - `test_timestamps.npy`
  - `meta.json`
- Graph and grid tasks must define their additional topology, mask, coordinate, or side-input artifacts explicitly rather than smuggling them through undocumented conventions.

### Task

- Every prediction task should define:
  - task-owned preprocessing
  - label construction
  - model instantiation rules
  - metric surface
- The current concrete implementation lives in `easytsf/task/mtsf.py`.
- The current sequence model interface is `forward(var_x, marker_x, marker_y)`.

### Workflow

- `easytsf/workflow/experiment.py` contains single-experiment orchestration
- `easytsf/workflow/benchmark.py` contains benchmark orchestration
- `easytsf/workflow/report.py` summarizes benchmark outputs
- Keep workflow logic out of `data`, `task`, and `model`
- When planning a new prediction task, specify how experiment, benchmark, and report should observe it.

## Adaptation Playbook

When the user wants to adapt a model from another repository:

1. Read the external model code first: constructor, `forward`, helper modules, and config fragments.
2. Classify the target task as `sequence_prediction`, `graph_prediction`, or `grid_prediction`.
3. Read the EasyTSF contract before editing: data contract, task contract, model interface, and workflow surface.
4. If the source model aligns with the current sequence path, map external constructor arguments onto explicit `Model.__init__` parameters that can be passed from flat experiment config keys.
5. If the source model requires graph or grid inputs, write the missing repository layers explicitly instead of collapsing the task into the current sequence path.
6. Normalize tensor layout inside the model when needed, but keep the target task interface explicit.
7. Wire the model into the appropriate task surface, add one example preset, and sync public docs if the maintained surface changed.

## Add a Maintained Prediction Task or Model

Definition of done for a maintained prediction path:

1. Define the data contract.
2. Define the task contract.
3. Define the model interface.
4. Define the workflow surface.
5. Add runnable code only for the layers the repository is actually implementing now.
6. Update `README.md` and `docs/readme_cn.md` when public onboarding or support status changed.
7. Run `python -m compileall easytsf` when Python code changed.

Keep model-private helpers inside the model file unless there is real cross-model reuse.

## Change Guidelines

- Change the minimum surface necessary for the target prediction contract.
- If a code path exists only for historical compatibility and the current path does not use it, delete it.
- Do not add defensive pre-validation for cases that Python, NumPy, or PyTorch already reject naturally.
- Keep explicit checks for structure or semantic mismatches that would otherwise fail silently.
- Documentation must describe the current repository state, not removed features.

## Validation Guidelines

Prefer lightweight validation:

1. `python -m compileall easytsf`
2. A small smoke experiment if local data is available

Do not run `pytest`.
