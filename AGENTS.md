# AGENTS.md

This repository is optimized for explicit prediction-task contracts and low-overhead research workflows. Keep it small, explicit, and honest about which task layers already exist versus which ones still need extension.

## Hard Boundaries

- Do not pretend one existing runtime path is the universal contract for every prediction task.
- Do not expand the repository contract just to mimic an external codebase.
- Prefer explicit config, direct code paths, and flat constructor arguments over framework-style indirection.
- Delete dead compatibility code instead of preserving legacy abstractions.
- When adding a new prediction task, define its data contract, task contract, model interface, and workflow surface explicitly.

If an external model or dataset needs behavior beyond the current `mtsf` or maintained `grid3d_forecasting` runtimes, surface the missing repository layers explicitly instead of forcing the task into an existing path.

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
- Current runnable sequence presets use `task: mtsf`
- Current maintained Grid3D presets use `task: grid3d_forecasting`
- Current model constructor arguments are read from flat config keys by name via `BaseForecastTask._instantiate_registered_model()`

### Data

- Every prediction task must make its data artifacts explicit.
- The current concrete implementations live in `easytsf/data/mts_data_module.py` and `easytsf/data/grid3d_data_module.py`.
- The current runnable layouts are directory-based with:
  - `train_data.npy`
  - `val_data.npy`
  - `test_data.npy`
  - `train_timestamps.npy`
  - `val_timestamps.npy`
  - `test_timestamps.npy`
  - `meta.json`
- Grid3D may also use `coord.npy`, `axes.npz`, and `stats.npz` according to its documented data contract.
- Graph tasks and custom grid semantics must define additional topology, mask, coordinate, or side-input artifacts explicitly rather than smuggling them through undocumented conventions.

### Task

- Every prediction task should define:
  - task-owned preprocessing
  - label construction
  - model instantiation rules
  - metric surface
- The current concrete task implementations live in `easytsf/task/mtsf.py` and `easytsf/task/grid3d_forecasting.py`.
- The current sequence model interface is `forward(var_x, marker_x, marker_y)`.
- The current Grid3D model interface is `forward(x, coords=None)`.

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
4. If the source model aligns with `mtsf` or `grid3d_forecasting`, map external constructor arguments onto explicit `Model.__init__` parameters that can be passed from flat experiment config keys.
5. If the source model requires graph inputs or grid semantics outside maintained Grid3D forecasting, write the missing repository layers explicitly instead of collapsing the task into an existing path.
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

## Lightweight Pytest Guidelines

Pytest is an engineering guardrail for this research repository, not a way to prove paper claims. Keep tests small, deterministic, and focused on interfaces that break during cleanup.

Use pytest for:

1. task registry and task/model/metric preflight checks
2. tiny synthetic MTSF and Grid3D datamodule/task smoke tests
3. representative small-tensor model forward shape checks
4. config and benchmark loading checks that do not start training

Do not use pytest for:

1. full training runs
2. GPU scheduling, sweep orchestration, or historical recipes
3. validating scientific claims or reproducing reported metrics
4. exhaustive model/dataset matrices unless a touched shared contract requires it

Preferred local validation order:

1. `python -m compileall -q easytsf`
2. `python -m pytest`
3. a small smoke experiment only when local data is available and the task needs runtime confirmation

Keep `pytest` in the optional test dependency group. Normal experiment environments should not require it unless the user is actively running the test suite.
