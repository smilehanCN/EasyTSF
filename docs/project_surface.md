# EasyTSF Project Surface

## Summary

EasyTSF has a small stable workflow surface and task-specific runtime layers. New research ideas should plug into the task layer rather than expanding workflow orchestration.

## Stable Workflow API

```bash
python -m easytsf.workflow.experiment <experiment-yaml> [--set KEY=VALUE ...]
python -m easytsf.workflow.benchmark <benchmark.py> [--no-resume] [--verbose]
python -m easytsf.workflow.report <benchmark.py> [--results-dir DIR] [--out FILE]
```

## Runtime Tasks

| Task | Family | Data output | Model call | Primary metrics |
| --- | --- | --- | --- | --- |
| `mtsf` | sequence | `[B,T,C]` plus timestamp features | `model(var_x, marker_x, marker_y)` | `val/loss`, `test/mae`, `test/mse` |
| `grid3d_forecasting` | grid | `[B,T,C,Y,X,Z]` plus optional coords | `model(x, coords=coords)` | `val/loss`, `test/mae`, `test/mse` |

## Extension Rule

When adding a new task, first define:

- required data artifacts and tensor layout
- task-owned preprocessing and labels
- exact model forward signature
- metric names visible to checkpointing, benchmark, and report
- one minimal experiment preset

Do not add a workflow feature when a task-local implementation is enough.
