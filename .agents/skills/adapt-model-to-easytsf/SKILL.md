---
name: adapt-model-to-easytsf
description: Inspect external prediction model implementations and adapt them to EasyTSF task contracts. Use when the user provides model code, class definitions, forward logic, or config fragments and wants Codex to classify the target task as `sequence_prediction`, `graph_prediction`, or `grid_prediction`, determine the current repository fit, and produce either a direct adaptation plan or a repository extension plan.
---

# Adapt Model to EasyTSF

## Overview

Use this skill to adapt external prediction models to EasyTSF by task, not by forcing everything into one existing runtime path. Work from code first. Use papers only to recover intent when the implementation is incomplete or ambiguous.

Before editing code, read:

- `references/task-taxonomy.md`
- `references/repo-contract.md`
- `references/model-interface-mapping.md`
- `references/adaptation-playbook.md`

Use `assets/model_stub.py`, the task-specific experiment stubs, and `assets/benchmark_stub.py` when you need a stable starting point for generated output.

## Workflow

1. Inspect the source model code.
   Read the external `__init__`, `forward`, helper modules, and config fragments. Extract:
   - constructor parameters
   - forward inputs
   - tensor layout assumptions
   - side inputs or runtime state
   - output shape
2. Classify the target prediction task.
   Decide whether the model most naturally fits `sequence_prediction`, `graph_prediction`, or `grid_prediction`.
3. Check the current repository fit.
   If the model aligns with the current `mtsf` or maintained `grid3d_forecasting` path, produce a direct adaptation plan. If it depends on graph inputs or grid semantics outside the maintained Grid3D contract, produce a repository extension plan instead of treating those inputs as automatic blockers.
4. Map the model onto the target EasyTSF contract.
   - convert constructor arguments into explicit task-aware parameters
   - map those parameters onto flat experiment config keys
   - define the target model interface for the classified task
   - spell out which datamodule outputs and task responsibilities are required
   - choose the correct task-specific experiment stub and the shared benchmark stub
   - identify benchmark `param_space` candidates from source code, scripts, or close EasyTSF baselines
   - define the target benchmark file placement under `config/benchmarks/<model_id>/`
5. Produce adaptation output.
   Default deliverable:
   - task classification
   - current repository fit
   - target `Model.__init__` parameter list
   - target model interface and shape mapping
   - required repository additions, if any
   - one example experiment preset draft
   - one example benchmark config draft

## Stop Conditions

Stop and report blockers when any of the following is true:

- the source model is not a prediction model
- the source code is too incomplete to classify the task or infer the interface
- the requested adaptation depends on hidden runtime behavior that is not visible in code or config
- the user asks for invented inputs that do not exist in the source task

Do not stop just because the model needs graph inputs, grid tensors, or richer side inputs. Those should trigger an extension plan.

## Output Format

When giving the adaptation result, organize it in this order:

1. Task classification
2. Current repository fit
3. EasyTSF constructor mapping
4. Target model interface and shape mapping
5. Required repository additions
6. Example experiment preset fields
7. Example benchmark config draft

When drafting benchmark output, follow these rules:

- always produce a complete `benchmark_config` draft, even when the task still needs repository extensions
- point `experiment` at the matching experiment preset draft path, for example `config/experiments/<model_id>/<dataset>.yaml`
- place `search_save_dir` under `save/benchmarks/<benchmark_name>`
- prefer benchmark search keys and ranges that are visible in source code, original scripts, or close EasyTSF baselines
- if no trustworthy search range exists, keep the search narrow or single-valued and mark uncertain ranges with a short comment such as `replace with validated search range`
- keep `search_config` aligned with the current EasyTSF benchmark surface instead of inventing a model-specific runner contract
- for graph tasks or custom grid tasks that are not yet runnable, clearly mark the benchmark draft as a planning draft rather than a runnable config

If the user asks for implementation, generate the model, experiment preset, and benchmark sample config from the mapped contract instead of copying the source project structure wholesale.
