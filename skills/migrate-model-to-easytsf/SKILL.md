---
name: migrate-model-to-easytsf
description: Migrate external forecasting model implementations into EasyTSF. Use when the user provides model code, class definitions, forward logic, or config snippets from another project and wants Codex to assess compatibility with the current EasyTSF mtsf path, map constructor parameters and tensor shapes, and produce an EasyTSF-ready model plus one example experiment preset. Stop and report blockers when the source model depends on inputs or abstractions that the maintained mtsf path does not provide.
---

# Migrate Model to EasyTSF

## Overview

Use this skill to move an external forecasting model into the maintained EasyTSF `mtsf` path. Work from code first. Use the paper only to recover intent when the implementation is incomplete or ambiguous.

Before editing code, read:

- `references/repo-contract.md`
- `references/shape-and-config-mapping.md`
- `references/migration-playbook.md`

Use `assets/model_stub.py` and `assets/experiment_stub.yaml` when you need a stable starting point for generated output.

## Workflow

1. Inspect the source model code.
   Read the external `__init__`, `forward`, helper modules, and config fragments. Extract:
   - constructor parameters
   - forward inputs
   - tensor layout assumptions
   - side inputs or runtime state
   - output shape
2. Check compatibility against the current EasyTSF contract.
   Treat the maintained path as sequence-only `mtsf`. The repository does not automatically inject graph side input, decoder caches, custom datamodule state, or extra tensors beyond `var_x`, `marker_x`, and `marker_y`.
3. Stop early when the source model is out of bounds.
   If the source model depends on unsupported inputs or abstractions, do not invent a new repository contract. Produce an incompatibility report and stop.
4. If compatible, map the model onto EasyTSF.
   - Convert constructor arguments into explicit `Model.__init__` parameters
   - Map those parameters onto flat experiment config keys
   - Adapt `forward` to `forward(var_x, marker_x, marker_y)`
   - Normalize tensor layout internally if needed
   - Return predictions in label-compatible shape, typically `[B, pred_len, N]`
5. Produce migration output.
   Default deliverable:
   - compatibility verdict
   - target `Model.__init__` parameter list
   - `forward(var_x, marker_x, marker_y)` adaptation plan
   - input and output shape mapping
   - required flat config keys
   - one example experiment preset draft

## Stop Conditions

Stop and report blockers when the source model needs any of the following and the user did not explicitly approve expanding the repository contract:

- graph side input
- spatial adjacency injected outside the current config flow
- decoder caches or recurrent runtime state not represented by the maintained task
- extra tensors beyond `var_x`, `marker_x`, and `marker_y`
- custom dataset layout or metadata beyond the current dataset contract

When stopping, name the exact unsupported input or abstraction and the file or method where it appears.

## Output Format

When giving the migration result, organize it in this order:

1. Compatibility verdict
2. Blocking inputs or assumptions, if any
3. EasyTSF constructor mapping
4. Forward adaptation and shape mapping
5. Required config keys
6. Example experiment preset fields

If the user asks for implementation, generate the model and preset from the mapped contract instead of copying the source project structure wholesale.
