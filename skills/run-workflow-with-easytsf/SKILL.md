---
name: run-workflow-with-easytsf
description: Plan or run EasyTSF experiment, benchmark, and report workflows for prediction tasks. Use when the user wants task-aware config guidance, exact CLI commands, or a workflow extension plan for `sequence_prediction`, `graph_prediction`, or `grid_prediction`. Classify the task first, then decide whether the current repository can run it directly or whether workflow, config, and task-layer expansion is required.
---

# Run Workflow with EasyTSF

## Overview

Use this skill for the full EasyTSF workflow surface: single experiment, benchmark search, and report generation. Start from the target prediction task and current repository state, not from assumptions.

Before answering, read:

- `references/task-taxonomy.md`
- `references/workflow-contract.md`
- `references/current-repo-surface.md`

## Workflow

1. Classify the prediction task.
   Decide whether the requested workflow targets `sequence_prediction`, `graph_prediction`, or `grid_prediction`.
2. Inspect the current repository surface.
   Read the model registry, existing experiment presets, benchmark configs, and workflow entrypoints before suggesting commands or edits.
3. Decide whether the workflow is directly runnable today.
   If the task aligns with the current sequence path, produce exact experiment, benchmark, and report commands. If the task needs graph or grid support, produce the workflow-extension plan instead of stopping.
4. Keep the config surface explicit.
   Spell out:
   - target task classification
   - required experiment config keys
   - benchmark param-space implications
   - report grouping consequences
5. Separate current runtime advice from future task design.
   The current codebase can run the existing sequence path through `mtsf`. Graph and grid tasks should receive a concrete workflow expansion plan across `data`, `task`, config, and reporting layers.

## Stop Conditions

Stop and report the blocker when any of the following is true:

- the request is not a prediction workflow
- the task cannot be classified from the provided artifacts
- the base model or dataset information is too incomplete to propose either a runnable path or an extension plan
- the user asks for hidden magic instead of explicit task, data, and workflow contracts

## Output Format

When returning the result, organize it in this order:

1. Task classification
2. Current repository fit
3. Experiment surface
4. Benchmark and report surface
5. Required workflow extensions or blockers
