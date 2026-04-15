---
name: prepare-data-for-easytsf
description: Inspect local prediction datasets and map them onto EasyTSF task contracts. Use when the user provides dataset folders, array files, graph structure files, grid tensors, or metadata and wants Codex to classify the task as `sequence_prediction`, `graph_prediction`, or `grid_prediction`, determine whether the current repository can use it directly, and produce either concrete config mapping or a repository extension plan.
---

# Prepare Data for EasyTSF

## Overview

Use this skill to inspect prediction data before touching configs, workflows, or model code. Start from artifacts on disk, not from dataset names or paper summaries.

Before answering, read:

- `references/task-taxonomy.md`
- `references/sequence-data-contract.md`
- `references/graph-data-contract.md`
- `references/grid-data-contract.md`

## Workflow

1. Inspect the dataset directory.
   Read the directory contents, `meta.json`, and the shapes of the available arrays before making any recommendation.
2. Classify the prediction task.
   Decide whether the data most naturally fits:
   - `sequence_prediction`
   - `graph_prediction`
   - `grid_prediction`
3. Map the data onto the current repository surface.
   If the data fits the current sequence path, produce concrete config fields and note that the closest runnable implementation today is the existing `mtsf` workflow.
4. Produce an extension plan when the task is not yet implemented.
   For graph or grid data, do not stop just because the repository lacks a runnable path today. Instead, spell out:
   - required new data artifacts
   - datamodule outputs
   - task-level inputs
   - config fields the repository would need
5. Keep unknowns explicit.
   Distinguish observed facts from inferred recommendations. `hist_len`, `pred_len`, and model hyperparameters are experiment decisions, not raw dataset facts.

## Stop Conditions

Stop and report the blocker when any of the following is true:

- the artifacts are too incomplete to classify the task
- split files or metadata contradict one another in a way that prevents even a draft contract
- the data appears to target a non-prediction task outside `sequence_prediction`, `graph_prediction`, and `grid_prediction`
- the user asks for synthetic recovery of missing files or silently invented metadata

## Output Format

When returning the result, organize it in this order:

1. Task classification
2. Observed artifacts and shapes
3. Current repository fit
4. Config mapping or extension plan
5. Missing information or blockers
