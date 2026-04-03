# Adaptation Playbook

## Step 1: Inventory the external model

Extract:

- model class name and file location
- constructor parameters
- forward signature
- tensor layout assumptions
- required side inputs
- output shape
- config keys that materially affect behavior

## Step 2: Classify the prediction task

Decide whether the model is best treated as:

- `sequence_prediction`
- `graph_prediction`
- `grid_prediction`

## Step 3: Determine the current repository fit

- if the model aligns with the current sequence path, produce a direct adaptation plan
- if the model depends on graph or grid inputs, produce a repository extension plan

Do not downgrade the task just to fit today's runtime.

## Step 4: Build the EasyTSF mapping

Produce:

- target constructor parameters
- target model interface
- input/output shape mapping
- required config keys
- repository additions in `data`, `task`, and workflow layers when the task is not yet implemented

## Step 5: Draft the task-aware experiment and benchmark surface

Choose the correct stubs:

- `assets/sequence_prediction_experiment_stub.yaml`
- `assets/graph_prediction_experiment_stub.yaml`
- `assets/grid_prediction_experiment_stub.yaml`
- `assets/benchmark_stub.py`

Produce:

- one experiment preset draft
- one benchmark config draft with `experiment` and `param_space`

Use the experiment stubs and the shared benchmark stub as contract sketches, not as proof that the current runtime already supports the task.
