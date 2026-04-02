# Graph Prediction Data Contract

This contract is a design target for repository expansion.

## Minimum artifacts

Recommended dataset layout:

```text
<data_root>/<dataset_name>/
  train_data.npy
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  graph.npy or edge_index.npy
  meta.json
```

Optional but common:

- `node_features.npy`
- `edge_features.npy`
- masks or split-specific side inputs when they are truly task-defining

## Core expectations

- temporal values should make the node or edge axis explicit
- graph topology must be provided as a first-class artifact
- `meta.json` should describe:
  - temporal frequency
  - graph representation type
  - node or edge semantics
  - timestamp feature descriptions when time markers exist

## Extension-plan outputs

When the current repository cannot run the task yet, the skill should specify:

- required datamodule outputs beyond sequence values
- whether the model needs adjacency, `edge_index`, or richer graph structure
- candidate config fields such as:
  - `task: graph_prediction`
  - graph topology reference
  - node and edge feature switches
- which current sequence assumptions must be lifted
