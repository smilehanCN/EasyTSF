# Prediction Task Taxonomy

Use this taxonomy before making any data recommendation.

## `sequence_prediction`

Signals:

- one temporal axis
- values organized as variables, channels, or nodes over time
- optional timestamp features
- no required graph topology or grid coordinates

Typical artifacts:

- `*_data.npy`
- `*_timestamps.npy`
- `meta.json`

## `graph_prediction`

Signals:

- temporal values are attached to graph nodes or edges
- the task needs explicit topology such as adjacency, `edge_index`, or edge lists
- graph structure is part of the prediction contract, not incidental metadata

Typical artifacts:

- temporal arrays for node or edge values
- graph topology files
- optional node or edge feature files
- timestamp metadata

## `grid_prediction`

Signals:

- temporal values live on a spatial grid
- arrays carry explicit height and width axes or equivalent cell indexing
- prediction quality depends on preserving spatial layout

Typical artifacts:

- grid tensors or cell-indexed tensors
- optional coordinate metadata or masks
- timestamp metadata

## Current repository note

The current runnable EasyTSF task registry exposes `mtsf` for sequence prediction and `grid3d_forecasting` for maintained 3D grid prediction. Graph prediction and grid data outside the maintained Grid3D contract should be treated as contract-extension targets, not as immediate runtime failures.
