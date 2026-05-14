# Prediction Task Taxonomy

Classify the external model before adapting it.

## `sequence_prediction`

Typical signals:

- history and future values over one temporal axis
- optional time features
- no required graph topology or grid structure

## `graph_prediction`

Typical signals:

- node or edge time series
- required graph topology in initialization or forward
- optional node or edge features

## `grid_prediction`

Typical signals:

- temporal tensors with height and width axes
- spatial neighborhoods are part of the model behavior
- optional masks, coordinates, or static grid features

## Current implementation note

EasyTSF currently has runnable paths through `mtsf` for sequence prediction and `grid3d_forecasting` for maintained 3D grid prediction. Graph prediction and grid semantics outside the maintained Grid3D contract are design targets for repository expansion.
