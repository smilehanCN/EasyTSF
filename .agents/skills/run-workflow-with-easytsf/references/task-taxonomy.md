# Prediction Task Taxonomy

Classify the workflow before selecting commands or configs.

- `sequence_prediction`: temporal values with optional time features and no required graph or grid structure
- `graph_prediction`: prediction depends on explicit graph topology
- `grid_prediction`: prediction depends on preserved grid structure

The current runnable EasyTSF task registry exposes `mtsf` for sequence prediction and `grid3d_forecasting` for maintained 3D grid prediction. Graph prediction and grid semantics outside the maintained Grid3D contract are extension targets.
