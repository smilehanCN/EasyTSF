# Prediction Task Taxonomy

Classify the workflow before selecting commands or configs.

- `sequence_prediction`: temporal values with optional time features and no required graph or grid structure
- `graph_prediction`: prediction depends on explicit graph topology
- `grid_prediction`: prediction depends on preserved grid structure

The current runnable EasyTSF code only implements a sequence-oriented path through `mtsf`, but workflow planning should still treat graph and grid tasks as first-class prediction tasks.
