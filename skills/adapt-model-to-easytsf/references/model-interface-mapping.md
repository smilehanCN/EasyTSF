# Model Interface Mapping

## Constructor mapping rule

EasyTSF favors explicit constructor parameters over opaque config objects.

Current runnable code uses:

- flat config keys
- parameter-name matching in `MTSFTask._build_model()`

Preserve that explicitness when proposing new tasks.

## Target interfaces by task

### `sequence_prediction`

Current concrete interface:

```python
forward(var_x, marker_x, marker_y)
```

Typical shapes:

- `var_x`: `[B, hist_len, N]`
- `marker_x`: `[B, hist_len, T]` or `None`
- `marker_y`: `[B, pred_len, T]` or `None`
- output: `[B, pred_len, N]`

### `graph_prediction`

Recommended target interface:

```python
forward(var_x, graph, marker_x=None, marker_y=None, node_features=None, edge_features=None)
```

Typical expectations:

- `var_x` keeps temporal node values explicit
- `graph` carries adjacency, `edge_index`, or equivalent topology
- optional features stay explicit instead of leaking through global state
- output shape should match the graph prediction label contract

### `grid_prediction`

Recommended target interface:

```python
forward(var_x, marker_x=None, marker_y=None, spatial_context=None)
```

Typical expectations:

- `var_x` preserves grid axes
- `spatial_context` carries masks, coordinates, or static grid features when they are task-defining
- output shape should match the grid prediction label contract

## Config-section mapping

Keep experiment presets flat and grouped by:

- `# model`
- `# data`
- `# train`
- `# runtime`

Task-specific side inputs should remain explicit config keys rather than hidden inside undocumented conventions.
