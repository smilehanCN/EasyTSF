# EasyTSF Repo Contract

## Public Contract Shape

EasyTSF describes every maintained prediction path in four layers:

1. data contract
2. task contract
3. model interface
4. workflow surface

Each task should make those layers explicit instead of hiding behavior inside one monolithic runtime path.

## Task Taxonomy

- `sequence_prediction`
- `grid_prediction`
- `graph_prediction`

`graph_prediction` is currently an extension target. Do not present it as runnable unless the repository has a real graph data module, task, model interface, and preset.

## Current Runnable Paths

```text
sequence dataset -> MTSDataModule -> MTSFTask -> sequence model -> experiment/benchmark/report
grid3d dataset   -> Grid3DDataModule -> Grid3DForecastingTask -> grid model -> experiment/benchmark/report
```

## Config Flow

- Experiment presets live under `config/experiments/<model_id>/`.
- Benchmark configs live under `config/benchmarks/<model_id>/`.
- Runtime merge order is fixed: `experiment preset < runtime overrides < benchmark param_space`.
- Model constructor requirements should be flat, explicit config keys.
- Benchmark files are Python modules that export `benchmark_config`.

## Boundary Rule

Do not treat sequence assumptions as universal. If a source model needs graph topology, grid coordinates beyond the maintained Grid3D path, masks, static fields, selected target channels, or physical metrics, surface the required task/data/model layers explicitly.
