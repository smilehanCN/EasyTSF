# EasyTSF Repo Contract

## Public contract shape

EasyTSF documents prediction work in four layers:

1. `data contract`
2. `task contract`
3. `model interface`
4. `workflow surface`

Each prediction task should make those layers explicit instead of hiding behavior inside one monolithic runtime path.

## Task taxonomy

The documented task taxonomy is:

- `sequence_prediction`
- `graph_prediction`
- `grid_prediction`

## Current repository note

The current runnable implementation still centers on the existing sequence-oriented `mtsf` path:

```text
dataset -> MTSDataModule -> MTSFTask -> Model -> experiment/benchmark/report
```

Use that path as the closest concrete implementation for `sequence_prediction` today. For graph and grid models, the correct output is a repository extension plan, not a forced downgrade into sequence assumptions.

## Config flow

- experiment presets live under `config/experiments/<model_id>/`
- benchmark configs live under `config/benchmarks/<model_id>/`
- runtime merge order is fixed: `experiment preset < runtime overrides`
- the current `MTSFTask._build_model()` path reads flat constructor keys by parameter name

When adapting new prediction tasks, keep constructor requirements explicit and avoid hiding them inside opaque nested objects or `**kwargs`.

## Boundary rule

Do not treat sequence assumptions as universal. If the target model needs graph topology, grid tensors, or richer task state, surface the missing repository layers explicitly:

- data artifacts
- task inputs
- model interface
- workflow/config additions
