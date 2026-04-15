# Current Repo Surface

Inspect the repository before finalizing a workflow answer.

## Current runnable implementation

- task registry currently exposes the existing `mtsf` path
- workflow entrypoints already exist for:
  - `experiment`
  - `benchmark`
  - `report`

## Current registered models

The source of truth is `easytsf/model/registry.py`.

At the moment it registers:

- `iTransformer`
- `TQNet`
- `STGCN`
- `STID`
- `SparseTSF`

## Current experiment presets

The maintained preset examples currently live under `config/experiments/tqnet/`.

## Planning rule

If the request targets graph or grid prediction, keep the current sequence implementation visible as a reference point, but produce the missing workflow layers explicitly instead of pretending the task is already runnable.
