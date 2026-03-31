# Migration Playbook

## Step 1: Inventory the External Model

Extract the following from the source project:

- model class name and file location
- constructor parameters
- forward signature
- expected input tensor layouts
- optional branches that depend on time features
- auxiliary losses or custom optimizer parameter groups
- config keys that materially affect behavior

Ignore training boilerplate until the model contract is clear.

## Step 2: Make a Compatibility Decision

Ask one question first:

Can this model run on the current EasyTSF `mtsf` path with only `var_x`, `marker_x`, `marker_y`, flat config keys, and the current dataset layout?

If no, stop and produce an incompatibility report. Do not silently add new repository abstractions.

## Step 3: Build the EasyTSF Mapping

For compatible models, produce:

- target `Model.__init__` parameters
- target `forward(var_x, marker_x, marker_y)` behavior
- internal transpose or reshape steps needed to match source layout
- output normalization back to `[B, pred_len, N]`
- config keys that belong in `# model`, `# data`, `# train`, and `# runtime`

Favor explicit constructor parameters over generic config containers.

## Step 4: Create the EasyTSF Surface

Minimum maintained migration output:

1. `easytsf/model/<model_id>.py`
2. registry entry in `easytsf/model/registry.py`
3. one example preset in `config/experiments/<model_id>/`

Benchmark wiring is optional and should only be added when the migrated path is ready for search.

## Step 5: Validate Lightly

Prefer lightweight validation:

```bash
python -m compileall easytsf
```

If local data is available, run a small smoke experiment on the new preset.

## Reporting Style

When the user asks for an assessment rather than implementation, keep the answer structured:

1. Compatibility verdict
2. Unsupported inputs or assumptions
3. Constructor mapping
4. Forward adaptation
5. Config keys
6. Example preset fields
