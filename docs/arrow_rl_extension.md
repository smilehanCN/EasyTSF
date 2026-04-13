# ARROW RL Extension Boundary

EasyTSF now integrates ARROW stage 1 under the existing `weatherbench` task.

The adaptive rollout scheduler from ARROW stage 2 is not wired into the current runtime because it needs workflow surfaces that EasyTSF does not yet expose:

- action-space configuration for allowed rollout intervals
- scheduler checkpoints coupled to a pretrained weather backbone
- replay buffer and off-policy update flow
- reward configuration and trajectory logging
- inference workflows that can switch between fixed and adaptive routing

A future stage-2 integration should add a dedicated weather-rollout workflow or task surface instead of overloading the supervised `weatherbench` path.
