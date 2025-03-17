from ray import tune

param_space = {
    "lr": tune.grid_search([0.01, 0.001, 0.0001]),
}
