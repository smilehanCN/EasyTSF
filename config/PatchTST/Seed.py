from ray import tune

param_space = {
    "seed": tune.grid_search([0, 1, 2, 3]),
}
