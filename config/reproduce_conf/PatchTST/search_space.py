from ray import tune

param_space = {
    "d_model": tune.grid_search([128, 256, 512]),
    "e_layers": tune.grid_search([2, 3]),
    "dropout": tune.grid_search([0.1, 0.2]),
    "lr": tune.grid_search([0.01, 0.001, 0.0001]),
}
