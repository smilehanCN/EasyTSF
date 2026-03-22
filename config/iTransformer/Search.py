from ray import tune

param_space = {
    "pred_len": tune.grid_search([96, 336]),
    "d_model": tune.grid_search([128, 256]),
    "e_layers": tune.grid_search([2, 3]),
    "dropout": tune.grid_search([0.2, 0.3]),
    
    "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    "seed": tune.grid_search([0]),
}
