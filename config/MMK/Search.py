from ray import tune

param_space = {
    "pred_len": tune.grid_search([96, 192, 336, 720]),
    
    "lr": tune.grid_search([0.01, 0.004, 0.001, 0.0004, 0.0001]),
    "seed": tune.grid_search([0]),
}
