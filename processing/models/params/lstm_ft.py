parameters = {
    "domain_adversarial": False,
    "domain_weights": False,
    "da_lambda": 0,
    "hist": 180,

    # model hyperparameters
    "hidden": [256, 256],
    "dropout_weights": 0.0,
    "dropout_layer": 0.0,
    "epochs": 500,
    "batch_size": 50,  # 50,
    "lr": 1e-3,
    "l2": 1e-4,
    "patience": 50,
}

search = {}
