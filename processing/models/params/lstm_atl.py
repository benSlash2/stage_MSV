parameters = {
    "domain_adversarial": True,
    "da_lambda": 10**(-1.75),
    "domain_weights": True,
    "hist": 180,

    # model hyperparameters
    "hidden": [256,256], #[[16,16,16],[64,64],[256],[256,256]],

    # training hyperparameters
    # "dropout": 0.0,
    # "recurrent_dropout": 0.3, #0.9, #0.5,
    "dropout_weights": 0.0,
    "dropout_layer": 0.0,
    "epochs": 500,
    "batch_size": 50,
    "lr": 1e-3,
    "l2": 1e-4, #[0.0, 1e-5, 1e-4, 1e-3], #0.0,
    "patience": 50,
}

search = {
}

