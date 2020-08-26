parameters = {
    "domain_adversarial": False, #True,
    "da_lambda": 0, #10**(-2.5),
    "domain_weights": False, #True,
    "hist": 180,

    # model hyperparameters
    "n_features_emb": 16, #64
    "n_hidden_rnn": 128,
    "n_layers_rnn": 1,
    "reverse_time": False,
    "bidirectional": False,

    # training_old hyperparameters
    "emb_dropout": 0.0,
    "ctx_dropout": 0.0,
    "dropout_layer": 0.0,
    "epochs": 500,
    "batch_size": 50,
    "lr": 1e-3,
    "l2": 0.0,
    # "l2": 1e-4,
    "patience": 25, #100
}

search = {
}
