parameters = {
    "hist": 180,

    "n_estimators": 10000,
    "max_depth": None,

    """ 250 for idiab and 2000 for ohio """
    # "min_samples_split": 250,  # IDIAB
    "min_samples_split": 2000,  # OhioT1DM

    "learning_rate": 0.1,
    "loss": "ls",
    "n_iter_no_change": 10,
}

search = {}
