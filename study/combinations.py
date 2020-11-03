import itertools
from misc.utils import printd


def all_features(dataset):
    if dataset == "idiab":
        return ["CHO", "insulin", "mets", "heartrate", "steps", "CPB", "IOB", "AOB"]
    elif dataset == "ohio":
        return ["CHO", "insulin", "CPB", "IOB"]
    elif dataset == "t1dms":
        return ["CHO", "insulin", "CPB", "IOB"]


def combinations(dataset, model, params, mode, ph, features_comb, number_comb, patients):
    """
    Return a set of combinations which will be used during the processing phase.
    :param dataset: samples Dataframe
    :param model: constant for model
    :param params: choose to display the
    :param mode:
    :param ph:
    :param features_comb:
    :param number_comb:
    :param patients:
    :return: list of combinations, list of patients
    """
    if features_comb is None:
        all_feat = all_features(dataset)
    else:
        all_feat = features_comb.split(',')

    combs = []
    if number_comb is None:
        number_comb = range(0, len(all_feat) + 1)
    else:
        number_comb = list(map(int, number_comb.split(',')))

    for i in number_comb:
        els = [list(x) for x in itertools.combinations(all_feat, i)]
        combs.extend(els)

    combs = [ele for ele in combs if
             ("CPB" not in ele or "CHO" not in ele) and ("IOB" not in ele or "insulin" not in ele) and (
                     "AOB" not in ele or "steps" not in ele)]
    # 107 combinations * 6 patients * 5 seeds * 5 sets = 32100 models to train !!

    if patients is None:
        patients = range(1, 7)
    else:
        patients = list(map(int, patients.split(',')))

    printd("Dataset:", dataset, "-------- Patients:", ", ".join(str(patient) for patient in patients),
           "-------- Features:", "glucose,", ", ".join(all_feat), "-------- Model:", model, "-------- Params:", params,
           "-------- Mode:", mode, "-------- Horizon:", ph, "minutes")
    return all_feat, combs, patients
