from postprocessing.results import ResultsSubject, ResultsAllPatientsAllExp, ResultsAllExp
from postprocessing.postprocessing import postprocessing
from preprocessing.preprocessing import preprocessing_idiab_full, preprocessing_idiab_select
import argparse
import itertools
import torch
import numpy as np
import misc.constants as cs
from misc.utils import printd
from processing.cross_validation import make_predictions
from misc.utils import locate_params, locate_model
import os


def main(dataset, model, params, mode, ph, number_comb=None, features_comb=None, patients=None):

    """ FEATURES COMBINATIONS """
    if features_comb is None:
        all_feat = ["CHO", "insulin", "mets", "heartrate", "steps", "CPB", "IOB", "AOB"]
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

    printd("Dataset:", dataset, "-------- Model:", model, "-------- Params:", params, "-------- Mode:", mode,
           "-------- Horizon:", ph, "minutes")

    # retrieve model's parameters
    params = locate_params(params)
    model_class = locate_model(model)

    # scale variables in minutes to the benchmark sampling frequency
    ph_f = ph // cs.freq
    hist_f = params["hist"] // cs.freq
    day_len_f = cs.day_len // cs.freq

    if patients is None:
        patients = range(1, 7)
    else:
        patients = list(map(int, patients.split(',')))

    for i in patients:
        dir = os.path.join(cs.path, "study", dataset, model, mode, "patient " + str(i))
        """ PREPROCESSING ALL FEATURES"""
        printd("Preprocessing patient " + str(i))
        data = preprocessing_idiab_full(dataset, str(i), ph_f, hist_f, day_len_f)

        for ele in combs:
            printd("Preprocessing patient", str(i), "with features " + " + ".join(ele))
            train, valid, test, scalers = preprocessing_idiab_select(data, dataset, day_len_f, ele)

            for j in range(5):
                torch.manual_seed(j)
                """ MODEL TRAINING & TUNING """
                if not ele:
                    file = os.path.join(dir, "reference", "seed " + str(j), "weights", "weights")
                else:
                    file = os.path.join(dir, " + ".join(ele), "seed " + str(j), "weights", "weights")
                raw_results = make_predictions(str(i), model_class, params, ph_f, train, valid, test, mode=mode,
                                               save_model_file=file)
                """ POST-PROCESSING """
                raw_results = postprocessing(raw_results, scalers, dataset)
                """ EVALUATION """
                if not ele:
                    file_save = os.path.join("reference", "seed " + str(j))
                else:
                    file_save = os.path.join(" + ".join(ele), "seed " + str(j))
                results = ResultsSubject(model, file_save, ph, dataset, str(i), params=params,
                                         results=raw_results, study=True, mode=mode)
                printd(results.compute_mean_std_results())

    """ GLOBAL RESULTS"""
    experiments = [" + ".join(ele) if len(ele) > 1 else ele[0] if len(ele) == 1 else "reference" for ele in combs]
    ResultsAllPatientsAllExp(model, mode, experiments, ph, dataset)
    for i in patients:
        ResultsAllExp(model, mode, experiments, ph, dataset, str(i))

    """ RESULTS VISUALIZATION"""
    for i in patients:
        printd("-------------------------------- Patient", str(i), "--------------------------------")
        file = os.path.join(cs.path, "study", "idiab", "lstm", "valid", "patient " + str(i), "results.npy")
        param, results = np.load(file, allow_pickle=True)
        mean_RMSE = {key:results[key][0]["RMSE"] for key in results.keys()}
        min_RMSE = min(mean_RMSE, key = lambda k:mean_RMSE[k])
        printd("Ref", results["reference"][0]["RMSE"])
        printd(results[min_RMSE][0]["RMSE"] / results["reference"][0]["RMSE"])
        printd("The best RMSE model for patient", str(i), "is", min_RMSE, "with ", results[min_RMSE])
        mean_MAPE = {key: results[key][0]["MAPE"] for key in results.keys()}
        min_MAPE = min(mean_MAPE, key=lambda k: mean_MAPE[k])
        printd("The best MAPE model for patient", str(i), "is", min_MAPE, "with ", results[min_MAPE])
        mean_MASE = {key: results[key][0]["MASE"] for key in results.keys()}
        min_MASE = min(mean_MASE, key=lambda k: mean_MASE[k])
        printd("The best MASE model for patient", str(i), "is", min_MASE, "with ", results[min_MASE])

    printd("-------------------------------- Global -------------------------------")
    file = os.path.join(cs.path, "study", "idiab", "lstm", "valid", "metrics.npy")
    param, results = np.load(file, allow_pickle=True)
    mean_RMSE = {key: results[key][0]["RMSE"] for key in results.keys()}
    min_RMSE = min(mean_RMSE, key=lambda k: mean_RMSE[k])
    printd("The best global RMSE model is", min_RMSE, "with ", results[min_RMSE])
    mean_MAPE = {key: results[key][0]["MAPE"] for key in results.keys()}
    min_MAPE = min(mean_MAPE, key=lambda k: mean_MAPE[k])
    printd("The best global MAPE model is", min_MAPE, "with ", results[min_MAPE])
    mean_MASE = {key: results[key][0]["MASE"] for key in results.keys()}
    min_MASE = min(mean_MASE, key=lambda k: mean_MASE[k])
    printd("The best global MASE model is", min_MASE, "with ", results[min_MASE])


if __name__ == "__main__":
    """ The main function contains the following optional parameters:
            --dataset: which dataset to use, should be referenced in misc/datasets.py;
            --subject: which subject to use, part of the dataset, use the spelling in misc/datasets.py;
            --model: model on which the benchmark will be run (e.g., "svr"); need to be lowercase; 
            --params: parameters of the model, usually has the same name as the model (e.g., "svr"); need to be lowercase; 
            --ph: the prediction horizon of the models; default 30 minutes;
            --exp: experimental folder in which the data will be stored, inside the results directory;
            --mode: specify is the model is tested on the validation "valid" set or testing "test" set ;

        Example:
            python main_run.py --dataset=idiab --model=lstm --params=lstm_ft --ph=30 --mode=valid 
    """

    """ ARGUMENTS HANDLER """
    # retrieve and process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--params", type=str)
    parser.add_argument("--ph", type=int)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--number_comb", type=str)
    parser.add_argument("--features_comb", type=str)
    parser.add_argument("--patients", type=str)

    args = parser.parse_args()

    main(model=args.model,
         ph=args.ph,
         params=args.params,
         dataset=args.dataset,
         mode=args.mode,
         number_comb=args.number_comb,
         features_comb=args.features_comb,
         patients=args.patients)
