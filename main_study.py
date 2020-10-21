from postprocessing.results import ResultsSubject, ResultsAllPatientsAllExp, ResultsAllExp
from postprocessing.postprocessing import postprocessing
from preprocessing.preprocessing import preprocessing_idiab_full, preprocessing_idiab_select
import argparse
import itertools
import torch
import operator
import numpy as np
import misc.constants as cs
from misc.utils import printd, print_latex
from processing.cross_validation import make_predictions
from misc.utils import locate_params, locate_model
import os


def combinations(dataset, model, params, mode, ph, features_comb, number_comb, patients):
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

    if patients is None:
        patients = range(1, 7)
    else:
        patients = list(map(int, patients.split(',')))

    printd("Dataset:", dataset, "-------- Patients:", ", ".join(str(patient) for patient in patients),
           "-------- Features:", "glucose,", ", ".join(all_feat), "-------- Model:", model, "-------- Params:", params,
           "-------- Mode:", mode, "-------- Horizon:", ph, "minutes")
    return combs, patients


def study(dataset, model, params, mode, ph, patients, combs):
    # retrieve model's parameters
    params = locate_params(params)
    model_class = locate_model(model)

    # scale variables in minutes to the benchmark sampling frequency
    ph_f = ph // cs.freq
    hist_f = params["hist"] // cs.freq
    day_len_f = cs.day_len // cs.freq

    # full processing
    for i in patients:
        dir = os.path.join(cs.path, "study", dataset, model, mode, "patient " + str(i))
        """ PREPROCESSING ALL FEATURES"""
        printd("Preprocessing patient " + str(i))
        data = preprocessing_idiab_full(dataset, str(i), ph_f, hist_f, day_len_f)

        for ele in combs:
            printd("Preprocessing patient", str(i), "with features glucose " + " + ".join(ele))
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


def results_metrics(combs, model, mode, ph, dataset, patients):
    experiments = [" + ".join(ele) if len(ele) > 1 else ele[0] if len(ele) == 1 else "reference" for ele in combs]
    ResultsAllPatientsAllExp(model, mode, experiments, ph, dataset)
    for i in patients:
        ResultsAllExp(model, mode, experiments, ph, dataset, str(i))


def top_model_all(mode, metrics, patients=None):
    best = {}
    if patients is None:
        printd("-------------------------------- Global -------------------------------")
        file = os.path.join(cs.path, "study", "idiab", "lstm", mode, "metrics.npy")
        param, results = np.load(file, allow_pickle=True)
        best["global"] = top_model(results, metrics)
    else:
        for i in patients:
            printd("-------------------------------- Patient", str(i), "--------------------------------")
            file = os.path.join(cs.path, "study", "idiab", "lstm", mode, "patient " + str(i), "results.npy")
            param, results = np.load(file, allow_pickle=True)
            best["patient " + str(i)] = top_model(results, metrics)
    return best


def top_model(results, metrics):
    best = {}
    tops = ["RMSE", "MAPE", "TG", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
    for top in tops:
        mean = {key: results[key][0][top] for key in results.keys()}
        mean_sorted = sorted(mean.items(), key=operator.itemgetter(1))[:5]
        mean_sorted.insert(0, ("reference", results["reference"][0][top]))
        stats = {key: {key_1: (results[key][0][key_1],
                               (100 * (results[key][0][key_1] / results["reference"][0][key_1] - 1)))
                       for key_1 in metrics} for key, val in mean_sorted}
        print_dict_latex(stats)
        best[top] = stats
    return best


def print_dict_latex(stats):
    print("\\begin{tabular}{l|| *{6}{c|}}")
    print("\\hline \\hline", "\n\\textbf{Models} & \\textbf{RMSE} & \\textbf{MAPE} & \\textbf{TG} & \\textbf{AP} "
                             "\\textbf{BE} & \\textbf{EP} \\\\ \n\\hline \\hline")
    cpt = 0
    for key in stats.keys():
        mean = [stats[key][key_1][0] for key_1 in stats[key].keys()]
        diff = [stats[key][key_1][1] for key_1 in stats[key].keys()]
        if cpt == 0:
            print_latex(mean, diff, key)
            print("\\hline")
        else:
            print_latex(mean, diff, str(cpt) + "/" + " glucose + " + key)
        cpt += 1
    print("\\hline \\hline")
    print("\\end{tabular}")


def print_dict_latex_physio(stats):
    print("\\begin{tabular}{l|| *{6}{c|}}")
    print("\\hline \\hline", "\n\\textbf{Features impact} & \\textbf{RMSE} & \\textbf{MAPE} & \\textbf{TG} & "
                             "\\textbf{AP} & \\textbf{BE} & \\textbf{EP} \\\\ \n\\hline")
    for key in stats.keys():
        mean = [stats[key][key_1][0] for key_1 in stats[key].keys()]
        diff = [stats[key][key_1][1] for key_1 in stats[key].keys()]
        print_latex(mean, diff, key)
    print("\\hline \\hline")
    print("\\end{tabular}")


def print_dict_stats_physio(stats):
    print("\\begin{tabular}{l|| *{6}{c|}}")
    print("\\hline \\hline", "\n\\textbf{Features impact} & \\textbf{RMSE} & \\textbf{MAPE} & \\textbf{TG} & "
                             "\\textbf{AP} & \\textbf{BE} & \\textbf{EP} \\\\ \n\\hline")
    for key in stats.keys():
        mean = [stats[key][key_1] for key_1 in stats[key].keys()]
        diff = [stats[key][key_1] for key_1 in stats[key].keys()]
        print_latex(mean, diff, " vs ".join(key))
    print("\\hline \\hline")
    print("\\end{tabular}")


def comparison(results, variable, metrics):
    key_variable = [ele for ele in results.keys() if variable in ele]
    variable_ante = {"CPB": "CHO", "IOB": "insulin", "AOB": "steps"}
    var_ante = variable_ante[variable]
    key_variable_ante = [ele for ele in results.keys() if var_ante in ele]
    dict_stats = {(key_, key): {key_1: 100 * (results[key][0][key_1] / results[key_][0][key_1] - 1)
                                for key_1 in metrics}
                  for key, key_ in zip(key_variable, key_variable_ante)}
    dict_mean = {key: (np.mean([ele[key] for ele in list(dict_stats.values())]),
                       np.std([ele[key] for ele in list(dict_stats.values())]))
                 for key in metrics}
    return dict_stats, dict_mean


def comparison_all(mode, variables, metrics, patients=None):
    compare_dict = {}
    compare_mean = {}
    if patients is None:
        printd("-------------------------------- Global -------------------------------")
        file = os.path.join(cs.path, "study", "idiab", "lstm", mode, "metrics.npy")
        param, results = np.load(file, allow_pickle=True)
        compare_dict["global"] = {}
        compare_mean["global"] = {}
        for variable in variables:
            compare_dict["global"][variable], compare_mean["global"][variable] = comparison(results, variable, metrics)
            print_dict_stats_physio(compare_dict["global"][variable])
        print_dict_latex_physio(compare_mean["global"])

    else:
        for i in patients:
            printd("-------------------------------- Patient", str(i), "--------------------------------")
            file = os.path.join(cs.path, "study", "idiab", "lstm", mode, "patient " + str(i), "results.npy")
            param, results = np.load(file, allow_pickle=True)
            compare_dict["patient " + str(i)] = {}
            compare_mean["patient " + str(i)] = {}
            for variable in variables:
                compare_dict["patient " + str(i)][variable], compare_mean["patient " + str(i)][variable] = \
                    comparison(results, variable, metrics)
                print_dict_stats_physio(compare_dict["patient " + str(i)][variable])
            print_dict_latex_physio(compare_mean["patient " + str(i)])
    return compare_dict, compare_mean


def visualization(patients, mode, var_physio=None, metrics=None):
    if metrics is None:
        metrics = ["RMSE", "MAPE", "TG", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
    else:
        metrics = metrics.split(',')
    best_dict = top_model_all(mode, metrics, patients)
    best_dict["global"] = top_model_all(mode, metrics)["global"]
    if var_physio is None:
        var_physio = ["CPB", "IOB", "AOB"]
    else:
        var_physio = var_physio.split(',')
    compare_dict, compare_mean = comparison_all(mode, var_physio, metrics, patients)
    comp_dict, comp_mean = comparison_all(mode, var_physio, metrics)
    compare_dict["global"], compare_mean["global"] = comp_dict["global"], comp_mean["global"]
    return best_dict, compare_dict, compare_mean


def visualization_old(patients, mode):
    for i in patients:
        printd("-------------------------------- Patient", str(i), "--------------------------------")
        file = os.path.join(cs.path, "study", "idiab", "lstm", mode, "patient " + str(i), "results.npy")
        param, results = np.load(file, allow_pickle=True)
        mean_rmse = {key: results[key][0]["RMSE"] for key in results.keys()}
        min_rmse = min(mean_rmse, key=lambda k: mean_rmse[k])
        printd("Ref", results["reference"][0]["RMSE"])
        printd(results[min_rmse][0]["RMSE"] / results["reference"][0]["RMSE"])
        printd("The best RMSE model for patient", str(i), "is", min_rmse, "with ", results[min_rmse])
        mean_mape = {key: results[key][0]["MAPE"] for key in results.keys()}
        min_mape = min(mean_mape, key=lambda k: mean_mape[k])
        printd("The best MAPE model for patient", str(i), "is", min_mape, "with ", results[min_mape])
        mean_mase = {key: results[key][0]["MASE"] for key in results.keys()}
        min_mase = min(mean_mase, key=lambda k: mean_mase[k])
        printd("The best MASE model for patient", str(i), "is", min_mase, "with ", results[min_mase])

    printd("-------------------------------- Global -------------------------------")
    file = os.path.join(cs.path, "study", "idiab", "lstm", mode, "metrics.npy")
    param, results = np.load(file, allow_pickle=True)
    mean_rmse = {key: results[key][0]["RMSE"] for key in results.keys()}
    min_rmse = min(mean_rmse, key=lambda k: mean_rmse[k])
    printd("The best global RMSE model is", min_rmse, "with ", results[min_rmse])
    mean_mape = {key: results[key][0]["MAPE"] for key in results.keys()}
    min_mape = min(mean_mape, key=lambda k: mean_mape[k])
    printd("The best global MAPE model is", min_mape, "with ", results[min_mape])
    mean_mase = {key: results[key][0]["MASE"] for key in results.keys()}
    min_mase = min(mean_mase, key=lambda k: mean_mase[k])
    printd("The best global MASE model is", min_mase, "with ", results[min_mase])


def main(dataset, model, params, mode, ph, number_comb=None, features_comb=None, patients=None):
    """ main study """
    """ FEATURES COMBINATIONS AND PATIENTS """
    combs, patients = combinations(dataset, model, params, mode, ph, features_comb, number_comb, patients)
    #
    # """ FULL PROCESSING PIPELINE """
    study(dataset, model, params, mode, ph, patients, combs)
    #
    """ GLOBAL RESULTS"""
    # results_metrics(combs, model, mode, ph, dataset, patients)

    """ RESULTS VISUALIZATION"""
    # visualization(patients, mode)


if __name__ == "__main__":
    """ The main function contains the following optional parameters:
            --dataset: which dataset to use, should be referenced in misc/datasets.py;
            --subject: which subject to use, part of the dataset, use the spelling in misc/datasets.py;
            --model: model on which the benchmark will be run (e.g., "svr"); need to be lowercase; 
            --params: parameters of the model, usually has the same name as the model (e.g., "svr"); 
              need to be lowercase; 
            --ph: the prediction horizon of the models; default 30 minutes;
            --exp: experimental folder in which the data will be stored, inside the results directory;
            --mode: specify is the model is tested on the validation "valid" set or testing "test" set ;

        Example:
            python main_study.py --dataset=idiab --model=lstm --params=lstm_ft --ph=30 --mode=valid 
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
