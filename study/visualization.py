import operator
import numpy as np
import misc.constants as cs
from misc.utils import printd, print_latex
import os


def print_dict_latex(stats, top, patient):
    print("\\begin{table}[ht]")
    print("\\resizebox{\\textwidth}{!}{\\begin{tabular}{l|| *{6}{c|}}")
    print("\\hline \\hline", "\n\\textbf{Models} & \\textbf{RMSE} & \\textbf{MAPE} & \\textbf{TG} & \\textbf{AP} "
                             "& \\textbf{BE} & \\textbf{EP} \\\\ \n\\hline \\hline")
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
    print("\\end{tabular}}")
    if patient is None:
        print("\\caption{\\detokenize{Top 5", top, "global models}}")
    else:
        print("\\caption{\\detokenize{Top 5", top, "models for patient", str(patient) + "}}")
    print("\\end{table}")


def top_model(results, metrics, patient=None):
    best = {}
    tops = ["RMSE", "MAPE", "TG", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
    for top in tops:
        mean = {key: results[key][0][top] for key in results.keys()}
        mean_sorted = sorted(mean.items(), key=operator.itemgetter(1))[:5]
        mean_sorted.insert(0, ("reference", results["reference"][0][top]))
        stats = {key: {key_1: (results[key][0][key_1],
                               (100 * (results[key][0][key_1] / results["reference"][0][key_1] - 1)))
                       for key_1 in metrics} for key, val in mean_sorted}
        print_dict_latex(stats, top, patient)
        best[top] = stats
    return best


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
            best["patient " + str(i)] = top_model(results, metrics, i)
    return best


def print_dict_latex_physio(stats, patient=None):
    print("\\begin{table}[ht]")
    print("\\resizebox{\\textwidth}{!}{\\begin{tabular}{l|| *{6}{c|}}")
    print("\\hline \\hline", "\n\\textbf{Features impact} & \\textbf{RMSE} & \\textbf{MAPE} & \\textbf{TG} & "
                             "\\textbf{AP} & \\textbf{BE} & \\textbf{EP} \\\\ \n\\hline")
    for key in stats.keys():
        mean = [stats[key][key_1][0] for key_1 in stats[key].keys()]
        diff = [stats[key][key_1][1] for key_1 in stats[key].keys()]
        print_latex(mean, diff, key)
    print("\\hline \\hline")
    print("\\end{tabular}}")
    if patient is None:
        print("\\caption{All features global impact on all the patients}")
    else:
        print("\\caption{All features global impact for patient", str(patient) + "}")
    print("\\end{table}")


def print_dict_stats_physio(stats, variable, patient=None):
    print("\\begin{table}[ht]")
    print("\\resizebox{\\textwidth}{!}{\\begin{tabular}{l|| *{6}{c|}}")
    print("\\hline \\hline", "\n\\textbf{Features impact} & \\textbf{RMSE} & \\textbf{MAPE} & \\textbf{TG} & "
                             "\\textbf{AP} & \\textbf{BE} & \\textbf{EP} \\\\ \n\\hline")
    for key in stats.keys():
        mean = [stats[key][key_1] for key_1 in stats[key].keys()]
        diff = [stats[key][key_1] for key_1 in stats[key].keys()]
        print_latex(mean, diff, " vs ".join(key))
    print("\\hline \\hline")
    print("\\end{tabular}}")
    if patient is None:
        print("\\caption{" + variable, "impact on all the patients}")
    else:
        print("\\caption{" + variable, "impact for patient", str(patient) + "}")
    print("\\end{table}")


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
            print_dict_stats_physio(compare_dict["global"][variable], variable)
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
                print_dict_stats_physio(compare_dict["patient " + str(i)][variable], variable, i)
            print_dict_latex_physio(compare_mean["patient " + str(i)], i)
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
