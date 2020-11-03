from postprocessing.results import ResultsSubject
from postprocessing.postprocessing import postprocessing
from preprocessing.preprocessing import preprocessing_full, preprocessing_select
from misc.utils import locate_params, locate_model
import misc.constants as cs
import torch
import os
from misc.utils import printd
from processing.cross_validation import make_predictions


def study(dataset, model, params, mode, ph, all_feat, patients, combs):
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
        data = preprocessing_full(dataset, str(i), ph_f, hist_f, day_len_f, all_feat)

        for ele in combs:
            printd("Preprocessing patient", str(i), "with features glucose " + " + ".join(ele))
            train, valid, test, scalers = preprocessing_select(data, dataset, day_len_f, all_feat, ele)

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
