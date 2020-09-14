from postprocessing.results import ResultsSubject
from postprocessing.postprocessing import postprocessing
from preprocessing.preprocessing import preprocessing, preprocessing_idiab_full, preprocessing_idiab_select
import sys
import argparse
import itertools
import time
from os.path import join
import torch
import numpy as np
from pydoc import locate
import misc.constants as cs
from misc.utils import printd
from processing.cross_validation import make_predictions, find_best_hyperparameters
from misc.utils import locate_params, locate_model, locate_search
import os


def main(dataset, model, params, exp, mode, log, ph, plot, save=False):
    all_feat = ["CHO", "insulin", "calories", "mets", "heartrate", "steps", "CPB", "IOB", "AOB"]
    combs = []
    for i in range(1, len(all_feat) + 1):
        els = [list(x) for x in itertools.combinations(all_feat, i)]
        combs.extend(els)

    for ele in combs:
        if ("CPB" in ele and "CHO" in ele) or ("IOB" in ele and "insulin" in ele) or ("AOB" in ele and "steps" in ele):
            combs.remove(ele)

    for i in range(1, 7):
        """ PREPROCESSING """
        printd(dataset, "Patient" + str(i), model, params, exp, mode, log, ph, plot)
        # retrieve model's parameters
        search = locate_search(params)
        params = locate_params(params)
        model_class = locate_model(model)

        # scale variables in minutes to the benchmark sampling frequency
        ph_f = ph // cs.freq
        hist_f = params["hist"] // cs.freq
        day_len_f = cs.day_len // cs.freq

        data = preprocessing_idiab_full(dataset, i, ph_f, hist_f, day_len_f)
        for ele in combs:
            train, valid, test, scalers = preprocessing_idiab_select(data, dataset,day_len_f, ele)
            for j in range(10):
                torch.manual_seed(j)
                """ MODEL TRAINING & TUNING """
                if search:
                    params = find_best_hyperparameters(i, model_class, params, search, ph_f, train, valid, test)

                if save:
                    dir = os.path.join(cs.path, "processing", "models", "weights", model_class.__name__, exp)
                    file = os.path.join(dir, model_class.__name__ + "_" + dataset + subject + ".pt")
                else:
                    file = None

                raw_results = make_predictions(i, model_class, params, ph_f, train, valid, test, mode=mode, save_model_file=file)
                """ POST-PROCESSING """
                raw_results = postprocessing(raw_results, scalers, dataset)

                """ EVALUATION """
                results = ResultsSubject(model, exp, ph, dataset, i, params=params, results=raw_results)
                printd(results.compute_mean_std_results())

                if plot:
                    results.plot(0)


if __name__ == "__main__":
    """ The main function contains the following optional parameters:
            --dataset: which dataset to use, should be referenced in misc/datasets.py;
            --subject: which subject to use, part of the dataset, use the spelling in misc/datasets.py;
            --model: model on which the benchmark will be run (e.g., "svr"); need to be lowercase; 
            --params: parameters of the model, usually has the same name as the model (e.g., "svr"); need to be lowercase; 
            --ph: the prediction horizon of the models; default 30 minutes;
            --exp: experimental folder in which the data will be stored, inside the results directory;
            --mode: specify is the model is tested on the validation "valid" set or testing "test" set ;
            --plot: if a plot of the predictions shall be made at the end of the training;
            --log: file where the standard outputs will be redirected to; default: logs stay in stdout; 

        Example:
            python main_tl.py --dataset=ohio --subject=559 --model=base --params=base --ph=30 
                        --exp=myexp --mode=valid --plot=1 --log=mylog
    """

    """ ARGUMENTS HANDLER """
    # retrieve and process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--params", type=str)
    parser.add_argument("--ph", type=int)
    parser.add_argument("--exp", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--plot", type=int)
    parser.add_argument("--log", type=str)
    parser.add_argument("--save", type=int)
    args = parser.parse_args()

    # compute stdout redirection to log file
    if args.log:
        sys.stdout = open(join(cs.path, "logs", args.log + ".log"), "w")

    main(log=args.log,
         subject=args.subject,
         model=args.model,
         ph=args.ph,
         params=args.params,
         exp=args.exp,
         dataset=args.dataset,
         mode=args.mode,
         plot=args.plot,
         save=args.save)