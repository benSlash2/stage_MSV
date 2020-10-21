from misc.utils import locate_model, locate_params, printd
import sys
import argparse
import os
import torch
from misc.constants import *
from preprocessing.preprocessing import preprocessing, preprocessing_source_multi
from processing.cross_validation import make_predictions_tl
from postprocessing.postprocessing import postprocessing
from postprocessing.results import ResultsSubject
torch.manual_seed(0)


def main_target_training(source_dataset, target_dataset, target_subject, model, params, eval_mode, exp, plot):
    hist_f = params["hist"] // freq
    train, valid, test, scalers = preprocessing(target_dataset, target_subject, ph_f, hist_f, day_len_f)
    raw_results = make_predictions_tl(target_subject, model, params, ph_f, train, valid, test,
                                      eval_mode=eval_mode, fit=True, save_model_file=None)

    return evaluation(raw_results, scalers, source_dataset, target_dataset, target_subject, model, params, exp, plot,
                      "target_training")


def main_source_training(source_dataset, target_dataset, target_subject, model, params, weights_exp, eval_mode):
    hist_f = params["hist"] // freq
    save_file = compute_weights_file(model, source_dataset, target_dataset, target_subject, weights_exp)

    train, valid, test, scalers = preprocessing_source_multi(source_dataset, target_dataset, target_subject, ph_f,
                                                             hist_f, day_len_f)
    make_predictions_tl(target_subject, model, params, ph_f, train, valid, test,
                        eval_mode=eval_mode, fit=True, save_model_file=save_file)


def main_target_global(source_dataset, target_dataset, target_subject, model, params, weights_exp, eval_mode, exp,
                       plot):
    hist_f = params["hist"] // freq
    weights_file = compute_weights_file(model, source_dataset, target_dataset, target_subject, weights_exp)

    train, valid, test, scalers = preprocessing(target_dataset, target_subject, ph_f, hist_f, day_len_f)

    raw_results = make_predictions_tl(target_subject, model, params, ph_f, train, valid, test,
                                      weights_file=weights_file, eval_mode=eval_mode, fit=False, save_model_file=None)

    return evaluation(raw_results, scalers, source_dataset, target_dataset, target_subject, model, params, exp, plot,
                      "target_global")


def main_target_finetuning(source_dataset, target_dataset, target_subject, model, params, weights_exp, eval_mode, exp,
                           plot, save=False):
    hist_f = params["hist"] // freq
    weights_file = compute_weights_file(model, source_dataset, target_dataset, target_subject, weights_exp)
    if save:
        save_file = compute_weights_file(model, source_dataset, target_dataset, target_subject, weights_exp + "_ft")
    else:
        save_file = None

    train, valid, test, scalers = preprocessing(target_dataset, target_subject, ph_f, hist_f, day_len_f)

    raw_results = make_predictions_tl(target_subject, model, params, ph_f, train, valid, test,
                                      weights_file=weights_file, eval_mode=eval_mode, fit=True,
                                      save_model_file=save_file)

    return evaluation(raw_results, scalers, source_dataset, target_dataset, target_subject, model, params, exp, plot,
                      "target_finetuning")


def end_to_end(source_dataset, target_dataset, target_subject, model, params, weights_exp, eval_mode, exp,
               plot):
    hist_f = params["hist"] // freq
    save_file = compute_weights_file(model, source_dataset, target_dataset, target_subject, weights_exp)

    train_m, valid_m, test_m, scalers_m = preprocessing_source_multi(source_dataset, target_dataset, target_subject,
                                                                     ph_f, hist_f, day_len_f)
    make_predictions_tl(target_subject, model, params, ph_f, train_m, valid_m, test_m,
                        eval_mode=eval_mode, fit=True, save_model_file=save_file)

    train, valid, test, scalers = preprocessing(target_dataset, target_subject, ph_f, hist_f, day_len_f)

    raw_results = make_predictions_tl(target_subject, model, params, ph_f, train, valid, test,
                                      weights_file=save_file, eval_mode=eval_mode, fit=False, save_model_file=None)

    evaluation(raw_results, scalers, source_dataset, target_dataset, target_subject, model, params, exp, plot,
               "target_global")

    raw_results_2 = make_predictions_tl(target_subject, model, params, ph_f, train, valid, test,
                                        weights_file=save_file, eval_mode=eval_mode, fit=True,
                                        save_model_file=None)

    return evaluation(raw_results_2, scalers, source_dataset, target_dataset, target_subject, model, params, exp, plot,
                      "target_finetuning")


def evaluation(raw_results, scalers, source_dataset, target_dataset, target_subject, model, params, exp, plot, tl_mode):
    raw_results = postprocessing(raw_results, scalers, target_dataset)

    exp += "_" + tl_mode.split("_")[1]
    exp = os.path.join(source_dataset + "_2_" + target_dataset, exp)
    results = ResultsSubject(model.__name__, exp, ph, target_dataset, target_subject, params=params,
                             results=raw_results)

    res_mean = results.compute_mean_std_results()
    printd(res_mean)
    if plot:
        results.plot(0)

    return res_mean


def compute_weights_file(model, source_dataset, target_dataset, target_subject, weights_exp):
    dir = os.path.join(path, "processing", "models", "weights", source_dataset + "_2_" + target_dataset, weights_exp)
    file = os.path.join(dir, model.__name__ + "_" + target_dataset + target_subject + ".pt")
    return file


def process_main_args(args_):
    model = locate_model(args_.model)
    params = locate_params(args_.params)

    # redirect the logs to a file if specified
    if args_.log is not None:
        log_file = args_.log
        log_path = os.path.join(path, "logs", log_file)
        sys.stdout = open(log_path, "w")

    sbj_msg = args_.source_dataset + "_2_" + args_.target_dataset, " " + args_.target_subject
    if args_.tl_mode == "source_training":
        printd("source_training", sbj_msg)
        main_source_training(args_.source_dataset, args_.target_dataset, args_.target_subject, model, params,
                             args_.weights, args_.eval_mode)
    elif args_.tl_mode == "source_training_test":
        printd("source_training_test", sbj_msg)
        main_source_training(args_.source_dataset, args_.target_dataset, args_.target_subject, model, params,
                             args_.weights, args_.eval_mode)
        main_target_global(args_.source_dataset, args_.target_dataset, args_.target_subject, model, params,
                           args_.weights, args_.eval_mode, args_.exp, args_.plot)
    elif args_.tl_mode == "end_to_end_0":
        printd("end_to_end_0", sbj_msg)
        end_to_end(args_.source_dataset, args_.target_dataset, args_.target_subject, model, params,
                   args_.weights, args_.eval_mode, args_.exp, args_.plot)
    elif args_.tl_mode == "target_training":
        printd("target_training", sbj_msg)
        main_target_training(args_.source_dataset, args_.target_dataset, args_.target_subject, model, params,
                             args_.eval_mode, args_.exp, args_.plot)
    elif args_.tl_mode == "target_global":
        printd("target_global", sbj_msg)
        main_target_global(args_.source_dataset, args_.target_dataset, args_.target_subject, model, params,
                           args_.weights, args_.eval_mode, args_.exp, args_.plot)
    elif args_.tl_mode == "target_finetuning":
        printd("target_finetuning", sbj_msg)
        main_target_finetuning(args_.source_dataset, args_.target_dataset, args_.target_subject, model, params,
                               args_.weights, args_.eval_mode, args_.exp, args_.plot)
    elif args_.tl_mode == "end_to_end" and args_.params_ft is not None:
        printd("end_to_end", sbj_msg)

        params_ft = locate_params(args_.params_ft)

        main_source_training(args_.source_dataset, args_.target_dataset, args_.target_subject, model, params,
                             args_.weights, args_.eval_mode)
        main_target_global(args_.source_dataset, args_.target_dataset, args_.target_subject, model, params_ft,
                           args_.weights, args_.eval_mode, args_.exp, args_.plot)
        main_target_finetuning(args_.source_dataset, args_.target_dataset, args_.target_subject, model, params_ft,
                               args_.weights, args_.eval_mode, args_.exp, args_.plot, args_.save)


if __name__ == "__main__":
    """
        --tl_mode: 5 modes 
                "source_training":      train a model on source dataset minus the target subject
                "target_training":      train a model on the target subject only
                "target_global":        use a model trained with the "source_training" mode to make the prediction for 
                                        the target subject. --weights_file must be set.
                "target_finetuning":    finetune a model trained with the "source_training" mode on the target subject
                "end_to_end":           perform "source_training", then "target_global" and finally "target_finetuning"
        --source_dataset:
                dataset used in the "source_training" mode, can be either "IDIAB", "Ohio" or "all"
        --target_dataset and --target_subject:
                specify the subject used in the "target_X" modes and removed from the "source_training" if needed
        --model:
                specify the model used in all the modes
        --params:
                name of the hyperparameters file to use in the processing/models/params directory
        --params_ft:
                in the case of "end_to_end" mode, secondary parameter file for finetuning
        --weights:
                specify the files to be used in the "target_global" and "target_finetuning" modes
        --eval_mode:
                specify the evaluation_old set to be used, in the "target_X" modes, either "valid" or "test". default:
                "valid".
        --log:
                specify the file where the logs shall be redirected to. default: None
        --exp:
                name of the experimental settings, results or weights will be saved under this name
        --plot:
                if set, plot the results after the training. default: True  
                
        Examples:
         
        --mode=source_training --source_dataset=IDIAB --target_dataset=IDIAB --target_subject=1 --model=DA_FCN 
        --eval=valid --save=test
        
        --mode=target_global --source_dataset=IDIAB --target_dataset=IDIAB --target_subject=1 --model=FCN 
        --eval=valid --weights=test --save=test
        
        --mode=target_finetuning --source_dataset=IDIAB --target_dataset=IDIAB --target_subject=1 --model=FCN 
        --eval=valid --weights=test --save=test
        
        --mode=target_training --source_dataset=IDIAB --target_dataset=IDIAB --target_subject=1 --model=FCN 
        --eval=valid --save=test
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tl_mode", type=str)
    parser.add_argument("--source_dataset", type=str)
    parser.add_argument("--target_dataset", type=str)
    parser.add_argument("--target_subject", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--params", type=str)
    parser.add_argument("--params_ft", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--eval_mode", type=str)
    parser.add_argument("--log", type=str)
    parser.add_argument("--exp", type=str)
    parser.add_argument("--plot", type=bool)
    parser.add_argument("--save", type=bool)
    args = parser.parse_args()

    process_main_args(args)
