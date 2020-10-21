import misc.constants as cs
from main_tl import *
from misc.utils import locate_search
from processing.hyperparameters_tuning import compute_coarse_params_grid


def main_gs(args_):
    model = locate_model(args_.model)
    params_original = locate_params(args_.params)
    search = locate_search(args_.params)

    hist_f = params_original["hist"] // freq

    save_file = compute_weights_file(model, args_.source_dataset, args_.target_dataset, args_.target_subject,
                                     args_.weights)
    weights_file = compute_weights_file(model, args_.source_dataset, args_.target_dataset, args_.target_subject,
                                        args_.weights)

    # redirect the logs to a file if specified
    if args_.log is not None:
        log_file = args_.log
        log_path = os.path.join(path, "logs", log_file)
        sys.stdout = open(log_path, "w")

    """ compute the params grid """
    params_grid = compute_coarse_params_grid(params_original, search)

    train_source, valid_source, test_source, scalers_source = preprocessing_source_multi(args_.source_dataset,
                                                                                         args_.target_dataset,
                                                                                         args_.target_subject, ph_f,
                                                                                         hist_f, day_len_f)
    train_target, valid_target, test_target, scalers_target = preprocessing(args_.target_dataset, args_.target_subject,
                                                                            ph_f, hist_f, day_len_f)

    for params in params_grid:
        sbj_msg = args_.source_dataset + "_2_" + args_.target_dataset, " " + args_.target_subject

        params_ft = params.copy()
        params_ft["lr"] /= 10

        """ Source training """
        make_predictions_tl(args_.target_subject, model, params, ph_f, train_source, valid_source, test_source,
                            tl_mode="source_training", save_model_file=save_file, eval_mode=args_.eval_mode)

        """ Target global """
        raw_results_global = make_predictions_tl(args_.target_subject, model, params, ph_f, train_target, valid_target,
                                                 test_target,
                                                 weights_file=weights_file, tl_mode="target_global",
                                                 eval_mode=args_.eval_mode)
        res_global = evaluation(raw_results_global, scalers_target, args_.source_dataset, args_.target_dataset,
                                args_.target_subject, model, params, args_.exp, args_.plot,
                                "target_global")

        """ Target finetuning """

        raw_results_ft = make_predictions_tl(args_.target_subject, model, params, ph_f, train_target, valid_target,
                                             test_target,
                                             weights_file=weights_file, tl_mode="target_finetuning",
                                             eval_mode=args_.eval_mode)
        res_ft = evaluation(raw_results_ft, scalers_target, args_.source_dataset, args_.target_dataset,
                            args_.target_subject, model, params, args_.exp, args_.plot,
                            "target_finetuning")

        with open(os.path.join(cs.path, "logs", args_.exp + "_gs_log.txt"), 'a+') as f:
            print(sbj_msg)
            print("params:", params, file=f)
            print("res_global:", res_global, file=f)
            print("res_ft:", res_ft, "\n", file=f)


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
    args = parser.parse_args()

    main_gs(args)
