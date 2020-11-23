from study.combinations import combinations
from study.results_metrics import results_metrics
from study.study import study
from study.visualization import visualization
import argparse


def main(dataset, model, params, mode, ph, number_comb=None, features_comb=None, patients=None):

    """ FEATURES COMBINATIONS AND PATIENTS """
    all_feat, combs, patients = combinations(dataset, model, params, mode, ph, features_comb, number_comb, patients)

    """ FULL PROCESSING PIPELINE """
    study(dataset, model, params, mode, ph, all_feat, patients, combs)

    """ GLOBAL RESULTS"""
    results_metrics(combs, model, mode, ph, dataset, patients)

    """ RESULTS VISUALIZATION"""
    visualization(patients, mode)


if __name__ == "__main__":
    """ The main function contains the following optional parameters:
            --dataset: which dataset to use, should be referenced in misc/datasets.py;
            --subject: which subject to use, part of the dataset, use the spelling in misc/datasets.py;
            --model: model on which the benchmark will be run (e.g., "svr"); need to be lowercase; 
            --params: parameters of the model, usually has the same name as the model (e.g., "svr"); 
              need to be lowercase; 
            --mode: specify is the model is tested on the validation "valid" set or testing "test" set;              
            --ph: the prediction horizon of the models; default 30 minutes;
            --number_comb: the number of features combination to consider; can be a tuple (no space, no parenthesis)
            --features_comb: the features among which combinations will be taken; no space, no parenthesis;
            --patients: which patients to use; can be a tuple (no space, no parenthesis)

        Example:
            python main_study.py --dataset=idiab --model=lstm --params=lstm_ft --mode=valid --ph=30 --number_comb = 1,2
                                 --patients = 3,5
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
