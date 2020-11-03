from postprocessing.results import ResultsAllPatientsAllExp, ResultsAllExp


def results_metrics(combs, model, mode, ph, dataset, patients):
    experiments = [" + ".join(ele) if len(ele) > 1 else ele[0] if len(ele) == 1 else "reference" for ele in combs]
    ResultsAllPatientsAllExp(model, mode, experiments, ph, dataset)
    for i in patients:
        ResultsAllExp(model, mode, experiments, ph, dataset, str(i))
