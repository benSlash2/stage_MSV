from misc.utils import print_latex, printd
import pandas as pd
import os
import numpy as np
from postprocessing.metrics import *
import misc.datasets
import misc.constants as cs
from pathlib import Path
import misc.datasets


class ResultsDataset:
    def __init__(self, model, experiment, ph, dataset):
        """
        Object that compute all the performances of a given dataset for a given model and experiment and prediction
        horizon
        :param model: name of the model (e.g., "base")
        :param experiment: name of the experiment (e.g., "test")
        :param ph: prediction horizons in minutes (e.g., 30)
        :param dataset: name of the dataset (e.g., "ohio")
        """

        self.model = model
        self.experiment = experiment
        self.ph = ph
        self.dataset = dataset
        self.freq = np.max([misc.constants.freq, misc.datasets.datasets[dataset]["glucose_freq"]])
        self.subjects = misc.datasets.datasets[self.dataset]["subjects"]

    def compute_results(self, details=False, study=False, mode="valid"):
        """
        Loop through the subjects of the dataset, and compute the mean performances
        :return: mean of metrics, std of metrics
        """
        res = []
        for subject in self.subjects:
            res_subject = ResultsSubject(self.model, self.experiment, self.ph, self.dataset, subject, study=study,
                                         mode=mode).compute_mean_std_results()
            if details:
                print(self.dataset, subject, res_subject)

            res.append(res_subject[0])  # only the mean

        keys = list(res[0].keys())
        res = [list(res_.values()) for res_ in res]
        mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
        return dict(zip(keys, mean)), dict(zip(keys, std))

    def compute_average_params(self, study=False, mode="valid"):
        params = []
        for subject in self.subjects:
            res_subject = ResultsSubject(self.model, self.experiment, self.ph, self.dataset, subject, study=study,
                                         mode=mode)
            params.append(res_subject.params)

        return dict(zip(params[0].keys(), np.mean([list(_.values()) for _ in params], axis=0)))

    def to_latex(self, table="acc", model_name=None):
        """
        Format the results into a string for the paper in LATEX
        :param table: either "acc" or "cg_ega", corresponds to the table
        :param model_name: prefix of the string, name of the model
        :return:
        """
        mean, std = self.compute_results()
        if table == "cg_ega":
            keys = ["CG_EGA_AP_hypo", "CG_EGA_BE_hypo", "CG_EGA_EP_hypo", "CG_EGA_AP_eu", "CG_EGA_BE_eu",
                    "CG_EGA_EP_eu", "CG_EGA_AP_hyper", "CG_EGA_BE_hyper", "CG_EGA_EP_hyper"]
            mean = [mean[k] * 100 for k in keys]
            std = [std[k] * 100 for k in keys]
        elif table == "general":
            acc_keys = ["RMSE", "MAPE", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
            mean = [mean[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else mean[k] * 100 for k in acc_keys]
            std = [std[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else std[k] * 100 for k in acc_keys]

        print_latex(mean, std, label=self.model)


class ResultsAllPatientsAllExp:
    def __init__(self, model, mode, experiments, ph, dataset):
        """
        Object that compute all the performances of a given dataset for a given model and experiment and prediction
        horizon
        :param model: name of the model (e.g., "base")
        :param experiments: names of the experiments (e.g., "test")
        :param ph: prediction horizons in minutes (e.g., 30)
        :param dataset: name of the dataset (e.g., "ohio")
        """

        self.model = model
        self.mode = mode
        self.experiments = experiments
        self.ph = ph
        self.dataset = dataset
        self.freq = np.max([misc.constants.freq, misc.datasets.datasets[dataset]["glucose_freq"]])
        self.subjects = misc.datasets.datasets[self.dataset]["subjects"]
        self.save_raw_results()

    def save_raw_results(self):
        """
        Save the results and params
        :return:
        """
        dir = os.path.join(cs.path, "study", self.dataset, self.model, self.mode)
        Path(dir).mkdir(parents=True, exist_ok=True)
        savable_results = self.compute_results()
        printd("Global metrics for all patients and all experiments saved at ", dir)
        np.save(os.path.join(dir, "metrics.npy"), [self.compute_params(), savable_results])

    def compute_results(self, details=False):
        """
        Loop through the subjects of the dataset, and compute the mean performances
        :return: mean of metrics, std of metrics
        """
        res = []
        for experiment in self.experiments:
            res_subject = ResultsAllPatients(self.model, self.mode, experiment, self.ph, self.dataset).compute_results()
            if details:
                print(self.dataset, experiment, res_subject)
            res.append(res_subject)

        return dict(zip(self.experiments, res))

    def compute_params(self):
        exp = os.path.join(self.experiments[0], "seed 0")
        res_subject = ResultsSubject(self.model, exp, self.ph, self.dataset, "1", study=True, mode=self.mode)
        return res_subject.params

    def to_latex(self, table="acc"):
        """
        Format the results into a string for the paper in LATEX
        :param table: either "acc" or "cg_ega", corresponds to the table
        :return:
        """
        mean, std = self.compute_results()
        if table == "cg_ega":
            keys = ["CG_EGA_AP_hypo", "CG_EGA_BE_hypo", "CG_EGA_EP_hypo", "CG_EGA_AP_eu", "CG_EGA_BE_eu",
                    "CG_EGA_EP_eu", "CG_EGA_AP_hyper", "CG_EGA_BE_hyper", "CG_EGA_EP_hyper"]
            mean = [mean[k] * 100 for k in keys]
            std = [std[k] * 100 for k in keys]
        elif table == "general":
            acc_keys = ["RMSE", "MAPE", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
            mean = [mean[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else mean[k] * 100 for k in acc_keys]
            std = [std[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else std[k] * 100 for k in acc_keys]

        print_latex(mean, std, label=self.model)


class ResultsAllExp:
    def __init__(self, model, mode, experiments, ph, dataset, subject):
        """
        Object that compute all the performances of a given dataset for a given model and experiment and prediction
        horizon
        :param model: name of the model (e.g., "base")
        :param experiments: names of the experiments (e.g., "test")
        :param ph: prediction horizons in minutes (e.g., 30)
        :param dataset: name of the dataset (e.g., "ohio")
        """

        self.model = model
        self.mode = mode
        self.experiments = experiments
        self.ph = ph
        self.dataset = dataset
        self.freq = np.max([misc.constants.freq, misc.datasets.datasets[dataset]["glucose_freq"]])
        self.subject = subject
        self.save_raw_results()

    def save_raw_results(self):
        """
        Save the results and params
        :return:
        """
        dir = os.path.join(cs.path, "study", self.dataset, self.model, self.mode, "patient " + self.subject)
        Path(dir).mkdir(parents=True, exist_ok=True)
        savable_results = self.compute_results()
        printd("Global results for patient", self.subject, " with all experiments saved at", dir)
        np.save(os.path.join(dir, "results.npy"), [self.compute_params(), savable_results])

    def compute_results(self, details=False):
        """
        Loop through the subjects of the dataset, and compute the mean performances
        :return: mean of metrics, std of metrics
        """
        res = []
        for experiment in self.experiments:
            res_subject = ResultsAllSeeds(self.model, self.mode, experiment, self.ph, self.dataset,
                                          self.subject).compute_results()
            if details:
                print(self.dataset, experiment, res_subject)
            res.append(res_subject)

        return dict(zip(self.experiments, res))

    def compute_params(self):
        exp = os.path.join(self.experiments[0], "seed 0")
        res_subject = ResultsSubject(self.model, exp, self.ph, self.dataset, "1", study=True,
                                     mode=self.mode)
        return res_subject.params

    def to_latex(self, table="acc"):
        """
        Format the results into a string for the paper in LATEX
        :param table: either "acc" or "cg_ega", corresponds to the table
        :return:
        """
        mean, std = self.compute_results()
        if table == "cg_ega":
            keys = ["CG_EGA_AP_hypo", "CG_EGA_BE_hypo", "CG_EGA_EP_hypo", "CG_EGA_AP_eu", "CG_EGA_BE_eu",
                    "CG_EGA_EP_eu", "CG_EGA_AP_hyper", "CG_EGA_BE_hyper", "CG_EGA_EP_hyper"]
            mean = [mean[k] * 100 for k in keys]
            std = [std[k] * 100 for k in keys]
        elif table == "general":
            acc_keys = ["RMSE", "MAPE", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
            mean = [mean[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else mean[k] * 100 for k in acc_keys]
            std = [std[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else std[k] * 100 for k in acc_keys]

        print_latex(mean, std, label=self.model)


class ResultsAllPatients:
    def __init__(self, model, mode, experiment, ph, dataset):
        """
        Object that compute all the performances of a given dataset for a given model and experiment and prediction
        horizon
        :param model: name of the model (e.g., "base")
        :param experiment: name of the experiment (e.g., "test")
        :param ph: prediction horizons in minutes (e.g., 30)
        :param dataset: name of the dataset (e.g., "ohio")
        """

        self.model = model
        self.mode = mode
        self.experiment = experiment
        self.ph = ph
        self.dataset = dataset
        self.freq = np.max([misc.constants.freq, misc.datasets.datasets[dataset]["glucose_freq"]])
        self.subjects = misc.datasets.datasets[self.dataset]["subjects"]

    def compute_results(self, details=False):
        """
        Loop through the subjects of the dataset, and compute the mean performances
        :return: mean of metrics, std of metrics
        """
        res = []
        for subject in self.subjects:
            res_subject = ResultsAllSeeds(self.model, self.mode, self.experiment, self.ph, self.dataset,
                                          subject).compute_results()
            if details:
                print(self.dataset, subject, res_subject)

            res.append(res_subject[0])  # only the mean

        keys = list(res[0].keys())
        res = [list(res_.values()) for res_ in res]
        mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
        return dict(zip(keys, mean)), dict(zip(keys, std))

    def compute_params(self):
        exp = os.path.join(self.experiment, "seed 0")
        res_subject = ResultsSubject(self.model, exp, self.ph, self.dataset, "1", study=True,
                                     mode=self.mode)
        return res_subject.params

    def to_latex(self, table="acc"):
        """
        Format the results into a string for the paper in LATEX
        :param table: either "acc" or "cg_ega", corresponds to the table
        :return:
        """
        mean, std = self.compute_results()
        if table == "cg_ega":
            keys = ["CG_EGA_AP_hypo", "CG_EGA_BE_hypo", "CG_EGA_EP_hypo", "CG_EGA_AP_eu", "CG_EGA_BE_eu",
                    "CG_EGA_EP_eu", "CG_EGA_AP_hyper", "CG_EGA_BE_hyper", "CG_EGA_EP_hyper"]
            mean = [mean[k] * 100 for k in keys]
            std = [std[k] * 100 for k in keys]
        elif table == "general":
            acc_keys = ["RMSE", "MAPE", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
            mean = [mean[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else mean[k] * 100 for k in acc_keys]
            std = [std[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else std[k] * 100 for k in acc_keys]

        print_latex(mean, std, label=self.model)


class ResultsAllSeeds:
    def __init__(self, model, mode, experiment, ph, dataset, subject):
        """
        Object that compute all the performances of a given dataset for a given model and experiment and prediction
        horizon
        :param model: name of the model (e.g., "base")
        :param experiment: name of the experiment (e.g., "test")
        :param ph: prediction horizons in minutes (e.g., 30)
        :param dataset: name of the dataset (e.g., "ohio")
        """

        self.model = model
        self.mode = mode
        self.experiment = experiment
        self.ph = ph
        self.dataset = dataset
        self.freq = np.max([misc.constants.freq, misc.datasets.datasets[dataset]["glucose_freq"]])
        self.subject = subject
        self.save_raw_results()

    def save_raw_results(self):
        """
        Save the results and params
        :return:
        """
        dir = os.path.join(cs.path, "study", self.dataset, self.model, self.mode, "patient " + self.subject,
                           self.experiment)
        Path(dir).mkdir(parents=True, exist_ok=True)
        savable_results = self.compute_results()
        printd("Global results for patient", self.subject, "with features", self.experiment, "\n", savable_results)
        np.save(os.path.join(dir, "results_metrics.npy"), [self.compute_params(), savable_results])

    def compute_results(self, details=False):
        """
        Loop through the subjects of the dataset, and compute the mean performances
        :return: mean of metrics, std of metrics
        """
        res = []
        for seed in range(5):
            exp = os.path.join(self.experiment, "seed " + str(seed))
            res_subject = ResultsSubject(self.model, exp, self.ph, self.dataset, self.subject, study=True,
                                         mode=self.mode).compute_mean_std_results()
            if details:
                print(self.dataset, "Seed ", seed, res_subject)

            res.append(res_subject[0])  # only the mean

        keys = list(res[0].keys())
        res = [list(res_.values()) for res_ in res]
        mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
        return dict(zip(keys, mean)), dict(zip(keys, std))

    def compute_params(self):
        exp = os.path.join(self.experiment, "seed 0")
        res_subject = ResultsSubject(self.model, exp, self.ph, self.dataset, self.subject, study=True,
                                     mode=self.mode)
        return res_subject.params

    def to_latex(self, table="acc"):
        """
        Format the results into a string for the paper in LATEX
        :param table: either "acc" or "cg_ega", corresponds to the table
        :return:
        """
        mean, std = self.compute_results()
        if table == "cg_ega":
            keys = ["CG_EGA_AP_hypo", "CG_EGA_BE_hypo", "CG_EGA_EP_hypo", "CG_EGA_AP_eu", "CG_EGA_BE_eu",
                    "CG_EGA_EP_eu", "CG_EGA_AP_hyper", "CG_EGA_BE_hyper", "CG_EGA_EP_hyper"]
            mean = [mean[k] * 100 for k in keys]
            std = [std[k] * 100 for k in keys]
        elif table == "general":
            acc_keys = ["RMSE", "MAPE", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
            mean = [mean[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else mean[k] * 100 for k in acc_keys]
            std = [std[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else std[k] * 100 for k in acc_keys]

        print_latex(mean, std, label=self.model)


class ResultsDatasetTransfer(ResultsDataset):
    """ Convenient class, child of ResultsDataset, that overwrites the to_latex function """

    def __init__(self, model, experiment, ph, source_dataset, target_dataset):
        experiment = source_dataset + "_2_" + target_dataset + "\\" + experiment
        super().__init__(model, experiment, ph, target_dataset)

    def to_latex(self, table="acc", model_name=None):
        """
        Format the results into a string for the paper in LATEX
        :param table: either "acc" or "cg_ega", corresponds to the table
        :param model_name: prefix of the string, name of the model
        :return:
        """
        mean, std = self.compute_results()
        if table == "cg_ega":
            keys = ["CG_EGA_AP_hypo", "CG_EGA_BE_hypo", "CG_EGA_EP_hypo", "CG_EGA_AP_eu", "CG_EGA_BE_eu",
                    "CG_EGA_EP_eu", "CG_EGA_AP_hyper", "CG_EGA_BE_hyper", "CG_EGA_EP_hyper"]
            mean = [mean[k] * 100 for k in keys]
            std = [std[k] * 100 for k in keys]
        elif table == "general":
            acc_keys = ["RMSE", "MAPE", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
            mean = [mean[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else mean[k] * 100 for k in acc_keys]
            std = [std[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else std[k] * 100 for k in acc_keys]

        print_latex(mean, std, label=self.model)


class ResultsSubject:
    def __init__(self, model, experiment, ph, dataset, subject, params=None, results=None, study=False,
                 mode="valid"):
        """
        Object that compute all the performances of a given subject for a given model and experiment and prediction
        horizon
        :param model: name of the model (e.g., "base")
        :param experiment: name of the experiment (e.g., "test")
        :param ph: prediction horizons in minutes (e.g., 30)
        :param dataset: name of the dataset (e.g., "ohio")
        :param subject: name of the subject (e.g., "559")
        :param params: if params and results  are given, performances are directly compute on them, and both are saved
        into a file
        :param results: see params
        """
        self.model = model
        self.experiment = experiment
        self.ph = ph
        self.dataset = dataset
        self.subject = subject
        self.freq = np.max([misc.constants.freq, misc.datasets.datasets[dataset]["glucose_freq"]])

        if results is None and params is None:
            self.params, self.results = self.load_raw_results(study, mode)
        else:
            self.results = results
            self.params = params
            self.save_raw_results(study, mode)

    def load_raw_results(self, from_study, mode, transfer=False):
        """
        Load the results from previous instance of ResultsSubject that has saved the them
        :param from_study
        :param mode
        :param transfer
        :return: (params dictionary), dataframe with ground truths and predictions
        """
        if not from_study:
            file = self.dataset + "_" + self.subject + ".npy"
            if not transfer:
                path = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph), file)
            else:
                path = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph), file)
        else:
            if not transfer:
                path = os.path.join(cs.path, "study", self.dataset, self.model, mode, "patient " + self.subject,
                                    self.experiment, "results.npy")
            else:
                path = os.path.join(cs.path, "study", self.dataset, self.model, mode, "patient " + self.subject,
                                    self.experiment, "results.npy")

        params, results = np.load(path, allow_pickle=True)
        dfs = []
        for result in results:
            df = pd.DataFrame(result, columns=["datetime", "y_true", "y_pred"])
            df = df.set_index("datetime")
            df = df.astype("float32")
            dfs.append(df)
        return params, dfs

    def save_raw_results(self, to_study, mode):
        """
        Save the results and params
        :return:
        """
        if not to_study:
            dir = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph))
            Path(dir).mkdir(parents=True, exist_ok=True)
            savable_results = np.array([res.reset_index().to_numpy() for res in self.results])
            np.save(os.path.join(dir, self.dataset + "_" + self.subject + ".npy"), [self.params, savable_results])
        else:
            dir = os.path.join(cs.path, "study", self.dataset, self.model, mode, "patient " + self.subject,
                               self.experiment)
            Path(dir).mkdir(parents=True, exist_ok=True)
            savable_results = np.array([res.reset_index().to_numpy() for res in self.results])
            np.save(os.path.join(dir, "results.npy"), [self.params, savable_results])

    def compute_raw_results(self, split_by_day=False):
        """
        Compute the raw metrics results for every split (or day, if split_by_day)
        :param split_by_day: whether the results are computed first by day and averaged, or averaged globally
        :return: dictionary of arrays of scores for the metrics
        """
        if split_by_day:
            results = []
            for res in self.results:
                for group in res.groupby(res.index.day):
                    results.append(group[1])
        else:
            results = self.results

        rmse_score = [rmse.rmse(res_day) for res_day in results]
        mape_score = [mape.mape(res_day) for res_day in results]
        mase_score = [mase.mase(res_day, self.ph, self.freq) for res_day in results]
        tg_score = [time_lag.time_gain(res_day, self.ph, self.freq, "mse") for res_day in results]
        cg_ega_score = np.array([cg_ega.CgEGA(res_day, self.freq).simplified() for res_day in results])
        cg_ega_score2 = np.array([cg_ega.CgEGA(res_day, self.freq).reduced() for res_day in results])
        p_ega_score = np.array([p_ega.PEga(res_day, self.freq).mean() for res_day in results])
        p_ega_a_plus_b_score = [p_ega.PEga(res_day, self.freq).a_plus_b() for res_day in results]
        r_ega_score = np.array([r_ega.REga(res_day, self.freq).mean() for res_day in results])
        r_ega_a_plus_b_score = [r_ega.REga(res_day, self.freq).a_plus_b() for res_day in results]

        return {
            "RMSE": rmse_score,
            "MAPE": mape_score,
            "MASE": mase_score,
            "TG": tg_score,
            "CG_EGA_AP": cg_ega_score2[:, 0],
            "CG_EGA_BE": cg_ega_score2[:, 1],
            "CG_EGA_EP": cg_ega_score2[:, 2],
            "CG_EGA_AP_hypo": cg_ega_score[:, 0],
            "CG_EGA_BE_hypo": cg_ega_score[:, 1],
            "CG_EGA_EP_hypo": cg_ega_score[:, 2],
            "CG_EGA_AP_eu": cg_ega_score[:, 3],
            "CG_EGA_BE_eu": cg_ega_score[:, 4],
            "CG_EGA_EP_eu": cg_ega_score[:, 5],
            "CG_EGA_AP_hyper": cg_ega_score[:, 6],
            "CG_EGA_BE_hyper": cg_ega_score[:, 7],
            "CG_EGA_EP_hyper": cg_ega_score[:, 8],
            "P_EGA_A+B": p_ega_a_plus_b_score,
            "P_EGA_A": p_ega_score[:, 0],
            "P_EGA_B": p_ega_score[:, 1],
            "P_EGA_C": p_ega_score[:, 2],
            "P_EGA_D": p_ega_score[:, 3],
            "P_EGA_E": p_ega_score[:, 4],
            "R_EGA_A+B": r_ega_a_plus_b_score,
            "R_EGA_A": r_ega_score[:, 0],
            "R_EGA_B": r_ega_score[:, 1],
            "R_EGA_uC": r_ega_score[:, 2],
            "R_EGA_lC": r_ega_score[:, 3],
            "R_EGA_uD": r_ega_score[:, 4],
            "R_EGA_lD": r_ega_score[:, 5],
            "R_EGA_uE": r_ega_score[:, 6],
            "R_EGA_lE": r_ega_score[:, 7],
        }

    def compute_mean_std_results(self, split_by_day=False):
        """
        From the raw metrics scores, compute the mean and std
        :param split_by_day: whether the results are computed first by day and averaged, or averaged globally
        :return: mean of dictionary of metrics, std of dictionary of metrics
        """
        raw_results = self.compute_raw_results(split_by_day=split_by_day)

        mean = {key: val for key, val in zip(list(raw_results.keys()), np.mean(list(raw_results.values()), axis=1))}
        std = {key: val for key, val in zip(list(raw_results.keys()), np.std(list(raw_results.values()), axis=1))}

        return mean, std

    def plot(self, day_number=0):
        """
        Plot a given day
        :param day_number: day (int) to plot
        :return: /
        """
        cg_ega.CgEga(self.results[0], self.freq).plot(day_number)
