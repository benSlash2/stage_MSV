import matplotlib.pyplot as plt
import os
import numpy as np
import misc
import misc.constants as cs
import pickle
from pathlib import Path
from preprocessing.preprocessing import preprocessing
from processing.models.retain_atl import RETAIN_ATL
from misc.utils import locate_params
import pandas as pd


class RetainAnalyzer():
    """
        Implements all the visualisation described in the publication:
            - plot sample contribution
            - plot mean absolute normalized contribution
            - plot max absolute normalized contribution
            - plot evolution of stimulti through time
    """

    def __init__(self, dataset, ph, hist, experiment, params):
        self.dataset = dataset
        self.ph = ph // cs.freq
        self.hist = hist // cs.freq
        self.exp = experiment
        self.params = locate_params(params)
        self.train, self.valid, self.test, self.scalers = {}, {}, {}, {}

    def _load_subject_data(self, subject):
        if not subject in list(self.train.keys()):
            train_sbj, valid_sbj, test_sbj, scalers_sbj = preprocessing(self.dataset, subject, self.ph,
                                                                        self.hist, cs.day_len_f)
            self.train[subject] = train_sbj
            self.valid[subject] = valid_sbj
            self.test[subject] = test_sbj
            self.scalers[subject] = scalers_sbj

    def _load_all_subjects_data(self):
        for subject in misc.datasets.datasets[self.dataset]["subjects"]:
            self._load_all_subjects_data(subject)

    def _create_models(self, subject):
        models = []
        for train_i, valid_i, test_i in zip(self.train[subject], self.valid[subject], self.test[subject]):
            model = RETAIN_ATL(subject, self.ph, self.params, train_i, valid_i, test_i)
            file_name = "RETAIN_ATL_" + self.dataset + subject + ".pt"
            file_path = os.path.join(cs.path, "processing", "models", "weights", self.exp, file_name)
            model.load(file_path)
            models.append(model)

        return models

    def get_contribution_subject_sample(self, subject, sample=0, split=None, eval_set="test"):
        self._load_subject_data(subject)
        models = self._create_models(subject)
        if split is None:
            contrib = []
            for model, scaler in zip(models, self.scalers[subject]):
                contrib_split = model.contribution(eval_set)
                contrib_split = contrib_split * scaler.scale_[-1]
                contrib.append(contrib_split)
            contrib = np.mean(contrib, axis=0)
        else:
            contrib = models[split].contribution(eval_set)
            contrib = contrib * self.scalers[subject][split].scale_[-1]

        # return contrib[sample], self._data_from_eval_set(subject, eval_set)[0]

        data = self._data_from_eval_set(subject, eval_set)[0].values[:, 1:]
        scaler = self.scalers[subject][0]
        data = data * scaler.scale_ + scaler.mean_

        return contrib[sample], data[sample]

    def compute_contribution_subject(self, subject, eval_set="test"):
        self._load_subject_data(subject)

        models = self._create_models(subject)

        contrib_an = []
        for model in models:
            contrib_an.append(model.contribution_an(eval_set))

        return contrib_an

    def compute_max_contrib(self, subject, eval_set):
        contrib_an = self.compute_contribution_subject(subject, eval_set)
        max_contrib = [np.max(contrib_an_split, axis=0) for contrib_an_split in contrib_an]
        max_contrib = [np.flip(max_contrib_split, axis=0) for max_contrib_split in max_contrib]
        mean_max_contrib, std_max_contrib = np.mean(max_contrib, axis=0), np.std(max_contrib, axis=0)
        return mean_max_contrib, std_max_contrib

    def compute_mean_std_max_contrib(self, subject, eval_set="test"):
        if subject == "all":
            max_contrib = []
            for sbj in misc.datasets.datasets[self.dataset]["subjects"]:
                mean_max_contrib_sbj, _ = self.compute_max_contrib(sbj, eval_set)
                max_contrib.append(mean_max_contrib_sbj)
            mean_max_contrib, std_max_contrib = np.mean(max_contrib, axis=0), np.std(max_contrib, axis=0)
        else:
            mean_max_contrib, std_max_contrib = self.compute_max_contrib(subject, eval_set)

        return mean_max_contrib, std_max_contrib

    def _data_from_eval_set(self, subject, eval_set):
        if eval_set == "train":
            return self.train[subject]
        elif eval_set == "valid":
            return self.valid[subject]
        elif eval_set == "test":
            return self.test[subject]

    def _compute_stimuli_indexes_subject(self, stimuli, subject, eval_set="test", max_lag=5):
        self._load_subject_data(subject)

        data = self._data_from_eval_set(subject, eval_set)

        last_idx = self.params["hist"] // cs.freq - 1

        stimuli_col_idx = np.where(data[0].columns == stimuli + "_" + str(last_idx))[0][0]
        stimuli_data = [data_split.loc[:, stimuli + "_" + str(last_idx)] for data_split in data]
        stimuli_data = [
            stimuli_data_split * scalers_split.scale_[stimuli_col_idx - 1] + scalers_split.mean_[stimuli_col_idx - 1]
            for stimuli_data_split, scalers_split in zip(stimuli_data, self.scalers[subject])]

        non_zero_stimuli_idx = [np.where(~np.isclose(stimuli_data_split, 0))[0] for stimuli_data_split in stimuli_data]
        toofar_idx = [_ + max_lag > len(data[0]) for _ in non_zero_stimuli_idx]
        non_zero_stimuli_idx = [_[~toofar_idx_split] for _, toofar_idx_split in zip(non_zero_stimuli_idx, toofar_idx)]
        return non_zero_stimuli_idx

    def compute_contrib_after_stimuli_subject(self, stimuli, subject, eval_set="test", lag=0):
        stimuli_idx = self._compute_stimuli_indexes_subject(stimuli, subject, eval_set)
        contrib = self.compute_contribution_subject(subject, eval_set)
        contrib_after_stimuli = [contrib_split[stimuli_idx_split + lag] for stimuli_idx_split, contrib_split in
                                 zip(stimuli_idx, contrib)]
        mean, std = np.mean(np.mean(contrib_after_stimuli, axis=0), axis=0), np.mean(
            np.std(contrib_after_stimuli, axis=0),
            axis=0)

        mean, std = np.flip(mean, axis=0), np.flip(std, axis=0)
        return mean, std

    def compute_mean_std_contrib_after_stimuli(self, stimuli, subject, eval_set="test", lag=0):
        if subject == "all":
            max_contrib = []
            for sbj in misc.datasets.datasets[self.dataset]["subjects"]:
                mean_max_contrib_sbj, _ = self.compute_contrib_after_stimuli_subject(stimuli, sbj, eval_set, lag)
                max_contrib.append(mean_max_contrib_sbj)
            mean_max_contrib, std_max_contrib = np.mean(max_contrib, axis=0), np.std(max_contrib, axis=0)
        else:
            mean_max_contrib, std_max_contrib = self.compute_contrib_after_stimuli_subject(stimuli, subject, eval_set,
                                                                                           lag)
        return mean_max_contrib, std_max_contrib

    def compute_mean_std_contrib_after_stimuli_all_lags(self, stimuli, subject, eval_set="test", max_lag=0):
        mean_contrib_after_stimuli, std_contrib_after_stimuli = [], []
        for lag in range(max_lag):
            mean_contrib_after_stimuli_lag, std_contrib_after_stimuli_lag = self.compute_mean_std_contrib_after_stimuli(
                stimuli, subject, eval_set, lag)
            mean_contrib_after_stimuli.append(mean_contrib_after_stimuli_lag)
            std_contrib_after_stimuli.append(std_contrib_after_stimuli_lag)
        return mean_contrib_after_stimuli, std_contrib_after_stimuli

    def format_all_evolution_lags(self, stimuli, subject, eval_set="test", max_lag=0):
        mean_contrib_after_stimuli, std_contrib_after_stimuli = self.compute_mean_std_contrib_after_stimuli_all_lags(
            stimuli, subject, eval_set, max_lag)

        hist_f = self.params["hist"] // cs.freq
        mean_contrib_after_stimuli = np.transpose(mean_contrib_after_stimuli, (1, 0, 2)).reshape(hist_f, -1)
        std_contrib_after_stimuli = np.transpose(std_contrib_after_stimuli, (1, 0, 2)).reshape(hist_f, -1)

        columns = np.c_[["glucose_" + str(i) for i in range(max_lag)], ["CHO_" + str(i) for i in range(max_lag)],
                        ["insulin_" + str(i) for i in range(max_lag)], ["mets_" + str(i) for i in range(max_lag)], ["calories_" + str(i) for i in range(max_lag)], ["heartrate_" + str(i) for i in range(max_lag)]]
        columns = np.reshape(columns, (-1))

        df_mean = pd.DataFrame(data=mean_contrib_after_stimuli, columns=columns)
        df_std = pd.DataFrame(data=std_contrib_after_stimuli, columns=columns)

        return df_mean, df_std

    def filter_contributions(self, filter, subject="all", eval_set="test"):
        """ three filters supported : hypo, eu, hyper"""

        def compute_filter_mask(data,scaler,filter):
            mask = None
            y_scaled = data.y * scaler.scale_[-1] + scaler.mean_[-1]
            if filter=="hypo":
                mask = y_scaled <= 70.0+1e-5
            elif filter=="eu":
                mask = (y_scaled > 70.0+1e-5) & (y_scaled <= 180.0+1e-5)
            elif filter == "hyper":
                mask = y_scaled > 180.0+1e-5
            return mask

        def compute_contrib_subject(sbj):
            contrib_sbj = self.compute_contribution_subject(sbj, eval_set)
            data = self._data_from_eval_set(sbj, eval_set)
            masks = [compute_filter_mask(data_split, scaler_split, filter) for data_split, scaler_split in zip(data,self.scalers[sbj])]
            contrib_sbj = [contrib_sbj_split[mask_split] for contrib_sbj_split, mask_split in zip(contrib_sbj, masks)]
            return contrib_sbj

        if subject == "all":
            contrib = []
            for sbj in misc.datasets.datasets[self.dataset]["subjects"]:
                contrib_sbj = compute_contrib_subject(sbj)
                contrib_sbj = np.mean(contrib_sbj, axis=0)
                contrib.append(contrib_sbj)
            mean_contrib = contrib
        else:
            contrib_sbj = compute_contrib_subject(subject)
            mean_contrib = np.mean(contrib_sbj, axis=0)

        return mean_contrib

    def mean_contrib_glycemia_region(self,region,subject,eval_set="test"):
        contrib = self.filter_contributions(region,subject,eval_set)

        if subject == "all":
            mean_contrib_sbj = [np.mean(contrib_sbj,axis=0) for contrib_sbj in contrib]
            mean_contrib, std_contrib = np.mean(mean_contrib_sbj,axis=0), np.std(mean_contrib_sbj,axis=0)
        else:
            mean_contrib, std_contrib = np.mean(contrib,axis=0), np.std(contrib,axis=0)

        return mean_contrib, std_contrib

    def plot_evolution_after_stimuli(self, stimuli, max_lag=5, subject="all", eval_set="test", history_limit=40):
        mean_contrib_after_stimuli, std_contrib_after_stimuli = self.compute_mean_std_contrib_after_stimuli_all_lags(
            stimuli, subject, eval_set, max_lag)

        fig, axes = plt.subplots(ncols=max_lag, nrows=1, figsize=(21, 5))
        history_limit_f = history_limit // cs.freq
        time = np.arange(0, self.params["hist"], cs.freq)[:history_limit_f]

        for i, (mean, std) in enumerate(zip(mean_contrib_after_stimuli, std_contrib_after_stimuli)):
            # mean, std = np.flip(mean, axis=0), np.flip(std, axis=0)

            axes[i].plot(time, mean[:history_limit_f, 0], color="blue", label="glycemia")
            axes[i].fill_between(time, mean[:history_limit_f, 0] - std[:history_limit_f, 0],
                                 mean[:history_limit_f, 0] + std[:history_limit_f, 0], alpha=0.2, edgecolor='blue',
                                 facecolor="blue")

            axes[i].plot(time, mean[:history_limit_f, 2], color="green", label="insulin")
            axes[i].fill_between(time, mean[:history_limit_f, 2] - std[:history_limit_f, 2],
                                 mean[:history_limit_f, 2] + std[:history_limit_f, 2], alpha=0.2, edgecolor='green',
                                 facecolor="green")

            axes[i].plot(time, mean[:history_limit_f, 1], color="red", label="CHO")
            axes[i].fill_between(time, mean[:history_limit_f, 1] - std[:history_limit_f, 1],
                                 mean[:history_limit_f, 1] + std[:history_limit_f, 1], alpha=0.2, edgecolor='red',
                                 facecolor="red")

            axes[i].plot(time, mean[:history_limit_f, 3], color="yellow", label="mets")
            axes[i].fill_between(time, mean[:history_limit_f, 3] - std[:history_limit_f, 3],
                                 mean[:history_limit_f, 3] + std[:history_limit_f, 3], alpha=0.2, edgecolor='yellow',
                                 facecolor="yellow")

            axes[i].plot(time, mean[:history_limit_f, 4], color="purple", label="calories")
            axes[i].fill_between(time, mean[:history_limit_f, 4] - std[:history_limit_f, 4],
                                 mean[:history_limit_f, 4] + std[:history_limit_f, 4], alpha=0.2, edgecolor='purple',
                                 facecolor="purple")

            axes[i].plot(time, mean[:history_limit_f, 5], color="orange", label="heartrate")
            axes[i].fill_between(time, mean[:history_limit_f, 5] - std[:history_limit_f, 5],
                                 mean[:history_limit_f, 5] + std[:history_limit_f, 5], alpha=0.2, edgecolor='orange',
                                 facecolor="orange")

    def plot_max_contribution(self, subject="all", eval_set="test"):
        mean_max_contrib, std_max_contrib = self.compute_mean_std_max_contrib(subject, eval_set)

        time = np.arange(0, self.params["hist"], cs.freq)

        plt.figure()
        plt.plot(time, mean_max_contrib[:, 0], color="blue", label="glycemia")
        plt.fill_between(time, mean_max_contrib[:, 0] - std_max_contrib[:, 0],
                         mean_max_contrib[:, 0] + std_max_contrib[:, 0], alpha=0.5, edgecolor='blue', facecolor="blue")

        plt.plot(time, mean_max_contrib[:, 2], color="green", label="insulin")
        plt.fill_between(time, mean_max_contrib[:, 2] - std_max_contrib[:, 2],
                         mean_max_contrib[:, 2] + std_max_contrib[:, 2], alpha=0.5, edgecolor='green',
                         facecolor="green")

        plt.plot(time, mean_max_contrib[:, 1], color="red", label="CHO")
        plt.fill_between(time, mean_max_contrib[:, 1] - std_max_contrib[:, 1],
                         mean_max_contrib[:, 1] + std_max_contrib[:, 1], alpha=0.5, edgecolor='red', facecolor="red")

        plt.plot(time, mean_max_contrib[:, 3], color="yellow", label="mets")
        plt.fill_between(time, mean_max_contrib[:, 3] - std_max_contrib[:, 3],
                         mean_max_contrib[:, 3] + std_max_contrib[:, 3], alpha=0.5, edgecolor='yellow', facecolor="yellow")

        plt.plot(time, mean_max_contrib[:, 4], color="purple", label="calories")
        plt.fill_between(time, mean_max_contrib[:, 4] - std_max_contrib[:, 4],
                         mean_max_contrib[:, 4] + std_max_contrib[:, 4], alpha=0.5, edgecolor='purple', facecolor="purple")

        plt.plot(time, mean_max_contrib[:, 5], color="orange", label="heartrate")
        plt.fill_between(time, mean_max_contrib[:, 5] - std_max_contrib[:, 5],
                         mean_max_contrib[:, 5] + std_max_contrib[:, 5], alpha=0.5, edgecolor='orange', facecolor="orange")

        plt.xlabel("History [min]")
        plt.ylabel("Maximum absolute normalized contribution")
        plt.legend()
        plt.title("Maximum absolute normalized contribution for dataset " + self.dataset + " and subject " + subject)
