import matplotlib.pyplot as plt
import os
import numpy as np
import misc
import misc.constants as cs
from preprocessing.preprocessing import preprocessing
from misc.utils import locate_params, locate_model


class TreeAnalyzer:
    """
        Analyse tree-based models through Gini feature importance or permutation feature importance
    """

    def __init__(self, dataset, ph, hist, experiment, model, params):
        self.dataset = dataset
        self.ph = ph // cs.freq
        self.hist = hist // cs.freq
        self.exp = experiment
        self.params = locate_params(params)
        self.model_class = locate_model(model)
        self.train, self.valid, self.test, self.scalers = {}, {}, {}, {}

    def _load_subject_data(self, subject):
        if subject not in list(self.train.keys()):
            train_sbj, valid_sbj, test_sbj, scalers_sbj = preprocessing(self.dataset, subject, self.ph,
                                                                        self.hist, cs.day_len_f)
            self.train[subject] = train_sbj
            self.valid[subject] = valid_sbj
            self.test[subject] = test_sbj
            self.scalers[subject] = scalers_sbj

    def _load_all_subjects_data(self):
        for subject in misc.datasets.datasets[self.dataset]["subjects"]:
            self._load_subject_data(subject)

    def _create_models(self, subject):
        models = []
        for i, (train_i, valid_i, test_i) in enumerate(
                zip(self.train[subject], self.valid[subject], self.test[subject])):
            model = self.model_class(subject, self.ph, self.params, train_i, valid_i, test_i)
            file_name = self.model_class.__name__ + "_" + self.dataset + subject + "_" + str(i)
            file_path = os.path.join(cs.path, "processing", "models", "weights", self.model_class.__name__, self.exp,
                                     file_name)
            model.load(file_path)
            models.append(model)
        return models

    def compute_feature_importance_subject(self, subject, method="gini", eval_set="test"):
        self._load_subject_data(subject)
        models = self._create_models(subject)

        feature_importance = []
        for model in models:
            features_importance_split = model.feature_importances(method, eval_set)
            features_importance_split = np.reshape(features_importance_split,
                                                   (-1, self.params["hist"] // cs.freq)).transpose(1, 0)
            feature_importance.append(features_importance_split)

        # return feature_importance

        feature_importance = np.flip(feature_importance, axis=1)
        mean_feature_importance, std_feature_importance = np.mean(feature_importance, axis=0), \
                                                          np.std(feature_importance, axis=0)
        return mean_feature_importance, std_feature_importance

    def compute_mean_std_feature_importance(self, subject, method="gini", eval_set="test"):
        if subject == "all":
            feature_importance = []
            for sbj in misc.datasets.datasets[self.dataset]["subjects"]:
                mean_feature_importance_sbj, _ = self.compute_feature_importance_subject(sbj, method, eval_set)
                feature_importance.append(mean_feature_importance_sbj)
            mean_feature_importance, std_feature_importance = np.mean(feature_importance, axis=0), np.std(
                feature_importance, axis=0)
        else:
            mean_feature_importance, std_feature_importance = self.compute_feature_importance_subject(subject, eval_set)

        return mean_feature_importance, std_feature_importance

    def plot_feature_importance(self, subject="all", method="gini", eval_set="test", plot_std=False):
        mean_max_contrib, std_max_contrib = self.compute_mean_std_feature_importance(subject, method, eval_set)

        time = np.arange(0, self.params["hist"], cs.freq)

        plt.figure()
        plt.yscale("log")
        plt.plot(time, mean_max_contrib[:, 0], color="blue", label="glycemia")

        plt.plot(time, mean_max_contrib[:, 2], color="green", label="insulin")

        plt.plot(time, mean_max_contrib[:, 1], color="red", label="CHO")

        if plot_std:
            plt.fill_between(time, mean_max_contrib[:, 0] - std_max_contrib[:, 0],
                             mean_max_contrib[:, 0] + std_max_contrib[:, 0], alpha=0.5, edgecolor='blue',
                             facecolor="blue")
            plt.fill_between(time, mean_max_contrib[:, 2] - std_max_contrib[:, 2],
                             mean_max_contrib[:, 2] + std_max_contrib[:, 2], alpha=0.5, edgecolor='green',
                             facecolor="green")
            plt.fill_between(time, mean_max_contrib[:, 1] - std_max_contrib[:, 1],
                             mean_max_contrib[:, 1] + std_max_contrib[:, 1], alpha=0.5, edgecolor='red',
                             facecolor="red")

        plt.xlabel("History [min]")
        plt.ylabel("Feature " + method + " importance")
        plt.legend()
        plt.title("Feature " + method + " importance for dataset " + self.dataset + " and subject " + subject)
