import pandas as pd
from .cleaning.unit_scaling import scaling_t1dms
from misc import constants as cs
from preprocessing.cleaning.nans_removal import remove_nans
from preprocessing.cleaning.nans_filling import fill_nans
from preprocessing.loading.loading import load
from preprocessing.data_augmentation.physiological_features import aob, cpb, iob
import misc.datasets
from misc.utils import printd
from preprocessing.resampling import resample
from preprocessing.samples_creation import create_samples
from preprocessing.splitting import split
from preprocessing.standardization import standardize
from .cleaning.last_day_removal import remove_last_day
from .cleaning.anomalies_removal import remove_anomalies


def preprocessing_ohio(dataset, subject, ph, hist, day_len, n_days_test):
    """
    OhioT1DM dataset preprocessing pipeline:
    loading -> samples creation -> cleaning (1st) -> splitting -> cleaning (2nd) -> standardization

    First cleaning is done before splitting to speedup the preprocessing

    :param dataset: name of the dataset, e.g. "ohio"
    :param subject: id of the subject, e.g. "559"
    :param ph: prediction horizon, e.g. 30
    :param hist: history length, e.g. 60
    :param day_len: length of a day normalized by sampling frequency, e.g. 288 (1440/5)
    :param n_days_test:
    :return: training_old folds, validation folds, testing folds, list of scaler (one per fold)
    """
    data = load(dataset, subject)
    data = resample(data, cs.freq)
    data = create_samples(data, ph, hist, day_len)
    data = fill_nans(data, day_len, n_days_test)
    train, valid, test = split(data, day_len, n_days_test, cs.cv)
    [train, valid, test] = [remove_nans(set_) for set_ in [train, valid, test]]
    train, valid, test, scalers = standardize(train, valid, test)
    print(test[0].shape)
    return train, valid, test, scalers


def preprocessing_t1dms(dataset, subject, ph, hist, day_len, n_days_test):
    """
    T1DMS dataset preprocessing pipeline (valid for adult, adolescents and children):
    loading -> samples creation -> splitting -> standardization

    :param dataset: name of the dataset, e.g. "t1dms"
    :param subject: id of the subject, e.g. "1"
    :param ph: prediction horizon, e.g. 30
    :param hist: history length, e.g. 60
    :param day_len: length of a day normalized by sampling frequency, e.g. 1440 (1440/1)
    :param n_days_test:
    :return: training_old folds, validation folds, testing folds, list of scaler (one per fold)
    """
    data = load(dataset, subject, day_len)
    data = scaling_t1dms(data)
    data = resample(data, cs.freq)
    data = create_samples(data, ph, hist, day_len)
    train, valid, test = split(data, day_len, n_days_test, cs.cv)
    train, valid, test, scalers = standardize(train, valid, test)
    return train, valid, test, scalers


def preprocessing_idiab(dataset, subject, ph, hist, day_len, n_days_test):
    """
    Idiab dataset preprocessing pipeline:
    loading -> remove anomalies -> resample -> remove last day -> samples creation -> cleaning (1st) -> features
    selection -> splitting -> cleaning (2nd) -> standardization

    First cleaning is done before splitting to speedup the preprocessing

    :param dataset: name of the dataset, e.g. "idiab"
    :param subject: id of the subject, e.g. "1"
    :param ph: prediction horizon, e.g. 30
    :param hist: history length, e.g. 60
    :param day_len: length of a day normalized by sampling frequency, e.g. 288 (1440/5)
    :param n_days_test:
    :return: training folds, validation folds, testing folds, list of scaler (one per fold)
    """
    printd("Preprocessing " + dataset + subject + "...")
    data = load(dataset, subject)
    data = remove_anomalies(data)
    data = resample(data, cs.freq)
    data = remove_last_day(data)
    # data["CHO"] = CPB(data, cs.C_bio, cs.t_max)
    # data["insulin"] = IOB(data, cs.K_DIA)
    # data["steps"] = AOB(data, cs.k_s)
    data = create_samples(data, ph, hist, day_len)
    data = fill_nans(data, day_len, n_days_test)
    to_drop = ["calories", "heartrate", "mets", "steps"]
    for col in data.columns:
        for ele in to_drop:
            if ele in col:
                data = data.drop(col, axis=1)
                break

    train, valid, test = split(data, day_len, n_days_test, cs.cv)
    [train, valid, test] = [remove_nans(set_) for set_ in [train, valid, test]]
    train, valid, test, scalers = standardize(train, valid, test)
    print(test[0].shape)
    return train, valid, test, scalers


def preprocessing_full(dataset, subject, ph, hist, day_len, all_feat):
    """
    Full dataset samples creation pipeline:
    loading -> selecting features -> remove anomalies -> resample -> remove last day -> samples creation
    -> cleaning (1st)

    First cleaning is done before splitting to speedup the preprocessing

    :param dataset: name of the dataset, e.g. "idiab"
    :param subject: id of the subject, e.g. "1"
    :param ph: prediction horizon, e.g. 30
    :param hist: history length, e.g. 60
    :param day_len: length of a day normalized by sampling frequency, e.g. 288 (1440/5)
    :param all_feat:
    :return: dataframe of samples
    """
    data = load(dataset, subject)

    features = [feature for feature in list(data.columns) if feature not in ["datetime", "glucose"]]
    to_drop = [feature for feature in features if feature not in all_feat]
    data = data.drop(to_drop, axis=1)

    if "idiab" in dataset:
        data = remove_anomalies(data)
    if "t1dms" in dataset:
        data = scaling_t1dms(data)

    data = resample(data, cs.freq)

    if "idiab" in dataset:
        data = remove_last_day(data)

    if "CPB" in all_feat:
        data["CPB"] = cpb(data, cs.C_bio, cs.t_max, True)
    if "IOB" in all_feat:
        data["IOB"] = iob(data, cs.K_DIA, True)
    if "AOB" in all_feat:
        data["AOB"] = aob(data, cs.k_s, True)

    data = create_samples(data, ph, hist, day_len)
    n_days_test = misc.datasets.datasets[dataset]["n_days_test"]

    if "idiab" in dataset or "ohio" in dataset:
        data = fill_nans(data, day_len, n_days_test)

    return data


def preprocessing_select(data, dataset, day_len, all_feat, features):
    """
    Dataset train, valid, test creation for specific features, after samples creation:
    features selection -> splitting -> cleaning (2nd) -> standardization

    :param data: samples creation after first cleaning
    :param dataset: name of the dataset, e.g. "idiab"
    :param day_len: length of a day normalized by sampling frequency, e.g. 288 (1440/5)
    :param all_feat:
    :param features: features to be used by the models during the processing phase
    :return: training folds, validation folds, testing folds, list of scaler (one per fold)
    """
    to_drop = [ele for ele in all_feat if ele not in features]
    for col in data.columns:
        for ele in to_drop:
            if ele in col:
                data = data.drop(col, axis=1)
                break

    train, valid, test = split(data, day_len, misc.datasets.datasets[dataset]["n_days_test"], cs.cv)
    if "idiab" in dataset or "ohio" in dataset:
        [train, valid, test] = [remove_nans(set_) for set_ in [train, valid, test]]
    train, valid, test, scalers = standardize(train, valid, test)
    return train, valid, test, scalers


preprocessing_per_dataset = {
    "t1dms": preprocessing_t1dms,
    "t1dms_adolescent": preprocessing_t1dms,
    "t1dms_child": preprocessing_t1dms,
    "ohio": preprocessing_ohio,
    "idiab": preprocessing_idiab,
}


def preprocessing(target_dataset, target_subject, ph, hist, day_len):
    """
    associate every dataset with a specific pipeline - which should be consistent with the others

    :param target_dataset: name of dataset (e.g., "ohio")
    :param target_subject: name of subject (e.g., "559")
    :param ph: prediction horizon in minutes (e.g., 5)
    :param hist: length of history in minutes (e.g., 60)
    :param day_len: typical length of a day in minutes standardized to the sampling frequency (e.g. 288 for 1440 min at
    freq=5 minutes)
    :return: train, valid, test folds
    """
    n_days_test = misc.datasets.datasets[target_dataset]["n_days_test"]
    return preprocessing_per_dataset[target_dataset](target_dataset, target_subject, ph, hist, day_len, n_days_test)


def preprocessing_source_multi(source_datasets, target_dataset, target_subject, ph, hist, day_len):
    """
    Preprocessing for multi-source training :
    - preprocess all the subjects from the source dataset, excluding the target subject if it is from the same dataset;
    - affect a class number to every subject;
    - merge the training and validation sets, and set the testing set as validation;
    - merge the sets from all the patients.
    :param source_datasets: name of the source datasets, separated by a "+" if several (e.g., "idiab+ohio")
    :param target_dataset: target dataset (i.e., "idiab" or "ohio")
    :param target_subject: target subject within target dataset (e.g, "559" if target_dataset is "ohio")
    :param ph: prediction horizon
    :param hist: history length
    :param day_len: length of day
    :return:
    """
    train_ds, valid_ds, test_ds, scalers_ds = [], [], [], []
    subject_domain = 0
    for source_dataset in source_datasets.split("+"):
        for source_subject in misc.datasets.datasets[source_dataset]["subjects"]:
            if target_dataset == source_dataset and target_subject == source_subject:
                continue

            # printd("Preprocessing " + source_dataset + source_subject + "...")

            n_days_test = misc.datasets.datasets[source_dataset]["n_days_test"]
            train_sbj, valid_sbj, test_sbj, scalers_sbj = preprocessing_per_dataset[source_dataset](source_dataset,
                                                                                                    source_subject, ph,
                                                                                                    hist,
                                                                                                    day_len,
                                                                                                    n_days_test)

            # no cross-validation when source training, train and valid are concatenated, and we evaluate on test
            train, valid, test = pd.concat([train_sbj[0], valid_sbj[0]]).sort_values("datetime"), test_sbj[0], test_sbj[
                0]

            # add subject domain
            train["domain"], valid["domain"], test["domain"] = subject_domain, subject_domain, subject_domain
            subject_domain += 1

            for ds, set_ in zip([train_ds, valid_ds, test_ds, scalers_ds], [train, valid, test, scalers_sbj[0]]):
                ds.append(set_)

    train_ds, valid_ds, test_ds = [pd.concat(ds) for ds in [train_ds, valid_ds, test_ds]]

    return [train_ds], [valid_ds], [test_ds], scalers_ds
