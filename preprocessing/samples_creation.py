import numpy as np
import pandas as pd


def create_samples(data, ph, hist, day_len):
    """
    Create samples consisting in glucose, insulin and CHO histories (past hist-length values)
    :param data: dataframe
    :param ph: prediction horizon in minutes in sampling frequency scale
    :param hist: history length in sampling frequency scale
    :param day_len: length of day in sampling frequency scale
    :return: dataframe of samples
    """
    n_samples = data.shape[0] - ph - hist + 1
    y = data.loc[ph + hist - 1:, "glucose"].values.reshape(-1, 1)
    d = pd.DatetimeIndex(data.loc[ph + hist - 1:, "datetime"].values)
    t = np.concatenate([np.arange(day_len) for _ in range(len(data) // day_len)], axis=0)[ph + hist - 1:].reshape(-1, 1)
    x = {}

    for feature in list(data.columns):
        if feature != "datetime":
            x[feature] = np.array([data.loc[i:i + n_samples - 1, feature] for i in range(hist)]).transpose()

    new_col_features = np.concatenate([[ele + "_" + str(i) for i in range(hist)] for ele in list(x.keys())], axis=None)
    new_columns = np.r_[["time"], new_col_features, ["y"]]

    data_features = np.c_[tuple(x.values())]
    data_full = np.c_[t, data_features, y]
    new_data = pd.DataFrame(data=np.c_[data_full], columns=new_columns)
    new_data["datetime"] = d
    new_data = new_data.loc[:, np.r_[["datetime"], new_columns]]  # reorder the columns, with datetime first

    return new_data

# hist_2 = hist //2
# m = np.array([data.loc[i + hist_2:i + hist_2 + n_samples - 1, "mets"] for i in range(hist_2)]).transpose()
# cal = np.array([data.loc[i + hist_2:i + hist_2 + n_samples - 1, "calories"] for i in range(hist_2)]).transpose()
# h = np.array([data.loc[i + hist_2:i + hist_2 + n_samples - 1, "heartrate"] for i in range(hist_2)]).transpose()
# new_columns = np.r_[["time"], ["glucose_" + str(i) for i in range(hist)], ["CHO_" + str(i) for i in range(hist)],
# ["insulin_" + str(i) for i in range(hist)], ["mets_" + str(i) for i in range(hist_2)],
# ["calories_" + str(i) for i in range(hist_2)], ["heartrate_" + str(i) for i in range(hist_2)], ["y"]]
