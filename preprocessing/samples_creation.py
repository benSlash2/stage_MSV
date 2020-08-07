import numpy as np
import pandas as pd

def create_samples_idiab(data, ph, hist, day_len):
    """
    Create samples consisting in glucose, insulin and CHO histories (past hist-length values)
    :param data: dataframe
    :param ph: prediction horizon in minutes in sampling frequency scale
    :param hist: history length in sampling frequency scale
    :param day_len: length of day in sampling frequency scale
    :return: dataframe of samples
    """
    n_samples = data.shape[0] - ph - hist + 1
    # hist_2 = hist //2
    y = data.loc[ph + hist - 1:, "glucose"].values.reshape(-1, 1)
    d = pd.DatetimeIndex(data.loc[ph + hist - 1:, "datetime"].values)
    t = np.concatenate([np.arange(day_len) for _ in range(len(data) // day_len)], axis=0)[ph + hist - 1:].reshape(-1, 1)
    g = np.array([data.loc[i:i + n_samples - 1, "glucose"] for i in range(hist)]).transpose()
    c = np.array([data.loc[i:i + n_samples - 1, "CHO"] for i in range(hist)]).transpose()
    i = np.array([data.loc[i:i + n_samples - 1, "insulin"] for i in range(hist)]).transpose()
    m = np.array([data.loc[i:i + n_samples - 1, "mets"] for i in range(hist)]).transpose()
    cal = np.array([data.loc[i:i + n_samples - 1, "calories"] for i in range(hist)]).transpose()
    h = np.array([data.loc[i:i + n_samples - 1, "heartrate"] for i in range(hist)]).transpose()
    # m = np.array([data.loc[i + hist_2:i + hist_2 + n_samples - 1, "mets"] for i in range(hist_2)]).transpose()
    # cal = np.array([data.loc[i + hist_2:i + hist_2 + n_samples - 1, "calories"] for i in range(hist_2)]).transpose()
    # h = np.array([data.loc[i + hist_2:i + hist_2 + n_samples - 1, "heartrate"] for i in range(hist_2)]).transpose()
    new_columns = np.r_[["time"], ["glucose_" + str(i) for i in range(hist)], ["CHO_" + str(i) for i in range(hist)], [
        "insulin_" + str(i) for i in range(hist)], ["mets_" + str(i) for i in range(hist)], ["calories_" + str(i) for
                                                                                             i in range(hist)],
                        ["heartrate_" + str(i) for i in range(hist)], ["y"]]
    # new_columns = np.r_[["time"], ["glucose_" + str(i) for i in range(hist)], ["CHO_" + str(i) for i in range(hist)], [
    #     "insulin_" + str(i) for i in range(hist)], ["mets_" + str(i) for i in range(hist_2)], ["calories_" + str(i) for i in range(hist_2)],
    #                     ["heartrate_" + str(i) for i in range(hist_2)], ["y"]]

    new_data = pd.DataFrame(data=np.c_[t, g, c, i, m, cal, h, y], columns=new_columns)
    new_data["datetime"] = d
    new_data = new_data.loc[:, np.r_[["datetime"], new_columns]]  # reorder the columns, with datetime first

    return new_data


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
    hist_2 = hist //2
    y = data.loc[ph + hist - 1:, "glucose"].values.reshape(-1, 1)
    d = pd.DatetimeIndex(data.loc[ph + hist - 1:, "datetime"].values)
    t = np.concatenate([np.arange(day_len) for _ in range(len(data) // day_len)], axis=0)[ph + hist - 1:].reshape(-1, 1)
    g = np.array([data.loc[i:i + n_samples - 1, "glucose"] for i in range(hist)]).transpose()
    c = np.array([data.loc[i:i + n_samples - 1, "CHO"] for i in range(hist)]).transpose()
    i = np.array([data.loc[i:i + n_samples - 1, "insulin"] for i in range(hist)]).transpose()

    new_columns = np.r_[["time"], ["glucose_" + str(i) for i in range(hist)], ["CHO_" + str(i) for i in range(hist)], [
        "insulin_" + str(i) for i in range(hist)], ["y"]]
    new_data = pd.DataFrame(data=np.c_[t, g, c, i, y], columns=new_columns)
    new_data["datetime"] = d
    new_data = new_data.loc[:, np.r_[["datetime"], new_columns]]  # reorder the columns, with datetime first

    return new_data