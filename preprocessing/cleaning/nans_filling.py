import numpy as np


def fill_nans(data, day_len, n_days_test):
    """
    Fill NaNs inside the dataframe of samples following:
    - CHO and insulin values are filled with 0
    - glucose history are interpolated (linearly) when possible and extrapolated if not
    :param data: sample dataframe
    :param day_len: length of day, scaled to sampling frequency
    :param n_days_test: number of test days
    :return: cleaned sample dataframe
    """
    data_nan = data.copy()

    # fill insulin and CHO nans with 0
    for col in data.columns:
        if "insulin" in col or "CHO" in col or "steps" in col:
            data[col] = data[col].fillna(0)
        if "mets" in col:
            data[col] = data[col].fillna(10)
        if "calories" in col:
            data[col] = data[col].fillna(1.1296000480651855)
        if "heartrate" in col:
            heart_mean = data.drop(data.loc[data[col] < 0].index, axis=0).mean()[col]
            data[col] = data[col].fillna(heart_mean)
            data.loc[data[col] < 0, col] = heart_mean

    # fill glucose nans
    g_col = [col for col in data.columns if "glucose" in col]
    g_mean = data.y.mean()
    for i in range(len(data.index)):
        g_i = data.loc[i, g_col]
        if g_i.notna().all():  # no nan
            pass
        elif g_i.isna().all():  # all nan
            if i > len(data) - n_days_test * day_len + 1:
                last_known_gs = data_nan.loc[:i - 1, "glucose_0"][data_nan.loc[:i - 1, "glucose_0"].notna()]
                if len(last_known_gs) >= 1:
                    for col in g_col:
                        # data.loc[i,col] = last_known_gs.iloc[-1]
                        data.loc[i, col] = g_mean
                else:
                    for col in g_col:
                        data.loc[i, col] = g_mean
        else:  # some nan
            # compute in-sample nan indices, and last out-sample + in-sample non-nan indices
            is_na_idx = i + np.where(g_i.isna())[0]
            not_na_idx = i + np.where(g_i.notna())[0]
            if data_nan.loc[:i - 1, "glucose_0"].notna().any():
                mask = data_nan.loc[:i - 1, "glucose_0"].notna().values
                not_na_idx = np.r_[data_nan.loc[:i - 1, "glucose_0"][mask].index[-2:], not_na_idx]

            # for all nan
            for is_na_i in is_na_idx:
                # get the two closest non nan values
                idx_diff = not_na_idx - is_na_i

                if np.any(idx_diff > 0) and np.any(idx_diff < 0):
                    # we got a start and an end
                    start = not_na_idx[np.where(idx_diff < 0, idx_diff, -np.inf).argmax()]
                    end = not_na_idx[np.where(idx_diff > 0, idx_diff, np.inf).argmin()]

                    start_idx = _compute_indexes(i, start, len(data_nan))
                    end_idx = _compute_indexes(i, end, len(data_nan))

                    start_val = data_nan.loc[start_idx]
                    end_val = data_nan.loc[end_idx]

                    # interpolate between them
                    rate = (end_val - start_val) / (end - start)
                    data.loc[i, g_col[is_na_i - i]] = data_nan.loc[start_idx] + rate * (is_na_i - start)
                elif np.any(idx_diff > 0):
                    # we only have end(s)
                    # backward extrapolation - only used in very first day where there is no start
                    if len(idx_diff) >= 2:
                        # we have two last values so we can compute a rate
                        end1, end2 = not_na_idx[0], not_na_idx[1]
                        [end1_idx, end2_idx] = [_compute_indexes(i, _, len(data_nan)) for _ in [end1, end2]]
                        end1_val, end2_val = data_nan.loc[end1_idx], data_nan.loc[end2_idx]
                        rate = (end2_val - end1_val) / (end2 - end1)
                        data.loc[i, g_col[is_na_i - i]] = data_nan.loc[end1_idx] - rate * (end1 - is_na_i)
                    else:
                        # we have only one value so we cannot compute a rate
                        end = not_na_idx[0]
                        end_idx = _compute_indexes(i, end, len(data_nan))
                        end_val = data_nan.loc[end_idx]
                        data.loc[i, g_col[is_na_i - i]] = end_val
                elif np.any(idx_diff < 0):
                    # forward extrapolation
                    if len(idx_diff) >= 2:
                        end1, end2 = not_na_idx[-2], not_na_idx[-1]
                        [end1_idx, end2_idx] = [_compute_indexes(i, _, len(data_nan)) for _ in [end1, end2]]
                        end1_val, end2_val = data_nan.loc[end1_idx], data_nan.loc[end2_idx]
                        rate = (end2_val - end1_val) / (end2 - end1)
                        data.loc[i, g_col[is_na_i - i]] = data_nan.loc[end1_idx] - rate * (end1 - is_na_i)
                    else:
                        # we only have one value, so we cannot compute a rate
                        last_val = g_i[g_i.notna()][-1]
                        data.loc[i, g_col[is_na_i - i]] = last_val

    return data


def _compute_indexes(i, index, length):
    if index >= length:
        return i, "glucose_" + str(index - i)
    else:
        return index, "glucose_0"
