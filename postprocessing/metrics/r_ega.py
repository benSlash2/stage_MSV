import numpy as np
from postprocessing.metrics.tools.cg_ega_tools import reshape_results, extract_columns_from_results
from .tools.cg_ega_tools import _all, _any


class REga:
    """
        The Rate-Error Grid Analysis (P-EGA) gives an estimation of the clinical acceptability of the glucose
        predictions based on their rate-of-change-accuracy (the accuracy of the predicted variations).
        Every prediction is given a mark from {"A", "B", "uC", "lC", "uD", "lD", "uE", "lE"}

        The implementation is taken from "Evaluating the accuracy of continuous glucose-monitoring sensors: continuous
        glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et al., 2004.
    """

    def __init__(self, results, freq):
        """
        Instantiate the P-EGA object.
        :param results: dataframe with predictions and ground truths
        :param freq: prediction frequency in minutes (e.g., 5)
        """

        self.freq = freq
        self.results = reshape_results(results, self.freq)

    def full(self):
        """
            Full version of the R-EGA, which consists of an array giving for every prediction (row), its mark vector
            (column). There are 8 columns representing the mark A, B, uC, lC, uD, lD, uE, and lE.

            :return: numpy array of shape (number of predictions, 8)
        """
        y_true, y_pred, dy_true, dy_pred = extract_columns_from_results(self.results)

        a = _any([
            _all([  # upper and lower
                np.greater_equal(dy_pred, dy_true - 1),
                np.less_equal(dy_pred, dy_true + 1)
            ]),
            _all([  # left
                np.less_equal(dy_pred, dy_true / 2),
                np.greater_equal(dy_pred, dy_true * 2)
            ]),
            _all([  # right
                np.less_equal(dy_pred, dy_true * 2),
                np.greater_equal(dy_pred, dy_true / 2)
            ])
        ])

        b = _all([
            np.equal(a, False),  # not in A but satisfies the cond below
            _any([
                _all([
                    np.less_equal(dy_pred, -1),
                    np.less_equal(dy_true, -1)
                ]),
                _all([
                    np.less_equal(dy_pred, dy_true + 2),
                    np.greater_equal(dy_pred, dy_true - 2)
                ]),
                _all([
                    np.greater_equal(dy_pred, 1),
                    np.greater_equal(dy_true, 1)
                ])
            ])
        ])

        uc = _all([
            np.less(dy_true, 1),
            np.greater_equal(dy_true, -1),
            np.greater(dy_pred, dy_true + 2)
        ])

        lc = _all([
            np.less_equal(dy_true, 1),
            np.greater(dy_true, -1),
            np.less(dy_pred, dy_true - 2)
        ])

        ud = _all([
            np.less_equal(dy_pred, 1),
            np.greater_equal(dy_pred, -1),
            np.greater(dy_pred, dy_true + 2)
        ])

        ld = _all([
            np.less_equal(dy_pred, 1),
            np.greater_equal(dy_pred, -1),
            np.less(dy_pred, dy_true - 2)
        ])

        ue = _all([
            np.greater(dy_pred, 1),
            np.less(dy_true, -1)
        ])

        le = _all([
            np.less(dy_pred, -1),
            np.greater(dy_true, 1)
        ])

        return np.concatenate([a, b, uc, lc, ud, ld, ue, le], axis=1)

    def mean(self):
        return np.mean(self.full(), axis=0)

    def a_plus_b(self):
        full = self.full()
        a_plus_b = full[:, 0] + full[:, 1]
        return np.sum(a_plus_b) / len(a_plus_b)
