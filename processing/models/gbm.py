from sklearn.inspection import permutation_importance
from processing.models.predictor import Predictor
from sklearn.ensemble import GradientBoostingRegressor
import misc.constants as cs
import os
from joblib import dump, load


class GBM(Predictor):
    def fit(self):
        x_train, y_train, _ = self._str2dataset("train")
        x_valid, y_valid, _ = self._str2dataset("valid")
        x = np.r_[x_train, x_valid]
        y = np.r_[y_train, y_valid]

        # define the model
        self.model = GradientBoostingRegressorCustom(
            n_estimators=self.params["n_estimators"],
            criterion="mse",
            max_depth=self.params["max_depth"],
            min_samples_split=self.params["min_samples_split"],
            learning_rate=self.params["learning_rate"],
            loss=self.params["loss"],
            n_iter_no_change=self.params["n_iter_no_change"],
            validation_fraction=len(x_valid) / (len(x_train) + len(x_valid)),
            random_state=cs.seed,
            # verbose=1,
        )

        # fit the model
        self.model.fit(x, y)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)

        # predict
        y_pred = self.model.predict(x)
        y_true = y.values

        return self._format_results(y_true, y_pred, t)

    def save(self, file):
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        dump(self.model, filename=file)

    def load(self, file):
        self.model = load(filename=file)

    def feature_importances(self, method="gini", dataset="test"):
        x, y, t = self._str2dataset(dataset)

        self.model.apply(x)

        if method == "gini":
            feature_importances = self.model.feature_importances_
        else:
            feature_importances = permutation_importance(self.model, x, y, n_repeats=10,
                                                         random_state=0).importances_mean

        return feature_importances


from sklearn.utils import *
from sklearn.tree._tree import *
from sklearn.base import *
from sklearn.model_selection._split import *


class GradientBoostingRegressorCustom(GradientBoostingRegressor):
    """
        This class implements a custom variant of the sklearn GradientBoostingRegressor class.

        In particular, it disables the shuffling of training set when using the early stopping training.
        As a consequence, when we already have a custom training and validation set and we concatenate the two into the
        training set of the model, our own validation is the one used by the model (because no shuffling).

        This change results in only one line change :

        X, X_val, y, y_val, sample_weight, sample_weight_val = (
                train_test_split(X, y, sample_weight,
                                 random_state=self.random_state,
                                 test_size=self.validation_fraction,
                                 stratify=stratify
                                 )
            )

        Into

        X, X_val, y, y_val, sample_weight, sample_weight_val = (
                train_test_split(X, y, sample_weight,
                                 random_state=self.random_state,
                                 test_size=self.validation_fraction,
                                 stratify=stratify,
                                 shuffle=False
                                 )
            )
    """

    def fit(self, X, y, sample_weight=None, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        y : array-like, shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like, shape (n_samples,) or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshoting.

        Returns
        -------
        self : object
        """
        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()

        # Check input
        # Since check_array converts both X and y to the same dtype, but the
        # trees use different types for X and y, checking them separately.
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], dtype=DTYPE)
        n_samples, self.n_features_ = X.shape

        sample_weight_is_none = sample_weight is None
        if sample_weight_is_none:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)
            sample_weight_is_none = False

        check_consistent_length(X, y, sample_weight)

        y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        y = column_or_1d(y, warn=True)
        y = self._validate_y(y, sample_weight)

        if self.n_iter_no_change is not None:
            stratify = y if is_classifier(self) else None
            X, X_val, y, y_val, sample_weight, sample_weight_val = (
                train_test_split(X, y, sample_weight,
                                 random_state=self.random_state,
                                 test_size=self.validation_fraction,
                                 stratify=stratify,
                                 shuffle=False
                                 )
            )
            if is_classifier(self):
                if self.n_classes_ != np.unique(y).shape[0]:
                    # We choose to error here. The problem is that the init
                    # estimator would be trained on y, which has some missing
                    # classes now, so its predictions would not have the
                    # correct shape.
                    raise ValueError(
                        'The training data after the early stopping split '
                        'is missing some classes. Try using another random '
                        'seed.'
                    )
        else:
            X_val = y_val = sample_weight_val = None

        self._check_params()

        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model and initialize raw predictions
            if self.init_ == 'zero':
                raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                           dtype=np.float64)
            else:
                # XXX clean this once we have a support_sample_weight tag
                if sample_weight_is_none:
                    self.init_.fit(X, y)
                else:
                    msg = ("The initial estimator {} does not support sample "
                           "weights.".format(self.init_.__class__.__name__))
                    try:
                        self.init_.fit(X, y, sample_weight=sample_weight)
                    except TypeError:  # regular estimator without SW support
                        raise ValueError(msg)
                    except ValueError as e:
                        if "pass parameters to specific steps of " \
                           "your pipeline using the " \
                           "stepname__parameter" in str(e):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = \
                    self.loss_.get_init_raw_predictions(X, self.init_)

            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _decision_function (called in two lines
            # below) are more constrained than fit. It accepts only CSR
            # matrices.
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            raw_predictions = self._raw_predict(X)
            self._resize_state()

        X_idx_sorted = None

        # fit the boosting stages
        n_stages = self._fit_stages(
            X, y, raw_predictions, sample_weight, self._rng, X_val, y_val,
            sample_weight_val, begin_at_stage, monitor, X_idx_sorted)

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self
