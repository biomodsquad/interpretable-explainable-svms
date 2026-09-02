"""Ranking, scoring, kernel, and numerical utility classes for MISTIC."""

import copy

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.metrics.pairwise import pairwise_kernels


class combined_rank:
    """Blend perturbation and frozen-objective feature rankings.

    Parameters
    ----------
    weight : float, default=0.75
        Weight assigned to the perturbation ranking.
    number_samples : int, default=100
        Number of synthetic samples used when ``set_for_rank="sample"``.
    random_seed : int, default=0
        Seed used to generate synthetic samples.

    Attributes
    ----------
    weight : float
        Perturbation-ranking weight; feature importance receives ``1-weight``.
    number_samples : int
        Number of samples generated for synthetic ranking.
    random_seed : int
        Random seed for synthetic ranking data.
    """

    def __init__(self, weight=0.75, number_samples=100, random_seed=0):
        """Configure the contribution weight and synthetic sample generator.

        Parameters are documented on :class:`combined_rank`.

        Returns
        -------
        None
        """
        self.weight = weight
        self.number_samples = number_samples
        self.random_seed = random_seed

    def compute(self, svmSet, model_index, set_for_rank):
        """Return consensus ranks for one fitted ensemble member.

        Parameters
        ----------
        svmSet : mistic.svmSet.svmSet
            Fitted ensemble whose features are ranked.
        model_index : int
            Index of the ensemble member to inspect.
        set_for_rank : str
            ``"sample"`` for synthetic observations, otherwise the name of a
            cross-validation index collection such as ``"train"``.

        Returns
        -------
        numpy.ndarray
            Zero-based consensus rank for each feature.
        """
        if set_for_rank == "sample":
            np.random.seed(self.random_seed)
            X_for_rank = np.zeros([self.number_samples, svmSet.cv.X.shape[1]])
            source = svmSet.cv.X
            if svmSet._is_one_class():
                source = svmSet.cv.X[svmSet.cv.y == 1]
            for i in range(svmSet.cv.X.shape[1]):
                X_for_rank[:, i] = np.random.normal(
                    loc=np.mean(source[:, i]), scale=np.std(source[:, i]), size=self.number_samples
                )
        else:
            rank_indices = np.asarray(getattr(svmSet.cv, set_for_rank)[model_index], dtype=int)
            if svmSet._is_one_class():
                rank_indices = rank_indices[svmSet.cv.y[rank_indices] == 1]
                if not len(rank_indices):
                    raise ValueError("one-class contribution ranking requires inlier samples")
            X_for_rank = svmSet.cv.X[rank_indices]
            # X_for_rank = svmSet.cv.X[getattr(svmSet.cv.sets[model_index], set_for_rank)]

        model = svmSet.models[model_index]
        uses_probability = (
            not svmSet._is_one_class() and getattr(model, "probability", False) is True
        )
        if uses_probability:
            feature_contribution = svmSet.probability_perturbation_(model_index, X_for_rank)
        else:
            feature_contribution = svmSet.decision_perturbation_(model_index, X_for_rank)
        if svmSet._is_one_class():
            current = np.asarray(
                svmSet.decision_function(X_for_rank, model_index=model_index), dtype=float
            )
            perturbed = current[:, np.newaxis] - feature_contribution
            current_dispersion = np.var(current)
            perturbed_dispersion = np.var(perturbed, axis=0)
            forward = getattr(svmSet, "_selection_direction_", None) == "forward"
            compression_gain = (
                (current_dispersion - perturbed_dispersion)
                if forward
                else (perturbed_dispersion - current_dispersion)
            )

            # Only rank a candidate as eligible when the perturbed model
            # retains the OneClassSVM coverage target on true-class samples.
            coverage = np.mean(perturbed >= 0, axis=0)
            target = 1.0 - float(svmSet.models[model_index].nu)
            eligible = coverage >= target
            order = np.lexsort((compression_gain, eligible.astype(int)))
            contribution_rank = np.empty(len(order), dtype=int)
            contribution_rank[order] = np.arange(len(order))
        else:
            if uses_probability:
                # Probability perturbations already express sensitivity on a
                # bounded, calibrated scale. Accumulate their magnitudes
                # directly rather than squaring them as decision margins are.
                cummulative_contribution = np.sum(np.abs(feature_contribution), axis=0)
            else:
                cummulative_contribution = np.sum(feature_contribution**2, axis=0)
            contribution_rank = rank_items(cummulative_contribution)

        feature_importance = svmSet.feature_importance_(model_index)
        if svmSet._is_one_class():
            feature_rank = rank_items(feature_importance)
        else:
            feature_rank = rank_items(feature_importance, descending=True)

        consensus_rank = self.weight * contribution_rank + (1 - self.weight) * feature_rank
        rank = rank_items(consensus_rank)

        return rank


class dotdict(dict):
    """Dictionary supporting attribute-style key access.

    Attributes
    ----------
    keys : object
        Dictionary keys are exposed dynamically as attributes.
    """

    def __getattr__(self, name):
        """Return a keyed value or raise the standard attribute error.

        Parameters
        ----------
        name : str
            Dictionary key requested through attribute syntax.

        Returns
        -------
        object
            Value stored under ``name``.

        Raises
        ------
        AttributeError
            If the mapping does not contain ``name``.
        """
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        """Create a recursive copy while retaining attribute access.

        Parameters
        ----------
        memo : dict or None, default=None
            Object-identity memo used by :func:`copy.deepcopy`.

        Returns
        -------
        dotdict
            Independent recursive copy of this mapping.
        """
        return dotdict(copy.deepcopy(dict(self), memo=memo))


class paramSet:
    """Pair estimator parameters with precomputed-kernel parameters.

    Parameters
    ----------
    model : mapping
        Parameters passed to the scikit-learn SVM estimator.
    kernel : mapping
        Parameters passed to the pairwise-kernel computation.

    Attributes
    ----------
    model : mapping
        Estimator parameter mapping.
    kernel : mapping
        Kernel parameter mapping.
    """

    def __init__(self, model, kernel):
        """Store model and kernel parameter mappings.

        Parameters are documented on :class:`paramSet`.

        Returns
        -------
        None
        """
        self.model = model
        self.kernel = kernel


class score_svc:
    """Score binary SVC members by discrimination and calibration.

    Parameters
    ----------
    weight : float, default=0.5
        AUC weight within the discrimination score; F1 receives ``1-weight``.
    calibration_weight : float, default=0.2
        Weight assigned to calibrated probability performance.

    Attributes
    ----------
    weight : float
        AUC contribution to discrimination performance.
    calibration_weight : float
        Calibration contribution to aggregate performance.
    """

    def __init__(self, weight=0.5, calibration_weight=0.2):
        """Configure AUC/F1 and optional probability-calibration weights.

        Parameters are documented on :class:`score_svc`.

        Returns
        -------
        None
        """
        if not 0 <= weight <= 1:
            raise ValueError("weight must be between 0 and 1")
        if not 0 <= calibration_weight <= 1:
            raise ValueError("calibration_weight must be between 0 and 1")
        self.weight = weight
        self.calibration_weight = calibration_weight

    def score(self, svmSet, model_index):
        """Return F1, AUC, Brier loss, and aggregate score for one member.

        Parameters
        ----------
        svmSet : mistic.svmSet.svmSet
            Fitted ensemble containing the member and validation data.
        model_index : int
            Index of the member to score.

        Returns
        -------
        dict
            ``f1``, ``auc``, ``brier``, ``calibration``, and aggregate
            ``score`` values. Calibration entries are NaN when probability
            estimates are unavailable.
        """
        if svmSet.separate_feature_sets | svmSet.separate_parameters:
            kernel_matrix = svmSet._get_kernel_matrix(
                svmSet.cv.test[model_index], svmSet.X_ind[model_index], model_index
            )
        else:
            kernel_matrix = svmSet._get_kernel_matrix(
                svmSet.cv.test[model_index], svmSet.X_ind[model_index]
            )

        y_pred = svmSet.models[model_index].predict(kernel_matrix)

        _tn, fp, fn, tp = confusion_matrix(svmSet.cv.y[svmSet.cv.test[model_index]], y_pred).ravel()

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        model = svmSet.models[model_index]
        if getattr(model, "probability", False) is True:
            # SVC orders probability columns according to ``classes_``.  The
            # second class is also the positive side of the binary decision
            # function used by the existing scorer.
            probability = model.predict_proba(kernel_matrix)[:, 1]
            curve_score = probability
            brier = brier_score_loss(
                svmSet.cv.y[svmSet.cv.test[model_index]], probability, pos_label=model.classes_[1]
            )
            calibration = 1 - brier
        else:
            curve_score = model.decision_function(kernel_matrix)
            brier = np.nan
            calibration = np.nan
        auc = roc_auc_score(svmSet.cv.y[svmSet.cv.test[model_index]], curve_score)

        discrimination = self.weight * auc + (1 - self.weight) * f1
        score = (
            (1 - self.calibration_weight) * discrimination + self.calibration_weight * calibration
            if np.isfinite(calibration)
            else discrimination
        )

        return {"f1": f1, "auc": auc, "brier": brier, "calibration": calibration, "score": score}


class score_ocsvm:
    """Score a one-class SVM using sklearn's ``-1``/``+1`` convention.

    With labeled inliers and outliers, ``score`` combines ROC AUC and inlier
    F1 in the same way :class:`score_svc` does.  A validation set containing
    only inliers is scored by the fraction retained inside the boundary.

    Parameters
    ----------
    weight : float, default=0.5
        AUC weight in the aggregate score; F1 receives ``1-weight``.

    Attributes
    ----------
    weight : float
        AUC contribution to aggregate validation performance.
    """

    def __init__(self, weight=0.5):
        """Configure the AUC weight; F1 receives the remaining weight.

        Parameters are documented on :class:`score_ocsvm`.

        Returns
        -------
        None
        """
        self.weight = weight

    def score(self, svmSet, model_index):
        """Return inlier rate, F1, AUC, and aggregate score for one member.

        Parameters
        ----------
        svmSet : mistic.svmSet.svmSet
            Fitted one-class ensemble containing validation data.
        model_index : int
            Index of the ensemble member to score.

        Returns
        -------
        dict
            Inlier rate, F1, AUC, and aggregate score. AUC is NaN when the
            validation subset contains inliers only.
        """
        if svmSet.separate_feature_sets | svmSet.separate_parameters:
            kernel_matrix = svmSet._get_kernel_matrix(
                svmSet.cv.test[model_index], svmSet.X_ind[model_index], model_index
            )
        else:
            kernel_matrix = svmSet._get_kernel_matrix(
                svmSet.cv.test[model_index], svmSet.X_ind[model_index]
            )

        y_true = np.asarray(svmSet.cv.y[svmSet.cv.test[model_index]])
        y_pred = svmSet.models[model_index].predict(kernel_matrix)
        labels = np.unique(y_true)
        if not np.all(np.isin(labels, [-1, 1])):
            raise ValueError("one-class validation labels must be -1 or +1")

        inlier_rate = float(np.mean(y_pred == 1))
        if labels.size == 1:
            if labels[0] != 1:
                raise ValueError(
                    "one-class validation requires inliers when only one label is present"
                )
            return {
                "inlier_rate": inlier_rate,
                "f1": inlier_rate,
                "auc": np.nan,
                "score": inlier_rate,
            }

        decision = svmSet.models[model_index].decision_function(kernel_matrix)
        auc = roc_auc_score(y_true, decision)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        score = self.weight * auc + (1 - self.weight) * f1
        return {"inlier_rate": inlier_rate, "f1": f1, "auc": auc, "score": score}


class score_svr:
    """Score SVR members with correlation, R-squared, and RMSE.

    Parameters
    ----------
    weight : float, default=0.5
        Squared-Pearson-correlation weight; nonnegative R-squared receives the
        remaining weight.

    Attributes
    ----------
    weight : float
        Correlation contribution to the aggregate score.
    """

    def __init__(self, weight=0.5):
        """Configure the squared-correlation weight in the aggregate score.

        Parameters are documented on :class:`score_svr`.

        Returns
        -------
        None
        """
        self.weight = weight

    def score(self, svmSet, model_index):
        """Return RMSE, squared Pearson correlation, R-squared, and score.

        Parameters
        ----------
        svmSet : mistic.svmSet.svmSet
            Fitted regression ensemble containing validation data.
        model_index : int
            Index of the ensemble member to score.

        Returns
        -------
        dict
            ``rmse``, squared ``pearson`` correlation, ``r2``, and aggregate
            ``score`` values.
        """
        if svmSet.separate_feature_sets | svmSet.separate_parameters:
            kernel_matrix = svmSet._get_kernel_matrix(
                svmSet.cv.test[model_index], svmSet.X_ind[model_index], model_index
            )
        else:
            kernel_matrix = svmSet._get_kernel_matrix(
                svmSet.cv.test[model_index], svmSet.X_ind[model_index]
            )

        y_pred = svmSet.models[model_index].predict(kernel_matrix)

        if len(np.unique(y_pred)) <= 2:
            pearson = 0.00001
            coef_det = 0.00001
            rmse = 1e12
        else:
            pearson = pearsonr(svmSet.cv.y[svmSet.cv.test[model_index]], y_pred).statistic ** 2
            coef_det = r2_score(svmSet.cv.y[svmSet.cv.test[model_index]], y_pred)
            rmse = root_mean_squared_error(svmSet.cv.y[svmSet.cv.test[model_index]], y_pred)

        score = self.weight * float(pearson) + (1 - self.weight) * max(0.00001, coef_det)

        return {"rmse": rmse, "pearson": pearson, "r2": coef_det, "score": score}


class kernelWrapper:
    """Compute pairwise kernels and supported analytical derivatives.

    Parameters
    ----------
    type : str, default="rbf"
        Pairwise-kernel metric name.

    Attributes
    ----------
    type : str
        Kernel metric passed to scikit-learn.
    """

    def __init__(self, type="rbf"):
        """Select a kernel accepted by scikit-learn pairwise kernels.

        Parameters are documented on :class:`kernelWrapper`.

        Returns
        -------
        None
        """
        # [‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’]
        self.type = type

    def compute(self, X, feature_index, parameters=None, Y=None):
        """Compute a kernel matrix over the selected feature columns.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples_x, n_features)
            Left-hand input matrix.
        feature_index : array-like of int
            Feature columns included in the kernel.
        parameters : dict or None, default=None
            Keyword parameters passed to ``pairwise_kernels``.
        Y : numpy.ndarray or None, default=None
            Optional right-hand input matrix. An empty value computes the
            symmetric kernel of ``X``.

        Returns
        -------
        numpy.ndarray
            Pairwise kernel matrix.
        """

        if Y is None:
            Y = []
        if parameters is None:
            parameters = {}
        if len(Y) == 0:
            if not bool(parameters):
                kernel_matrix = pairwise_kernels(X[:, feature_index], metric=self.type)
            else:
                kernel_matrix = pairwise_kernels(
                    X[:, feature_index], metric=self.type, **parameters
                )
        else:
            if not bool(parameters):
                kernel_matrix = pairwise_kernels(
                    X[:, feature_index], Y=Y[:, feature_index], metric=self.type
                )
            else:
                kernel_matrix = pairwise_kernels(
                    X[:, feature_index], Y=Y[:, feature_index], metric=self.type, **parameters
                )

        return kernel_matrix

    def compute_gradient(self, X, feature_index, wrt, parameters, Y=None):
        """Differentiate a supported kernel with respect to one feature.

        Parameters
        ----------
        X : numpy.ndarray
            Left-hand input matrix.
        feature_index : array-like of int
            Feature columns included in the kernel.
        wrt : int
            Original feature column with respect to which to differentiate.
        parameters : dict
            Kernel parameters, including ``gamma`` or ``degree`` as required.
        Y : numpy.ndarray or None, default=None
            Right-hand input matrix.

        Returns
        -------
        numpy.ndarray
            Kernel derivative matrix with one row per ``X`` sample and one
            column per ``Y`` sample.
        """
        if Y is None:
            Y = []
        if self.type == "rbf":
            K = self.compute(X, feature_index=feature_index, parameters=parameters, Y=Y)

            kernel_gradient = (
                2 * parameters["gamma"] * (X[:, wrt, np.newaxis] - Y[np.newaxis, :, wrt]) * K
            )

        elif self.type == "linear":
            kernel_gradient = np.broadcast_to(X[:, wrt, np.newaxis], (len(X), len(Y)))

        elif self.type == "polynomial":
            # K(X, Y) = (gamma <X, Y> + coef0) ^ degree
            d_parameters = copy.deepcopy(parameters)
            d_parameters["degree"] = parameters["degree"] - 1

            kernel_gradient = (
                parameters["degree"]
                * X[:, wrt, np.newaxis]
                * self.compute(X, feature_index, d_parameters, Y)
            )

        else:
            raise NameError("NoGradientMethod")

        return kernel_gradient


def rank_items(score, descending=False):
    """Convert numeric scores into zero-based ordinal ranks.

    Parameters
    ----------
    score : array-like
        Numeric values to rank.
    descending : bool, default=False
        Rank the greatest value first when true.

    Returns
    -------
    numpy.ndarray
        Zero-based rank at each original input position.
    """
    if descending:
        sign = -1
    else:
        sign = 1

    return np.argsort(sign * score).argsort()


def svc_dec2(x, svmSet, model_index, n_to_opt=None, xref=None):
    """Return a squared decision value for boundary-point optimization.

    Parameters
    ----------
    x : array-like
        Complete candidate point, or values for the optimized columns.
    svmSet : mistic.svmSet.svmSet
        Fitted ensemble used to evaluate the decision function.
    model_index : int
        Index of the ensemble member to evaluate.
    n_to_opt : array-like of int or None, default=None
        Columns replaced in ``xref`` by ``x``. If omitted, ``x`` is treated
        as the complete point.
    xref : numpy.ndarray or None, default=None
        Reference point modified when only selected columns are optimized.

    Returns
    -------
    numpy.ndarray
        Squared decision-function value for the candidate point.
    """
    if n_to_opt is None:
        xstar = x
    else:
        xref[:, n_to_opt] = x
        xstar = xref

    if svmSet.separate_parameters:
        parameters = svmSet.parameters_[model_index].kernel
    else:
        parameters = svmSet.parameters_.kernel

    if svmSet.separate_feature_sets:
        feature_index = svmSet.features[model_index]
    else:
        feature_index = svmSet.features

    xstar = np.reshape(xstar, [1, -1])
    K = svmSet.kernel.compute(
        X=xstar,
        feature_index=feature_index,
        parameters=parameters,
        Y=svmSet.cv.X[svmSet.X_ind[model_index], :],
    )

    y = svmSet.models[model_index].decision_function(K)
    y_squared = y**2

    return y_squared


def perDiff(dat):
    """Return mean relative differences for every pair of columns.

    Pairs are evaluated in SciPy condensed-matrix order and in bounded chunks,
    avoiding both pandas row-wise callbacks and an unbounded
    ``n_rows x n_columns x n_columns`` temporary.

    Parameters
    ----------
    dat : array-like or pandas.DataFrame of shape (n_rows, n_columns)
        Numeric observations whose column pairs are compared.

    Returns
    -------
    numpy.ndarray
        Mean relative difference for each column pair in SciPy condensed-
        matrix order.
    """
    values = np.asarray(dat, dtype=float)
    if values.ndim != 2:
        raise ValueError("dat must be a two-dimensional array or DataFrame")

    n_rows, n_columns = values.shape
    left_columns, right_columns = np.triu_indices(n_columns, k=1)
    differences = np.empty(len(left_columns), dtype=float)

    # The numerator and denominator are both n_rows x chunk_size arrays.
    # Keeping each at roughly 16 MB bounds peak temporary memory.
    pairs_per_chunk = max(1, 2_000_000 // max(1, n_rows))
    for start in range(0, len(left_columns), pairs_per_chunk):
        stop = min(start + pairs_per_chunk, len(left_columns))
        left = values[:, left_columns[start:stop]]
        right = values[:, right_columns[start:stop]]
        numerator = np.abs(left - right)
        denominator = np.maximum(np.abs(left), np.abs(right))

        relative_difference = np.full(numerator.shape, np.nan)
        np.divide(numerator, denominator, out=relative_difference, where=denominator != 0)

        valid = ~np.isnan(relative_difference)
        valid_count = np.sum(valid, axis=0)
        chunk_means = np.full(stop - start, np.nan)
        np.divide(
            np.nansum(relative_difference, axis=0),
            valid_count,
            out=chunk_means,
            where=valid_count != 0,
        )
        differences[start:stop] = chunk_means

    return differences
