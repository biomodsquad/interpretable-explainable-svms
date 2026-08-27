"""Public API and end-to-end smoke tests."""

import pickle

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mistic
from mistic import cvSet, kernelWrapper, paramSet, score_svc, svmSet
from mistic.utility import combined_rank, rank_items


def _fitted_ensemble():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X[:150, :6])
    y = y[:150]
    splits = cvSet(X, y)
    splits.classification(num_sets=2, validation_size=0.25, random_seed=7)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=kernelWrapper("linear"),
    )
    ensemble.tune_models([paramSet({"C": 1.0}, {})])
    return ensemble


def test_public_api_and_version():
    assert mistic.__version__ == "0.1.0"
    assert set(mistic.__all__) == {
        "combined_rank", "cvSet", "kernelWrapper", "paramSet",
        "score_svc", "score_svr", "svmSet",
    }


def test_cv_and_ensemble_are_pickleable():
    ensemble = _fitted_ensemble()
    restored = pickle.loads(pickle.dumps(ensemble))
    np.testing.assert_array_equal(restored.cv.X, ensemble.cv.X)


def test_tuning_prediction_and_ranking():
    ensemble = _fitted_ensemble()
    predictions = ensemble.predict(ensemble.cv.X[:5])
    assert predictions.shape == (5,)
    ranks = combined_rank(number_samples=5).compute(ensemble, 0, "train")
    assert sorted(ranks.tolist()) == list(range(ensemble.cv.X.shape[1]))


def test_rank_items_orders_values():
    np.testing.assert_array_equal(rank_items(np.array([10, 5, 20])), [1, 0, 2])
    np.testing.assert_array_equal(rank_items(np.array([10, 5, 20]), descending=True), [1, 2, 0])
