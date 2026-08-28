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
    assert mistic.__version__ == "0.1.1"
    assert set(mistic.__all__) == {
        "combined_rank", "cvSet", "kernelWrapper", "paramSet", "perDiff",
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


def test_greedy_forward_selection_evaluates_singletons_and_orders_features():
    ensemble = _fitted_ensemble()
    parameter_grid = [paramSet({"C": 1.0}, {})]

    ensemble.greedy_forward_selection(parameter_grid, reduction_factor=0.5)

    assert len(ensemble.singleton_performance_) == ensemble.cv.X.shape[1]
    assert sorted(ensemble.sorted_features.tolist()) == list(range(ensemble.cv.X.shape[1]))
    assert len(ensemble.features) >= 1
    scores = [row["score"] for row in ensemble.feature_performance_.values()]
    expected_size = ensemble.feature_performance_[int(np.argmax(scores))]["num_features"]
    assert len(ensemble.features) == expected_size


def test_forward_decision_perturbation_has_opposite_sign():
    ensemble = _fitted_ensemble()
    X = ensemble.cv.X[:5]
    ensemble._set_features([0, 1], update_kernel=False)
    ensemble.tune_models([paramSet({"C": 1.0}, {})])
    backward = ensemble.decision_perturbation_(0, X)[:, 1]

    ensemble._set_features([0], update_kernel=False)
    ensemble._selection_direction_ = "forward"
    try:
        forward = ensemble.decision_perturbation_(0, X)[:, 0]
    finally:
        del ensemble._selection_direction_

    np.testing.assert_allclose(forward, -backward)


def test_enrichment_score_sorts_feature_counts_descending():
    ensemble = _fitted_ensemble()
    forward_history = {
        0: {"num_features": 1, "score": 0.5},
        1: {"num_features": 2, "score": 0.75},
    }
    backward_history = dict(reversed(forward_history.items()))

    ensemble.feature_performance_ = forward_history
    forward_score = ensemble.enrichment_score()
    ensemble.feature_performance_ = backward_history
    backward_score = ensemble.enrichment_score()

    assert forward_score == 0.3125
    assert backward_score == forward_score
