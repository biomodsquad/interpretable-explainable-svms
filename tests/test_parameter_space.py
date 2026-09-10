"""Tests for space-filling hyperparameter generation."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from mistic import (
    categorical,
    cvSet,
    integer,
    kernelWrapper,
    loguniform,
    parameterSpace,
    score_svc,
    svmSet,
    uniform,
)


def test_latin_hypercube_is_reproducible_stratified_and_bounded():
    space = parameterSpace(
        model={
            "C": loguniform(1e-3, 1e3),
            "tol": uniform(1e-5, 1e-2),
            "degree": integer(2, 5),
            "class_weight": categorical([None, "balanced"]),
            "probability": True,
        },
        kernel={"gamma": loguniform(1e-6, 1e1)},
    )

    first = space.sample(16, random_state=7, optimization=None)
    second = space.sample(16, random_state=7, optimization=None)
    assert [(item.model, item.kernel) for item in first] == [
        (item.model, item.kernel) for item in second
    ]
    assert len(first) == 16
    assert all(1e-3 <= item.model["C"] < 1e3 for item in first)
    assert all(1e-5 <= item.model["tol"] < 1e-2 for item in first)
    assert all(2 <= item.model["degree"] <= 5 for item in first)
    assert all(item.model["class_weight"] in {None, "balanced"} for item in first)
    assert all(item.model["probability"] is True for item in first)
    assert all(1e-6 <= item.kernel["gamma"] < 1e1 for item in first)

    # LHS places one continuous C coordinate in each marginal stratum.
    normalized_log_c = (np.log10([item.model["C"] for item in first]) + 3) / 6
    assert sorted(np.floor(normalized_log_c * 16).astype(int).tolist()) == list(range(16))


def test_latin_hypercube_removes_discrete_duplicates_and_groups_kernels():
    space = parameterSpace(
        model={"C": categorical([1.0, 10.0])},
        kernel={"gamma": categorical([0.1, 1.0])},
    )
    candidates = space.sample(64, random_state=3)
    signatures = [(item.model["C"], item.kernel["gamma"]) for item in candidates]
    assert len(signatures) == len(set(signatures)) == 4
    gammas = [item.kernel["gamma"] for item in candidates]
    assert all(gammas[index] <= gammas[index + 1] for index in range(len(gammas) - 1))


def test_fixed_space_returns_one_candidate_and_invalid_inputs_are_rejected():
    candidate = parameterSpace(model={"C": 1.0}, kernel={"gamma": 0.2}).sample(10)
    assert len(candidate) == 1
    assert candidate[0].model == {"C": 1.0}
    assert candidate[0].kernel == {"gamma": 0.2}

    for invalid in (0, -1, 1.5, True):
        with np.testing.assert_raises((TypeError, ValueError)):
            parameterSpace().sample(invalid)
    with np.testing.assert_raises(ValueError):
        parameterSpace().sample(2, strategy="random")
    with np.testing.assert_raises(ValueError):
        loguniform(0, 1)
    with np.testing.assert_raises(ValueError):
        uniform(1, 1)
    with np.testing.assert_raises(ValueError):
        categorical([])


def test_latin_hypercube_candidates_tune_an_rbf_ensemble():
    X, y = make_classification(n_samples=80, n_features=5, n_informative=3, random_state=11)
    splits = cvSet(X, y)
    splits.classification(num_sets=2, validation_size=0.25, random_seed=5)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=kernelWrapper("rbf"),
    )
    candidates = parameterSpace(
        model={"C": loguniform(0.1, 10)},
        kernel={"gamma": loguniform(1e-3, 1)},
    ).sample(6, random_state=13)

    ensemble.tune_models(candidates)
    assert np.isfinite(ensemble.performance_.score)
    assert ensemble.parameters_ in candidates
