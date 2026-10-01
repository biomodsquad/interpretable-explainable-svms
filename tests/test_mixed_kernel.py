import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

from mistic import (
    MixedKernel,
    MixedkernelWrapper,
    cvSet,
    kernelWrapper,
    paramSet,
    score_svc,
    svmSet,
)


@pytest.fixture
def data():
    X = np.array([[0.2, -0.5, 1.0], [1.2, 0.4, -0.3]])
    Y = np.array([[-0.1, 0.8, 0.7], [0.9, -0.2, 0.5], [0.3, 0.1, -0.6]])
    return X, Y


def test_expression_matches_manual_sum_product_and_power(data):
    X, Y = data
    linear = kernelWrapper("linear")
    rbf = kernelWrapper("rbf", name="radial")
    expression = (2.0 * linear + rbf) * rbf**2
    parameters = {"radial__gamma": 0.4}

    actual = expression.compute(X, [0, 1, 2], parameters, Y)
    linear_matrix = linear.compute(X, [0, 1, 2], {}, Y)
    rbf_matrix = rbf.compute(X, [0, 1, 2], {"gamma": 0.4}, Y)
    np.testing.assert_allclose(actual, (2 * linear_matrix + rbf_matrix) * rbf_matrix**2)


def test_per_kernel_features_intersect_selected_features(data):
    X, Y = data
    mixed = MixedKernel.weighted_sum(
        [kernelWrapper("linear"), kernelWrapper("rbf")],
        weights=[0.25, 0.75],
        feature_sets=[[0], [1, 2]],
    )
    actual = mixed.compute(X, [0, 1], {"gamma": 0.3}, Y)
    expected = 0.25 * kernelWrapper("linear").compute(X, [0], {}, Y)
    expected += 0.75 * kernelWrapper("rbf").compute(X, [1], {"gamma": 0.3}, Y)
    np.testing.assert_allclose(actual, expected)


def test_inactive_kernel_is_removed_from_expression(data):
    X, Y = data
    mixed = MixedKernel.weighted_sum(
        [kernelWrapper("linear"), kernelWrapper("rbf")],
        weights=[0.25, 0.75],
        feature_sets=[[0], [1]],
    )
    expected = 0.25 * kernelWrapper("linear").compute(X, [0], {}, Y)
    np.testing.assert_allclose(mixed.compute(X, [0], {"gamma": 0.3}, Y), expected)


@pytest.mark.parametrize("kernel_type, parameters", [
    ("linear", {}),
    ("rbf", {"gamma": 0.4}),
    ("polynomial", {"gamma": 0.3, "degree": 3, "coef0": 0.2}),
    ("sigmoid", {"gamma": 0.2, "coef0": -0.1}),
])
def test_base_kernel_analytical_gradient(kernel_type, parameters, data):
    X, Y = data
    kernel = kernelWrapper(kernel_type)
    epsilon = 1e-6
    plus, minus = Y.copy(), Y.copy()
    plus[:, 1] += epsilon
    minus[:, 1] -= epsilon
    numerical = (kernel.compute(X, [0, 1, 2], parameters, plus) - kernel.compute(X, [0, 1, 2], parameters, minus)) / (2 * epsilon)
    analytical = kernel.compute_gradient(X, [0, 1, 2], 1, parameters, Y)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-6)


def test_tanimoto_kernel_matches_binary_jaccard_and_handles_empty_vectors():
    X = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0]], dtype=float)
    actual = kernelWrapper("tanimoto").compute(X, [0, 1, 2])
    expected = np.array(
        [[1.0, 1.0 / 3.0, 0.0], [1.0 / 3.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(actual, expected)


def test_tanimoto_gradient_matches_finite_difference():
    X = np.array([[0.2, 0.0, 1.0], [1.0, 0.5, 0.1]])
    Y = np.array([[0.4, 0.7, 0.3], [0.8, 0.2, 0.6]])
    kernel = kernelWrapper("tanimoto")
    epsilon = 1e-6
    plus, minus = Y.copy(), Y.copy()
    plus[:, 1] += epsilon
    minus[:, 1] -= epsilon
    numerical = (
        kernel.compute(X, [0, 1, 2], {}, plus)
        - kernel.compute(X, [0, 1, 2], {}, minus)
    ) / (2 * epsilon)
    analytical = kernel.compute_gradient(X, [0, 1, 2], 1, {}, Y)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-6)


def test_tanimoto_rbf_uses_tanimoto_distance_and_handles_empty_vectors():
    X = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0]], dtype=float)
    gamma = 0.7
    actual = kernelWrapper("tanimoto_rbf").compute(
        X, [0, 1, 2], {"gamma": gamma}
    )
    tanimoto = kernelWrapper("tanimoto").compute(X, [0, 1, 2])
    expected = np.exp(-gamma * (1.0 - tanimoto))
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(np.diag(actual), 1.0)


@pytest.mark.parametrize("kernel_type", ["tanimoto_rbf", "rbf_tanimoto"])
def test_tanimoto_rbf_gradient_matches_finite_difference(kernel_type):
    X = np.array([[0.2, 0.0, 1.0], [1.0, 0.5, 0.1]])
    Y = np.array([[0.4, 0.7, 0.3], [0.8, 0.2, 0.6]])
    kernel = kernelWrapper(kernel_type)
    parameters = {"gamma": 0.8}
    epsilon = 1e-6
    plus, minus = Y.copy(), Y.copy()
    plus[:, 1] += epsilon
    minus[:, 1] -= epsilon
    numerical = (
        kernel.compute(X, [0, 1, 2], parameters, plus)
        - kernel.compute(X, [0, 1, 2], parameters, minus)
    ) / (2 * epsilon)
    analytical = kernel.compute_gradient(X, [0, 1, 2], 1, parameters, Y)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-6)


def test_tanimoto_rbf_validates_gamma_and_nonnegative_features():
    kernel = kernelWrapper("tanimoto_rbf")
    with pytest.raises(ValueError, match="nonnegative features"):
        kernel.compute(np.array([[1.0, -1.0]]), [0, 1], {"gamma": 1.0})
    with pytest.raises(ValueError, match="gamma must be nonnegative"):
        kernel.compute(np.array([[1.0, 0.0]]), [0, 1], {"gamma": -1.0})
    with pytest.raises(TypeError, match="gamma must be a nonnegative number"):
        kernel.compute(np.array([[1.0, 0.0]]), [0, 1], {"gamma": "scale"})


def test_tanimoto_rbf_routes_named_gamma_in_mixed_expression():
    X = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    Y = np.array([[1.0, 1.0, 0.0]])
    expression = MixedKernel.weighted_sum(
        [
            kernelWrapper("tanimoto_rbf", name="fingerprint"),
            kernelWrapper("linear", name="linear"),
        ],
        weights=[0.75, 0.25],
    )
    parameters = {"fingerprint__gamma": 0.6}
    actual = expression.compute(X, [0, 1, 2], parameters, Y)
    expected = 0.75 * kernelWrapper("tanimoto_rbf").compute(
        X, [0, 1, 2], {"gamma": 0.6}, Y
    )
    expected += 0.25 * kernelWrapper("linear").compute(X, [0, 1, 2], {}, Y)
    np.testing.assert_allclose(actual, expected)


def test_tanimoto_rejects_negative_features():
    with pytest.raises(ValueError, match="nonnegative"):
        kernelWrapper("tanimoto").compute(np.array([[1.0, -1.0]]), [0, 1])


def test_tanimoto_is_available_in_mixed_expressions():
    X = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    Y = np.array([[1.0, 1.0, 0.0]])
    expression = (
        0.7 * kernelWrapper("tanimoto", name="fingerprint")
        + 0.3 * kernelWrapper("linear", name="linear")
    )
    actual = expression.compute(X, [0, 1, 2], {}, Y)
    expected = 0.7 * kernelWrapper("tanimoto").compute(X, [0, 1, 2], {}, Y)
    expected += 0.3 * kernelWrapper("linear").compute(X, [0, 1, 2], {}, Y)
    np.testing.assert_allclose(actual, expected)


def test_mixed_gradient_obeys_chain_and_product_rules(data):
    X, Y = data
    expression = (kernelWrapper("linear") + kernelWrapper("rbf", name="r")) ** 2
    parameters = {"r__gamma": 0.4}
    epsilon = 1e-6
    plus, minus = Y.copy(), Y.copy()
    plus[:, 2] += epsilon
    minus[:, 2] -= epsilon
    numerical = (expression.compute(X, [0, 1, 2], parameters, plus) - expression.compute(X, [0, 1, 2], parameters, minus)) / (2 * epsilon)
    analytical = expression.compute_gradient(X, [0, 1, 2], 2, parameters, Y)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-6)


def test_named_weights_are_hyperparameters(data):
    X, Y = data
    mixed = MixedKernel.weighted_sum(
        [kernelWrapper("linear"), kernelWrapper("rbf")],
        weight_parameters=["linear_weight", "rbf_weight"],
    )
    first = mixed.compute(X, [0, 1, 2], {"gamma": 0.2, "linear_weight": 1, "rbf_weight": 0}, Y)
    expected = kernelWrapper("linear").compute(X, [0, 1, 2], {}, Y)
    np.testing.assert_allclose(first, expected)


def test_draft_compatibility_constructor(data):
    X, Y = data
    mixed = MixedkernelWrapper(
        [kernelWrapper("linear"), kernelWrapper("rbf")],
        weights=[0.2, 0.8],
        mix_type="weightedsum",
    )
    expected = 0.2 * kernelWrapper("linear").compute(X, [0, 1, 2], {}, Y)
    expected += 0.8 * kernelWrapper("rbf").compute(X, [0, 1, 2], {"gamma": 0.3}, Y)
    np.testing.assert_allclose(mixed.compute(X, [0, 1, 2], {"gamma": 0.3}, Y), expected)


def test_fractional_and_negative_powers_are_rejected():
    kernel = MixedKernel(kernelWrapper("linear"))
    for invalid_power in (-1, 0, 0.5):
        with pytest.raises(ValueError):
            kernel**invalid_power


def test_mapping_routes_features_independently(data):
    X, Y = data
    mixed = kernelWrapper("linear", name="linear") + kernelWrapper("rbf", name="radial")
    mapping = {"linear": [0, 1], "radial": [1, 2]}
    actual = mixed.compute(X, mapping, {"radial__gamma": 0.3}, Y)
    expected = kernelWrapper("linear").compute(X, [0, 1], {}, Y)
    expected += kernelWrapper("rbf").compute(X, [1, 2], {"gamma": 0.3}, Y)
    np.testing.assert_allclose(actual, expected)

    gradient = mixed.compute_gradient(X, mapping, 0, {"radial__gamma": 0.3}, Y)
    expected_gradient = kernelWrapper("linear").compute_gradient(X, [0, 1], 0, {}, Y)
    np.testing.assert_allclose(gradient, expected_gradient)


def test_svmset_removes_feature_from_only_one_kernel_and_predicts():
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[:90, :6], y[:90]
    splits = cvSet(X, y)
    splits.classification(num_sets=3)
    mixed = MixedKernel.weighted_sum(
        [kernelWrapper("linear", name="linear"), kernelWrapper("rbf", name="radial")],
        weights=[0.5, 0.5],
    )
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_method=score_svc().score,
        kernel=mixed,
        kernel_feature_selection="independent",
    )
    ensemble.remove_kernel_features("radial", [0], update_kernel=False)

    assert 0 in ensemble.kernel_features["linear"]
    assert 0 not in ensemble.kernel_features["radial"]
    assert 0 in ensemble.features
    ensemble.tune_models([paramSet({"C": 1.0}, {"gamma": 0.01})])
    assert ensemble.predict(X[:4]).shape == (4,)
    gradient = ensemble.decision_gradient_(0, X[:4])
    assert gradient.shape == (4, len(ensemble.features))
    assert np.all(np.isfinite(gradient))


def test_selection_units_keep_kernel_perturbation_sets_independent():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:60, :3], y[:60])
    splits.classification(num_sets=2)
    mixed = kernelWrapper("linear", name="linear") + kernelWrapper("rbf", name="radial")
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=mixed,
        kernel_feature_selection="independent",
        perturbation_sets=[[0, 1], [2]],
    )
    radial_group = next(
        index
        for index, unit in enumerate(ensemble.selection_space_)
        if unit.kernel_id == "radial" and unit.features == (0, 1)
    )

    ensemble.remove_selection_units([radial_group], update_kernel=False)

    np.testing.assert_array_equal(ensemble.kernel_features["linear"], [0, 1, 2])
    np.testing.assert_array_equal(ensemble.kernel_features["radial"], [2])
    np.testing.assert_array_equal(ensemble.features, [0, 1, 2])


def test_svmset_kernel_candidate_features_restrict_selection_space():
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[:80, :4], y[:80]
    splits = cvSet(X, y, feature_medoids=[0, 2])
    splits.classification(num_sets=2, validation_size=0.25, random_seed=3)
    kernel = (
        kernelWrapper("linear", name="continuous")
        + kernelWrapper("rbf", name="radial")
    )
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=kernel,
        perturbation_sets=[[0, 1], [2, 3]],
        kernel_candidate_features={"continuous": [0, 1], "radial": [2, 3]},
    )

    assert [(unit.kernel_id, unit.features) for unit in ensemble.selection_space_] == [
        ("continuous", (0, 1)),
        ("radial", (2, 3)),
    ]
    np.testing.assert_array_equal(ensemble.kernel_features["continuous"], [0, 1])
    np.testing.assert_array_equal(ensemble.kernel_features["radial"], [2, 3])


def test_kernel_candidate_features_validate_names_and_indices():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:40, :3], y[:40], num_feature_medoids=0)
    splits.classification(num_sets=2)
    kernel = kernelWrapper("linear", name="linear") + kernelWrapper("rbf", name="radial")

    with pytest.raises(ValueError, match="unknown kernels"):
        svmSet(
            SVC(kernel="precomputed"), splits, score_svc().score, kernel=kernel,
            kernel_candidate_features={"missing": [0]},
        )
    with pytest.raises(ValueError, match="out-of-range"):
        svmSet(
            SVC(kernel="precomputed"), splits, score_svc().score, kernel=kernel,
            kernel_candidate_features={"linear": [3]},
        )


def test_mixed_forward_initial_units_honor_explicit_medoids():
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[:70, :3], y[:70]
    splits = cvSet(X, y, feature_medoids=[2])
    splits.classification(num_sets=2, validation_size=0.25, random_seed=2)
    kernel = kernelWrapper("linear", name="linear") + kernelWrapper("rbf", name="radial")
    ensemble = svmSet(
        SVC(kernel="precomputed"), splits, score_svc().score, kernel=kernel,
        kernel_candidate_features={"linear": [0, 1], "radial": [2]},
    )
    ensemble.greedy_forward_selection(
        [paramSet({"C": 1.0}, {"radial__gamma": 0.01})],
        max_features=1,
        num_initial_medoids=1,
        post_find_knee=False,
    )

    assert ensemble.selected_kernel_features_ == (("radial", 2),)


def test_selection_state_is_separate_for_each_fold_when_requested():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:60, :2], y[:60])
    splits.classification(num_sets=2)
    mixed = kernelWrapper("linear", name="linear") + kernelWrapper("rbf", name="radial")
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=mixed,
        kernel_feature_selection="independent",
        feature_set_policy="per_model",
    )

    ensemble.remove_selection_units([0], model_index=0, update_kernel=False)

    assert 0 not in ensemble.selection_state_[0].active
    assert 0 in ensemble.selection_state_[1].active
    assert 0 not in ensemble.kernel_features[0]["linear"]
    assert 0 in ensemble.kernel_features[1]["linear"]


def test_shared_policy_uses_per_model_states_kept_in_sync():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:60, :2], y[:60])
    splits.classification(num_sets=2)
    mixed = kernelWrapper("linear", name="linear") + kernelWrapper("rbf", name="radial")
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=mixed,
        feature_set_policy="shared",
    )

    ensemble.remove_selection_units([0], update_kernel=False)

    assert isinstance(ensemble.selection_state_, list)
    assert len(ensemble.selection_state_) == ensemble.num_models
    assert all(0 not in state.active for state in ensemble.selection_state_)
    assert all(state == ensemble.selection_state_[0] for state in ensemble.selection_state_)


def test_feature_set_policy_validates_deprecated_alias_conflicts():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:60, :2], y[:60])
    splits.classification(num_sets=2)

    with np.testing.assert_raises_regex(ValueError, "conflicts"):
        svmSet(
            SVC(kernel="precomputed"),
            splits,
            score_svc().score,
            separate_feature_sets=True,
            feature_set_policy="shared",
        )


def test_unified_feature_policy_validates_supported_values():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:60, :2], y[:60])
    splits.classification(num_sets=3)

    with pytest.raises(ValueError, match="unified_feature_policy"):
        svmSet(
            SVC(kernel="precomputed"),
            splits,
            score_svc().score,
            unified_feature_policy="all",
        )


@pytest.mark.parametrize(
    ("policy", "expected_active"),
    [("any", {0, 1, 2, 3}), ("majority", {0})],
)
def test_unified_policy_aggregates_kernel_specific_selection_units(
    policy, expected_active
):
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:60, :2], y[:60])
    splits.classification(num_sets=3)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=kernelWrapper("linear", name="linear")
        + kernelWrapper("rbf", name="radial"),
        kernel_feature_selection="independent",
        feature_set_policy="per_model",
        unified_feature_policy=policy,
    )
    for model_index, active in enumerate(({0, 1}, {0, 2}, {0, 3})):
        ensemble.set_selection_units(active, model_index=model_index, update_kernel=False)

    unified = ensemble._unified_kernel_features()

    assert set(ensemble.unified_selection_units_) == expected_active
    expected_state = ensemble.selection_state_[0].only(expected_active)
    for kernel_name in ensemble.kernel.kernel_names:
        np.testing.assert_array_equal(
            unified[kernel_name], expected_state.kernel_features[kernel_name]
        )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("any", [0, 1]), ("majority", [0])],
)
def test_unified_policy_aggregates_shared_kernel_perturbation_sets(policy, expected):
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:60, :2], y[:60])
    splits.classification(num_sets=3)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=kernelWrapper("linear"),
        kernel_feature_selection="shared",
        feature_set_policy="per_model",
        unified_feature_policy=policy,
    )
    for model_index, features in enumerate(([0, 1], [0], [0])):
        ensemble._set_features(features, model_index=model_index, update_kernel=False)

    np.testing.assert_array_equal(ensemble._unified_kernel_features(), expected)


def test_forward_kernel_selection_chooses_kernel_feature_pairs():
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[:60, :3], y[:60]
    splits = cvSet(X, y)
    splits.classification(num_sets=2)
    mixed = kernelWrapper("linear", name="linear") + kernelWrapper("rbf", name="radial")
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_method=score_svc().score,
        kernel=mixed,
        kernel_feature_selection="independent",
    )
    ensemble.greedy_forward_kernel_selection(
        [paramSet({"C": 1.0}, {"gamma": 0.001})], max_kernel_features=2
    )

    assert len(ensemble.selected_kernel_features_) == 2
    assert sum(map(len, ensemble.kernel_features.values())) == 2
    assert len(ensemble.kernel_feature_performance_) == 2
    assert ensemble.kernel_feature_performance_[0]["num_kernel_features"] == 1
    assert ensemble.predict(X[:3]).shape == (3,)


def test_forward_kernel_selection_addition_factor_batches_remaining_budget():
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[:60, :3], y[:60]
    splits = cvSet(X, y)
    splits.classification(num_sets=2)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_method=score_svc().score,
        kernel=kernelWrapper("linear", name="linear")
        + kernelWrapper("rbf", name="radial"),
        kernel_feature_selection="independent",
    )
    ensemble.greedy_forward_kernel_selection(
        [paramSet({"C": 1.0}, {"gamma": 0.001})],
        max_kernel_features=6,
        addition_factor=0.5,
        num_initial_kernel_features=3,
    )

    pair_counts = [
        row["num_kernel_features"]
        for row in ensemble.kernel_feature_performance_.values()
    ]
    assert pair_counts == [3, 4, 5, 6]
    assert len(ensemble.kernel_feature_performance_[0]["added_kernel"]) == 3


def test_forward_kernel_selection_accepts_kernel_unit_feature_ranker():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:60, :2], y[:60])
    splits.classification(num_sets=2)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_method=score_svc().score,
        kernel=kernelWrapper("linear", name="linear")
        + kernelWrapper("rbf", name="radial"),
        kernel_feature_selection="independent",
    )
    calls = []

    def ranker(model, model_index, set_for_rank):
        state = model._selection_state_for()
        candidates = [i for i in range(len(state.selection_space)) if i not in state.active]
        if not candidates:
            candidates = sorted(state.active)
        calls.append((model_index, set_for_rank, len(candidates)))
        return np.arange(len(candidates))

    ensemble.greedy_forward_kernel_selection(
        [paramSet({"C": 1.0}, {"gamma": 0.001})],
        max_kernel_features=2,
        addition_factor=0,
        feature_ranker=ranker,
    )

    assert calls
    assert all(set_name == "train" for _, set_name, _ in calls)
    assert len(ensemble.selected_selection_units_) == 2
    assert ensemble.selection_state_[0].active == frozenset(ensemble.selected_selection_units_)
    assert all(state == ensemble.selection_state_[0] for state in ensemble.selection_state_)


def test_forward_kernel_selection_validates_addition_factor():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:40, :2], y[:40])
    splits.classification(num_sets=2)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_method=score_svc().score,
        kernel=kernelWrapper("linear") + kernelWrapper("rbf"),
        kernel_feature_selection="independent",
    )
    grid = [paramSet({"C": 1.0}, {"gamma": 0.001})]
    with pytest.raises(ValueError):
        ensemble.greedy_forward_kernel_selection(grid, addition_factor=-0.1)
    with pytest.raises(TypeError):
        ensemble.greedy_forward_kernel_selection(grid, addition_factor="many")
    with pytest.raises(ValueError):
        ensemble.greedy_forward_kernel_selection(
            grid,
            max_kernel_features=2,
            num_initial_kernel_features=3,
        )
    with pytest.raises(TypeError):
        ensemble.greedy_forward_kernel_selection(
            grid,
            num_initial_kernel_features=1.5,
        )


def test_unified_forward_and_backward_selection_support_mixed_kernel_units():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:70, :2], y[:70])
    splits.classification(num_sets=2)
    grid = [paramSet({"C": 1.0}, {"gamma": 0.001})]
    mixed = kernelWrapper("linear", name="linear") + kernelWrapper("rbf", name="radial")

    forward = svmSet(
        SVC(kernel="precomputed"), splits, score_svc().score, kernel=mixed
    )
    forward.greedy_forward_selection(
        grid, max_features=2, addition_factor=0, post_find_knee=False
    )
    assert len(forward.selected_selection_units_) == 2
    assert sum(map(len, forward.kernel_features.values())) == 2

    backward = svmSet(
        SVC(kernel="precomputed"), splits, score_svc().score, kernel=mixed
    )
    backward.greedy_backward_selection(
        grid, reduction_factor=0.5, post_find_knee=False
    )
    assert backward.selected_selection_units_
    assert backward.selection_state_[0].active == frozenset(
        backward.selected_selection_units_
    )


def test_per_model_forward_and_backward_select_different_mixed_kernel_units():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:70, :2], y[:70])
    splits.classification(num_sets=2)
    grid = [paramSet({"C": 1.0}, {"gamma": 0.001})]
    mixed = kernelWrapper("linear", name="linear") + kernelWrapper("rbf", name="radial")

    def opposing_ranker(model, model_index, set_for_rank):
        candidates = model._candidate_selection_unit_indices(model_index)
        ranks = np.arange(len(candidates), dtype=float)
        return ranks if model_index == 0 else ranks[::-1]

    forward = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=mixed,
        feature_set_policy="per_model",
    )
    forward.greedy_forward_selection(
        grid,
        max_features=2,
        addition_factor=0,
        num_initial_medoids=1,
        feature_ranker=opposing_ranker,
        post_find_knee=False,
    )

    assert forward.separate_parameters is False
    assert len(forward.selected_selection_units_) == forward.num_models
    assert forward.selected_selection_units_[0] != forward.selected_selection_units_[1]
    assert tuple(forward.selection_state_[0].active) != tuple(
        forward.selection_state_[1].active
    )
    assert all(
        "selection_units_per_model" in row
        for row in forward.feature_performance_.values()
    )

    backward = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=mixed,
        feature_set_policy="per_model",
    )
    backward.greedy_backward_selection(
        grid,
        reduction_factor=0.5,
        feature_ranker=opposing_ranker,
        post_find_knee=False,
    )

    recorded_states = [
        row["selection_units_per_model"]
        for row in backward.feature_performance_.values()
    ]
    assert any(states[0] != states[1] for states in recorded_states[1:])
    assert backward.separate_parameters is False


def test_single_kernel_is_normalized_to_weight_one_mixed_kernel():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:50, :2], y[:50])
    splits.classification(num_sets=2)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=kernelWrapper("linear"),
    )

    assert isinstance(ensemble.kernel, MixedKernel)
    assert len(ensemble.kernel.base_kernels) == 1
    assert ensemble.kernel_feature_selection == "independent"


def test_mixed_performance_knee_uses_and_restores_selection_feature_count():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:50, :2], y[:50])
    splits.classification(num_sets=2)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=kernelWrapper("linear", name="linear")
        + kernelWrapper("rbf", name="radial"),
    )
    ensemble.feature_performance_ = {
        0: {"num_features": 1, "num_selection_features": 1, "score": 0.5,
            "selection_units": (0,)},
        1: {"num_features": 2, "num_selection_features": 2, "score": 0.9,
            "selection_units": (0, 1)},
        2: {"num_features": 2, "num_selection_features": 3, "score": 1.0,
            "selection_units": (0, 1, 2)},
    }

    ensemble.plot_performance()
    np.testing.assert_array_equal(plt.gca().lines[-1].get_xdata(), [1, 2, 3])
    assert ensemble.find_knee() == 2
    assert ensemble.knee_count_ == "num_selection_features"

    ensemble.tune_models = lambda parameter_grid: None
    ensemble.set_num_selection_features(2, [])
    assert ensemble.selection_state_[0].active == frozenset({0, 1})


def test_per_model_selection_state_restoration_uses_exact_fold_snapshots():
    X, y = load_breast_cancer(return_X_y=True)
    splits = cvSet(X[:50, :2], y[:50])
    splits.classification(num_sets=2)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=kernelWrapper("linear", name="linear")
        + kernelWrapper("rbf", name="radial"),
        feature_set_policy="per_model",
    )
    ensemble.feature_performance_ = {
        0: {
            "num_features": 1,
            "num_selection_features": 2,
            "score": 0.8,
            "selection_units_per_model": ((0, 1), (2, 3)),
        }
    }

    ensemble.set_num_selection_features(
        2, [paramSet({"C": 1.0}, {"gamma": 0.001})]
    )

    assert ensemble.selection_state_[0].active == frozenset({0, 1})
    assert ensemble.selection_state_[1].active == frozenset({2, 3})
    assert ensemble.selected_selection_units_ == [(0, 1), (2, 3)]
    assert ensemble.separate_parameters is False
