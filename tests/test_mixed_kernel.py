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
        separate_feature_sets=True,
    )

    ensemble.remove_selection_units([0], model_index=0, update_kernel=False)

    assert 0 not in ensemble.selection_state_[0].active
    assert 0 in ensemble.selection_state_[1].active
    assert 0 not in ensemble.kernel_features[0]["linear"]
    assert 0 in ensemble.kernel_features[1]["linear"]


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
