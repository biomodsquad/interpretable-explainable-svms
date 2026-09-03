"""Public API and end-to-end smoke tests."""

import copy
import pickle
import runpy
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, OneClassSVM

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised in Python 3.10 CI
    import tomli as tomllib

import mistic
from mistic import (
    BoundaryCounterfactualResult,
    IntegratedGradientsResult,
    cvSet,
    kernelWrapper,
    paramSet,
    score_ocsvm,
    score_svc,
    score_svr,
    svmSet,
)
from mistic.utility import combined_rank, dotdict, rank_items


def test_documentation_version_and_public_api_are_synchronized():
    """Keep package metadata, Sphinx, and the documented exports aligned."""
    repository_root = Path(__file__).resolve().parents[1]
    with (repository_root / "pyproject.toml").open("rb") as project_file:
        project_version = tomllib.load(project_file)["project"]["version"]

    sphinx_config = runpy.run_path(repository_root / "docs" / "conf.py")
    assert mistic.__version__ == project_version
    assert sphinx_config["version"] == project_version
    assert sphinx_config["release"] == project_version

    api_reference = (repository_root / "docs" / "api.rst").read_text()
    assert all(f"mistic.{name}" in api_reference for name in mistic.__all__)


def test_dotdict_missing_attributes_follow_python_lookup_protocol():
    """Allow pickle and introspection to probe absent special methods."""
    values = dotdict({"score": 1.0})
    assert values.score == 1.0
    with np.testing.assert_raises(AttributeError):
        _ = values.__setstate__


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
        "combined_rank",
        "cvSet",
        "BoundaryCounterfactualResult",
        "IntegratedGradientsResult",
        "kernelWrapper",
        "paramSet",
        "perDiff",
        "score_ocsvm",
        "score_svc",
        "score_svr",
        "svmSet",
    }


def test_cv_and_ensemble_are_pickleable():
    ensemble = _fitted_ensemble()
    restored = pickle.loads(pickle.dumps(ensemble))
    np.testing.assert_array_equal(restored.cv.X, ensemble.cv.X)
    np.testing.assert_array_equal(
        restored.predict(restored.cv.X[:5]), ensemble.predict(ensemble.cv.X[:5])
    )


def test_legacy_pickle_without_unified_model_falls_back_to_set_prediction():
    ensemble = _fitted_ensemble()
    for attribute in ("unified_model_", "unified_parameters_", "unified_prediction_features_"):
        ensemble.__dict__.pop(attribute)
    restored = pickle.loads(pickle.dumps(ensemble))

    np.testing.assert_array_equal(
        restored.predict(restored.cv.X[:5]),
        restored.predict(restored.cv.X[:5], prediction_mode="set"),
    )


def test_mean_performance_averages_separate_model_results():
    ensemble = _fitted_ensemble()
    ensemble.separate_parameters = True
    ensemble.performance_ = [
        {"score": 0.6, "auc": np.float64(0.7), "kernel": "linear"},
        {"score": 0.8, "auc": np.float64(0.9), "kernel": "linear"},
    ]

    performance = ensemble.mean_performance()

    assert performance == {"score": 0.7, "auc": 0.8, "kernel": "linear"}


def test_mean_performance_preserves_existing_aggregate():
    ensemble = _fitted_ensemble()
    expected = copy.deepcopy(ensemble.performance_)

    result = ensemble.mean_performance()

    assert result == expected
    assert result is not ensemble.performance_


def test_cv_feature_medoids_default_to_twenty_or_feature_count():
    rng = np.random.default_rng(7)
    large = cvSet(rng.normal(size=(40, 25)), np.arange(40))
    small = cvSet(rng.normal(size=(40, 6)), np.arange(40))

    assert len(large.feature_medoids_) == 20
    assert len(np.unique(large.feature_medoids_)) == 20
    np.testing.assert_array_equal(small.feature_medoids_, np.arange(6))


def test_cv_feature_medoids_are_deterministic_and_validate_count():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 24))
    first = cvSet(X, np.arange(30), num_feature_medoids=5)
    second = cvSet(X, np.arange(30), num_feature_medoids=5)

    np.testing.assert_array_equal(first.feature_medoids_, second.feature_medoids_)
    with np.testing.assert_raises(ValueError):
        cvSet(X, np.arange(30), num_feature_medoids=0)
    with np.testing.assert_raises(TypeError):
        cvSet(X, np.arange(30), num_feature_medoids=2.5)


def test_cv_ensemble_validation_set_is_excluded_from_all_splits():
    X, y = load_breast_cancer(return_X_y=True)
    X = X[:100, :6]
    y = y[:100]
    splits = cvSet(X, y, ensemble_validation_size=0.2, ensemble_validation_random_seed=13)

    assert len(splits.ensemble_validation_indices_) == 20
    assert len(splits.development_indices_) == 80
    assert not np.intersect1d(splits.ensemble_validation_indices_, splits.development_indices_).size

    for configure in (
        lambda: splits.classification(num_sets=2, random_seed=3),
        lambda: splits.k_fold(num_folds=3),
        lambda: splits.independent(num_sets=2, random_seed=3),
    ):
        configure()
        used = np.unique(np.concatenate(splits.train + splits.test))
        assert not np.intersect1d(used, splits.ensemble_validation_indices_).size
        assert set(used).issubset(set(splits.development_indices_))


def test_cv_ensemble_validation_set_validates_fraction():
    X = np.zeros((10, 2))
    y = np.arange(10)
    for invalid_size in (-0.1, 1.0):
        with np.testing.assert_raises(ValueError):
            cvSet(X, y, ensemble_validation_size=invalid_size)
    with np.testing.assert_raises(TypeError):
        cvSet(X, y, ensemble_validation_size="0.2")
    with np.testing.assert_raises(TypeError):
        cvSet(X, y, ensemble_validation_stratify="yes")


def test_cv_ensemble_validation_set_can_be_stratified_and_reused():
    X = np.arange(200).reshape(100, 2)
    y = np.repeat([0, 1], [70, 30])
    options = {
        "ensemble_validation_size": 0.2,
        "ensemble_validation_random_seed": 13,
        "ensemble_validation_stratify": True,
    }
    first = cvSet(X, y, **options)
    second = cvSet(X, y, **options)

    np.testing.assert_array_equal(
        first.ensemble_validation_indices_,
        second.ensemble_validation_indices_,
    )
    validation_y = y[first.ensemble_validation_indices_]
    np.testing.assert_array_equal(np.bincount(validation_y, minlength=2), [14, 6])


def test_tuning_prediction_and_ranking():
    ensemble = _fitted_ensemble()
    assert np.isfinite(ensemble.decision_value_cutoff_)
    calibration_indices = ensemble.cv.development_indices_
    expected_cutoff = ensemble._optimal_f1_cutoff(
        ensemble.decision_function(ensemble.cv.X[calibration_indices], prediction_mode="set"),
        ensemble.cv.y[calibration_indices],
        positive_class=ensemble.models[0].classes_[1],
    )
    assert ensemble.decision_value_cutoff_ == expected_cutoff
    predictions = ensemble.predict(ensemble.cv.X[:5])
    assert predictions.shape == (5,)
    ranks = combined_rank(number_samples=5).compute(ensemble, 0, "train")
    assert sorted(ranks.tolist()) == list(range(ensemble.cv.X.shape[1]))


def _fitted_probability_ensemble():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X[:150, :6])
    y = y[:150]
    splits = cvSet(X, y)
    splits.classification(num_sets=2, validation_size=0.25, random_seed=7)
    ensemble = svmSet(
        SVC(kernel="precomputed", probability=True, random_state=7),
        splits,
        score_svc().score,
        kernel=kernelWrapper("linear"),
    )
    ensemble.tune_models([paramSet({"C": 1.0}, {})])
    return ensemble


def test_probability_svc_scores_calibrated_curve_and_ranks_sensitivity():
    ensemble = _fitted_probability_ensemble()

    model_index = 0
    test = ensemble.cv.test[model_index]
    kernel = ensemble._get_kernel_matrix(test, ensemble.X_ind[model_index])
    probability = ensemble.models[model_index].predict_proba(kernel)[:, 1]
    expected_auc = roc_auc_score(ensemble.cv.y[test], probability)
    expected_brier = brier_score_loss(
        ensemble.cv.y[test], probability, pos_label=ensemble.models[model_index].classes_[1]
    )
    result = ensemble.score(ensemble, model_index)
    assert result["auc"] == expected_auc
    assert result["brier"] == expected_brier
    assert result["calibration"] == 1 - expected_brier
    expected_discrimination = 0.5 * result["auc"] + 0.5 * result["f1"]
    assert result["score"] == (0.8 * expected_discrimination + 0.2 * result["calibration"])

    X_probe = ensemble.cv.X[:8]
    contribution = ensemble.probability_perturbation_(model_index, X_probe)
    probability = ensemble.models[model_index].predict_proba(
        ensemble._inference_kernel(X_probe, model_index)
    )[:, 1]
    slope = -ensemble.models[model_index].probA_[0] * probability * (1 - probability)
    np.testing.assert_allclose(
        contribution, ensemble.decision_perturbation_(model_index, X_probe) * slope[:, np.newaxis]
    )


def test_probability_perturbation_requires_probability_enabled_svc():
    ensemble = _fitted_ensemble()
    with np.testing.assert_raises_regex(ValueError, "probability=True"):
        ensemble.probability_perturbation_(0, ensemble.cv.X[:2])


def test_probability_svc_calibration_weight_is_validated():
    for invalid in (-0.1, 1.1):
        with np.testing.assert_raises(ValueError):
            score_svc(calibration_weight=invalid)


def test_probability_svc_predict_proba_unified_and_set():
    ensemble = _fitted_probability_ensemble()
    X = ensemble.cv.X[:7]
    np.testing.assert_allclose(
        ensemble.predict_proba(X),
        ensemble.unified_model_.predict_proba(ensemble._unified_inference_kernel(X)),
    )
    expected_set = np.mean(
        [
            model.predict_proba(ensemble._inference_kernel(X, index))
            for index, model in enumerate(ensemble.models)
        ],
        axis=0,
    )
    np.testing.assert_allclose(ensemble.predict_proba(X, prediction_mode="set"), expected_set)
    np.testing.assert_allclose(expected_set.sum(axis=1), 1.0)


def test_probability_integrated_gradients_satisfy_completeness():
    ensemble = _fitted_probability_ensemble()
    X = ensemble.cv.X[:5]
    reference = np.zeros(X.shape[1])
    references = np.broadcast_to(reference, X.shape)
    values = ensemble.integrated_gradient(
        X, reference_point=reference, num_steps=200, output="probability"
    )
    expected = (
        ensemble.predict_proba(X, prediction_mode="set")[:, 1]
        - ensemble.predict_proba(references, prediction_mode="set")[:, 1]
    )
    # libsvm notes that calibrated probabilities may be slightly inconsistent
    # with its decision values; the chain-rule attribution remains close.
    np.testing.assert_allclose(values.sum(axis=1), expected, atol=2e-3)


def test_optimal_decision_cutoff_maximizes_f1_and_predict_uses_it():
    decision_values = np.array([-2.0, -1.0, 0.1, 0.2, 0.3])
    y_true = np.array([0, 0, 0, 1, 1])

    cutoff = svmSet._optimal_f1_cutoff(decision_values, y_true, positive_class=1)

    assert cutoff == 0.1
    ensemble = _fitted_ensemble()
    ensemble.decision_value_cutoff_ = cutoff
    ensemble.decision_function = lambda X, model_index=None, prediction_mode="unified": (
        decision_values
    )
    np.testing.assert_array_equal(
        ensemble.predict(np.zeros((len(decision_values), 6)), prediction_mode="set"),
        [0, 0, 0, 1, 1],
    )


def test_unified_classifier_is_default_and_set_prediction_is_available():
    ensemble = _fitted_ensemble()
    X = ensemble.cv.X[:8]
    kernel = ensemble._unified_inference_kernel(X)

    np.testing.assert_array_equal(ensemble.predict(X), ensemble.unified_model_.predict(kernel))
    np.testing.assert_allclose(
        ensemble.decision_function(X), ensemble.unified_model_.decision_function(kernel)
    )
    expected_set_decision = np.mean(
        [
            ensemble.models[index].decision_function(ensemble._inference_kernel(X, index))
            for index in range(ensemble.num_models)
        ],
        axis=0,
    )
    np.testing.assert_allclose(
        ensemble.decision_function(X, prediction_mode="set"), expected_set_decision
    )


def test_unified_regressor_is_default_and_set_prediction_is_available():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(60, 5))
    y = 2 * X[:, 0] - X[:, 2] + rng.normal(scale=0.1, size=len(X))
    splits = cvSet(X, y)
    splits.k_fold(num_folds=3)
    ensemble = svmSet(
        SVR(kernel="precomputed"),
        splits,
        score_svr(weight=0.0).score,
        kernel=kernelWrapper("linear"),
    )
    ensemble.tune_models([paramSet({"C": 1.0, "epsilon": 0.1}, {})])
    probe = X[:7]

    np.testing.assert_allclose(
        ensemble.predict(probe),
        ensemble.unified_model_.predict(ensemble._unified_inference_kernel(probe)),
    )
    expected_set = np.mean(
        [
            ensemble.models[index].predict(ensemble._inference_kernel(probe, index))
            for index in range(ensemble.num_models)
        ],
        axis=0,
    )
    np.testing.assert_allclose(ensemble.predict(probe, prediction_mode="set"), expected_set)


def test_integrated_gradient_averages_models_with_separate_feature_sets():
    ensemble = _fitted_ensemble()
    ensemble.separate_feature_sets = True
    ensemble.features = [np.array([0, 2]), np.array([1, 2, 4])]
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    reference = np.zeros(X.shape[1])

    def constant_gradient(model_index, x_steps):
        values = [1.0, 2.0] if model_index == 0 else [3.0, 4.0, 5.0]
        return np.tile(values, (len(x_steps), 1))

    ensemble.decision_gradient_ = constant_gradient
    ensemble.decision_function = lambda X, model_index=None: np.zeros(len(X))

    model_0 = ensemble.integrated_gradient(X, model_index=0, reference_point=reference)
    model_1 = ensemble.integrated_gradient(X, model_index=1, reference_point=reference)
    combined = ensemble.integrated_gradient(X, reference_point=reference)

    expected = np.zeros((1, 4))
    expected[:, [0, 2]] += model_0
    expected[:, [1, 2, 3]] += model_1
    expected /= ensemble.num_models

    assert combined.shape == (1, 4)  # sorted union: [0, 1, 2, 4]
    np.testing.assert_array_equal(ensemble.unified_features, [0, 1, 2, 4])
    np.testing.assert_allclose(combined, expected)


def test_integrated_gradients_satisfy_svc_decision_completeness():
    ensemble = _fitted_ensemble()
    X = ensemble.cv.X[:5]
    reference = np.zeros(X.shape[1])

    values = ensemble.integrated_gradient(X, reference_point=reference, num_steps=4)
    expected = ensemble.decision_function(X, prediction_mode="set") - ensemble.decision_function(
        np.broadcast_to(reference, X.shape), prediction_mode="set"
    )

    np.testing.assert_allclose(values.sum(axis=1), expected, atol=1e-10)


def test_integrated_gradients_satisfy_svr_prediction_completeness():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(80, 4))
    y = 1.5 * X[:, 0] - 0.75 * X[:, 2] + rng.normal(scale=0.05, size=80)
    splits = cvSet(X, y)
    splits.k_fold(num_folds=2)
    ensemble = svmSet(
        SVR(kernel="precomputed"),
        splits,
        score_svr(weight=0.0).score,
        kernel=kernelWrapper("linear"),
    )
    ensemble.tune_models([paramSet({"C": 1.0, "epsilon": 0.05}, {})])
    explained = X[:5]
    reference = np.zeros(X.shape[1])
    reference_rows = np.broadcast_to(reference, explained.shape)

    values = ensemble.integrated_gradient(explained, reference_point=reference, num_steps=4)
    expected = ensemble.predict(explained, prediction_mode="set") - ensemble.predict(
        reference_rows, prediction_mode="set"
    )

    np.testing.assert_allclose(values.sum(axis=1), expected, atol=1e-10)


def test_integrated_gradients_result_metadata_and_plots():
    import matplotlib.pyplot as plt

    ensemble = _fitted_ensemble()
    X = ensemble.cv.X[:12]
    result = ensemble.explain_integrated_gradients(
        X,
        feature_names=[f"measurement_{i}" for i in range(X.shape[1])],
        reference_point=np.zeros(X.shape[1]),
        target=ensemble.cv.y[:12],
        num_steps=4,
    )

    assert isinstance(result, IntegratedGradientsResult)
    assert result.to_frame().columns.tolist() == list(result.feature_names)
    assert result.inputs.shape == result.values.shape
    assert result.interaction_scores().shape == (X.shape[1], X.shape[1])

    axes = [
        result.summary_plot(),
        result.heatmap(cluster=True),
        result.interaction_plot(),
        result.interaction_plot("measurement_0", "measurement_1"),
    ]
    assert all(axis.figure is not None for axis in axes)
    plt.close("all")


def test_boundary_counterfactuals_are_exposed_and_reused_by_integrated_gradients():
    import matplotlib.pyplot as plt

    ensemble = _fitted_ensemble()
    X = ensemble.cv.X[:4]
    calls = []

    def fixed_boundary(model_index, values, return_diagnostics=False):
        calls.append(model_index)
        points = np.asarray(values) * 0.5
        success = np.ones(len(values), dtype=bool)
        return (points, success) if return_diagnostics else points

    ensemble._find_boundary_points = fixed_boundary
    result = ensemble.explain_integrated_gradients(
        X, feature_names=[f"measurement_{i}" for i in range(X.shape[1])],
        target=ensemble.cv.y[:4], num_steps=4)

    assert isinstance(result.counterfactuals, BoundaryCounterfactualResult)
    assert calls == list(range(ensemble.num_models))
    assert result.reference_points.shape == (ensemble.num_models, *X.shape)
    np.testing.assert_allclose(
        result.counterfactuals.values,
        np.broadcast_to(X[np.newaxis] * 0.5, result.counterfactuals.values.shape),
    )
    np.testing.assert_allclose(
        result.counterfactuals.distances,
        np.linalg.norm(X * 0.5, axis=1)[np.newaxis].repeat(ensemble.num_models, axis=0),
    )
    assert result.counterfactuals.to_frame(model_index=0).shape == X.shape
    assert result.counterfactuals.summary_plot().figure is not None
    assert result.counterfactuals.sample_plot(0, model_index=0).figure is not None
    plt.close("all")


def test_boundary_counterfactuals_reject_regression_models():
    ensemble = _fitted_ensemble()
    ensemble.SVM = SVR(kernel="precomputed")
    with np.testing.assert_raises(TypeError):
        ensemble.explain_counterfactuals(ensemble.cv.X[:2])


def test_integrated_gradients_heatmap_supports_continuous_target_and_existing_axis():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(4)
    result = IntegratedGradientsResult(
        values=rng.normal(size=(20, 3)),
        inputs=rng.normal(size=(20, 3)),
        feature_indices=np.arange(3),
        feature_names=("a", "b", "c"),
        reference_points=None,
        model_indices=(0,),
        num_steps=20,
        target=np.linspace(-2, 2, 20),
    )
    figure, axis = plt.subplots()
    returned = result.heatmap(ax=axis)
    assert returned is axis
    target_strip = next(
        extra
        for extra in figure.axes
        if extra is not axis
        and extra.images
        and extra.images[0].get_array().shape == (len(result.values), 1)
    )
    assert target_strip.images[0].get_array().shape == (len(result.values), 1)
    assert any(extra.get_ylabel() == "Target" for extra in figure.axes)
    plt.close(figure)


def test_integrated_gradients_heatmap_sorts_targets_and_feature_contributions():
    import matplotlib.pyplot as plt

    values = np.array([[1.0, 9.0, 2.0], [2.0, 8.0, 1.0], [3.0, 7.0, 0.0]])
    target = np.array([20.0, 10.0, 30.0])
    result = IntegratedGradientsResult(
        values=values,
        inputs=values,
        feature_indices=np.arange(3),
        feature_names=("low", "high", "lowest"),
        reference_points=None,
        model_indices=(0,),
        num_steps=20,
        target=target,
    )

    axis = result.heatmap(cluster=False)
    # Rows follow target order [10, 20, 30]; columns follow mean |IG| [high, low, lowest].
    expected = values[np.ix_([1, 0, 2], [1, 0, 2])]
    np.testing.assert_array_equal(axis.images[0].get_array(), expected)
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["high", "low", "lowest"]
    plt.close(axis.figure)


def test_integrated_gradients_clustered_heatmap_displays_row_dendrogram():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(8)
    result = IntegratedGradientsResult(
        values=rng.normal(size=(12, 4)),
        inputs=rng.normal(size=(12, 4)),
        feature_indices=np.arange(4),
        feature_names=("a", "b", "c", "d"),
        reference_points=None,
        model_indices=(0,),
        num_steps=20,
        target=np.arange(12),
    )

    axis = result.heatmap(cluster=True)
    dendrogram_axes = [
        extra
        for extra in axis.figure.axes
        if extra is not axis and extra.collections and not extra.images and not extra.axison
    ]
    assert len(dendrogram_axes) == 1
    assert dendrogram_axes[0].axison is False
    plt.close(axis.figure)


def test_unified_feature_rank_averages_model_ranks():
    ensemble = _fitted_ensemble()
    ensemble.separate_feature_sets = True
    ensemble.features = [np.array([0, 2]), np.array([1, 2, 4])]
    ensemble.feature_rank = [
        np.array([0, 4, 1, 5, 2, 3]),
        np.array([5, 0, 2, 4, 1, 3]),
    ]

    ensemble._update_unified_feature_attributes()

    np.testing.assert_array_equal(ensemble.unified_features, [0, 1, 2, 4])
    np.testing.assert_allclose(ensemble.unified_feature_rank, [2.5, 2.0, 1.5, 4.5, 1.5, 3.0])
    np.testing.assert_array_equal(ensemble.unified_sorted_features, [2, 4, 1, 0, 5, 3])


def test_inference_kernel_computes_support_vectors_only():
    ensemble = _fitted_ensemble()
    X = ensemble.cv.X[:5]
    expected_decisions = 0
    for model_index, model in enumerate(ensemble.models):
        full_kernel = ensemble.kernel.compute(
            X,
            feature_index=ensemble.features,
            parameters=ensemble.parameters_.kernel,
            Y=ensemble.cv.X[ensemble.X_ind[model_index], :],
        )
        expected_decisions += model.decision_function(full_kernel)
    expected_decisions /= ensemble.num_models

    computed_against = []
    original_compute = ensemble.kernel.compute

    def track_compute(*args, **kwargs):
        computed_against.append(len(kwargs["Y"]))
        return original_compute(*args, **kwargs)

    ensemble.kernel.compute = track_compute
    actual_decisions = ensemble.decision_function(X, prediction_mode="set")

    np.testing.assert_allclose(actual_decisions, expected_decisions)
    assert computed_against == [len(model.support_) for model in ensemble.models]


def test_support_only_kernel_preserves_model_prediction():
    ensemble = _fitted_ensemble()
    X = ensemble.cv.X[:5]

    for model_index, model in enumerate(ensemble.models):
        full_kernel = ensemble.kernel.compute(
            X,
            feature_index=ensemble.features,
            parameters=ensemble.parameters_.kernel,
            Y=ensemble.cv.X[ensemble.X_ind[model_index], :],
        )
        support_kernel = ensemble._inference_kernel(X, model_index)

        assert support_kernel.shape == full_kernel.shape
        np.testing.assert_array_equal(
            support_kernel[:, model.support_], full_kernel[:, model.support_]
        )
        np.testing.assert_array_equal(model.predict(support_kernel), model.predict(full_kernel))


def test_rank_items_orders_values():
    np.testing.assert_array_equal(rank_items(np.array([10, 5, 20])), [1, 0, 2])
    np.testing.assert_array_equal(rank_items(np.array([10, 5, 20]), descending=True), [1, 2, 0])


def test_greedy_forward_selection_evaluates_singletons_and_orders_features():
    ensemble = _fitted_ensemble()
    parameter_grid = [paramSet({"C": 1.0}, {})]

    ensemble.greedy_forward_selection(parameter_grid, addition_factor=0.5, post_find_knee=False)

    assert len(ensemble.singleton_performance_) == ensemble.cv.X.shape[1]
    assert sorted(ensemble.sorted_features.tolist()) == list(range(ensemble.cv.X.shape[1]))
    assert len(ensemble.features) >= 1
    scores = [row["score"] for row in ensemble.feature_performance_.values()]
    expected_size = ensemble.feature_performance_[int(np.argmax(scores))]["num_features"]
    assert len(ensemble.features) == expected_size


def test_greedy_forward_selection_caps_search_and_ties_unselected_features():
    ensemble = _fitted_ensemble()
    parameter_grid = [paramSet({"C": 1.0}, {})]

    ensemble.greedy_forward_selection(
        parameter_grid, addition_factor=0.5, max_features=3, post_find_knee=False
    )

    feature_counts = [row["num_features"] for row in ensemble.feature_performance_.values()]
    assert feature_counts[-1] == ensemble.cv.X.shape[1]
    assert max(feature_counts[:-1]) <= 3
    assert len(ensemble.features) <= 3
    last_rank = np.max(ensemble.feature_rank)
    assert np.count_nonzero(ensemble.feature_rank == last_rank) == 3


def test_forward_addition_factor_controls_batch_size():
    parameter_grid = [paramSet({"C": 1.0}, {})]
    one_at_a_time = _fitted_ensemble()
    batched = _fitted_ensemble()

    one_at_a_time.greedy_forward_selection(
        parameter_grid,
        addition_factor=0,
        max_features=4,
        post_find_knee=False,
    )
    batched.greedy_forward_selection(
        parameter_grid,
        addition_factor=0.5,
        max_features=4,
        post_find_knee=False,
    )

    single_counts = [row["num_features"] for row in one_at_a_time.feature_performance_.values()][
        :-1
    ]
    batch_counts = [row["num_features"] for row in batched.feature_performance_.values()][:-1]
    np.testing.assert_array_equal(single_counts, [1, 2, 3, 4])
    assert batch_counts[1] - batch_counts[0] > 1


def test_forward_addition_factor_validates_input():
    ensemble = _fitted_ensemble()
    parameter_grid = [paramSet({"C": 1.0}, {})]

    with np.testing.assert_raises(ValueError):
        ensemble.greedy_forward_selection(parameter_grid, addition_factor=-0.1)
    with np.testing.assert_raises(TypeError):
        ensemble.greedy_forward_selection(parameter_grid, addition_factor="many")


def test_forward_singleton_round_uses_only_feature_medoids():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X[:120, :25])
    y = y[:120]
    splits = cvSet(X, y, num_feature_medoids=4)
    splits.classification(num_sets=1, validation_size=0.25, random_seed=7)
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        splits,
        score_svc().score,
        kernel=kernelWrapper("linear"),
    )

    ensemble.greedy_forward_selection(
        [paramSet({"C": 1.0}, {})], max_features=1, post_find_knee=False
    )

    assert len(ensemble.singleton_performance_) == 4
    np.testing.assert_array_equal(
        np.sort(np.concatenate(ensemble.singleton_candidates_)), splits.feature_medoids_
    )
    assert ensemble.feature_performance_[1]["num_features"] == 25


def test_greedy_forward_selection_validates_max_features():
    ensemble = _fitted_ensemble()
    parameter_grid = [paramSet({"C": 1.0}, {})]

    for invalid_count in (0, ensemble.cv.X.shape[1] + 1):
        with np.testing.assert_raises(ValueError):
            ensemble.greedy_forward_selection(parameter_grid, max_features=invalid_count)
    with np.testing.assert_raises(TypeError):
        ensemble.greedy_forward_selection(parameter_grid, max_features=2.5)


def test_greedy_forward_selection_stops_when_no_group_fits_cap():
    fitted = _fitted_ensemble()
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        fitted.cv,
        score_svc().score,
        kernel=kernelWrapper("linear"),
        perturbation_sets=[[0, 1], [2, 3], [4, 5]],
    )

    ensemble.greedy_forward_selection(
        [paramSet({"C": 1.0}, {})], max_features=3, post_find_knee=False
    )

    assert len(ensemble.features) == 2
    assert [row["num_features"] for row in ensemble.feature_performance_.values()] == [2, 6]
    assert np.count_nonzero(ensemble.feature_rank == np.max(ensemble.feature_rank)) == 4


def test_backward_selection_retunes_selected_subset_after_scaled_search():
    ensemble = _fitted_ensemble()
    parameter_grid = [
        paramSet({"C": 0.5}, {}),
        paramSet({"C": 1.0}, {}),
    ]
    tuned_feature_counts = []
    original_tune_models = ensemble.tune_models

    def track_tuning(grid):
        tuned_feature_counts.append(len(ensemble.features))
        return original_tune_models(grid)

    ensemble.tune_models = track_tuning
    ensemble.greedy_backward_selection(
        parameter_grid,
        reduction_factor=0.5,
        tune_models_each_step=False,
        post_find_knee=False,
    )

    assert len(tuned_feature_counts) == 2
    assert tuned_feature_counts[0] == ensemble.cv.X.shape[1]
    assert tuned_feature_counts[-1] == len(ensemble.features)


def test_forward_selection_retunes_selected_subset_after_scaled_search():
    ensemble = _fitted_ensemble()
    parameter_grid = [
        paramSet({"C": 0.5}, {}),
        paramSet({"C": 1.0}, {}),
    ]
    tuned_feature_counts = []
    original_tune_models = ensemble.tune_models

    def track_tuning(grid):
        tuned_feature_counts.append(len(ensemble.features))
        return original_tune_models(grid)

    ensemble.tune_models = track_tuning
    ensemble.greedy_forward_selection(
        parameter_grid,
        addition_factor=0.5,
        tune_models_each_step=False,
        max_features=3,
        post_find_knee=False,
    )

    # One exhaustive singleton tuning per feature, followed by one final
    # retuning pass on the selected best subset.
    assert len(tuned_feature_counts) == ensemble.cv.X.shape[1] + 1
    assert tuned_feature_counts[:-1] == ensemble.cv.X.shape[1] * [1]
    assert tuned_feature_counts[-1] == len(ensemble.features)


def test_greedy_searches_find_knee_before_final_retuning_by_default():
    parameter_grid = [paramSet({"C": 1.0}, {})]

    for direction in ("backward", "forward"):
        ensemble = _fitted_ensemble()
        events = []
        original_tune_models = ensemble.tune_models

        def find_knee(metric="score", _events=events):
            _events.append("find_knee")
            return 2

        def track_tuning(grid, _events=events, _tune=original_tune_models):
            _events.append("tune_models")
            return _tune(grid)

        ensemble.find_knee = find_knee
        ensemble.tune_models = track_tuning
        search = getattr(ensemble, f"greedy_{direction}_selection")
        factor = {"reduction_factor": 0.5} if direction == "backward" else {"addition_factor": 0.5}
        search(
            parameter_grid,
            tune_models_each_step=False,
            **factor,
        )

        assert events[-2:] == ["find_knee", "tune_models"]
        assert len(ensemble.features) == 2


def test_set_num_features_uses_ranking_and_retunes():
    ensemble = _fitted_ensemble()
    parameter_grid = [
        paramSet({"C": 0.1}, {}),
        paramSet({"C": 1.0}, {}),
    ]
    ensemble.sorted_features = np.array([4, 2, 5, 0, 1, 3])

    ensemble.set_num_features(3, parameter_grid)

    np.testing.assert_array_equal(ensemble.features, [2, 4, 5])
    assert ensemble.parameters_.model["C"] in {0.1, 1.0}
    assert ensemble.performance_.score is not None
    assert ensemble.kernel_matrix_.shape == (ensemble.num_samples, ensemble.num_samples)


def test_set_num_features_requires_selection_and_valid_count():
    ensemble = _fitted_ensemble()
    parameter_grid = [paramSet({"C": 1.0}, {})]

    with np.testing.assert_raises_regex(RuntimeError, "feature selection"):
        ensemble.set_num_features(2, parameter_grid)

    ensemble.sorted_features = np.arange(ensemble.cv.X.shape[1])
    for invalid_count in (0, ensemble.cv.X.shape[1] + 1):
        with np.testing.assert_raises(ValueError):
            ensemble.set_num_features(invalid_count, parameter_grid)
    with np.testing.assert_raises(TypeError):
        ensemble.set_num_features(2.5, parameter_grid)


def test_stochastic_selection_proposes_ranked_add_and_remove_moves():
    parameter_grid = [paramSet({"C": 1.0}, {})]

    addition = _fitted_ensemble()
    addition._set_features([0, 1], update_kernel=False)
    addition.tune_models(parameter_grid)
    addition.stochastic_feature_selection(
        parameter_grid, n_iterations=1, add_probability=1.0, random_seed=3
    )

    assert addition.stochastic_performance_[0].operation == "add"
    assert addition.stochastic_performance_[0].model_index is None
    assert len(addition.stochastic_performance_[0].features_changed) == 1
    assert addition.stochastic_best_score_ >= 0

    removal = _fitted_ensemble()
    removal.stochastic_feature_selection(
        parameter_grid, n_iterations=1, add_probability=0.0, random_seed=3
    )

    assert removal.stochastic_performance_[0].operation == "remove"
    assert len(removal.stochastic_performance_[0].features_changed) == 1


def test_stochastic_selection_preserves_separate_feature_set_structure():
    fitted = _fitted_ensemble()
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        fitted.cv,
        score_svc().score,
        kernel=kernelWrapper("linear"),
        separate_feature_sets=True,
    )
    parameter_grid = [paramSet({"C": 1.0}, {})]
    ensemble._set_features([0, 1], model_index=0, update_kernel=False)
    ensemble._set_features([0, 1], model_index=1, update_kernel=False)
    ensemble.tune_models(parameter_grid)

    ensemble.stochastic_feature_selection(
        parameter_grid, n_iterations=1, add_probability=1.0, random_seed=5
    )

    row = ensemble.stochastic_performance_[0]
    assert row.operation == "add"
    assert row.model_index in (0, 1)
    assert isinstance(ensemble.features, list)
    assert len(ensemble.features) == ensemble.num_models
    assert all(isinstance(features, np.ndarray) for features in ensemble.features)
    expected_union = np.unique(np.concatenate(ensemble.features))
    np.testing.assert_array_equal(ensemble.unified_features, expected_union)


def test_stochastic_selection_can_update_every_model_as_one_proposal():
    fitted = _fitted_ensemble()
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        fitted.cv,
        score_svc().score,
        kernel=kernelWrapper("linear"),
        separate_feature_sets=True,
    )
    parameter_grid = [paramSet({"C": 1.0}, {})]
    for model_index in range(ensemble.num_models):
        ensemble._set_features([0, 1], model_index=model_index, update_kernel=False)
    ensemble.tune_models(parameter_grid)
    tune_calls = []
    original_tune_models = ensemble.tune_models

    def track_tuning(grid):
        tune_calls.append(copy.deepcopy(ensemble.features))
        return original_tune_models(grid)

    ensemble.tune_models = track_tuning
    ensemble.stochastic_feature_selection(
        parameter_grid, n_iterations=1, add_probability=1.0, update_all_models=True, random_seed=11
    )

    row = ensemble.stochastic_performance_[0]
    assert len(row.moves) == ensemble.num_models
    assert [move.model_index for move in row.moves] == list(range(ensemble.num_models))
    assert all(move.operation == "add" for move in row.moves)
    assert all(len(features) == 3 for features in tune_calls[0])
    assert len(tune_calls) == 1


def test_stochastic_selection_can_optimize_ensemble_validation_score():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X[:150, :6])
    y = y[:150]
    splits = cvSet(X, y, ensemble_validation_size=0.2, ensemble_validation_random_seed=9)
    splits.classification(num_sets=2, validation_size=0.25, random_seed=7)
    ensemble = svmSet(
        SVC(kernel="precomputed"), splits, score_svc().score, kernel=kernelWrapper("linear")
    )
    parameter_grid = [paramSet({"C": 1.0}, {})]
    ensemble.tune_models(parameter_grid)

    ensemble.stochastic_feature_selection(
        parameter_grid,
        n_iterations=1,
        add_probability=0.0,
        use_ensemble_validation=True,
        random_seed=4,
    )

    row = ensemble.stochastic_performance_[0]
    assert row.objective == "ensemble_validation"
    assert np.isfinite(row.score)
    with np.testing.assert_raises_regex(ValueError, "non-empty"):
        _fitted_ensemble().stochastic_feature_selection(
            parameter_grid, n_iterations=1, use_ensemble_validation=True
        )


def test_stochastic_selection_can_propose_multiple_changes_per_model():
    parameter_grid = [paramSet({"C": 1.0}, {})]
    shared = _fitted_ensemble()
    shared._set_features([0], update_kernel=False)
    shared.tune_models(parameter_grid)
    shared.stochastic_feature_selection(
        parameter_grid,
        n_iterations=1,
        add_probability=1.0,
        expected_changes_per_model=100,
        random_seed=2,
    )

    shared_moves = shared.stochastic_performance_[0].moves
    assert len(shared_moves) == 5
    assert shared.stochastic_performance_[0].num_changes == 5
    assert shared.stochastic_performance_[0].num_changes_by_model == [5]
    assert all(move.operation == "add" for move in shared_moves)

    fitted = _fitted_ensemble()
    separate = svmSet(
        SVC(kernel="precomputed"),
        fitted.cv,
        score_svc().score,
        kernel=kernelWrapper("linear"),
        separate_feature_sets=True,
    )
    for model_index in range(separate.num_models):
        separate._set_features([0], model_index=model_index, update_kernel=False)
    separate.tune_models(parameter_grid)
    separate.stochastic_feature_selection(
        parameter_grid,
        n_iterations=1,
        add_probability=1.0,
        update_all_models=True,
        expected_changes_per_model=100,
        random_seed=2,
    )

    moves = separate.stochastic_performance_[0].moves
    assert len(moves) == 5 * separate.num_models
    assert separate.stochastic_performance_[0].num_changes_by_model == [5, 5]
    assert {move.model_index for move in moves} == set(range(separate.num_models))


def test_stochastic_selection_validates_expected_changes():
    ensemble = _fitted_ensemble()
    parameter_grid = [paramSet({"C": 1.0}, {})]
    for invalid_count in (0, 0.5, np.inf):
        with np.testing.assert_raises(ValueError):
            ensemble.stochastic_feature_selection(
                parameter_grid, n_iterations=1, expected_changes_per_model=invalid_count
            )
    with np.testing.assert_raises(TypeError):
        ensemble.stochastic_feature_selection(
            parameter_grid, n_iterations=1, expected_changes_per_model="2"
        )


def test_stochastic_swap_mode_preserves_every_model_feature_count():
    fitted = _fitted_ensemble()
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        fitted.cv,
        score_svc().score,
        kernel=kernelWrapper("linear"),
        separate_feature_sets=True,
    )
    parameter_grid = [paramSet({"C": 1.0}, {})]
    for model_index in range(ensemble.num_models):
        ensemble._set_features([0, 1, 2], model_index=model_index, update_kernel=False)
    ensemble.tune_models(parameter_grid)
    proposed_feature_sets = []
    original_tune_models = ensemble.tune_models

    def track_tuning(grid):
        proposed_feature_sets.append(copy.deepcopy(ensemble.features))
        return original_tune_models(grid)

    ensemble.tune_models = track_tuning
    ensemble.stochastic_feature_selection(
        parameter_grid, n_iterations=1, preserve_feature_count=True, random_seed=8
    )

    row = ensemble.stochastic_performance_[0]
    assert row.num_changes == 2 * ensemble.num_models
    assert row.num_changes_by_model == [2] * ensemble.num_models
    for model_index in range(ensemble.num_models):
        model_moves = [move for move in row.moves if move.model_index == model_index]
        assert [move.operation for move in model_moves] == ["remove", "add"]
    assert all(len(features) == 3 for features in proposed_feature_sets[0])
    assert all(len(features) == 3 for features in ensemble.features)


def test_stochastic_swap_mode_requires_single_expected_change():
    with np.testing.assert_raises_regex(ValueError, "must equal 1"):
        _fitted_ensemble().stochastic_feature_selection(
            [paramSet({"C": 1.0}, {})],
            n_iterations=1,
            preserve_feature_count=True,
            expected_changes_per_model=2,
        )


def test_stochastic_selection_stops_when_convergence_patience_is_reached():
    ensemble = _fitted_ensemble()
    parameter_grid = [paramSet({"C": 1.0}, {})]
    ensemble._set_features([0, 1, 2], update_kernel=False)
    ensemble.tune_models(parameter_grid)
    ensemble.stochastic_feature_selection(
        parameter_grid,
        n_iterations=10,
        preserve_feature_count=True,
        convergence_patience=2,
        convergence_min_delta=1.0,
        random_seed=6,
    )

    assert ensemble.stochastic_converged_ is True
    assert ensemble.stochastic_iterations_ == 2
    assert ensemble.stochastic_stop_reason_ == "converged"
    assert len(ensemble.stochastic_performance_) == 2
    assert ensemble.stochastic_performance_[-1].iterations_without_improvement == 2


def test_stochastic_selection_can_disable_and_validates_convergence():
    ensemble = _fitted_ensemble()
    parameter_grid = [paramSet({"C": 1.0}, {})]
    ensemble._set_features([0, 1, 2], update_kernel=False)
    ensemble.tune_models(parameter_grid)
    ensemble.stochastic_feature_selection(
        parameter_grid,
        n_iterations=3,
        preserve_feature_count=True,
        convergence_patience=None,
        convergence_min_delta=1.0,
        random_seed=6,
    )

    assert ensemble.stochastic_converged_ is False
    assert ensemble.stochastic_iterations_ == 3
    assert ensemble.stochastic_stop_reason_ == "max_iterations"

    for invalid_patience in (0, 1.5):
        error = ValueError if invalid_patience == 0 else TypeError
        with np.testing.assert_raises(error):
            ensemble.stochastic_feature_selection(
                parameter_grid, n_iterations=1, convergence_patience=invalid_patience
            )
    for invalid_delta in (-0.1, np.inf):
        with np.testing.assert_raises(ValueError):
            ensemble.stochastic_feature_selection(
                parameter_grid, n_iterations=1, convergence_min_delta=invalid_delta
            )


def test_ensemble_stochastic_selection_uses_one_global_candidate_pool():
    fitted = _fitted_ensemble()
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        fitted.cv,
        score_svc().score,
        kernel=kernelWrapper("linear"),
        separate_feature_sets=True,
    )
    parameter_grid = [paramSet({"C": 1.0}, {})]
    for model_index in range(ensemble.num_models):
        ensemble._set_features([0, 1], model_index=model_index, update_kernel=False)
    ensemble.tune_models(parameter_grid)

    ensemble.ensemble_stochastic_feature_selection(
        parameter_grid, n_iterations=1, add_probability=1.0, random_seed=4
    )

    row = ensemble.ensemble_stochastic_performance_[0]
    assert row.pool_size == ensemble.num_models * 6
    assert row.operation == "add"
    assert row.model_index in range(ensemble.num_models)
    assert np.isfinite(row.estimated_score)
    assert np.isfinite(row.score)
    assert ensemble.ensemble_stochastic_iterations_ == 1


def test_ensemble_stochastic_selection_requires_separate_feature_sets():
    with np.testing.assert_raises_regex(ValueError, "separate_feature_sets"):
        _fitted_ensemble().ensemble_stochastic_feature_selection(
            [paramSet({"C": 1.0}, {})], n_iterations=1
        )


def test_ensemble_stochastic_selection_scores_feature_and_prediction_diversity():
    fitted = _fitted_ensemble()
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        fitted.cv,
        score_svc().score,
        kernel=kernelWrapper("linear"),
        separate_feature_sets=True,
    )
    parameter_grid = [paramSet({"C": 1.0}, {})]
    for model_index in range(ensemble.num_models):
        ensemble._set_features([0, 1, 2], model_index=model_index, update_kernel=False)
    ensemble.tune_models(parameter_grid)

    ensemble.ensemble_stochastic_feature_selection(
        parameter_grid,
        n_iterations=1,
        preserve_feature_count=True,
        feature_diversity_weight=0.2,
        prediction_diversity_weight=0.1,
        max_feature_similarity=0.8,
        random_seed=12,
    )

    row = ensemble.ensemble_stochastic_performance_[0]
    assert np.isclose(
        row.estimated_objective,
        row.estimated_score
        + 0.2 * row.estimated_feature_diversity
        + 0.1 * row.estimated_prediction_diversity,
    )
    assert np.isclose(
        row.objective, row.score + 0.2 * row.feature_diversity + 0.1 * row.prediction_diversity
    )
    assert row.estimated_feature_diversity > 0
    assert row.max_feature_similarity <= 0.8


def test_ensemble_stochastic_selection_validates_diversity_settings():
    fitted = _fitted_ensemble()
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        fitted.cv,
        score_svc().score,
        kernel=kernelWrapper("linear"),
        separate_feature_sets=True,
    )
    ensemble.tune_models([paramSet({"C": 1.0}, {})])
    parameter_grid = [paramSet({"C": 1.0}, {})]

    for option in (
        "feature_diversity_weight",
        "prediction_diversity_weight",
        "performance_tolerance",
    ):
        with np.testing.assert_raises(ValueError):
            ensemble.ensemble_stochastic_feature_selection(
                parameter_grid, n_iterations=1, **{option: -0.1}
            )
    with np.testing.assert_raises(ValueError):
        ensemble.ensemble_stochastic_feature_selection(
            parameter_grid, n_iterations=1, max_feature_similarity=1.1
        )


def test_ensemble_stochastic_swap_mode_preserves_model_feature_counts():
    fitted = _fitted_ensemble()
    ensemble = svmSet(
        SVC(kernel="precomputed"),
        fitted.cv,
        score_svc().score,
        kernel=kernelWrapper("linear"),
        separate_feature_sets=True,
    )
    parameter_grid = [paramSet({"C": 1.0}, {})]
    for model_index in range(ensemble.num_models):
        ensemble._set_features([0, 1], model_index=model_index, update_kernel=False)
    ensemble.tune_models(parameter_grid)
    initial_counts = [len(features) for features in ensemble.features]

    ensemble.ensemble_stochastic_feature_selection(
        parameter_grid,
        n_iterations=3,
        preserve_feature_count=True,
        random_seed=4,
        convergence_patience=None,
    )

    assert [len(features) for features in ensemble.features] == initial_counts
    assert all(row.operation == "swap" for row in ensemble.ensemble_stochastic_performance_)
    for row in ensemble.ensemble_stochastic_performance_:
        assert len(row.features_removed) == len(row.features_added)


def test_ensemble_stochastic_swap_mode_validates_boolean():
    with np.testing.assert_raises(TypeError):
        _fitted_ensemble().ensemble_stochastic_feature_selection(
            [paramSet({"C": 1.0}, {})], n_iterations=1, preserve_feature_count=1
        )


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


def test_find_knee_is_independent_of_selection_direction():
    ensemble = _fitted_ensemble()
    points = [
        (1, 0.00),
        (2, 0.70),
        (3, 0.90),
        (4, 0.97),
        (5, 1.00),
    ]

    for ordered_points in (points, reversed(points)):
        ensemble.feature_performance_ = {
            index: {"num_features": count, "score": score}
            for index, (count, score) in enumerate(ordered_points)
        }
        assert ensemble.find_knee() == 2
        assert ensemble.knee_num_features_ == 2


def test_find_knee_requires_a_nonflat_curve_with_three_points():
    ensemble = _fitted_ensemble()
    with np.testing.assert_raises_regex(RuntimeError, "feature selection"):
        ensemble.find_knee()

    ensemble.feature_performance_ = {
        0: {"num_features": 1, "score": 0.5},
        1: {"num_features": 2, "score": 0.5},
        2: {"num_features": 3, "score": 0.5},
    }
    with np.testing.assert_raises_regex(ValueError, "flat curve"):
        ensemble.find_knee()


def _fitted_one_class_ensemble():
    rng = np.random.default_rng(42)
    inliers = rng.normal(0, 0.45, size=(48, 4))
    outliers = rng.normal(4, 0.35, size=(12, 4))
    X = np.vstack((inliers, outliers))
    y = np.concatenate((np.ones(len(inliers)), -np.ones(len(outliers))))
    splits = cvSet(X, y, num_feature_medoids=4)
    splits.one_class(num_sets=2, validation_size=0.25, random_seed=3)
    ensemble = svmSet(
        OneClassSVM(kernel="precomputed"), splits, score_ocsvm().score, kernel=kernelWrapper("rbf")
    )
    ensemble.tune_models([paramSet({"nu": 0.1}, {"gamma": 0.5})])
    return ensemble


def test_one_class_splits_never_train_on_outliers():
    ensemble = _fitted_one_class_ensemble()
    for train, test in zip(ensemble.cv.train, ensemble.cv.test):
        assert np.all(ensemble.cv.y[train] == 1)
        assert np.any(ensemble.cv.y[test] == -1)


def test_one_class_prediction_and_decision_match_sklearn_conventions():
    ensemble = _fitted_one_class_ensemble()
    X = ensemble.cv.X[:10]
    decision = ensemble.decision_function(X)
    prediction = ensemble.predict(X)
    np.testing.assert_array_equal(prediction, np.where(decision >= 0, 1, -1))

    set_decision = ensemble.decision_function(X, prediction_mode="set")
    set_prediction = ensemble.predict(X, prediction_mode="set")
    np.testing.assert_array_equal(set_prediction, np.where(set_decision >= 0, 1, -1))


def test_one_class_decision_perturbation_is_exact_frozen_model_change():
    ensemble = _fitted_one_class_ensemble()
    model_index = 0
    X = ensemble.cv.X[:7]
    actual = ensemble.decision_perturbation_(model_index, X)[:, 0]
    model = ensemble.models[model_index]
    support = ensemble._get_support_vectors(model_index)
    features = ensemble.features
    parameters = ensemble.parameters_.kernel
    full = ensemble.kernel.compute(support, feature_index=features, parameters=parameters, Y=X)
    reduced = ensemble.kernel.compute(
        support, feature_index=features[features != 0], parameters=parameters, Y=X
    )
    expected = model.dual_coef_[0] @ (full - reduced)
    np.testing.assert_allclose(actual, expected)


def test_one_class_importance_uses_quadratic_dual_objective_change():
    ensemble = _fitted_one_class_ensemble()
    model = ensemble.models[0]
    support = ensemble._get_support_vectors(0)
    features = ensemble.features
    parameters = ensemble.parameters_.kernel
    full = ensemble.kernel.compute(support, feature_index=features, parameters=parameters)
    reduced = ensemble.kernel.compute(
        support, feature_index=features[features != 0], parameters=parameters
    )
    alpha = model.dual_coef_[0]
    expected = abs(0.5 * alpha @ (full - reduced) @ alpha)
    np.testing.assert_allclose(ensemble.feature_importance_(0)[0], expected)


def test_one_class_combined_rank_prefers_coverage_then_compression():
    ensemble = _fitted_one_class_ensemble()
    model_index = 0
    indices = np.asarray(ensemble.cv.train[model_index])
    X = ensemble.cv.X[indices]
    current = ensemble.decision_function(X, model_index=model_index)
    perturbation = ensemble.decision_perturbation_(model_index, X)
    perturbed = current[:, np.newaxis] - perturbation
    compression_gain = np.var(perturbed, axis=0) - np.var(current)
    coverage = np.mean(perturbed >= 0, axis=0)
    eligible = coverage >= 1 - ensemble.models[model_index].nu
    order = np.lexsort((compression_gain, eligible.astype(int)))
    expected = np.empty(len(order), dtype=int)
    expected[order] = np.arange(len(order))

    actual = combined_rank(weight=1.0).compute(ensemble, model_index, "train")
    np.testing.assert_array_equal(actual, expected)
