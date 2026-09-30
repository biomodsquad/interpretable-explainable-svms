"""Unexecuted benchmark harness for predictive and feature-recovery comparisons."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mistic import (
    MixedKernel,
    categorical,
    combined_rank,
    cvSet,
    kernelWrapper,
    loguniform,
    parameterSpace,
    paramSet,
    score_svc,
    svmSet,
)


def _seaborn():
    """Import the optional plotting dependency only when figures are requested."""
    try:
        import seaborn
    except ModuleNotFoundError as error:
        raise ImportError(
            "benchmark plotting requires the optional 'seaborn' package"
        ) from error
    return seaborn


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_samples: int
    n_features: int
    scenario: str
    data_seed: int
    n_signal: int = 20


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_version: str = "permuted_signal_columns_v1"
    sample_sizes: tuple[int, ...] = (50, 100, 250, 500, 1000)
    feature_counts: tuple[int, ...] = (100, 250, 500)
    scenarios: tuple[str, ...] = (
        "clean_linear", "correlated", "nonlinear", "noisy_imbalanced")
    data_seeds: tuple[int, ...] = (2026, 2027, 2028)
    outer_folds: int = 5
    inner_folds: int = 3
    mistic_members: int = 5
    mistic_rank_weight: float = 0.90
    mistic_feature_fraction: float = 0.20
    mistic_num_perturbation_clusters: int = 40
    mistic_max_features: int | None = None
    mistic_class_weight: bool = True
    mistic_tune_models_each_step: bool = False
    mistic_addition_factor: float = 0.3
    mistic_score_weight: float = 1.0
    mistic_c_values: tuple[float, ...] = (0.25, 1.0, 4.0)
    mistic_gamma_values: tuple[float, ...] = (2**-9, 2**-7, 2**-5)
    mistic_parameter_strategy: str = "grid"
    mistic_kernel: str = "rbf"
    mistic_lhs_trials: int = 9
    mistic_c_bounds: tuple[float, float] = (0.25, 8.0)
    mistic_gamma_bounds: tuple[float, float] = (2**-9, 2**-1)
    mistic_mixed_linear_weights: tuple[float, ...] = (0.25, 0.5, 0.75)
    mistic_max_selection_features: int = 5
    mistic_num_initial_selection_units: int = 1
    random_seed: int = 42


FULL_CONFIG = BenchmarkConfig()
SMOKE_CONFIG = BenchmarkConfig(
    sample_sizes=(100,), feature_counts=(100,), scenarios=("correlated",),
    data_seeds=(2026,), outer_folds=2, inner_folds=2)

OPTIONAL_DEPENDENCIES = {
    "xgboost": "xgboost",
    "catboost": "catboost",
    "interpretml_ebm": "interpret",
    "tabpfn": "tabpfn",
}


def dependency_report():
    return pd.DataFrame([
        {"method": method, "package": package,
         "available": find_spec(package) is not None}
        for method, package in OPTIONAL_DEPENDENCIES.items()
    ])


def dataset_specs(config=FULL_CONFIG):
    return [
        DatasetSpec(
            name=f"{scenario}_n{n_samples}_p{n_features}_s{seed}",
            n_samples=n_samples, n_features=n_features,
            scenario=scenario, data_seed=seed)
        for scenario in config.scenarios
        for n_samples in config.sample_sizes
        for n_features in config.feature_counts
        for seed in config.data_seeds
    ]


def _nonlinear_dataset(spec):
    rng = np.random.default_rng(spec.data_seed)
    X = rng.normal(size=(spec.n_samples, spec.n_features))
    score = (
        1.6 * X[:, 0] * X[:, 1]
        + 1.2 * np.sin(X[:, 2] * 1.5)
        + 0.9 * (X[:, 3] ** 2 - 1)
        - 1.0 * X[:, 4] * X[:, 5]
        + 0.6 * X[:, 6:10].sum(axis=1)
        + rng.normal(scale=1.1, size=spec.n_samples)
    )
    y = (score > np.median(score)).astype(int)
    mixing = rng.normal(size=(10, 10))
    X[:, 10:20] = X[:, :10] @ mixing + rng.normal(
        scale=0.35, size=(spec.n_samples, 10))
    return X, y


def make_dataset(spec):
    if spec.scenario == "nonlinear":
        X, y = _nonlinear_dataset(spec)
    else:
        settings = {
            "clean_linear": dict(
                n_informative=20, n_redundant=0, class_sep=1.5,
                flip_y=0.01, weights=[0.5, 0.5]),
            "correlated": dict(
                n_informative=10, n_redundant=10, class_sep=1.0,
                flip_y=0.03, weights=[0.55, 0.45]),
            "noisy_imbalanced": dict(
                n_informative=10, n_redundant=10, class_sep=0.65,
                flip_y=0.10, weights=[0.85, 0.15]),
        }[spec.scenario]
        X, y = make_classification(
            n_samples=spec.n_samples, n_features=spec.n_features,
            n_repeated=0, n_classes=2, shuffle=False,
            random_state=spec.data_seed, **settings)
    # Keep signal location from becoming an index-order cue. The separate RNG
    # makes the permutation reproducible without depending on how many random
    # values a particular scenario consumed while generating observations.
    original_names = np.asarray([f"x{index:04d}" for index in range(spec.n_features)])
    original_signal_mask = np.arange(spec.n_features) < spec.n_signal
    permutation = np.random.default_rng(spec.data_seed + 104729).permutation(
        spec.n_features
    )
    X = X[:, permutation]
    feature_names = original_names[permutation]
    signal_mask = original_signal_mask[permutation]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y), signal_mask


def _safe_split_count(y, requested):
    return max(2, min(int(requested), int(np.bincount(np.asarray(y)).min())))


def _scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        probability = estimator.predict_proba(X)[:, 1]
        return probability, probability
    decision = np.asarray(estimator.decision_function(X), dtype=float)
    return decision, expit(decision)


def _permutation_ranking(estimator, X, y, seed):
    values = permutation_importance(
        estimator, X, y, scoring="roc_auc", n_repeats=10,
        random_state=seed, n_jobs=-1).importances_mean
    return np.asarray(values, dtype=float)


def _native_importance(estimator, n_features):
    model = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator
    if isinstance(model, Pipeline):
        selector = model.named_steps.get("select")
        final_model = model.steps[-1][1]
        if selector is not None and hasattr(selector, "support_") and hasattr(final_model, "coef_"):
            importance = np.zeros(n_features, dtype=float)
            importance[selector.support_] = np.abs(
                np.asarray(final_model.coef_)).mean(axis=0)
            return importance
        model = final_model
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_)).mean(axis=0)
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    return np.full(n_features, np.nan)


def sklearn_methods(seed, inner_folds):
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    return {
        "elastic_net_logistic": GridSearchCV(
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(
                    penalty="elasticnet", solver="saga", max_iter=5000,
                    class_weight="balanced", random_state=seed)),
            ]),
            {"model__C": [0.03, 0.1, 0.3, 1, 3, 10],
             "model__l1_ratio": [0.25, 0.5, 0.75, 1.0]},
            scoring="roc_auc", cv=inner, n_jobs=-1, refit=True),
        "rbf_svm": GridSearchCV(
            Pipeline([
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", class_weight="balanced",
                              probability=True, random_state=seed)),
            ]),
            {"model__C": [0.1, 0.3, 1, 3, 10],
             "model__gamma": ["scale", 2**-9, 2**-7, 2**-5]},
            scoring="roc_auc", cv=inner, n_jobs=-1, refit=True),
        "linear_svm_rfe_20": GridSearchCV(
            Pipeline([
                ("scale", StandardScaler()),
                ("select", RFE(
                    SVC(kernel="linear", C=1.0, class_weight="balanced"),
                    n_features_to_select=20, step=0.1)),
                ("model", SVC(
                    kernel="linear", class_weight="balanced",
                    probability=True, random_state=seed)),
            ]),
            {"select__estimator__C": [0.1, 1.0, 10.0],
             "model__C": [0.1, 1.0, 10.0]},
            scoring="roc_auc", cv=inner, n_jobs=-1, refit=True),
        "extra_trees": GridSearchCV(
            ExtraTreesClassifier(
                n_estimators=500, class_weight="balanced", n_jobs=-1,
                random_state=seed),
            {"max_features": ["sqrt", 0.25, 0.5],
             "min_samples_leaf": [1, 2, 5]},
            scoring="roc_auc", cv=inner, n_jobs=-1, refit=True),
        "hist_gradient_boosting": GridSearchCV(
            HistGradientBoostingClassifier(
                class_weight="balanced", early_stopping=True,
                random_state=seed),
            {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15, 31],
             "l2_regularization": [0, 1, 10]},
            scoring="roc_auc", cv=inner, n_jobs=-1, refit=True),
    }


def optional_methods(seed):
    methods = {}
    if find_spec("xgboost") is not None:
        from xgboost import XGBClassifier
        methods["xgboost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, random_state=seed)
    if find_spec("catboost") is not None:
        from catboost import CatBoostClassifier
        methods["catboost"] = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            verbose=False, random_seed=seed)
    if find_spec("interpret") is not None:
        from interpret.glassbox import ExplainableBoostingClassifier
        methods["explainable_boosting_machine"] = ExplainableBoostingClassifier(
            interactions=10, outer_bags=8, random_state=seed)
    if find_spec("tabpfn") is not None:
        from tabpfn import TabPFNClassifier
        methods["tabpfn"] = TabPFNClassifier(random_state=seed)
    return methods


class MisticClassifier:
    """Small benchmark adapter exposing the final unified MISTIC predictor."""

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def _kernel_and_grid(self):
        """Build the configured base kernel and its tuning candidates."""
        kernel = kernelWrapper(self.config.mistic_kernel)
        if self.config.mistic_parameter_strategy == "latin_hypercube":
            kernel_space = (
                {"gamma": loguniform(*self.config.mistic_gamma_bounds)}
                if self.config.mistic_kernel == "rbf" else {}
            )
            grid = parameterSpace(
                model={"C": loguniform(*self.config.mistic_c_bounds)},
                kernel=kernel_space,
            ).sample(
                n_trials=self.config.mistic_lhs_trials,
                strategy="latin_hypercube",
                random_state=self.seed,
            )
        elif self.config.mistic_parameter_strategy == "grid":
            if self.config.mistic_kernel == "rbf":
                grid = [
                    paramSet(model={"C": c}, kernel={"gamma": gamma})
                    for c in self.config.mistic_c_values
                    for gamma in self.config.mistic_gamma_values
                ]
            else:
                grid = [
                    paramSet(model={"C": c}, kernel={})
                    for c in self.config.mistic_c_values
                ]
        else:
            raise ValueError(
                "mistic_parameter_strategy must be 'grid' or 'latin_hypercube'"
            )
        return kernel, grid

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        scaled = self.scaler_.transform(X)
        feature_budget = max(
            1, int(round(self.config.mistic_feature_fraction * X.shape[1])))
        self.feature_budget_ = feature_budget
        num_feature_medoids = min(
            X.shape[1], self.config.mistic_num_perturbation_clusters
        )
        independent = getattr(self, "independent_kernel_features", False)
        max_features = (
            feature_budget if self.config.mistic_max_features is None
            else min(X.shape[1], self.config.mistic_max_features)
        )
        self.num_feature_medoids_ = num_feature_medoids
        self.max_features_ = max_features
        cluster_medoids, perturbation_sets = cvSet.cluster_features(
            scaled, num_feature_medoids
        )
        splits = cvSet(scaled, np.asarray(y), num_feature_medoids=num_feature_medoids,
                       ensemble_validation_size=0.0)
        # Use the same preprocessing clusters both as selectable perturbation
        # units and as the representative candidates that seed forward search.
        splits.feature_medoids_ = cluster_medoids
        splits.classification(
            num_sets=self.config.mistic_members, validation_size=0.20,
            random_seed=self.seed)
        kernel, grid = self._kernel_and_grid()
        self.ensemble_ = svmSet(
            SVC(
                kernel="precomputed",
                class_weight="balanced" if self.config.mistic_class_weight else None,
            ), splits,
            score_method=score_svc(weight=self.config.mistic_score_weight).score,
            kernel=kernel,
            feature_set_policy="shared" if independent else "per_model",
            kernel_feature_selection="independent" if independent else "shared",
            perturbation_sets=perturbation_sets,
        )
        if independent:
            pair_budget = min(
                self.config.mistic_max_selection_features,
                len(kernel.base_kernels) * len(splits.feature_medoids_),
            )
            self.ensemble_.greedy_forward_selection(
                parameter_grid=grid,
                max_features=pair_budget,
                addition_factor=self.config.mistic_addition_factor,
                num_initial_medoids=self.config.mistic_num_initial_selection_units,
                feature_ranker=combined_rank(
                    weight=self.config.mistic_rank_weight).compute,
            )
        else:
            self.ensemble_.greedy_forward_selection(
                parameter_grid=grid,
                addition_factor=self.config.mistic_addition_factor,
                num_initial_medoids=1,
                feature_ranker=combined_rank(
                    weight=self.config.mistic_rank_weight).compute,
                set_for_rank="sample",
                tune_models_each_step=self.config.mistic_tune_models_each_step,
                max_features=max_features,
                post_find_knee=False)
        if independent:
            self.selected_features_ = np.asarray(self.ensemble_.unified_features, dtype=int)
            self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
            pair_count = len(self.ensemble_.selected_kernel_features_)
            for rank, (_, feature) in enumerate(self.ensemble_.selected_kernel_features_):
                self.feature_importances_[feature] += pair_count - rank
        else:
            self.selected_features_ = np.asarray(
                self.ensemble_.unified_prediction_features_, dtype=int
            )
            ranks = np.asarray(self.ensemble_.unified_feature_rank, dtype=float)
            self.feature_importances_ = np.max(ranks) - ranks + 1
        return self

    def predict(self, X):
        return self.ensemble_.predict(self.scaler_.transform(X))

    def decision_function(self, X):
        return self.ensemble_.decision_function(self.scaler_.transform(X))


class MixedMisticClassifier(MisticClassifier):
    """Benchmark adapter for a tunable linear-plus-RBF MISTIC kernel."""

    def _kernel_and_grid(self):
        kernel = MixedKernel.weighted_sum(
            [
                kernelWrapper("linear", name="linear"),
                kernelWrapper("rbf", name="radial"),
            ],
            weight_parameters=["linear_weight", "rbf_weight"],
        )
        if self.config.mistic_parameter_strategy == "latin_hypercube":
            grid = parameterSpace(
                model={"C": loguniform(*self.config.mistic_c_bounds)},
                kernel={
                    "radial__gamma": loguniform(*self.config.mistic_gamma_bounds),
                    "linear_weight": categorical(self.config.mistic_mixed_linear_weights),
                },
            ).sample(
                n_trials=self.config.mistic_lhs_trials,
                strategy="latin_hypercube",
                random_state=self.seed,
            )
            for candidate in grid:
                candidate.kernel["rbf_weight"] = 1.0 - candidate.kernel["linear_weight"]
        elif self.config.mistic_parameter_strategy == "grid":
            grid = [
                paramSet(
                    model={"C": c},
                    kernel={
                        "radial__gamma": gamma,
                        "linear_weight": weight,
                        "rbf_weight": 1.0 - weight,
                    },
                )
                for c in self.config.mistic_c_values
                for gamma in self.config.mistic_gamma_values
                for weight in self.config.mistic_mixed_linear_weights
            ]
        else:
            raise ValueError(
                "mistic_parameter_strategy must be 'grid' or 'latin_hypercube'"
            )
        return kernel, grid


class IndependentMixedMisticClassifier(MixedMisticClassifier):
    """Mixed-kernel benchmark adapter with kernel-specific feature selection."""

    independent_kernel_features = True


def feature_metrics(importance, signal_mask):
    importance = np.nan_to_num(np.asarray(importance, dtype=float), nan=0.0)
    k = int(np.count_nonzero(signal_mask))
    top = np.argsort(importance, kind="stable")[::-1][:k]
    true = set(np.flatnonzero(signal_mask))
    overlap = len(set(top) & true)
    return {
        "signal_recall_at_k": overlap / k,
        "signal_precision_at_k": overlap / max(1, len(top)),
        "feature_average_precision": average_precision_score(signal_mask, importance),
        "top_features": json.dumps(top.tolist()),
    }


def prediction_metrics(y, prediction, score, probability):
    probability = np.clip(probability, 1e-7, 1-1e-7)
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "f1": f1_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "mcc": matthews_corrcoef(y, prediction),
        "accuracy": accuracy_score(y, prediction),
        "log_loss": log_loss(y, probability),
    }


def evaluate_method(name, estimator, X_train, y_train, X_test, y_test,
                    signal_mask, seed):
    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    prediction = estimator.predict(X_test)
    score, probability = _scores(estimator, X_test)
    importance = _native_importance(estimator, X_train.shape[1])
    importance_source = "native"
    if not np.isfinite(importance).any():
        # Evaluation-only permutation importance uses the outer fold after
        # predictions; it never changes the fitted model or hyperparameters.
        importance = _permutation_ranking(estimator, X_test, y_test, seed)
        importance_source = "outer_test_permutation_diagnostic"
    selected = getattr(estimator, "selected_features_", None)
    return {
        "method": name, "elapsed_seconds": time.perf_counter()-started,
        "importance_source": importance_source,
        "selected_feature_count": (len(selected) if selected is not None else np.nan),
        **prediction_metrics(y_test, prediction, score, probability),
        **feature_metrics(importance, signal_mask),
    }


def benchmark_dataset(spec, config=FULL_CONFIG, include_optional=True):
    X, y, signal_mask = make_dataset(spec)
    outer_folds = _safe_split_count(y, config.outer_folds)
    outer = StratifiedKFold(
        outer_folds, shuffle=True,
        random_state=config.random_seed + spec.data_seed)
    rows = []
    for fold, (train, test) in enumerate(outer.split(X, y)):
        inner_folds = _safe_split_count(y.iloc[train], config.inner_folds)
        seed = config.random_seed + 1000*spec.data_seed + fold
        methods = {
            "mistic_unified": MisticClassifier(config, seed),
            **sklearn_methods(seed, inner_folds),
        }
        if include_optional:
            methods.update(optional_methods(seed))
        for name, estimator in methods.items():
            try:
                result = {
                    "status": "ok", "error_type": None, "error_message": None,
                    **evaluate_method(
                        name, estimator, X.iloc[train], y.iloc[train],
                        X.iloc[test], y.iloc[test], signal_mask, seed),
                }
            except Exception as error:
                # A difficult p/n regime or optional backend must not erase
                # successful paired results from the other methods.
                result = {
                    "method": name, "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:1000],
                }
            rows.append({**asdict(spec), "outer_fold": fold, **result})
    return pd.DataFrame(rows)


def run_benchmark(config=FULL_CONFIG, output_dir=None, include_optional=True):
    """Run with per-dataset checkpoints. Not called on module import."""
    output_dir = Path(output_dir or Path(__file__).resolve().parent / "benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"config": asdict(config), "dependencies": dependency_report().to_dict("records")}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    completed = []
    for spec in dataset_specs(config):
        path = output_dir / f"{spec.name}.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            frame = benchmark_dataset(spec, config, include_optional)
            frame.to_csv(path, index=False)
        completed.append(frame)
    results = pd.concat(completed, ignore_index=True)
    results.to_csv(output_dir / "all_results.csv", index=False)
    return results


def rerun_mistic_only(
        config=FULL_CONFIG, output_dir=None,
        checkpoint_name="mistic_scaled20_checkpoints",
        results_name="all_results_scaled_mistic.csv"):
    """Rerun only MISTIC and merge it with completed reference results.

    New MISTIC checkpoints are stored separately from the original per-dataset
    files, so interruption cannot damage the existing benchmark. When every
    dataset is available, ``all_results_scaled_mistic.csv`` combines the new
    MISTIC rows with all original non-MISTIC rows.
    """
    output_dir = Path(output_dir or Path(__file__).resolve().parent / "benchmark_results")
    original_path = output_dir / "all_results.csv"
    if not original_path.exists():
        raise FileNotFoundError(
            f"reference results are required before a MISTIC-only rerun: {original_path}")
    original = pd.read_csv(original_path)
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for spec in dataset_specs(config):
        path = checkpoint_dir / f"{spec.name}.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            X, y, signal_mask = make_dataset(spec)
            outer_folds = _safe_split_count(y, config.outer_folds)
            outer = StratifiedKFold(
                outer_folds, shuffle=True,
                random_state=config.random_seed + spec.data_seed)
            rows = []
            for fold, (train, test) in enumerate(outer.split(X, y)):
                seed = config.random_seed + 1000*spec.data_seed + fold
                try:
                    result = {
                        "status": "ok", "error_type": None,
                        "error_message": None,
                        **evaluate_method(
                            "mistic_unified", MisticClassifier(config, seed),
                            X.iloc[train], y.iloc[train], X.iloc[test],
                            y.iloc[test], signal_mask, seed),
                    }
                except Exception as error:
                    result = {
                        "method": "mistic_unified", "status": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:1000],
                    }
                rows.append({
                    **asdict(spec), "outer_fold": fold,
                    "mistic_feature_budget": max(
                        1, int(round(config.mistic_feature_fraction * spec.n_features))),
                    **result,
                })
            frame = pd.DataFrame(rows)
            frame.to_csv(path, index=False)
        completed.append(frame)
        print(f"MISTIC complete: {spec.name}", flush=True)
    replacement = pd.concat(completed, ignore_index=True)
    references = original.loc[~original.method.eq("mistic_unified")]
    merged = pd.concat([references, replacement], ignore_index=True, sort=False)
    merged.to_csv(output_dir / results_name, index=False)
    return merged


def summarize(results):
    metrics = [
        "roc_auc", "pr_auc", "f1", "balanced_accuracy", "mcc",
        "signal_recall_at_k", "feature_average_precision", "elapsed_seconds",
    ]
    return (results.groupby(["scenario", "n_samples", "n_features", "method"])[metrics]
            .agg(["mean", "std", "count"]))


def feature_stability(results):
    """Calculate top-k Jaccard stability from successful, ranked folds only."""
    rows = []
    group_keys = ["name", "method"]
    usable = results.loc[
        results["status"].eq("ok") & results["top_features"].notna()
    ]
    for keys, frame in usable.groupby(group_keys):
        sets = [set(json.loads(value)) for value in frame.top_features]
        similarities = [
            len(left & right) / max(1, len(left | right))
            for index, left in enumerate(sets)
            for right in sets[index+1:]
        ]
        rows.append({"name": keys[0], "method": keys[1],
                     "mean_top_k_jaccard": (
                         np.mean(similarities) if similarities else np.nan)})
    return pd.DataFrame(rows)


def failure_summary(results):
    """Return one row per method/status/error combination."""
    columns = ["method", "status", "error_type", "error_message"]
    return (results[columns].fillna("").groupby(columns, dropna=False)
            .size().rename("fold_count").reset_index())


def _successful_seed_means(results):
    metrics = [
        "roc_auc", "pr_auc", "balanced_accuracy", "mcc",
        "signal_recall_at_k", "feature_average_precision", "elapsed_seconds",
    ]
    successful = results.loc[results["status"].eq("ok")].copy()
    return (successful.groupby(
        ["scenario", "n_samples", "n_features", "data_seed", "method"],
        as_index=False)[metrics].mean())


def plot_predictive_performance(results):
    """Plot ROC-AUC scaling across scenarios, sample sizes, and dimensions."""
    sns = _seaborn()
    data = _successful_seed_means(results)
    grid = sns.relplot(
        data=data, x="n_samples", y="roc_auc", hue="method",
        row="scenario", col="n_features", kind="line", marker="o",
        errorbar=("ci", 95), facet_kws={"margin_titles": True},
        height=2.7, aspect=1.15)
    grid.set(xscale="log", ylim=(0.35, 1.01))
    grid.set_axis_labels("Samples (log scale)", "Outer-fold ROC-AUC")
    grid.set_titles(row_template="{row_name}", col_template="{col_name} features")
    grid.figure.suptitle("Predictive performance across benchmark regimes", y=1.01)
    return grid.figure


def plot_feature_recovery(results):
    """Plot known-signal recall at k across the complete benchmark matrix."""
    sns = _seaborn()
    data = _successful_seed_means(results)
    grid = sns.relplot(
        data=data, x="n_samples", y="signal_recall_at_k", hue="method",
        row="scenario", col="n_features", kind="line", marker="o",
        errorbar=("ci", 95), facet_kws={"margin_titles": True},
        height=2.7, aspect=1.15)
    grid.set(xscale="log", ylim=(-0.02, 1.02))
    grid.set_axis_labels("Samples (log scale)", "Signal recall at k=20")
    grid.set_titles(row_template="{row_name}", col_template="{col_name} features")
    grid.figure.suptitle("Recovery of known signal features", y=1.01)
    return grid.figure


def plot_mistic_advantage(results, metric="roc_auc"):
    """Plot MISTIC minus the strongest non-MISTIC mean in every regime."""
    sns = _seaborn()
    successful = results.loc[results["status"].eq("ok")]
    means = (successful.groupby(
        ["scenario", "n_samples", "n_features", "method"])[metric].mean()
        .reset_index())
    mistic = means.loc[means.method.eq("mistic_unified")].set_index(
        ["scenario", "n_samples", "n_features"])[metric]
    competitors = (means.loc[~means.method.eq("mistic_unified")]
                   .groupby(["scenario", "n_samples", "n_features"])[metric]
                   .max())
    delta = (mistic-competitors).rename("delta").reset_index()
    scenarios = list(delta.scenario.drop_duplicates())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2*len(scenarios), 3.7),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    bound = max(0.01, float(np.nanmax(np.abs(delta.delta))))
    for axis, scenario in zip(axes, scenarios):
        matrix = (delta.loc[delta.scenario.eq(scenario)]
                  .pivot(index="n_features", columns="n_samples", values="delta")
                  .sort_index(ascending=False))
        sns.heatmap(matrix, annot=True, fmt="+.3f", center=0,
                    vmin=-bound, vmax=bound, cmap="vlag", cbar=axis is axes[-1],
                    ax=axis)
        axis.set_title(scenario)
        axis.set_xlabel("Samples")
        axis.set_ylabel("Features" if axis is axes[0] else "")
    fig.suptitle(f"MISTIC minus best competing method: {metric}", y=1.05)
    return fig


def plot_performance_recovery_tradeoff(results):
    """Plot predictive performance against known-signal recovery."""
    sns = _seaborn()
    data = (_successful_seed_means(results).groupby("method", as_index=False)
            [["roc_auc", "signal_recall_at_k"]].mean())
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    sns.scatterplot(data=data, x="signal_recall_at_k", y="roc_auc",
                    hue="method", s=100, ax=axis)
    for row in data.itertuples():
        axis.annotate(row.method, (row.signal_recall_at_k, row.roc_auc),
                      xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set(xlabel="Mean signal recall at k=20", ylabel="Mean outer-fold ROC-AUC",
             title="Prediction–feature recovery trade-off")
    axis.legend_.remove()
    return fig


def plot_feature_stability(results):
    """Plot mean outer-fold top-k Jaccard stability for each method."""
    sns = _seaborn()
    data = (feature_stability(results).groupby("method", as_index=False)
            ["mean_top_k_jaccard"].mean()
            .sort_values("mean_top_k_jaccard", ascending=False))
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    sns.barplot(data=data, x="mean_top_k_jaccard", y="method", ax=axis)
    axis.set(xlim=(0, 1), xlabel="Mean pairwise top-k Jaccard",
             ylabel="", title="Feature-ranking stability across outer folds")
    return fig


def plot_benchmark_results(results, output_dir=None):
    """Generate the standard benchmark figure set and optionally save PNGs."""
    figures = {
        "predictive_performance": plot_predictive_performance(results),
        "feature_recovery": plot_feature_recovery(results),
        "mistic_auc_advantage": plot_mistic_advantage(results, "roc_auc"),
        "performance_recovery_tradeoff": plot_performance_recovery_tradeoff(results),
        "feature_stability": plot_feature_stability(results),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    return figures
