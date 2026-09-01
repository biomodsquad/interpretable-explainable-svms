"""Training-only selection between MiSTIC set prediction and one unified SVM."""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
    root_mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, KFold, ShuffleSplit, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

REPO_ROOT = next(
    path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / "mistic" / "svmSet.py").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mistic import combined_rank, cvSet, kernelWrapper, paramSet, score_svc, score_svr, svmSet

N_SAMPLES = 500
N_FEATURES = 100
N_INFORMATIVE = 10
N_REDUNDANT = 10
DATA_SEED = 2026
BLIND_SET_SEED = 42
SELECTION_SEEDS = list(range(20))
BLIND_SIZE = 0.25
ARCHITECTURE_VALIDATION_SIZE = 0.25
INNER_VALIDATION_SIZE = 0.20
MAX_FEATURES = 20
ADDITION_FACTOR = 0.10
C_VALUES = [0.25, 1.0, 4.0]
GAMMA_VALUES = [2.0 ** exponent for exponent in (-9, -7, -5)]
EPSILON_VALUES = [0.05, 0.1, 0.2]


def ranked_unified_features(ensemble, limit):
    unified = set(np.asarray(ensemble.unified_features, dtype=int))
    ranked = [f for f in ensemble.unified_sorted_features if f in unified]
    return np.asarray(ranked[:min(int(limit), len(ranked))], dtype=int)


def classification_data():
    values, target = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=N_INFORMATIVE, n_redundant=N_REDUNDANT,
        n_classes=2, weights=[0.55, 0.45], class_sep=1.0, flip_y=0.03,
        shuffle=False, random_state=DATA_SEED,
    )
    X, y = pd.DataFrame(values), pd.Series(target)
    return train_test_split(
        X, y, test_size=BLIND_SIZE, stratify=y, random_state=BLIND_SET_SEED
    )


def regression_data():
    base, target = make_regression(
        n_samples=N_SAMPLES, n_features=90, n_informative=N_INFORMATIVE,
        noise=60.0, shuffle=False, random_state=DATA_SEED,
    )
    rng = np.random.default_rng(DATA_SEED)
    mixing = rng.normal(size=(N_INFORMATIVE, N_REDUNDANT))
    redundant = base[:, :N_INFORMATIVE] @ mixing
    redundant += rng.normal(scale=0.30 * redundant.std(axis=0), size=redundant.shape)
    values = np.column_stack([base[:, :N_INFORMATIVE], redundant, base[:, N_INFORMATIVE:]])
    X, y = pd.DataFrame(values), pd.Series(target)
    X_train, X_blind, y_train, y_blind = train_test_split(
        X, y, test_size=BLIND_SIZE, random_state=BLIND_SET_SEED
    )
    center, scale = y_train.mean(), y_train.std(ddof=0)
    return X_train, X_blind, (y_train-center)/scale, (y_blind-center)/scale


def fit_classification_mistic(X, y, seed):
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    splits = cvSet(scaled, np.asarray(y), num_feature_medoids=20,
                   ensemble_validation_size=0.0)
    splits.classification(num_sets=5, validation_size=INNER_VALIDATION_SIZE,
                          random_seed=seed)
    model = svmSet(
        SVC(kernel="precomputed", class_weight="balanced"), splits,
        score_method=score_svc(weight=1.0).score,
        kernel=kernelWrapper(type="rbf"), separate_feature_sets=True,
        separate_parameters=True,
    )
    grid = [paramSet(model={"C": c}, kernel={"gamma": g})
            for c in C_VALUES for g in GAMMA_VALUES]
    model.greedy_forward_selection(
        addition_factor=ADDITION_FACTOR, max_features=MAX_FEATURES,
        parameter_grid=grid, feature_ranker=combined_rank(weight=0.90).compute,
        set_for_rank="sample", tune_models_each_step=False, post_find_knee=True,
    )
    return scaler, model, ranked_unified_features(model, model.find_knee(metric="score"))


def fit_regression_mistic(X, y, seed):
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    splits = cvSet(scaled, np.asarray(y), num_feature_medoids=20,
                   ensemble_validation_size=0.0)
    generated = list(ShuffleSplit(
        n_splits=20, test_size=INNER_VALIDATION_SIZE, random_state=seed
    ).split(scaled, y))
    splits.train = [train for train, _ in generated]
    splits.test = [test for _, test in generated]
    splits.type = "regression_shuffle_split"
    model = svmSet(
        SVR(kernel="precomputed"), splits,
        score_method=score_svr(weight=0.0).score,
        kernel=kernelWrapper(type="linear"), separate_feature_sets=True,
        separate_parameters=True,
    )
    grid = [paramSet(model={"C": c, "epsilon": e}, kernel={})
            for c in C_VALUES for e in EPSILON_VALUES]
    model.greedy_forward_selection(
        addition_factor=ADDITION_FACTOR, max_features=MAX_FEATURES,
        parameter_grid=grid, feature_ranker=combined_rank(weight=0.75).compute,
        set_for_rank="sample", tune_models_each_step=False, post_find_knee=True,
    )
    return scaler, model, ranked_unified_features(model, model.find_knee(metric="score"))


def fit_classification_unified(X, y, features, seed):
    return GridSearchCV(
        Pipeline([("scale", StandardScaler()),
                  ("svc", SVC(kernel="rbf", class_weight="balanced"))]),
        {"svc__C": C_VALUES, "svc__gamma": GAMMA_VALUES}, scoring="roc_auc",
        cv=StratifiedKFold(4, shuffle=True, random_state=seed), n_jobs=-1,
    ).fit(X.iloc[:, features], y)


def fit_regression_unified(X, y, features, seed):
    return GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="linear"))]),
        {"svr__C": C_VALUES, "svr__epsilon": EPSILON_VALUES}, scoring="r2",
        cv=KFold(4, shuffle=True, random_state=seed), n_jobs=-1,
    ).fit(X.iloc[:, features], y)


def classification_metrics(y, prediction, decision):
    return dict(roc_auc=roc_auc_score(y, decision), f1=f1_score(y, prediction),
                balanced_accuracy=balanced_accuracy_score(y, prediction),
                accuracy=accuracy_score(y, prediction))


def regression_metrics(y, prediction):
    r = pearsonr(y, prediction).statistic
    return dict(r2=r2_score(y, prediction), rmse=root_mean_squared_error(y, prediction),
                mae=mean_absolute_error(y, prediction), pearson_r2=r*r)


def run_classification(seed):
    X_train, X_blind, y_train, y_blind = classification_data()
    X_dev, X_select, y_dev, y_select = train_test_split(
        X_train, y_train, test_size=ARCHITECTURE_VALIDATION_SIZE,
        stratify=y_train, random_state=10_000 + seed,
    )
    dev_scaler, dev_set, dev_features = fit_classification_mistic(X_dev, y_dev, seed)
    unified = fit_classification_unified(X_dev, y_dev, dev_features, seed)
    set_selection = roc_auc_score(y_select, dev_set.decision_function(dev_scaler.transform(X_select)))
    unified_selection = roc_auc_score(
        y_select, unified.decision_function(X_select.iloc[:, dev_features]))
    selected = "unified" if unified_selection >= set_selection else "svm_set"

    scaler, final_set, features = fit_classification_mistic(X_train, y_train, seed)
    final_unified = fit_classification_unified(X_train, y_train, features, seed)
    set_metrics = classification_metrics(
        y_blind, final_set.predict(scaler.transform(X_blind)),
        final_set.decision_function(scaler.transform(X_blind)))
    unified_metrics = classification_metrics(
        y_blind, final_unified.predict(X_blind.iloc[:, features]),
        final_unified.decision_function(X_blind.iloc[:, features]))
    return [
        {"task": "classification", "seed": seed, "method": method,
         "selected_architecture": selected, "selection_set_score": score,
         "num_features": len(features), **metrics}
        for method, score, metrics in [
            ("svm_set", set_selection, set_metrics),
            ("unified", unified_selection, unified_metrics),
            ("training_selected", max(set_selection, unified_selection),
             unified_metrics if selected == "unified" else set_metrics),
        ]
    ]


def run_regression(seed):
    X_train, X_blind, y_train, y_blind = regression_data()
    X_dev, X_select, y_dev, y_select = train_test_split(
        X_train, y_train, test_size=ARCHITECTURE_VALIDATION_SIZE,
        random_state=10_000 + seed,
    )
    dev_scaler, dev_set, dev_features = fit_regression_mistic(X_dev, y_dev, seed)
    unified = fit_regression_unified(X_dev, y_dev, dev_features, seed)
    set_selection = r2_score(y_select, dev_set.predict(dev_scaler.transform(X_select)))
    unified_selection = r2_score(y_select, unified.predict(X_select.iloc[:, dev_features]))
    selected = "unified" if unified_selection >= set_selection else "svm_set"

    scaler, final_set, features = fit_regression_mistic(X_train, y_train, seed)
    final_unified = fit_regression_unified(X_train, y_train, features, seed)
    set_metrics = regression_metrics(y_blind, final_set.predict(scaler.transform(X_blind)))
    unified_metrics = regression_metrics(
        y_blind, final_unified.predict(X_blind.iloc[:, features]))
    return [
        {"task": "regression", "seed": seed, "method": method,
         "selected_architecture": selected, "selection_set_score": score,
         "num_features": len(features), **metrics}
        for method, score, metrics in [
            ("svm_set", set_selection, set_metrics),
            ("unified", unified_selection, unified_metrics),
            ("training_selected", max(set_selection, unified_selection),
             unified_metrics if selected == "unified" else set_metrics),
        ]
    ]


def run_experiment():
    rows = []
    for task, runner in [("classification", run_classification),
                         ("regression", run_regression)]:
        for seed in SELECTION_SEEDS:
            started = time.perf_counter()
            rows.extend(runner(seed))
            print(f"{task} seed={seed:02d} complete in {time.perf_counter()-started:.1f}s", flush=True)
    results = pd.DataFrame(rows)
    output = REPO_ROOT / "validation/Synthetic100_nested_unified_selection_20seeds_results.csv"
    results.to_csv(output, index=False)
    return results


if __name__ == "__main__":
    run_experiment()
