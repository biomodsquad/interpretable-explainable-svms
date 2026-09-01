"""Second-stage one-feature-at-a-time forward selection on MiSTIC unions."""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
    root_mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

VALIDATION_DIR = Path(__file__).resolve().parent
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from synthetic100_nested_unified_selection import (
    C_VALUES, EPSILON_VALUES, GAMMA_VALUES, N_FEATURES, N_INFORMATIVE,
    N_REDUNDANT, classification_data, fit_classification_mistic,
    fit_regression_mistic, regression_data,
)

SEEDS = list(range(10))
SIGNAL_FEATURES = set(range(N_INFORMATIVE + N_REDUNDANT))


def classification_search(X, y, features, seed):
    return GridSearchCV(
        Pipeline([("scale", StandardScaler()),
                  ("model", SVC(kernel="rbf", class_weight="balanced"))]),
        {"model__C": C_VALUES, "model__gamma": GAMMA_VALUES},
        scoring="roc_auc", cv=StratifiedKFold(4, shuffle=True, random_state=seed),
        n_jobs=-1, refit=True,
    ).fit(X.iloc[:, features], y)


def regression_search(X, y, features, seed):
    return GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="linear"))]),
        {"model__C": C_VALUES, "model__epsilon": EPSILON_VALUES},
        scoring="r2", cv=KFold(4, shuffle=True, random_state=seed),
        n_jobs=-1, refit=True,
    ).fit(X.iloc[:, features], y)


def curve_knee(feature_counts, scores):
    counts = np.asarray(feature_counts, dtype=float)
    values = np.asarray(scores, dtype=float)
    if len(counts) < 3 or np.ptp(values) == 0:
        return int(counts[np.argmax(values)]), "max_cv_fallback"
    normalized_counts = (counts-counts.min()) / np.ptp(counts)
    normalized_values = (values-values.min()) / np.ptp(values)
    distance = normalized_values - normalized_counts
    interior = distance[1:-1]
    if not len(interior) or np.max(interior) <= np.finfo(float).eps:
        return int(counts[np.argmax(values)]), "max_cv_fallback"
    return int(counts[1 + np.argmax(interior)]), "knee"


def second_round_forward(X, y, candidates, seed, search_factory):
    """Exhaust every singleton, then greedily add exactly one candidate."""
    candidates = np.asarray(sorted(set(map(int, candidates))), dtype=int)
    selected = []
    remaining = candidates.tolist()
    curve = []
    fitted = {}
    while remaining:
        trials = []
        for candidate in remaining:
            features = np.asarray(selected + [candidate], dtype=int)
            search = search_factory(X, y, features, seed)
            trials.append((search.best_score_, candidate, features, search))
        score, added, features, model = max(trials, key=lambda row: row[0])
        selected.append(added)
        remaining.remove(added)
        curve.append({
            "num_features": len(selected), "cv_score": score,
            "feature_added": added, "selected_features": tuple(selected),
        })
        fitted[len(selected)] = model
    knee_count, rule = curve_knee(
        [row["num_features"] for row in curve],
        [row["cv_score"] for row in curve])
    return (np.asarray(selected[:knee_count], dtype=int), fitted[knee_count],
            pd.DataFrame(curve), rule)


def recovery(features):
    features = set(map(int, features))
    overlap = len(features & SIGNAL_FEATURES)
    return overlap / len(SIGNAL_FEATURES), overlap / len(features)


def classification_metrics(y, prediction, score):
    return dict(roc_auc=roc_auc_score(y, score), f1=f1_score(y, prediction),
                balanced_accuracy=balanced_accuracy_score(y, prediction),
                accuracy=accuracy_score(y, prediction))


def regression_metrics(y, prediction):
    correlation = pearsonr(y, prediction).statistic
    return dict(r2=r2_score(y, prediction),
                rmse=root_mean_squared_error(y, prediction),
                mae=mean_absolute_error(y, prediction),
                pearson_r2=correlation**2)


def row(task, seed, method, features, metrics, **extra):
    recall, precision = recovery(features)
    return {"task": task, "seed": seed, "method": method,
            "num_features": len(features), "signal_recall": recall,
            "signal_precision": precision, **extra, **metrics}


def classification_seed(seed):
    X_train, X_blind, y_train, y_blind = classification_data()
    scaler, ensemble, _ = fit_classification_mistic(X_train, y_train, seed)
    blind_scaled = scaler.transform(X_blind)
    union = np.asarray(ensemble.unified_features, dtype=int)
    default_features = np.asarray(ensemble.unified_prediction_features_, dtype=int)
    second_features, second_model, curve, knee_rule = second_round_forward(
        X_train, y_train, union, seed, classification_search)

    selector = RFE(SVC(kernel="linear", C=1.0, class_weight="balanced"),
                   n_features_to_select=20, step=0.1)
    selector.fit(StandardScaler().fit_transform(X_train), y_train)
    rfe_features = np.flatnonzero(selector.support_)
    references = {
        "sklearn RFE (20)": (rfe_features,
                             classification_search(X_train, y_train, rfe_features, seed)),
        "oracle known signal (20)": (np.arange(20),
                                     classification_search(X_train, y_train, np.arange(20), seed)),
    }
    rows = [
        row("classification", seed, "MiSTIC unified default", default_features,
            classification_metrics(
                y_blind, ensemble.predict(blind_scaled),
                ensemble.decision_function(blind_scaled))),
        row("classification", seed, "MiSTIC set aggregation", union,
            classification_metrics(
                y_blind, ensemble.predict(blind_scaled, prediction_mode="set"),
                ensemble.decision_function(blind_scaled, prediction_mode="set"))),
        row("classification", seed, "second-round forward knee", second_features,
            classification_metrics(
                y_blind, second_model.predict(X_blind.iloc[:, second_features]),
                second_model.decision_function(X_blind.iloc[:, second_features])),
            second_round_rule=knee_rule,
            second_round_max_cv=curve.cv_score.max()),
    ]
    for method, (features, model) in references.items():
        rows.append(row(
            "classification", seed, method, features,
            classification_metrics(
                y_blind, model.predict(X_blind.iloc[:, features]),
                model.decision_function(X_blind.iloc[:, features]))))
    curve.insert(0, "seed", seed)
    curve.insert(1, "task", "classification")
    return rows, curve


def regression_seed(seed):
    X_train, X_blind, y_train, y_blind = regression_data()
    scaler, ensemble, _ = fit_regression_mistic(X_train, y_train, seed)
    blind_scaled = scaler.transform(X_blind)
    union = np.asarray(ensemble.unified_features, dtype=int)
    default_features = np.asarray(ensemble.unified_prediction_features_, dtype=int)
    second_features, second_model, curve, knee_rule = second_round_forward(
        X_train, y_train, union, seed, regression_search)

    selector = RFE(SVR(kernel="linear", C=1.0, epsilon=0.1),
                   n_features_to_select=20, step=0.1)
    selector.fit(StandardScaler().fit_transform(X_train), y_train)
    rfe_features = np.flatnonzero(selector.support_)
    references = {
        "sklearn RFE (20)": (rfe_features,
                             regression_search(X_train, y_train, rfe_features, seed)),
        "oracle known signal (20)": (np.arange(20),
                                     regression_search(X_train, y_train, np.arange(20), seed)),
    }
    rows = [
        row("regression", seed, "MiSTIC unified default", default_features,
            regression_metrics(y_blind, ensemble.predict(blind_scaled))),
        row("regression", seed, "MiSTIC set aggregation", union,
            regression_metrics(
                y_blind, ensemble.predict(blind_scaled, prediction_mode="set"))),
        row("regression", seed, "second-round forward knee", second_features,
            regression_metrics(
                y_blind, second_model.predict(X_blind.iloc[:, second_features])),
            second_round_rule=knee_rule,
            second_round_max_cv=curve.cv_score.max()),
    ]
    for method, (features, model) in references.items():
        rows.append(row("regression", seed, method, features,
                        regression_metrics(
                            y_blind, model.predict(X_blind.iloc[:, features]))))
    curve.insert(0, "seed", seed)
    curve.insert(1, "task", "regression")
    return rows, curve


def run_experiment(seeds=SEEDS):
    rows, curves = [], []
    for task, runner in (("classification", classification_seed),
                         ("regression", regression_seed)):
        for seed in seeds:
            started = time.perf_counter()
            seed_rows, curve = runner(seed)
            rows.extend(seed_rows)
            curves.append(curve)
            print(f"{task} seed={seed} complete in {time.perf_counter()-started:.1f}s", flush=True)
    results = pd.DataFrame(rows)
    curves = pd.concat(curves, ignore_index=True)
    results.to_csv(VALIDATION_DIR / "Synthetic100_second_round_forward_selection_10seeds_results.csv", index=False)
    curves.to_csv(VALIDATION_DIR / "Synthetic100_second_round_forward_selection_10seeds_curves.csv", index=False)
    return results, curves


if __name__ == "__main__":
    run_experiment()
