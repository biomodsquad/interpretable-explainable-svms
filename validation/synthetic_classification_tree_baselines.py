"""Evaluate tree baselines on the fixed Synthetic-100 classification split."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

N_FEATURES = 100
SIGNAL_FEATURES = set(range(20))
INNER_SEEDS = range(10)
OUTPUT = Path(__file__).with_name("Synthetic100_tree_baselines_10seeds_results.csv")


def metrics(y_true, prediction, probability):
    """Return blind classification metrics."""
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "f1": f1_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "accuracy": accuracy_score(y_true, prediction),
    }


def signal_recovery(fitted, X_train, y_train, seed):
    """Return top-20 development permutation-importance recovery."""
    importance = permutation_importance(
        fitted,
        X_train,
        y_train,
        scoring="roc_auc",
        n_repeats=5,
        random_state=seed,
        n_jobs=-1,
    ).importances_mean
    ranked = np.argsort(importance)[-20:]
    recovered = len(set(ranked).intersection(SIGNAL_FEATURES))
    return recovered / len(SIGNAL_FEATURES), recovered / len(ranked)


def main():
    """Run paired tree searches and write one row per seed and method."""
    X, y = make_classification(
        n_samples=500,
        n_features=N_FEATURES,
        n_informative=10,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        weights=[0.55, 0.45],
        class_sep=1.0,
        flip_y=0.03,
        shuffle=False,
        random_state=2026,
    )
    X_train, X_blind, y_train, y_blind = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    rows = []
    for seed in INNER_SEEDS:
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        searches = {
            "sklearn: random forest": GridSearchCV(
                RandomForestClassifier(
                    class_weight="balanced", random_state=seed, n_jobs=-1
                ),
                {
                    "n_estimators": (300,),
                    "max_features": ("sqrt", 0.5),
                    "min_samples_leaf": (1, 3),
                },
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
            ),
            "sklearn: histogram gradient boosting": GridSearchCV(
                HistGradientBoostingClassifier(random_state=seed),
                {
                    "learning_rate": (0.05, 0.10),
                    "max_leaf_nodes": (15, 31),
                    "l2_regularization": (0.0, 1.0),
                },
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
            ),
        }
        for method, search in searches.items():
            search.fit(X_train, y_train)
            recall, precision = signal_recovery(search.best_estimator_, X_train, y_train, seed)
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "num_models": 1,
                    "num_features": N_FEATURES,
                    "signal_recall": recall,
                    "signal_precision": precision,
                    "cv_best_score": search.best_score_,
                    "best_params": repr(search.best_params_),
                    **metrics(
                        y_blind,
                        search.predict(X_blind),
                        search.predict_proba(X_blind)[:, 1],
                    ),
                }
            )
        print(f"seed {seed} complete", flush=True)

    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
