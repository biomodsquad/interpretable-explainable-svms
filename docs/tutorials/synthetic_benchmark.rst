Synthetic benchmark against scikit-learn models
================================================

This tutorial compares MISTIC with four familiar scikit-learn strategies on a
classification problem where the signal features are known:

* an RBF SVC using all features;
* a linear-SVM RFE selector followed by a tuned RBF SVC;
* a random forest;
* histogram gradient-boosted trees.

The goal is not to declare a universal winner. Synthetic data make it possible
to evaluate two different questions at once: **predictive generalization** and
**recovery of the variables that generated the outcome**.

Experimental design
-------------------

Use one development set for every model and one untouched blind set for the
final comparison. All scaling, feature selection, parameter search, and
importance estimation must occur without blind-set information.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   seed = 7
   X, y = make_classification(
       n_samples=500,
       n_features=100,
       n_informative=12,
       n_redundant=8,
       n_repeated=0,
       n_clusters_per_class=2,
       class_sep=1.0,
       flip_y=0.03,
       shuffle=False,       # known signal occupies columns 0 through 19
       random_state=seed,
   )
   signal_features = set(range(20))
   X_dev, X_blind, y_dev, y_blind = train_test_split(
       X, y, test_size=0.25, stratify=y, random_state=42
   )

Keeping the blind split fixed while changing model-building seeds measures
sensitivity to resampling and stochastic fitting. It does not estimate
uncertainty over independently sampled test populations. For that, repeat the
entire data-generation and blind-split process.

Shared metrics
--------------

Measure discrimination, threshold behavior, and feature recovery separately:

.. code-block:: python

   from sklearn.metrics import (
       balanced_accuracy_score,
       f1_score,
       roc_auc_score,
   )

   def evaluate(name, fitted, X_test, y_test, selected_features):
       if hasattr(fitted, "decision_function"):
           score = fitted.decision_function(X_test)
       else:
           score = fitted.predict_proba(X_test)[:, 1]
       prediction = fitted.predict(X_test)
       selected = set(map(int, selected_features))
       recovered = selected & signal_features
       return {
           "method": name,
           "n_features": len(selected),
           "roc_auc": roc_auc_score(y_test, score),
           "f1": f1_score(y_test, prediction),
           "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
           "signal_recall": len(recovered) / len(signal_features),
           "signal_precision": len(recovered) / len(selected),
       }

Signal recall asks how much true signal was recovered. Signal precision asks
how concentrated the selected set is in true signal. Neither is available on a
real dataset, which is why this synthetic benchmark is useful.

MISTIC forward selection
------------------------

MISTIC controls its own reusable cross-validation members. The following grid
is intentionally small enough for a tutorial; expand it for a substantive
benchmark.

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   from sklearn.svm import SVC
   from mistic import (
       combined_rank, cvSet, kernelWrapper, paramSet, score_svc, svmSet,
   )

   scaler = StandardScaler().fit(X_dev)
   X_dev_scaled = scaler.transform(X_dev)
   X_blind_scaled = scaler.transform(X_blind)

   splits = cvSet(X_dev_scaled, y_dev)
   splits.classification(num_sets=5, validation_size=0.2, random_seed=seed)
   grid = [
       paramSet(model={"C": C}, kernel={"gamma": gamma})
       for C in (0.5, 2.0, 8.0)
       for gamma in (2**-7, 2**-4, 2**-1)
   ]
   mistic_model = svmSet(
       SVC(kernel="precomputed", class_weight="balanced"),
       splits,
       score_method=score_svc().score,
       kernel=kernelWrapper("rbf"),
       separate_feature_sets=True,
       separate_parameters=True,
   )
   mistic_model.greedy_forward_selection(
       parameter_grid=grid,
       addition_factor=0.10,
       max_features=30,
       feature_ranker=combined_rank(weight=0.90, random_seed=seed).compute,
       set_for_rank="sample",
       tune_models_each_step=False,
   )

   mistic_selected = mistic_model.unified_prediction_features_
   mistic_result = evaluate(
       "MISTIC forward selection",
       mistic_model,
       X_blind_scaled,
       y_blind,
       mistic_selected,
   )

The unified feature set and final model are determined entirely from the
development workflow. Do not choose ``weight``, the stopping rule, or the
feature count after inspecting blind performance.

Plain RBF SVC
-------------

The all-feature SVC tests whether selection is useful at all. Scaling and
parameter tuning stay inside the cross-validation pipeline.

.. code-block:: python

   from sklearn.model_selection import GridSearchCV, StratifiedKFold
   from sklearn.pipeline import Pipeline

   inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
   svc_pipeline = Pipeline([
       ("scale", StandardScaler()),
       ("model", SVC(kernel="rbf", class_weight="balanced")),
   ])
   plain_svc = GridSearchCV(
       svc_pipeline,
       {
           "model__C": (0.5, 2.0, 8.0),
           "model__gamma": (2**-7, 2**-4, 2**-1),
       },
       scoring="roc_auc",
       cv=inner_cv,
       n_jobs=-1,
   ).fit(X_dev, y_dev)
   plain_result = evaluate(
       "RBF SVC: all features",
       plain_svc,
       X_blind,
       y_blind,
       range(X.shape[1]),
   )

SVM-RFE followed by RBF SVC
---------------------------

RFE requires an estimator with a linear coefficient or feature-importance
attribute, so a linear SVM performs selection and an RBF SVC performs final
prediction. Placing RFE *inside* ``GridSearchCV`` ensures each validation fold
learns its selector using only that fold's training partition.

.. code-block:: python

   from sklearn.feature_selection import RFE
   from sklearn.svm import LinearSVC

   rfe_pipeline = Pipeline([
       ("scale", StandardScaler()),
       ("rfe", RFE(
           estimator=LinearSVC(C=1.0, dual="auto", random_state=seed),
           n_features_to_select=20,
           step=0.10,
       )),
       ("model", SVC(kernel="rbf", class_weight="balanced")),
   ])
   rfe_svc = GridSearchCV(
       rfe_pipeline,
       {
           "model__C": (0.5, 2.0, 8.0),
           "model__gamma": (2**-7, 2**-4, 2**-1),
       },
       scoring="roc_auc",
       cv=inner_cv,
       n_jobs=-1,
   ).fit(X_dev, y_dev)
   rfe_mask = rfe_svc.best_estimator_.named_steps["rfe"].support_
   rfe_result = evaluate(
       "Linear-SVM RFE + RBF SVC",
       rfe_svc,
       X_blind,
       y_blind,
       np.flatnonzero(rfe_mask),
   )

Random forest and boosted trees
-------------------------------

Trees supply nonlinear references with different inductive biases. They do not
require scaling. Their top-20 feature-recovery summaries below use permutation
importance computed on development data only. These rankings are descriptive;
they are not fed back into model tuning.

.. code-block:: python

   from sklearn.ensemble import (
       HistGradientBoostingClassifier,
       RandomForestClassifier,
   )
   from sklearn.inspection import permutation_importance

   forest = GridSearchCV(
       RandomForestClassifier(
           class_weight="balanced", random_state=seed, n_jobs=-1
       ),
       {
           "n_estimators": (300,),
           "max_features": ("sqrt", 0.5),
           "min_samples_leaf": (1, 3),
       },
       scoring="roc_auc",
       cv=inner_cv,
       n_jobs=-1,
   ).fit(X_dev, y_dev)

   boosted = GridSearchCV(
       HistGradientBoostingClassifier(random_state=seed),
       {
           "learning_rate": (0.05, 0.10),
           "max_leaf_nodes": (15, 31),
           "l2_regularization": (0.0, 1.0),
       },
       scoring="roc_auc",
       cv=inner_cv,
       n_jobs=-1,
   ).fit(X_dev, y_dev)

   def top_permutation_features(fitted):
       importance = permutation_importance(
           fitted,
           X_dev,
           y_dev,
           scoring="roc_auc",
           n_repeats=10,
           random_state=seed,
           n_jobs=-1,
       ).importances_mean
       return np.argsort(importance)[-20:]

   forest_result = evaluate(
       "Random forest",
       forest,
       X_blind,
       y_blind,
       top_permutation_features(forest),
   )
   boosted_result = evaluate(
       "Histogram gradient boosting",
       boosted,
       X_blind,
       y_blind,
       top_permutation_features(boosted),
   )

Compare the results
-------------------

.. code-block:: python

   results = pd.DataFrame([
       mistic_result,
       plain_result,
       rfe_result,
       forest_result,
       boosted_result,
   ]).set_index("method")
   results.sort_values("roc_auc", ascending=False)

Do not collapse the comparison to one number. Read the columns together:

* If all-feature SVC performance is strong but signal precision is low, the
  model predicts well without isolating a concise signal set.
* If RFE improves precision, its linear selector is capturing useful marginal
  structure; failure can indicate nonlinear or interacting signal.
* Forests and boosted trees can excel on thresholds and interactions, but
  permutation ranks are not equivalent to MISTIC's kernel perturbations.
* MISTIC is most compelling when it preserves blind performance with a smaller,
  stable feature set and supplies useful local explanations.
* Similar performance does not imply similar decision logic. Compare selected
  variables, error overlap, calibration, and explanation stability.

Repeated benchmark
------------------

For a defensible comparison, wrap the entire development procedure in a seed
loop and retain one result row per seed and method. Keep the blind observations
fixed only when studying model-building sensitivity; regenerate data and the
blind split when estimating population-level uncertainty. Report paired
differences and distributions rather than only the best seed.

The repository's ``validation`` directory contains longer-running paired
MISTIC/SVM, RFE, oracle-feature, and standard-model studies. Those validation
artifacts remain separate from the installable examples because they are
benchmark evidence rather than introductory workflows.
