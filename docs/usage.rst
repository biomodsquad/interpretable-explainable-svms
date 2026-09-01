Usage
=====

Install MISTIC from PyPI with ``python -m pip install mistic-svm``. The
distribution installs the import package named ``mistic``.

A typical workflow creates cross-validation splits, configures a precomputed
kernel SVM, and tunes model/kernel parameter pairs::

   from sklearn.svm import SVC
   from mistic import cvSet, kernelWrapper, paramSet, score_svc, svmSet

   splits = cvSet(X, y)
   splits.classification(num_sets=5)

   ensemble = svmSet(
       SVC(kernel="precomputed"),
       splits,
       score_svc().score,
       kernel=kernelWrapper("rbf"),
   )
   ensemble.tune_models([
       paramSet(model={"C": 1.0}, kernel={"gamma": 0.1}),
   ])
   predictions = ensemble.predict(X_new)

By default, prediction uses one final SVM tuned on the existing CV splits and
fitted on the knee-ranked unified feature subset. The cross-validation member
set remains available explicitly::

   unified_predictions = ensemble.predict(X_new)
   set_predictions = ensemble.predict(X_new, prediction_mode="set")

The same option applies to ``decision_function``. Supplying ``model_index``
continues to evaluate one particular member model.

One-class models use sklearn's ``+1`` inlier and ``-1`` outlier convention.
The dedicated split keeps known outliers out of every training set::

   from sklearn.svm import OneClassSVM
   from mistic import score_ocsvm

   splits = cvSet(X, y)
   splits.one_class(num_sets=5)
   ensemble = svmSet(
       OneClassSVM(kernel="precomputed"), splits, score_ocsvm().score,
       kernel=kernelWrapper("rbf"),
   )
   ensemble.tune_models([
       paramSet(model={"nu": 0.1}, kernel={"gamma": 0.1}),
   ])

``decision_function`` returns positive values for inliers and negative values
for outliers. One-class feature contributions use the same signed decision
perturbations as SVC and SVR; objective importance uses the magnitude of the
frozen-coefficient quadratic dual change.

An optional ensemble-level validation set can be reserved when ``cvSet`` is
created. These samples are excluded from every later CV train/test split and
from feature-medoid construction::

   splits = cvSet(
       X,
       y,
       ensemble_validation_size=0.15,
       ensemble_validation_random_seed=7,
       ensemble_validation_stratify=True,
   )
   splits.classification(num_sets=5)

After running greedy feature selection, a specific number of the highest-ranked
features can be selected and the ensemble retuned with a parameter grid. The
knee of the stored performance curve provides a data-driven feature count::

   num_features = ensemble.find_knee()
   ensemble.set_num_features(num_features, parameter_grid)

Greedy forward and backward selection perform this knee selection and final
retuning automatically. Pass ``post_find_knee=False`` to either search method
to retain its best-scoring subset instead.

The selected subset can then be refined with rank-guided stochastic search.
The search can both add and remove perturbation groups, uses the same
``feature_ranker`` and ``set_for_rank`` interface as the greedy methods, and
supports both ensemble-wide and per-model feature sets::

   ensemble.stochastic_feature_selection(
       parameter_grid,
       n_iterations=100,
       temperature=0.05,
       cooling_rate=0.97,
       add_probability=0.5,
       random_seed=7,
   )

The best visited feature state is restored, while proposal diagnostics remain
available in ``ensemble.stochastic_performance_``.

For an ensemble configured with ``separate_feature_sets=True``, synchronous
refinement makes one independently ranked move for every model in each round.
All moves are evaluated as one proposal against the average cross-validation
score and are therefore accepted or rejected together::

   ensemble.stochastic_feature_selection(
       parameter_grid,
       n_iterations=100,
       update_all_models=True,
       use_ensemble_validation=True,
       expected_changes_per_model=2.5,
       random_seed=7,
   )

The number of changes for each selected model is stochastic, with the value
above specifying its expectation. At least one perturbation group is changed
for every eligible selected model, and the sampled count is capped when fewer
groups can be added or safely removed. The default value of ``1.0`` retains
one-group-per-model rounds.

When the greedy search has already selected an appropriate feature count, swap
mode keeps that count fixed. Every model with a separate feature set proposes
one ranked removal and one ranked addition per round; shared feature sets make
one swap that applies to the full ensemble::

   ensemble.stochastic_feature_selection(
       parameter_grid,
       n_iterations=100,
       preserve_feature_count=True,
       use_ensemble_validation=True,
       convergence_patience=20,
       convergence_min_delta=0.001,
       random_seed=7,
   )

Removal and addition groups must contain the same number of feature columns.
Refinement stops early when the best accepted score has not improved by more
than ``convergence_min_delta`` for ``convergence_patience`` proposals. The
result is summarized by ``stochastic_converged_``, ``stochastic_iterations_``,
and ``stochastic_stop_reason_``. Pass ``convergence_patience=None`` to disable
early stopping.

Ensemble-pooled refinement is available as a separate search method for
``separate_feature_sets=True`` ensembles. It estimates every feasible
model-feature addition and removal against one aggregate out-of-fold ensemble
prediction, pools those candidates globally, and fully tunes only the sampled
candidate::

   ensemble.ensemble_stochastic_feature_selection(
       parameter_grid,
       n_iterations=100,
       preserve_feature_count=True,
       feature_diversity_weight=0.02,
       prediction_diversity_weight=0.01,
       performance_tolerance=0.005,
       max_feature_similarity=0.8,
       convergence_patience=20,
       random_seed=7,
   )

Diagnostics are stored separately in
``ensemble.ensemble_stochastic_performance_`` so this method does not alter the
interface or history of ``stochastic_feature_selection``.

Feature diversity is the mean pairwise Jaccard distance between model feature
sets. Prediction diversity is one minus the mean absolute pairwise correlation
of overlapping out-of-fold decision values; regression uses out-of-fold
residual correlations. The combined proposal objective is the ensemble score
plus the two weighted diversity terms. ``performance_tolerance`` prevents the
search from trading away excessive predictive performance. A hard
``max_feature_similarity`` rejects moves above the requested pairwise Jaccard
similarity, while still allowing moves that reduce similarity when the initial
ensemble already exceeds the threshold.
