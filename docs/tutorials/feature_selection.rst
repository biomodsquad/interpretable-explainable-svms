Forward and backward feature selection
======================================

MISTIC's greedy searches alternate between a model-derived ranking and an
empirical validation check. Forward selection starts small and adds promising
groups; backward selection starts with all groups and removes the least useful.

Shared setup
------------

.. code-block:: python

   from sklearn.svm import SVC
   from mistic import (
       combined_rank, cvSet, kernelWrapper, paramSet, score_svc, svmSet,
   )

   splits = cvSet(X_development, y_development)
   splits.classification(num_sets=5, random_seed=7)

   grid = [
       paramSet(model={"C": C}, kernel={"gamma": gamma})
       for C in (0.5, 2.0, 8.0)
       for gamma in (2**-7, 2**-4)
   ]
   ranker = combined_rank(weight=0.9, random_seed=7)

   model = svmSet(
       SVC(kernel="precomputed", class_weight="balanced"),
       splits,
       score_method=score_svc().score,
       kernel=kernelWrapper("rbf"),
       separate_feature_sets=True,
       separate_parameters=True,
   )

Forward selection
-----------------

Forward selection is useful when relatively few groups are expected to carry
the signal. ``addition_factor`` controls the fraction of eligible groups
considered at a step, and ``max_features`` provides a computational or
scientific ceiling.

.. code-block:: python

   model.greedy_forward_selection(
       parameter_grid=grid,
       addition_factor=0.1,
       max_features=15,
       feature_ranker=ranker.compute,
       set_for_rank="sample",
       tune_models_each_step=False,
   )

Setting ``tune_models_each_step=True`` more fully accounts for parameter and
feature interactions, at a substantial computational cost.

Backward selection
------------------

Backward selection is useful when the full model is stable and redundancy is
the main concern. ``reduction_factor`` controls how aggressively groups are
removed.

.. code-block:: python

   model.greedy_backward_selection(
       parameter_grid=grid,
       reduction_factor=0.1,
       feature_ranker=ranker.compute,
       set_for_rank="sample",
       tune_models_each_step=False,
   )

Knee selection and final fitting
--------------------------------

Both greedy methods select a knee and retune by default. To inspect the search
before deciding, pass ``post_find_knee=False`` and then explicitly set the
feature count:

.. code-block:: python

   count = model.find_knee(metric="score")
   model.set_num_features(count, grid)

The final unified model uses the ranked unified subset. Member-specific feature
sets remain available for stability analysis and set-mode predictions.

Comparing the directions
------------------------

Forward and backward searches need not converge to the same subset. Correlated
features and nonlinear interactions make the path matter. Compare:

* cross-validated performance versus feature count;
* selected-group stability across members;
* agreement of global ranks and local explanations;
* blind performance only after the complete selection rule is frozen.

Use the forward and backward example notebooks as executable end-to-end
templates. For further refinement, MISTIC also provides stochastic selection,
but greedy paths are usually easier to audit and communicate.
