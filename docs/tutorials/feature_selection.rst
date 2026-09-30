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
       MixedKernel, combined_rank, cvSet, kernelWrapper, paramSet, score_svc,
       svmSet,
   )

   splits = cvSet(X_development, y_development)
   splits.classification(num_sets=5, random_seed=7)

   grid = [
       paramSet(model={"C": C}, kernel={"radial__gamma": gamma})
       for C in (0.5, 2.0, 8.0)
       for gamma in (2**-7, 2**-4)
   ]
   ranker = combined_rank(weight=0.9, random_seed=7)

   model = svmSet(
       SVC(kernel="precomputed", class_weight="balanced"),
       splits,
       score_method=score_svc().score,
       kernel=MixedKernel.weighted_sum(
           [kernelWrapper("rbf", name="radial")], weights=[1.0]
       ),
       kernel_feature_selection="shared",
       feature_set_policy="per_model",
       unified_feature_policy="majority",
   )

``feature_set_policy="per_model"`` gives every CV member its own selection
state. The members may therefore choose different kernel/perturbation units,
but MISTIC tunes one shared model/kernel parameter set against their aggregate
CV performance. Use ``"shared"`` to keep the member selections synchronized.

When selections are per-model, ``unified_feature_policy`` controls which
selection units enter the final model trained on all development observations:

* ``"any"`` (the default) retains a unit selected by at least one CV member;
* ``"majority"`` retains a unit only when more than half of the members select it.

Consensus is calculated over complete perturbation sets. For independent mixed
kernels, the kernel identifier is also part of the unit, so a majority vote for
``(linear, group_2)`` does not activate ``(radial, group_2)``.

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

Explicit medoids and kernel candidate pools
-------------------------------------------

Externally computed medoids can seed the initial forward-selection round
without invoking MISTIC's built-in feature clustering. Medoids are original
input-column indices. A perturbation set is eligible for the initial round
when it contains at least one supplied medoid.

.. code-block:: python

   splits = cvSet(
       X,
       y,
       feature_medoids=[continuous_medoid, fingerprint_medoid],
   )
   splits.classification(num_sets=5)

For mixed-type inputs, restrict each named base kernel to its own candidate
pool on ``svmSet``. Perturbation sets are intersected with these pools when the
kernel-aware selection space is constructed.

.. code-block:: python

   mixed = MixedKernel.weighted_sum(
       [
           kernelWrapper("rbf", name="continuous"),
           kernelWrapper("tanimoto", name="fingerprint"),
       ],
       weight_parameters=["continuous_weight", "fingerprint_weight"],
   )

   model = svmSet(
       SVC(kernel="precomputed"),
       splits,
       score_svc().score,
       kernel=mixed,
       kernel_feature_selection="independent",
       perturbation_sets=perturbation_sets,
       kernel_candidate_features={
           "continuous": continuous_columns,
           "fingerprint": fingerprint_columns,
       },
   )

Kernel names and column indices are validated during construction. Candidate
pools control eligibility, while ``model.kernel_features`` continues to show
the currently active subset for each kernel. Explicit medoids also constrain
the initial selection units for independent mixed kernels; later forward
rounds may add any unit in the corresponding candidate pools.

Knee selection and final fitting
--------------------------------

Both greedy methods select a knee and retune by default. To inspect the search
before deciding, pass ``post_find_knee=False`` and then explicitly set the
feature count:

.. code-block:: python

   count = model.find_knee(metric="score")
   model.set_num_selection_features(count, grid)

For mixed kernels with independent feature sets, each performance row stores
the exact state of every CV member. Knee restoration restores those snapshots
rather than forcing one member's units onto the others. The final unified model
uses either the any-appearance or majority consensus configured above.
Member-specific sets remain available for stability analysis and set-mode
predictions.

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

For the distinction between kernel-specific and model-specific feature sets,
and for the meaning of ``max_features`` in an independent mixed kernel, see
:doc:`mixed_kernels`.
