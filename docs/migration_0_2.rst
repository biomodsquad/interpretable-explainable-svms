Migrating to MISTIC 0.2
=======================

MISTIC 0.2 unifies conventional and mixed kernels under
:class:`mistic.MixedKernel` and unifies kernel-aware forward and backward
selection under the existing greedy method names.

Kernel construction
-------------------

A bare wrapper remains valid, but an explicit one-leaf mixed kernel is the
recommended representation for new analyses.

.. code-block:: python

   # 0.1-compatible form; still accepted
   kernel = kernelWrapper("rbf")

   # Recommended 0.2 form
   kernel = MixedKernel.weighted_sum(
       [kernelWrapper("rbf", name="radial")], weights=[1.0]
   )

Named leaves use prefixed parameters:

.. code-block:: python

   paramSet(model={"C": 2.0}, kernel={"radial__gamma": 0.1})

Use :meth:`mistic.MixedKernel.weighted_sum` with ``weight_parameters`` for
tunable mixture weights. See :doc:`tutorials/mixed_kernels` for sums,
products, powers, and complete tuning examples.

Feature-selection policies
--------------------------

The old ``separate_feature_sets`` Boolean is replaced by the clearer
``feature_set_policy`` argument:

.. code-block:: python

   model = svmSet(
       estimator,
       splits,
       scorer,
       kernel=kernel,
       kernel_feature_selection="independent",
       feature_set_policy="per_model",
       unified_feature_policy="majority",
   )

``kernel_feature_selection`` controls whether kernel leaves share features or
own independent assignments. ``feature_set_policy`` separately controls
whether CV members share one selection state. ``unified_feature_policy``
chooses any-appearance or majority consensus for the final development-data
model.

``separate_feature_sets=True`` remains a deprecated alias for
``feature_set_policy="per_model"``. ``separate_parameters`` is accepted for
compatibility, but 0.2 always tunes one parameter set against aggregate CV
performance.

Unified greedy selection
------------------------

Replace ``greedy_forward_kernel_selection`` with
``greedy_forward_selection``:

.. code-block:: python

   model.greedy_forward_selection(
       parameter_grid=grid,
       max_features=20,
       num_initial_medoids=1,
       addition_factor=0.25,
       feature_ranker=combined_rank().compute,
   )

For independent mixed kernels, ``max_features`` counts kernel-feature
assignments rather than unique input columns. A feature assigned to two leaves
therefore counts twice. ``num_initial_medoids`` controls the first batch;
``addition_factor`` controls later batches relative to the remaining distance
to the maximum. Backward selection uses the same kernel-aware units.

Candidate pools and grouped perturbations
-----------------------------------------

Use ``kernel_candidate_features`` to restrict named leaves to appropriate
input types, and ``perturbation_sets`` to add, remove, and rank related columns
as indivisible units:

.. code-block:: python

   model = svmSet(
       estimator,
       splits,
       scorer,
       kernel=mixed,
       kernel_feature_selection="independent",
       perturbation_sets=groups,
       kernel_candidate_features={
           "continuous": continuous_columns,
           "fingerprint": fingerprint_columns,
       },
   )

Existing medoid calculations can be reused with
``cvSet(X, y, feature_medoids=medoid_columns)``.

Inspecting selected features
----------------------------

``model.kernel_features`` maps each leaf name to its active original columns.
``model.features`` is their unique union. ``model.unified_prediction_features_``
is the unique union used by the final fitted model. Selection history may
therefore report more kernel-feature assignments than unique features.

Compatibility timeline
----------------------

The compatibility names ``MixedkernelWrapper`` and
``greedy_forward_kernel_selection`` remain available during the 0.2
transition. New code should use ``MixedKernel`` and
``greedy_forward_selection`` so it does not depend on their future removal.
