Mixed kernels and kernel-specific features
==========================================

MISTIC represents every kernel as a :class:`~mistic.MixedKernel` expression.
A conventional single kernel is a one-leaf expression with weight one; a
mixed kernel combines named leaves by addition, multiplication, or a positive
integer power. This common representation lets tuning, feature selection,
prediction, gradients, and integrated gradients use the same API.

Creating kernel expressions
---------------------------

Create leaves with :class:`~mistic.kernelWrapper`. Give every leaf a unique
name when it will have its own parameters or feature set.

.. code-block:: python

   from mistic import MixedKernel, kernelWrapper

   linear = kernelWrapper("linear", name="linear")
   radial = kernelWrapper("rbf", name="radial")

   additive = linear + radial
   multiplicative = linear * radial
   powered = radial**2
   expression = 0.25 * linear + 0.75 * radial**2

Python operators create fixed expressions. Kernel sums, products, nonnegative
scalings, and positive integer powers preserve positive semidefiniteness. A
negative coefficient and a non-positive or non-integer power are rejected.

For a single kernel, use the same representation explicitly:

.. code-block:: python

   kernel = MixedKernel.weighted_sum(
       [kernelWrapper("rbf", name="radial")],
       weights=[1.0],
   )

``svmSet`` also converts a bare ``kernelWrapper`` to this one-leaf form for
compatibility. New code should prefer the explicit form when it helps make a
configuration or serialized analysis self-describing.

Tunable weights and leaf parameters
-----------------------------------

Use :meth:`~mistic.MixedKernel.weighted_sum` with ``weight_parameters`` when
mixture weights must be tuned. The strings become kernel hyperparameter names.
Leaf parameters use ``leaf_name__parameter`` names. This prevents two RBF,
polynomial, or sigmoid leaves from accidentally sharing parameters.

.. code-block:: python

   mixed = MixedKernel.weighted_sum(
       [
           kernelWrapper("linear", name="linear"),
           kernelWrapper("rbf", name="radial"),
       ],
       weight_parameters=["linear_weight", "radial_weight"],
   )

   grid = [
       paramSet(
           model={"C": C},
           kernel={
               "radial__gamma": gamma,
               "linear_weight": linear_weight,
               "radial_weight": 1.0 - linear_weight,
           },
       )
       for C in (0.5, 2.0, 8.0)
       for gamma in (2**-7, 2**-4, 2**-1)
       for linear_weight in (0.25, 0.5, 0.75)
   ]

MISTIC tunes one parameter set against aggregate cross-validation performance.
Parameters may also be addressed by leaf position, such as
``kernel1__gamma``, and an unprefixed parameter such as ``gamma`` is shared by
all compatible leaves. Named parameters are clearest and are recommended for
mixed expressions. Weights must remain nonnegative; if a convex combination
is intended, construct the grid so the weights also sum to one.

RBF over Tanimoto distance
--------------------------

``kernelWrapper("tanimoto_rbf")`` implements an exponential kernel over
Tanimoto distance rather than the usual squared Euclidean distance:

.. math::

   K(x, y) = \exp\{-\gamma [1 - T(x, y)]\}.

Here ``T`` is Tanimoto similarity for nonnegative binary or count features.
This differs from multiplying independent RBF and Tanimoto leaves. ``gamma``
must be a nonnegative number; the scikit-learn strings ``"scale"`` and
``"auto"`` are not used because this distance is already normalized.

.. code-block:: python

   fingerprint_kernel = kernelWrapper("tanimoto_rbf", name="fingerprint")
   grid = [
       paramSet(
           model={"C": C},
           kernel={"fingerprint__gamma": gamma},
       )
       for C in (0.1, 1.0, 10.0)
       for gamma in (0.1, 0.3, 1.0, 3.0, 10.0)
   ]

The leaf can be used alone or inside any :class:`~mistic.MixedKernel`
expression. The alias ``"rbf_tanimoto"`` is accepted, but
``"tanimoto_rbf"`` is the canonical spelling.

Using a mixed kernel in ``svmSet``
----------------------------------

The estimator must use ``kernel="precomputed"`` because MISTIC evaluates the
expression.

.. code-block:: python

   from sklearn.svm import SVC
   from mistic import combined_rank, score_svc, svmSet

   model = svmSet(
       SVC(kernel="precomputed", class_weight="balanced"),
       splits,
       score_method=score_svc().score,
       kernel=mixed,
       kernel_feature_selection="independent",
       feature_set_policy="per_model",
       unified_feature_policy="majority",
   )
   model.tune_models(grid)

There are two separate feature-selection choices:

.. list-table:: Selection axes
   :header-rows: 1
   :widths: 27 28 45

   * - Argument
     - Values
     - Meaning
   * - ``kernel_feature_selection``
     - ``"shared"`` or ``"independent"``
     - Whether all leaves use one feature set, or each named leaf owns its own
       selectable kernel/perturbation assignments.
   * - ``feature_set_policy``
     - ``"shared"`` or ``"per_model"``
     - Whether cross-validation members use the same active assignments, or
       each member may select a different state.

These axes are independent. For example, ``kernel_feature_selection="independent"``
and ``feature_set_policy="shared"`` gives every leaf a separate feature set,
but synchronizes that mapping across CV members. With ``"per_model"``, the
final development-data model uses the union or majority consensus selected by
``unified_feature_policy``.

Kernel-specific candidate pools
-------------------------------

Mixed-type datasets often contain both continuous measurements and discrete
fingerprints. ``kernel_candidate_features`` specifies which original columns
each named leaf may consider. It controls eligibility; ``model.kernel_features``
records the currently active columns.

.. code-block:: python

   mixed = MixedKernel.weighted_sum(
       [
           kernelWrapper("rbf", name="continuous"),
           kernelWrapper("tanimoto", name="fingerprint"),
       ],
       weight_parameters=["continuous_weight", "fingerprint_weight"],
   )

   model = svmSet(
       SVC(kernel="precomputed", class_weight="balanced"),
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

The selection space contains one
:class:`~mistic.KernelFeatureSet` for every eligible ``(kernel, perturbation
set)`` pairing. Therefore, removing a group from ``continuous`` does not
remove it from ``fingerprint`` when both pools contain that group. A feature
can appear in multiple leaves, so the number of kernel-feature assignments can
exceed the number of unique input columns.

Leaf-level ``features=[...]`` and ``MixedKernel(..., feature_sets=...)`` also
place hard feature restrictions on expression evaluation. Candidate pools on
``svmSet`` are preferable for feature selection because they describe the
search space explicitly and are validated against leaf names.

Grouped perturbations, medoids, and greedy selection
----------------------------------------------------

``perturbation_sets`` makes a group of original columns indivisible during
ranking, addition, and removal. Externally computed medoids can seed forward
selection by passing original column indices to ``cvSet``.

.. code-block:: python

   splits = cvSet(X_dev, y_dev, feature_medoids=precomputed_medoids)
   splits.classification(num_sets=5, random_seed=7)

   model.greedy_forward_selection(
       parameter_grid=grid,
       max_features=20,
       num_initial_medoids=2,
       addition_factor=0.25,
       feature_ranker=combined_rank(weight=0.75).compute,
       set_for_rank="train",
   )

For independent mixed kernels, ``max_features`` counts kernel-feature
assignments, not unique columns. ``num_initial_medoids`` controls the first
selected batch. Later batches target ``addition_factor`` times the remaining
distance to ``max_features``; zero adds exactly one selection unit. The same
kernel-aware abstraction is used by :meth:`~mistic.svmSet.greedy_backward_selection`.

Inspecting and changing the active state
----------------------------------------

.. code-block:: python

   print(model.kernel_features)
   print(model.unified_prediction_features_)

   model.remove_kernel_features("radial", [2, 5])
   model.add_kernel_features("linear", [2])

``kernel_features`` is keyed by leaf name. ``features`` is the union of active
original columns, while ``unified_prediction_features_`` is the union used by
the final model. Selection histories distinguish the number of active
kernel-feature assignments from the number of unique features.

Gradients and integrated gradients
----------------------------------

MISTIC applies the sum, product, scale, and power differentiation rules to the
entire expression, so the usual explanation API works without extracting a
single leaf.

.. code-block:: python

   explanation = model.explain_integrated_gradients(
       X_blind,
       feature_names=feature_names,
       target=y_blind,
       reference_point=X_dev.mean(axis=0),
       num_steps=30,
   )
   explanation.summary_plot(max_features=20)

Tanimoto accepts nonnegative binary or count features. For binary
fingerprints, use an explicit binary reference and discrete feature-flip paths:

.. code-block:: python

   explanation = model.explain_integrated_gradients(
       X_fingerprint,
       reference_point=reference_fingerprint,
       discrete_features=fingerprint_columns,
       n_discrete_paths=32,
       random_seed=7,
   )

All active model features must be binary when discrete paths are requested;
hybrid continuous/discrete attribution paths are not currently supported.

Compatibility notes
-------------------

``MixedkernelWrapper`` and ``greedy_forward_kernel_selection`` remain as
transition aliases. New analyses should use ``MixedKernel`` and the unified
``greedy_forward_selection`` method. ``separate_feature_sets`` is likewise a
deprecated alias for ``feature_set_policy``. Separate per-member parameter
sets are no longer supported; one parameter set is selected from aggregate CV
performance.
