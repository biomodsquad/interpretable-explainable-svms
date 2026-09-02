Feature selection and perturbation analysis
===========================================

Feature selection searches for a smaller input set that preserves or improves
generalization. For SVMs, it can also make the fitted decision system easier to
inspect. Selection is not itself causal discovery: a selected variable may be
a proxy for correlated information, and a removed variable may still be useful
when its substitute is absent.

Why perturb features?
---------------------

MISTIC recomputes the kernel after removing a feature group while keeping the
fitted dual coefficients fixed. This isolates two related changes:

``feature_importance_``
   The change in the frozen SVM objective. It summarizes the group's role in
   the fitted model geometry.

``decision_perturbation_``
   The signed change in decision output for each sample. It preserves local
   direction and heterogeneity.

``probability_perturbation_``
   For probability-enabled binary SVCs, the local change in positive-class
   probability.

.. code-block:: python

   objective_change = model.feature_importance_(model_index=0)
   local_decision_change = model.decision_perturbation_(0, X_to_explain)
   local_probability_change = model.probability_perturbation_(0, X_to_explain)

Feature groups
--------------

Dummy variables, repeated measurements, or engineered representations can be
perturbed as indivisible groups:

.. code-block:: python

   groups = [
       [0, 1, 2],       # three encodings of one factor
       [3],
       [4, 5],          # a paired measurement
   ]
   model = svmSet(
       estimator,
       splits,
       scorer.score,
       kernel=kernel,
       perturbation_sets=groups,
       perturbation_normalization="per_feature",
   )

The normalization determines how group size affects ranking. Use
``"per_feature"`` when groups should compete on average contribution,
``"sqrt"`` when some group-size advantage is scientifically reasonable, and
``"none"`` when total group effect is the intended quantity.

Choosing the sample set
-----------------------

Ranking can use training or validation-oriented samples through
``set_for_rank``. A sample-based rank asks how consistently perturbation changes
outputs across observations. Keep the choice fixed while comparing selection
strategies; otherwise a change in results can reflect a change in the ranking
population rather than the search algorithm.

Reading perturbation values
---------------------------

A large objective change and a large local change answer different questions.
The first says the group is structurally important to the fitted SVM; the
second says it matters for particular observations. Look at the distribution,
not only the mean. Opposite signed effects can average to nearly zero even when
the group is locally influential.
