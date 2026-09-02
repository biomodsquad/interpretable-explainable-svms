Interpretations and explanations
================================

An interpretation describes model structure or global behavior; an explanation
attributes a particular output. MISTIC provides both, and each answers a
different question.

Feature rank
------------

Feature rank summarizes priority during selection. It is useful for reporting
which groups repeatedly enter or remain across member models. It is relative,
not a calibrated effect size, and correlated groups may exchange positions.

Report ranks alongside selection frequency and the validation-performance
curve. Avoid describing rank as a causal effect.

Support vectors
---------------

Support vectors are the training observations with nonzero dual coefficients.
They anchor the fitted boundary or regression tube. Inspecting their frequency,
class balance, and proximity to data-quality problems can reveal what the model
relies upon.

.. code-block:: python

   unified = model.unified_model_
   support_indices = unified.support_
   dual_coefficients = unified.dual_coef_

For precomputed kernels, scikit-learn's estimator does not store original
support-vector feature rows. Use ``support_`` to map back to the development
matrix retained by your analysis. Do not infer population prototypes: support
vectors are boundary-defining observations, not necessarily representative
ones.

Perturbation explanations
-------------------------

Decision and probability perturbations answer a discrete counterfactual:
*how would this fitted model's output change if this feature group contributed
no kernel information?* They are especially helpful for grouped variables and
nonlinear kernels.

.. code-block:: python

   local_margin_effect = model.decision_perturbation_(0, X_explain)

   # Binary probability-enabled SVC only:
   local_probability_effect = model.probability_perturbation_(0, X_explain)

Gradients
---------

``decision_gradient_`` and ``probability_gradient_`` measure infinitesimal
local sensitivity. Gradients depend on feature scale; compare them only after
considering preprocessing and units.

Integrated gradients
--------------------

Integrated gradients accumulate local gradients along a straight path from a
reference point to an observation. The attributions approximately sum to the
output difference between the observation and reference.

.. code-block:: python

   import numpy as np

   reference = np.zeros(X_explain.shape[1])  # meaningful after standardization
   result = model.explain_integrated_gradients(
       X_explain,
       feature_names=feature_names,
       target=y_explain,
       reference_point=reference,
       num_steps=100,
       output="decision",
   )

   attribution_table = result.to_frame()
   global_ig_importance = result.importance

For a binary probability-enabled SVC, use ``output="probability"`` to explain
the positive-class member-set probability. The reference is part of the
question: zero is convenient for standardized data, while a real baseline or
cohort median may be more scientifically meaningful.

Triangulating evidence
----------------------

Use feature rank to describe selection, support vectors to identify boundary
anchors, perturbations to test discrete removal, and integrated gradients to
allocate local output differences. When they disagree, investigate feature
correlation, interactions, saturation, and reference choice rather than
averaging the methods into one unexplained number.
