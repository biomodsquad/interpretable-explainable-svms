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

For a differentiable model output :math:`F`, input :math:`\mathbf{x}`, and
reference :math:`\mathbf{x}'`, the attribution to feature :math:`j` is
[#integratedgradients]_

.. math::

   \operatorname{IG}_j(\mathbf{x};\mathbf{x}')=
   (x_j-x_j')\int_0^1
   \frac{\partial F\!\left(\mathbf{x}'+
   \alpha(\mathbf{x}-\mathbf{x}')\right)}{\partial x_j}\,d\alpha.

The scalar :math:`\alpha` traces the straight path from the reference at zero
to the observation at one. Under the usual differentiability conditions, the
attributions satisfy the **completeness** property

.. math::

   \sum_{j=1}^{p}\operatorname{IG}_j(\mathbf{x};\mathbf{x}')
   =F(\mathbf{x})-F(\mathbf{x}').

MISTIC approximates each integral numerically with the configured
``num_steps``. The completeness residual is therefore a useful convergence
check: increase ``num_steps`` if the attribution sum is not sufficiently close
to the observed output difference.

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

Reference
---------

.. [#integratedgradients] Sundararajan, Taly, and Yan, `Axiomatic attribution
   for deep networks <https://proceedings.mlr.press/v70/sundararajan17a.html>`_,
   *Proceedings of Machine Learning Research* 70, 3319–3328 (2017).
