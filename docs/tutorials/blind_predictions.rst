Blind predictions: best practices
=================================

A blind set estimates performance only when it has not influenced
preprocessing, parameter tuning, rank weighting, feature selection, stopping,
threshold selection, or visualization choices.

Freeze the workflow
-------------------

Before opening blind labels, record:

* inclusion and exclusion criteria;
* transformations fitted on development data;
* split seeds and resampling strategy;
* kernel and estimator parameter grids;
* ranking metric, rank weight, and feature-group normalization;
* selection direction and stopping rule;
* final prediction mode and classification threshold;
* primary performance metric and uncertainty procedure.

Generate predictions once
-------------------------

Apply the development-fitted transformer and final model without refitting:

.. code-block:: python

   X_blind = scaler.transform(X_blind_raw)
   predicted = model.predict(X_blind)
   decision = model.decision_function(X_blind)

   # Binary probability-enabled SVC:
   probability = model.predict_proba(X_blind)
   classes = model.unified_model_.classes_

For member-set sensitivity analysis, request it explicitly and keep it separate
from the prespecified primary output:

.. code-block:: python

   member_probability = model.predict_proba(
       X_blind, prediction_mode="set"
   )

Create an auditable export
--------------------------

.. code-block:: python

   import pandas as pd

   blind_results = pd.DataFrame({
       "sample_id": blind_ids,
       "observed": y_blind,
       "predicted": predicted,
       "decision": decision,
       f"P({classes[0]})": probability[:, 0],
       f"P({classes[1]})": probability[:, 1],
   })
   blind_results.to_csv("blind_predictions.csv", index=False)

Include a stable sample identifier, preserve class-column order from
``classes_``, and store package version, model configuration, preprocessing
parameters, and random seeds with the output.

Evaluate the right quantities
-----------------------------

For classification, report discrimination and threshold-dependent metrics;
for probabilities, also report Brier loss and calibration. For imbalanced data,
include precision, recall, and the confusion matrix rather than accuracy alone.
For regression, report error in meaningful units and inspect residuals across
the target range. For one-class models, report inlier and novelty performance
separately.

Explain without tuning
----------------------

It is valid to explain blind predictions after they are generated, but do not
use those explanations to revise the model and still call the same evaluation
blind. Any revision starts a new development cycle and requires a new untouched
test cohort.

Check for drift
---------------

Compare missingness, ranges, categorical levels, and feature distributions
between development and blind cohorts without changing the fitted workflow.
Out-of-range values or schema changes should trigger a documented data-quality
decision, not silent clipping or retraining.
