Support vector machine foundations
==================================

An SVM learns a boundary whose position is controlled by a subset of training
observations called **support vectors**. The decision function is expressed in
terms of similarities between a new observation and those support vectors.
Kernel functions make nonlinear boundaries possible without explicitly
constructing every transformed feature.

Classification
--------------

A support vector classifier separates classes while balancing a wide margin
against training violations. ``C`` controls that tradeoff: large values
penalize violations more strongly; small values permit a softer margin.

.. code-block:: python

   from sklearn.svm import SVC

   classifier = SVC(
       kernel="precomputed",
       C=2.0,
       class_weight="balanced",
       probability=True,
   )

The signed decision value measures position relative to the boundary. For a
binary classifier, positive and negative signs correspond to the order in
``model.classes_``. A predicted probability is a calibrated transformation of
that score; it is useful for ranking risk but should be checked for calibration.

MISTIC's ``score_svc`` combines ROC AUC and F1. For probability-enabled SVCs it
can also reward calibration through ``1 - Brier loss``:

.. code-block:: python

   from mistic import score_svc

   scorer = score_svc(weight=0.5, calibration_weight=0.2)

Regression
----------

Support vector regression fits a function with an ``epsilon``-wide tube.
Errors inside the tube receive no penalty; observations outside it become
support vectors. ``C`` controls the cost of deviations and ``epsilon`` controls
the tube width.

.. code-block:: python

   from sklearn.svm import SVR
   from mistic import score_svr

   regressor = SVR(kernel="precomputed")
   scorer = score_svr(weight=0.5)

MISTIC's regression score combines squared Pearson correlation with
nonnegative R-squared; it also reports root mean squared error. Always interpret
regression explanations and errors in the units of the modeled target,
including any target transformation.

One-class classification
------------------------

A one-class SVM learns the region occupied by an **inlier** class. It does not
learn a conventional boundary between two labeled classes. Training uses
inliers only; known novelties can be retained for validation and blind
evaluation.

.. code-block:: python

   import numpy as np
   from sklearn.svm import OneClassSVM
   from mistic import cvSet, score_ocsvm

   # +1 means inlier and -1 means novelty.
   y_novelty = np.where(original_label == inlier_label, 1, -1)
   splits = cvSet(X_development, y_novelty)
   splits.one_class(num_sets=5, inlier_label=1)

   detector = OneClassSVM(kernel="precomputed", nu=0.05)
   scorer = score_ocsvm(weight=0.5)

Positive decision values indicate inliers and negative values indicate
novelties. ``nu`` jointly bounds the expected training-error fraction and
support-vector fraction. The chosen inlier population changes the scientific
question, preprocessing, and resulting explanations.

Kernel choices
--------------

* A **linear kernel** provides the most direct global geometry.
* An **RBF kernel** models local nonlinear similarity; ``gamma`` controls how
  quickly similarity decays.
* A **polynomial kernel** represents interactions up to its configured degree.

Tune kernel and estimator parameters within the same resampling workflow.
Comparing only training performance will favor overly flexible boundaries.
