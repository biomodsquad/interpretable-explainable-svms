MISTIC
======

**Model Informed Feature Selection Through Importance and Contribution**

MISTIC is a Python framework for building interpretable support vector machine
ensembles. It connects reproducible cross-validation, kernel-aware feature
selection, perturbation ranking, calibrated probability analysis, and local
integrated-gradient explanations in one workflow.

.. raw:: html

   <div class="hero-actions">
     <a class="primary-action" href="getting_started.html">Install and start</a>
     <a class="secondary-action" href="tutorials/index.html">Explore tutorials</a>
     <a class="secondary-action" href="api.html">Browse the API</a>
   </div>

What you can do
---------------

**Select features with the model.** Run forward or backward selection using
rankings that combine changes in model objective with sample-level decision or
probability perturbations.

**Explain nonlinear SVMs.** Inspect support vectors, global feature ranks,
per-sample perturbations, gradients, and integrated gradients without replacing
the trained SVM with a surrogate model.

**Keep evaluation honest.** Reuse controlled cross-validation splits during
tuning and selection, then evaluate the frozen workflow once on untouched
blind data.

.. code-block:: python

   from sklearn.svm import SVC
   from mistic import cvSet, kernelWrapper, paramSet, score_svc, svmSet

   splits = cvSet(X_train, y_train)
   splits.classification(num_sets=5)

   model = svmSet(
       SVC(kernel="precomputed", probability=True),
       splits,
       score_svc().score,
       kernel=kernelWrapper("rbf"),
   )
   model.tune_models([
       paramSet(model={"C": 1.0}, kernel={"gamma": 0.1}),
   ])
   predictions = model.predict(X_blind)

Where to begin
--------------

New to SVMs? Start with :doc:`tutorials/svm_foundations`. Ready to build a
model? Follow :doc:`getting_started`, then choose a curated
:doc:`examples/index`. For the concepts behind MISTIC's ranking and
explanation workflow, see :doc:`framework`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   framework
   tutorials/index
   examples/index
   api
