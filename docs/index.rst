MISTIC
======

.. warning::

   This is the MISTIC 0.2 beta documentation. The 0.1 series remains the
   default stable release. Install this beta explicitly with
   ``python -m pip install "mistic-svm==0.2.0b1"``. Stable documentation is
   available at the `site root
   <https://biomodsquad.org/interpretable-explainable-svms/>`_ and under
   `/stable/ <https://biomodsquad.org/interpretable-explainable-svms/stable/>`_;
   beta documentation is published under
   `/beta/ <https://biomodsquad.org/interpretable-explainable-svms/beta/>`_.

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

**Combine complementary kernels.** Build named sums, products, and powers of
linear, RBF, polynomial, sigmoid, and Tanimoto kernels, tune their parameters
and weights, and optionally select a different feature set for every leaf.

**Explain nonlinear SVMs.** Inspect support vectors, global feature ranks,
per-sample perturbations, gradients, and integrated gradients without replacing
the trained SVM with a surrogate model.

**Keep evaluation honest.** Reuse controlled cross-validation splits during
tuning and selection, then evaluate the frozen workflow once on untouched
blind data.

.. code-block:: python

   from sklearn.svm import SVC
   from mistic import MixedKernel, cvSet, kernelWrapper, paramSet, score_svc, svmSet

   splits = cvSet(X_train, y_train)
   splits.classification(num_sets=5)

   kernel = MixedKernel.weighted_sum(
       [kernelWrapper("rbf", name="radial")], weights=[1.0]
   )

   model = svmSet(
       SVC(kernel="precomputed", probability=True),
       splits,
       score_svc().score,
       kernel=kernel,
       kernel_feature_selection="shared",
   )
   model.tune_models([
       paramSet(model={"C": 1.0}, kernel={"radial__gamma": 0.1}),
   ])
   predictions = model.predict(X_blind)

Where to begin
--------------

New to SVMs? Start with :doc:`tutorials/svm_foundations`. Ready to build a
model? Follow :doc:`getting_started`, then choose a curated
:doc:`examples/index`. For the concepts behind MISTIC's ranking and
explanation workflow, see :doc:`framework`. To combine kernels or give their
leaves different candidate feature pools, read
:doc:`tutorials/mixed_kernels`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   migration_0_2
   framework
   tutorials/index
   examples/index
   api
