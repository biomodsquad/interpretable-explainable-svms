Installation and setup
======================

Requirements
------------

MISTIC supports Python 3.10 and newer. The distribution is named
``mistic-svm`` on PyPI; the Python package is imported as ``mistic``.

Create an isolated environment, then install the package:

.. code-block:: console

   $ python -m venv .venv
   $ source .venv/bin/activate        # Windows: .venv\Scripts\activate
   $ python -m pip install --upgrade pip
   $ python -m pip install mistic-svm

Confirm the installation:

.. code-block:: python

   import mistic
   print(mistic.__version__)

Development installation
------------------------

Clone the canonical repository and install the package in editable mode:

.. code-block:: console

   $ git clone https://github.com/biomodsquad/interpretable-explainable-svms.git
   $ cd interpretable-explainable-svms
   $ python -m pip install -e ".[dev]"
   $ pytest

The ``dev`` extra installs the test, lint, security, build, and documentation
tooling used by continuous integration.

Data preparation
----------------

MISTIC accepts NumPy-compatible two-dimensional feature arrays and
one-dimensional target arrays. Split off blind data *before* fitting any
transformer. Fit scaling and preprocessing only on development data.

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler

   X_dev_raw, X_blind_raw, y_dev, y_blind = train_test_split(
       X, y, test_size=0.2, stratify=y, random_state=7
   )
   scaler = StandardScaler().fit(X_dev_raw)
   X_dev = scaler.transform(X_dev_raw)
   X_blind = scaler.transform(X_blind_raw)

Minimal classification workflow
-------------------------------

MISTIC computes kernels itself and therefore expects a scikit-learn estimator
configured with ``kernel="precomputed"``.

.. code-block:: python

   from sklearn.svm import SVC
   from mistic import cvSet, kernelWrapper, paramSet, score_svc, svmSet

   splits = cvSet(X_dev, y_dev)
   splits.classification(num_sets=5, validation_size=0.2, random_seed=7)

   estimator = SVC(
       kernel="precomputed",
       class_weight="balanced",
       probability=True,
       random_state=7,
   )
   grid = [
       paramSet(model={"C": C}, kernel={"gamma": gamma})
       for C in (0.5, 2.0, 8.0)
       for gamma in (2**-7, 2**-4, 2**-1)
   ]

   model = svmSet(
       estimator,
       splits,
       score_method=score_svc(weight=0.5, calibration_weight=0.2).score,
       kernel=kernelWrapper("rbf"),
       separate_feature_sets=True,
       separate_parameters=True,
   )
   model.tune_models(grid)
   print(model.mean_performance())

Space-filling parameter search
------------------------------

For continuous hyperparameters, a Latin hypercube provides broader coverage
than a similarly sized regular grid. Positive scale parameters such as ``C``
and ``gamma`` should generally be sampled logarithmically.

.. code-block:: python

   from mistic import loguniform, parameterSpace

   grid = parameterSpace(
       model={"C": loguniform(1e-3, 1e3)},
       kernel={"gamma": loguniform(1e-6, 1e1)},
   ).sample(
       n_trials=24,
       strategy="latin_hypercube",
       random_state=7,
   )

   model.tune_models(grid)

Use ``uniform`` for linear continuous ranges, ``integer`` for inclusive
integer ranges, and ``categorical`` for discrete choices. Plain values in a
parameter space are held fixed. Discrete duplicate candidates are removed,
so the returned list can contain fewer entries than ``n_trials``.

Independent mixed-kernel features
---------------------------------

A mixed kernel can select assignments independently, so removing a feature
from one base kernel does not remove it from another::

   from mistic import MixedKernel

   mixed = MixedKernel.weighted_sum(
       [
           kernelWrapper("linear", name="linear"),
           kernelWrapper("rbf", name="radial"),
       ],
       weight_parameters=["linear_weight", "radial_weight"],
   )
   model = svmSet(
       estimator,
       splits,
       score_method=score_svc().score,
       kernel=mixed,
       kernel_feature_selection="independent",
   )
   model.remove_kernel_features("radial", [2, 5])
   model.add_kernel_features("linear", [2])

Use ``greedy_forward_kernel_selection(grid, max_kernel_features=10,
num_initial_kernel_features=1, addition_factor=0.3)`` to select
``(kernel, feature)`` pairs automatically. The initial parameter controls the
first screened batch independently. After that, the addition factor is
computed from the remaining distance to the maximum pair count; zero adds
exactly one pair per later iteration. The resulting
``kernel_features`` mapping records the assignments, while ``features``
remains the union of original input columns for reporting and explanations.

Next steps
----------

Use :doc:`tutorials/feature_selection` to reduce the feature set,
:doc:`tutorials/explanations` to explain the fitted model, and
:doc:`tutorials/blind_predictions` before evaluating on held-out observations.
