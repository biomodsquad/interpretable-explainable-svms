API reference
=============

The tutorials explain how the objects fit together. This page documents every
public class and function exposed by the package. Private methods beginning
with an underscore are implementation details unless a tutorial explicitly
identifies them as diagnostic tools.

Public package namespace
------------------------

These objects are available directly from :mod:`mistic`.

.. autosummary::
   :nosignatures:

   mistic.IntegratedGradientsResult
   mistic.BoundaryCounterfactualResult
   mistic.combined_rank
   mistic.cvSet
   mistic.kernelWrapper
   mistic.MixedKernel
   mistic.KernelFeatureSet
   mistic.KernelFeatureSelectionState
   mistic.parameterSpace
   mistic.uniform
   mistic.loguniform
   mistic.integer
   mistic.categorical
   mistic.paramSet
   mistic.perDiff
   mistic.score_ocsvm
   mistic.score_svc
   mistic.score_svr
   mistic.svmSet

API index
---------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Cross-validation
----------------

.. automodule:: mistic.cvSet
   :members:

SVM ensembles
-------------

.. automodule:: mistic.svmSet
   :members:

Explanation results
-------------------

.. automodule:: mistic.explanations
   :members:

Utilities
---------

.. automodule:: mistic.utility
   :members:

Mixed kernels
-------------

.. automodule:: mistic.mixed_kernel
   :members:

``mistic.MixedkernelWrapper`` is retained as a deprecated compatibility
constructor. New code should use :class:`mistic.MixedKernel`; see
:doc:`tutorials/mixed_kernels` for expression construction, tunable weights,
and kernel-specific feature sets.
