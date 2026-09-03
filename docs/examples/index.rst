Example notebooks
=================

The package includes six curated notebooks and their input datasets. After
installation, locate them without assuming a site-packages path:

.. code-block:: python

   from importlib.resources import files

   examples = files("mistic.examples")
   print(examples)

Run a notebook from a writable copy rather than editing the installed package.
The repository versions can also be opened directly on GitHub.

Classification
--------------

`Forward selection <https://github.com/biomodsquad/interpretable-explainable-svms/blob/main/mistic/examples/BreastCancer_classification_forward.ipynb>`_
   RBF classification, combined-rank forward selection, blind evaluation, and
   boundary-counterfactual and integrated-gradient plots.

`Backward selection <https://github.com/biomodsquad/interpretable-explainable-svms/blob/main/mistic/examples/BreastCancer_classification_backward.ipynb>`_
   Backward elimination across rank weights, with the same evaluation and
   boundary-counterfactual and attribution workflow for a direct comparison.

`Probability workflow <https://github.com/biomodsquad/interpretable-explainable-svms/blob/main/mistic/examples/BreastCancer_classification_probability.ipynb>`_
   Probability-enabled SVC tuning, Brier-aware scoring, calibrated blind
   probabilities, and probability integrated gradients.

`Additional visualizations <https://github.com/biomodsquad/interpretable-explainable-svms/blob/main/mistic/examples/BreastCancer_classification_extraPlots.ipynb>`_
   Pair plots, three-dimensional views, clustered attributions, and local
   feature-attribution relationships.

Regression
----------

`Boston housing regression <https://github.com/biomodsquad/interpretable-explainable-svms/blob/main/mistic/examples/BostonHousing_regression.ipynb>`_
   Target and feature transformations, linear/RBF/polynomial SVR comparison,
   and regression feature analysis.

One-class classification
------------------------

`Breast-cancer novelty detection <https://github.com/biomodsquad/interpretable-explainable-svms/blob/main/mistic/examples/BreastCancer_one_class.ipynb>`_
   Compares both choices of inlier class under controlled splits, with blind
   metrics, perturbation summaries, and integrated gradients.

Data files
----------

``wdbc.data`` supports the breast-cancer examples and
``boston-housing_train.csv`` supports the regression example. The datasets are
included for reproducibility; review their provenance and suitability before
using them beyond these demonstrations.

Synthetic model comparison
--------------------------

The :doc:`../tutorials/synthetic_benchmark` tutorial provides a reproducible
known-signal comparison between MISTIC, a plain RBF SVC, linear-SVM RFE followed
by an RBF SVC, random forests, and histogram gradient-boosted trees.
