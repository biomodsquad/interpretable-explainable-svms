Feature ranking in MISTIC
=========================

MISTIC ranks active groups for removal and inactive groups for addition. A rank
is a search heuristic: it prioritizes candidates, while validation performance
decides whether the resulting feature set is useful.

Objective-based ranking
-----------------------

``feature_importance_`` measures how the frozen dual objective changes when a
group is perturbed. This is model-specific and kernel-aware. It identifies
groups that support the current SVM geometry, but it does not describe whether
their effect raises or lowers a particular sample's output.

Decision-based ranking
----------------------

``decision_perturbation_`` measures each sample's signed decision change. The
ranking summarizes these changes across the selected sample population. It can
surface groups with consistent local influence even when their objective
change is modest.

Probability-based ranking
-------------------------

When a binary ``SVC`` has ``probability=True``, MISTIC can rank using changes in
positive-class probability. This evaluates influence on the calibrated output
that many downstream users consume.

Probability perturbations are not interchangeable with margin perturbations:
calibration is nonlinear, so the same decision change can have different
probability effects near and far from the boundary. Evaluate Brier loss and a
calibration curve alongside discrimination.

Combined ranking
----------------

``combined_rank`` converts objective and sample-level criteria to ordinal ranks
and blends those ranks:

.. math::

   r_{combined} = w\,r_{objective} + (1-w)\,r_{sample}

``weight`` is the objective-rank share. ``weight=1`` uses only objective rank;
smaller values give sample perturbations more influence.

.. code-block:: python

   from mistic import combined_rank

   ranker = combined_rank(
       weight=0.90,
       number_samples=100,
       random_seed=7,
   )

   model.greedy_forward_selection(
       parameter_grid=grid,
       feature_ranker=ranker.compute,
       set_for_rank="sample",
       addition_factor=0.1,
   )

Because the criteria are converted to ranks, the blend is not dominated merely
because one metric has larger numeric units. Rank differences still need not
represent equal effect-size differences.

Choosing and reporting a weight
-------------------------------

Treat ``weight`` as an analysis choice, not a hidden tuning constant. Compare a
small prespecified grid within development resampling, report the grid and
selection rule, and avoid choosing the weight from blind-set performance. The
forward and backward breast-cancer notebooks demonstrate performance curves
across several weights.
