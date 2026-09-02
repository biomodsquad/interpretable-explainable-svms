Deriving MISTIC's feature-ranking metrics
=========================================

MISTIC ranks active groups for removal and inactive groups for addition. The
rank is a search heuristic; validation performance determines whether a
candidate subset is retained. This chapter derives each criterion from the
shared sparse SVM decision function.

Notation and a frozen-model perturbation
----------------------------------------

Let :math:`F` be the current feature set and :math:`F_g` the set after group
:math:`g` is removed (backward selection) or added (forward selection). For the
same fitted support vectors and kernel parameters, define

.. math::

   K_F=[K_F(\mathbf{x}_i,\mathbf{x}_j)]_{i,j\in SV},\qquad
   \Delta K_g=K_F-K_{F_g}.

The corresponding support-vector kernel column for a sample :math:`\mathbf x`
is :math:`\mathbf{k}_F(\mathbf x)`. MISTIC freezes the fitted dual
coefficients :math:`\boldsymbol\beta` and intercept while changing only the
features used to recompute the kernel. This is a sensitivity calculation, not
a refit.

Group-size normalization uses

.. math::

   s_g=\begin{cases}
   |g| & \text{per_feature},\\
   \sqrt{|g|} & \text{sqrt},\\
   1 & \text{none}.
   \end{cases}

Decision perturbation
---------------------

All supported models have
:math:`f_F(\mathbf x)=\boldsymbol\beta^{\mathsf T}\mathbf{k}_F(\mathbf x)+b_0`.
Because the frozen intercept cancels, the normalized decision perturbation is

.. math::

   \delta_g(\mathbf x)
   =\frac{f_F(\mathbf x)-f_{F_g}(\mathbf x)}{s_g}
   =\frac{\boldsymbol\beta^{\mathsf T}
   [\mathbf{k}_F(\mathbf x)-\mathbf{k}_{F_g}(\mathbf x)]}{s_g}.

This is exactly what ``decision_perturbation_`` computes. For SVC and SVR,
MISTIC summarizes it over ranking samples by squared energy,

.. math::

   D_g=\sum_{r=1}^{m}\delta_g(\mathbf x_r)^2.

Squaring prevents positive and negative local effects from cancelling. The
ordinal contribution rank is ascending in :math:`D_g`: low-energy active groups
are candidates for early removal, while high-energy inactive groups are
preferred for addition.

Probability perturbation
------------------------

For a binary probability-enabled SVC, Platt calibration maps the margin to the
second class's probability [#platt]_:

.. math::

   p(\mathbf x)=\frac{1}{1+\exp(Af_F(\mathbf x)+B)}.

Differentiating the sigmoid gives

.. math::

   \frac{\partial p}{\partial f}=-A\,p(1-p).

A first-order Taylor expansion therefore converts the exact frozen-margin
perturbation into an approximate probability perturbation:

.. math::

   \Delta p_g(\mathbf x)
   \approx -A\,p(\mathbf x)[1-p(\mathbf x)]\,\delta_g(\mathbf x).

``probability_perturbation_`` implements this chain-rule expression using the
fitted SVC's ``probA_`` and predicted positive-class probability. It is most
sensitive where :math:`p(1-p)` is large and naturally shrinks near probabilities
zero and one. MISTIC summarizes the bounded probability-scale effects by

.. math::

   P_g=\sum_{r=1}^{m}|\Delta p_g(\mathbf x_r)|,

rather than squaring them. This preserves a directly interpretable total
absolute probability sensitivity. It remains a local linear approximation;
for a large perturbation, recomputing the calibrated probability under the
perturbed kernel can differ from the Taylor estimate.

The dual-objective criterion
----------------------------

For SVC and SVR, define :math:`\boldsymbol\beta=\boldsymbol\alpha\odot\mathbf y`
or :math:`\boldsymbol\alpha-\boldsymbol\alpha^*`, respectively. Holding all
dual variables fixed, the kernel-dependent part of either maximization dual is

.. math::

   Q_F=-\frac12\boldsymbol\beta^{\mathsf T}K_F\boldsymbol\beta.

The exact frozen quadratic change would consequently be

.. math::

   Q_F-Q_{F_g}
   =-\frac12\boldsymbol\beta^{\mathsf T}\Delta K_g\boldsymbol\beta.

MISTIC 0.1.1 retains its original SVC/SVR **kernel-mass proxy** instead:

.. math::

   M_g=-\frac{\lVert\boldsymbol\beta\rVert^2}{2s_g}
       \sum_{i,j}(\Delta K_g)_{ij}.

Thus the current criterion scales the total change in support-vector kernel
mass by the dual-coefficient energy; it does not apply the pair-specific
:math:`\beta_i\beta_j` weights of the exact dual change. This distinction is
important when interpreting the value. MISTIC uses :math:`M_g` for ordinal
search ranking, not as a reported change in the optimized dual objective.

For a one-class SVM, whose minimized dual quadratic term is
:math:`\frac12\boldsymbol\alpha^{\mathsf T}K\boldsymbol\alpha`, MISTIC does use
the exact frozen quadratic magnitude:

.. math::

   O_g=\frac{1}{s_g}\left|
       \frac12\boldsymbol\alpha^{\mathsf T}\Delta K_g\boldsymbol\alpha
       \right|.

Why frozen coefficients?
------------------------

Refitting after every group perturbation would mix two effects: the information
removed from the kernel and the optimizer's ability to compensate by changing
coefficients, support vectors, and intercept. Freezing isolates immediate model
sensitivity and makes ranking much cheaper. The greedy selection loop then
fits or retunes the chosen subset and evaluates its cross-validated score, so a
ranking proposal is not accepted solely because its frozen metric looked good.

Combined ranking
----------------

Raw objective and sample metrics have incompatible units. ``combined_rank``
first converts them to zero-based ordinal ranks. Let :math:`r_{sample,g}` be
the decision- or probability-contribution rank and :math:`r_{objective,g}` the
objective-criterion rank. The consensus score is

.. math::

   c_g=w\,r_{sample,g}+(1-w)\,r_{objective,g}.

Here ``weight`` is the **sample-perturbation share**. ``weight=1`` uses only
decision/probability behavior, while ``weight=0`` uses only the objective
criterion. This corrects a common reading error: ``weight`` is not the
objective share.

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

For backward selection, low consensus ranks are removed first. For forward
selection, high consensus ranks are added first. Because ordinal conversion
discards effect-size spacing, a one-rank difference need not represent the same
numeric difference at every position.

One-class contribution ranking
------------------------------

One-class ranking adds the fitted coverage target. MISTIC constructs the
normalized ranking score

.. math::

   \widetilde f_g(\mathbf x)=f_F(\mathbf x)-\delta_g(\mathbf x).

When :math:`s_g=1`, this equals the frozen perturbed decision; with group-size
normalization it is deliberately a normalized sensitivity score. MISTIC then
computes

.. math::

   C_g=\frac1m\sum_r\mathbf 1[f_{F_g}(\mathbf x_r)\geq0]

and compares it with :math:`1-\nu`. Candidates are ordered first by whether
they retain this inlier-coverage target and then by the change in decision-value
dispersion. The ordering direction protects coverage during backward removal
and favors coverage-preserving candidates during forward addition. This
one-class contribution rank is blended with the quadratic-objective rank using
the same :math:`w` formula above.

Choosing and reporting a weight
-------------------------------

Treat ``weight`` as an analysis choice rather than a hidden tuning constant.
Compare a prespecified grid within development resampling, report the grid and
selection rule, and never choose it from blind-set performance. Also report the
ranking population, perturbation normalization, direction of search, and
whether probability or decision perturbations were used.

References
----------

.. [#platt] Platt, `Probabilistic outputs for support vector machines and
   comparisons to regularized likelihood methods
   <https://www.cs.cornell.edu/courses/cs678/2007sp/platt.pdf>`_,
   *Advances in Large Margin Classifiers* (1999).
