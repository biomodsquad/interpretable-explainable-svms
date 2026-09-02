Support vector machine foundations
==================================

Support vector machines construct a decision rule from a subset of training
observations called **support vectors**. This chapter derives the linear
maximum-margin classifier, introduces the kernel representation, and then gives
the primal, dual, and decision functions for the three model families supported
by MISTIC.

From a separating plane to maximum margin
-----------------------------------------

For binary labels :math:`y_i\in\{-1,+1\}`, a linear classifier uses

.. math::

   f(\mathbf{x}) = \mathbf{w}^{\mathsf T}\mathbf{x}+b,
   \qquad \widehat y = \operatorname{sign}(f(\mathbf{x})).

Multiplying :math:`(\mathbf{w},b)` by a positive constant does not change the
plane :math:`f(\mathbf{x})=0`. We therefore choose a canonical scaling in which
the nearest observations obey
:math:`y_i(\mathbf{w}^{\mathsf T}\mathbf{x}_i+b)=1`. The two supporting planes
are :math:`f(\mathbf{x})=+1` and :math:`f(\mathbf{x})=-1`. Their perpendicular
distance is :math:`2/\lVert\mathbf{w}\rVert`, so maximizing the margin is
equivalent to minimizing :math:`\frac12\lVert\mathbf{w}\rVert^2`.

.. raw:: html

   <div class="svm-schematic" role="img" aria-label="Two classes separated by a maximum-margin hyperplane. Dashed parallel lines mark the margins and gold rings identify support vectors.">
     <div class="margin left"></div><div class="hyperplane"></div><div class="margin right"></div>
     <span class="label boundary-label">decision boundary f(x) = 0</span>
     <span class="label margin-label">margin width 2 / ||w||</span>
     <span class="label negative-label">class −1</span><span class="label positive-label">class +1</span>
     <i class="point negative" style="left:10%;top:24%"></i><i class="point negative" style="left:18%;top:48%"></i>
     <i class="point negative" style="left:28%;top:18%"></i><i class="point negative support" style="left:36%;top:55%"></i>
     <i class="point negative" style="left:23%;top:76%"></i><i class="point positive support" style="left:55%;top:35%"></i>
     <i class="point positive" style="left:68%;top:22%"></i><i class="point positive" style="left:73%;top:55%"></i>
     <i class="point positive" style="left:84%;top:72%"></i><i class="point positive" style="left:79%;top:35%"></i>
   </div>
   <p class="schematic-caption">Gold rings mark support vectors: the observations that constrain the margin. Non-support vectors do not appear in the final kernel expansion.</p>

Perfect separation is often impossible. Slack variables
:math:`\xi_i\geq 0` permit margin violations, and :math:`C` controls their
penalty. This gives the soft-margin primal problem [#cortes]_:

.. math::

   \min_{\mathbf{w},b,\boldsymbol\xi}
   \frac12\lVert\mathbf{w}\rVert^2+C\sum_{i=1}^{n}\xi_i

.. math::

   \text{subject to}\quad
   y_i(\mathbf{w}^{\mathsf T}\phi(\mathbf{x}_i)+b)\geq 1-\xi_i,
   \qquad \xi_i\geq 0.

The map :math:`\phi` is written explicitly because the same derivation applies
in a transformed feature space.

Lagrange multipliers and the classification dual
-------------------------------------------------

Introduce multipliers :math:`\alpha_i\geq0` for the margin constraints and
:math:`\mu_i\geq0` for nonnegative slack. Stationarity of the Lagrangian gives

.. math::

   \mathbf{w}=\sum_i\alpha_i y_i\phi(\mathbf{x}_i),\qquad
   \sum_i\alpha_i y_i=0,\qquad 0\leq\alpha_i\leq C.

Substituting these conditions removes :math:`\mathbf w`, :math:`b`, and the
slack variables, yielding the SVC dual:

.. math::

   \max_{\boldsymbol\alpha}
   \left[
   \sum_i\alpha_i-rac12\sum_{i,j}\alpha_i\alpha_j y_i y_j
   K(\mathbf{x}_i,\mathbf{x}_j)
   \right]

.. math::

   \text{subject to}\quad 0\leq\alpha_i\leq C,
   \qquad \sum_i\alpha_i y_i=0.

Only observations with :math:`\alpha_i>0` are support vectors. If
:math:`\beta_i=\alpha_i y_i`, the decision function is

.. math::

   f_{\mathrm{SVC}}(\mathbf{x})=
   \sum_{i\in SV}\beta_i K(\mathbf{x}_i,\mathbf{x})+b.

The kernel trick
----------------

The dual uses transformed observations only through inner products
:math:`\phi(\mathbf{x}_i)^{\mathsf T}\phi(\mathbf{x}_j)`. A kernel evaluates
that inner product directly,

.. math::

   K(\mathbf{x}_i,\mathbf{x}_j)
   =\phi(\mathbf{x}_i)^{\mathsf T}\phi(\mathbf{x}_j),

without explicitly constructing the possibly high-dimensional coordinates.
The optimization remains a linear maximum-margin problem in feature space even
when its boundary is nonlinear in the original inputs. MISTIC computes these
kernel matrices so it can recompute them after adding or removing feature
groups.

Support vector regression
-------------------------

SVR replaces the classification margin with an :math:`\varepsilon`-insensitive
tube around a regression function [#svr]_. Deviations inside the tube have zero
loss. The primal is

.. math::

   \min_{\mathbf w,b,\boldsymbol\xi,\boldsymbol\xi^*}
   \frac12\lVert\mathbf w\rVert^2
   +C\sum_i(\xi_i+\xi_i^*)

subject to

.. math::

   y_i-f(\mathbf{x}_i)\leq\varepsilon+\xi_i,
   \quad
   f(\mathbf{x}_i)-y_i\leq\varepsilon+\xi_i^*,
   \quad \xi_i,\xi_i^*\geq0.

With one multiplier for each side of the tube, the dual becomes

.. math::

   \max_{\boldsymbol\alpha,\boldsymbol\alpha^*}
   \left[
   -\frac12(\boldsymbol\alpha-\boldsymbol\alpha^*)^{\mathsf T}
   K(\boldsymbol\alpha-\boldsymbol\alpha^*)
   -\varepsilon\sum_i(\alpha_i+\alpha_i^*)
   +\sum_i y_i(\alpha_i-\alpha_i^*)
   \right]

.. math::

   \text{subject to}\quad
   0\leq\alpha_i,\alpha_i^*\leq C,
   \qquad \sum_i(\alpha_i-\alpha_i^*)=0.

Writing :math:`\beta_i=\alpha_i-\alpha_i^*`, the prediction function is

.. math::

   f_{\mathrm{SVR}}(\mathbf{x})=
   \sum_{i\in SV}\beta_i K(\mathbf{x}_i,\mathbf{x})+b.

MISTIC's regression score combines squared Pearson correlation with
nonnegative R-squared and also reports root mean squared error. Interpret errors
and explanations in the units of the modeled target, including any target
transformation.

One-class SVM
-------------

A one-class SVM estimates a region containing most of an inlier distribution
[#oneclass]_. It separates mapped observations from the origin. Its primal is

.. math::

   \min_{\mathbf w,\rho,\boldsymbol\xi}
   \frac12\lVert\mathbf w\rVert^2
   +\frac{1}{\nu n}\sum_i\xi_i-\rho

.. math::

   \text{subject to}\quad
   \mathbf w^{\mathsf T}\phi(\mathbf{x}_i)\geq\rho-\xi_i,
   \qquad \xi_i\geq0.

The dual is

.. math::

   \min_{\boldsymbol\alpha}\frac12\boldsymbol\alpha^{\mathsf T}
   K\boldsymbol\alpha
   \quad\text{subject to}\quad
   0\leq\alpha_i\leq\frac{1}{\nu n},\quad\sum_i\alpha_i=1,

and the signed decision function is

.. math::

   f_{\mathrm{OCSVM}}(\mathbf{x})=
   \sum_{i\in SV}\alpha_iK(\mathbf{x}_i,\mathbf{x})-\rho.

MISTIC and scikit-learn use :math:`+1` for inliers and :math:`-1` for
novelties. Positive decision values lie on the learned inlier side. The choice
of inlier population changes the scientific question, preprocessing, fitted
region, and explanations.

Connecting the three models
---------------------------

All three decision functions have the common sparse form

.. math::

   f(\mathbf{x})=\boldsymbol\beta^{\mathsf T}
   \mathbf{k}(\mathbf{x})+b_0,

where :math:`\mathbf{k}(\mathbf{x})` contains kernels between support vectors
and :math:`\mathbf{x}`. The meanings of :math:`\boldsymbol\beta` and
:math:`b_0` differ by model, but the shared form is what permits MISTIC's
kernel perturbations and analytical gradients.

References
----------

.. [#cortes] Cortes and Vapnik, `Support-vector networks
   <https://doi.org/10.1007/BF00994018>`_, *Machine Learning* 20, 273–297
   (1995).
.. [#svr] Smola and Schölkopf, `A tutorial on support vector regression
   <https://doi.org/10.1023/B:STCO.0000035301.49549.88>`_, *Statistics and
   Computing* 14, 199–222 (2004).
.. [#oneclass] Schölkopf et al., `Estimating the support of a high-dimensional
   distribution <https://doi.org/10.1162/089976601750264965>`_, *Neural
   Computation* 13, 1443–1471 (2001).
