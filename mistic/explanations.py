"""Integrated-gradient results and visualizations."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm


@dataclass(frozen=True)
class IntegratedGradientsResult:
    """Values and metadata produced by an integrated-gradients explanation.

    Plot methods accept an existing Matplotlib ``ax`` and return the primary
    axes. Any additional Matplotlib keyword arguments can be supplied through
    ``scatter_kwargs`` or ``imshow_kwargs`` and the returned artists remain
    fully editable.

    Attributes
    ----------
    values : numpy.ndarray of shape (n_samples, n_features)
        Integrated-gradient attribution assigned to each input value.
    inputs : numpy.ndarray of shape (n_samples, n_features)
        Input values corresponding to the attribution matrix.
    feature_indices : numpy.ndarray of shape (n_features,)
        Column indices in the original model input.
    feature_names : tuple of str
        Display names corresponding to the attribution columns.
    reference_points : numpy.ndarray or None
        Baseline points used by the integration paths.
    model_indices : tuple of int
        Ensemble members included in the explanation.
    num_steps : int
        Number of numerical integration steps.
    target : numpy.ndarray or None
        Optional class labels or regression targets for sample annotation.
    """

    values: np.ndarray
    inputs: np.ndarray
    feature_indices: np.ndarray
    feature_names: tuple
    reference_points: np.ndarray | None
    model_indices: tuple
    num_steps: int
    target: np.ndarray | None = None

    def __post_init__(self):
        """Validate array shapes and normalize immutable result metadata.

        Returns
        -------
        None
        """
        values = np.asarray(self.values, dtype=float)
        inputs = np.asarray(self.inputs, dtype=float)
        indices = np.asarray(self.feature_indices, dtype=int)
        if values.ndim != 2 or inputs.shape != values.shape:
            raise ValueError("values and inputs must be equally shaped 2D arrays")
        if len(indices) != values.shape[1] or len(self.feature_names) != values.shape[1]:
            raise ValueError("feature metadata must match the attribution columns")
        if self.target is not None and len(self.target) != len(values):
            raise ValueError("target must contain one value per sample")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "feature_names", tuple(map(str, self.feature_names)))
        if self.target is not None:
            object.__setattr__(self, "target", np.asarray(self.target))

    def to_frame(self):
        """Return attributions as a labeled DataFrame.

        Returns
        -------
        pandas.DataFrame
            Attribution matrix with feature names as columns.
        """
        return pd.DataFrame(self.values, columns=self.feature_names)

    def _feature_position(self, feature):
        """Resolve a feature name or integer position to a column index.

        Parameters
        ----------
        feature : str, int, or None
            Feature name, result-column position, or ``None``.

        Returns
        -------
        int or None
            Resolved column position, or ``None`` when no feature was given.
        """
        if feature is None:
            return None
        if isinstance(feature, str):
            try:
                return self.feature_names.index(feature)
            except ValueError as exc:
                raise KeyError(f"unknown feature: {feature!r}") from exc
        position = int(feature)
        if not 0 <= position < self.values.shape[1]:
            raise IndexError("feature position is out of range")
        return position

    @property
    def importance(self):
        """Mean absolute attribution for each feature.

        Returns
        -------
        numpy.ndarray
            Mean absolute attribution for every feature column.
        """
        return np.mean(np.abs(self.values), axis=0)

    def interaction_scores(self):
        """Return a symmetric matrix of heuristic pairwise interaction scores.

        Each attribution is linearly residualized against its own feature
        value. Absolute residual/other-feature correlations are then averaged
        in both directions. This is intended to nominate plots, not to provide
        a statistical interaction test.

        Returns
        -------
        pandas.DataFrame
            Symmetric feature-by-feature interaction score matrix.
        """
        n_features = self.values.shape[1]
        directed = np.zeros((n_features, n_features), dtype=float)
        for i in range(n_features):
            design = np.column_stack((np.ones(len(self.inputs)), self.inputs[:, i]))
            fitted = design @ np.linalg.lstsq(design, self.values[:, i], rcond=None)[0]
            residual = self.values[:, i] - fitted
            residual_sd = np.std(residual)
            if residual_sd == 0:
                continue
            for j in range(n_features):
                if i != j and np.std(self.inputs[:, j]) > 0:
                    directed[i, j] = abs(np.corrcoef(residual, self.inputs[:, j])[0, 1])
        scores = (directed + directed.T) / 2
        np.fill_diagonal(scores, 0)
        return pd.DataFrame(scores, index=self.feature_names, columns=self.feature_names)

    def summary_plot(
        self,
        ax=None,
        max_features=None,
        jitter=0.22,
        cmap="coolwarm",
        random_state=0,
        scatter_kwargs=None,
    ):
        """Draw an attribution summary (beeswarm-style) plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default=None
            Axes to draw on; a new axes is created when omitted.
        max_features : int or None, default=None
            Maximum number of highest-importance features to show.
        jitter : float, default=0.22
            Maximum vertical jitter applied to each sample point.
        cmap : str or matplotlib.colors.Colormap, default="coolwarm"
            Colormap used for feature values.
        random_state : int, default=0
            Seed controlling deterministic point jitter.
        scatter_kwargs : dict or None, default=None
            Additional keyword arguments passed to ``Axes.scatter``.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the summary plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        count = (
            self.values.shape[1]
            if max_features is None
            else min(max_features, self.values.shape[1])
        )
        order = np.argsort(self.importance)[-count:]
        rng = np.random.default_rng(random_state)
        options = {"s": 24, "alpha": 0.75, **(scatter_kwargs or {})}
        artist = None
        for row, feature in enumerate(order):
            artist = ax.scatter(
                self.values[:, feature],
                row + rng.uniform(-jitter, jitter, len(self.values)),
                c=self.inputs[:, feature],
                cmap=cmap,
                **options,
            )
        ax.set_yticks(range(count), [self.feature_names[i] for i in order])
        ax.set_xlabel("Integrated gradient")
        ax.set_ylabel("Feature")
        if artist is not None:
            ax.figure.colorbar(artist, ax=ax, label="Feature value")
        return ax

    def heatmap(
        self,
        ax=None,
        target=None,
        cmap="coolwarm",
        center=0.0,
        cluster=False,
        imshow_kwargs=None,
        target_cmap=None,
        attribution_colorbar_kwargs=None,
        target_colorbar_kwargs=None,
        target_strip_width=0.10,
        strip_pad=0.04,
        colorbar_width=0.16,
        colorbar_pad=0.08,
        colorbar_gap=0.06,
        dendrogram_width=0.75,
        dendrogram_pad=0.04,
        dendrogram_linewidth=0.8,
        dendrogram_kwargs=None,
    ):
        """Draw an attribution heatmap with a class/target annotation bar.

        ``cluster=True`` hierarchically orders samples and displays their row
        dendrogram. With ``cluster=False``, samples are sorted by the supplied
        or stored target (and retain input order when no target is available).
        In both modes, features are sorted from greatest to least mean absolute
        attribution. Discrete targets receive a categorical colorbar;
        continuous values receive a continuous one.
        The strip widths and gaps are measured in inches, so their spacing is
        independent of figure size. The two ``*_colorbar_kwargs`` mappings are
        passed to :meth:`matplotlib.figure.Figure.colorbar`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default=None
            Main heatmap axes, created automatically when omitted.
        target : array-like or None, default=None
            Sample annotation overriding :attr:`target`.
        cmap, target_cmap : str or matplotlib.colors.Colormap
            Attribution and target-strip colormaps, respectively.
        center : float, default=0.0
            Center of the symmetric attribution color scale.
        cluster : bool, default=False
            Whether to cluster samples hierarchically.
        imshow_kwargs : dict or None, default=None
            Extra keyword arguments passed to ``Axes.imshow``.
        attribution_colorbar_kwargs, target_colorbar_kwargs : dict or None
            Extra keyword arguments for the two colorbars.
        target_strip_width, strip_pad, colorbar_width, colorbar_pad, colorbar_gap : float
            Fixed layout dimensions in inches.
        dendrogram_width, dendrogram_pad, dendrogram_linewidth : float
            Dendrogram layout dimensions and line width.
        dendrogram_kwargs : dict or None, default=None
            Extra keyword arguments passed to SciPy's ``dendrogram``.

        Returns
        -------
        matplotlib.axes.Axes
            Main axes containing the attribution heatmap.
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

        if ax is None:
            _, ax = plt.subplots()
        values = self.values
        annotation = self.target if target is None else np.asarray(target)
        if annotation is not None and len(annotation) != len(values):
            raise ValueError("target must contain one value per sample")
        row_order = np.arange(len(values))
        column_order = np.argsort(self.importance, kind="stable")[::-1]
        row_linkage = None
        if cluster and len(values) > 1:
            row_linkage = linkage(values, method="average")
            row_order = leaves_list(row_linkage)
        elif annotation is not None:
            row_order = np.argsort(annotation, kind="stable")
        shown = values[np.ix_(row_order, column_order)]
        limit = np.nanmax(np.abs(shown - center))
        options = {"aspect": "auto", "interpolation": "nearest", **(imshow_kwargs or {})}
        image = ax.imshow(shown, cmap=cmap, vmin=center - limit, vmax=center + limit, **options)
        ax.set_xticks(
            range(len(column_order)), [self.feature_names[i] for i in column_order], rotation=90
        )
        # Clustered row numbers are not meaningful sample identifiers. Hiding
        # them also leaves a clean margin for the per-sample target strip.
        ax.set_yticks([])
        ax.set_ylabel("Sample")
        divider = make_axes_locatable(ax)
        target_colorbar_ax = None
        if annotation is not None:
            # Targets describe samples, so draw one color per heatmap row on
            # the sample axis rather than across the feature columns.
            strip = divider.append_axes(
                "left", size=Fixed(target_strip_width), pad=Fixed(strip_pad)
            )
            target_colorbar_ax = divider.append_axes(
                "right", size=Fixed(colorbar_width), pad=Fixed(colorbar_pad)
            )
            attribution_colorbar_ax = divider.append_axes(
                "right", size=Fixed(colorbar_width), pad=Fixed(colorbar_gap)
            )
        else:
            attribution_colorbar_ax = divider.append_axes(
                "right", size=Fixed(colorbar_width), pad=Fixed(colorbar_pad)
            )

        if row_linkage is not None:
            dendrogram_ax = divider.append_axes(
                "left", size=Fixed(dendrogram_width), pad=Fixed(dendrogram_pad)
            )
            dendrogram_options = {
                "orientation": "left",
                "no_labels": True,
                "color_threshold": 0,
                "above_threshold_color": "black",
                **(dendrogram_kwargs or {}),
            }
            dendrogram(row_linkage, ax=dendrogram_ax, **dendrogram_options)
            for collection in dendrogram_ax.collections:
                collection.set_linewidth(dendrogram_linewidth)
            # scipy places leaves at 5, 15, ... from bottom to top, whereas
            # imshow places its first row at the top. Reversing this axis keeps
            # every branch aligned with the corresponding heatmap sample.
            dendrogram_ax.set_ylim(len(values) * 10, 0)
            dendrogram_ax.set_axis_off()

        attribution_bar_options = {
            "label": "Integrated gradient",
            **(attribution_colorbar_kwargs or {}),
        }
        # Layout is controlled by fixed-size cax objects rather than the
        # figure-relative pad/fraction parameters accepted by colorbar().
        attribution_bar_options.pop("pad", None)
        attribution_bar_options.pop("fraction", None)
        ax.figure.colorbar(image, cax=attribution_colorbar_ax, **attribution_bar_options)

        if annotation is not None:
            ordered = np.asarray(annotation)[row_order]
            unique = np.unique(ordered)
            categorical = len(unique) <= min(10, max(2, len(ordered) // 5))
            if categorical:
                encoded = np.searchsorted(unique, ordered)
                chosen_cmap = target_cmap or "tab10"
                norm = BoundaryNorm(np.arange(len(unique) + 1) - 0.5, len(unique))
                target_image = strip.imshow(
                    encoded[:, np.newaxis], aspect="auto", cmap=chosen_cmap, norm=norm
                )
                target_bar_options = {
                    "ticks": np.arange(len(unique)),
                    "label": "Class",
                    **(target_colorbar_kwargs or {}),
                }
                target_bar_options.pop("pad", None)
                target_bar_options.pop("fraction", None)
                bar = ax.figure.colorbar(target_image, cax=target_colorbar_ax, **target_bar_options)
                bar.ax.set_yticklabels([str(value) for value in unique])
            else:
                target_image = strip.imshow(
                    ordered[:, np.newaxis], aspect="auto", cmap=target_cmap or "viridis"
                )
                target_bar_options = {
                    "label": "Target",
                    **(target_colorbar_kwargs or {}),
                }
                target_bar_options.pop("pad", None)
                target_bar_options.pop("fraction", None)
                ax.figure.colorbar(target_image, cax=target_colorbar_ax, **target_bar_options)
            strip.set_axis_off()
        return ax

    def interaction_plot(
        self, feature=None, interaction_feature=None, ax=None, cmap="viridis", scatter_kwargs=None
    ):
        """Plot attribution dependence for a specified or automatic pair.

        Parameters
        ----------
        feature : str, int, or None, default=None
            Feature shown on the horizontal axis; selected automatically when
            omitted.
        interaction_feature : str, int, or None, default=None
            Feature mapped to point color; selected automatically when omitted.
        ax : matplotlib.axes.Axes or None, default=None
            Axes to draw on; a new axes is created when omitted.
        cmap : str or matplotlib.colors.Colormap, default="viridis"
            Colormap used for the interaction feature.
        scatter_kwargs : dict or None, default=None
            Additional keyword arguments passed to ``Axes.scatter``.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the dependence plot.
        """
        if self.values.shape[1] < 2:
            raise ValueError("an interaction plot requires at least two features")
        first = self._feature_position(feature)
        second = self._feature_position(interaction_feature)
        scores = self.interaction_scores().to_numpy()
        if first is None and second is None:
            first, second = np.unravel_index(np.argmax(scores), scores.shape)
            if first == second:  # all scores are zero
                first, second = np.argsort(self.importance)[-2:]
        elif first is None:
            first = int(np.argmax(scores[:, second]))
        elif second is None:
            second = int(np.argmax(scores[first]))
        if first == second:
            raise ValueError("feature and interaction_feature must differ")
        if ax is None:
            _, ax = plt.subplots()
        options = {"s": 32, "alpha": 0.8, **(scatter_kwargs or {})}
        artist = ax.scatter(
            self.inputs[:, first],
            self.values[:, first],
            c=self.inputs[:, second],
            cmap=cmap,
            **options,
        )
        ax.set_xlabel(self.feature_names[first])
        ax.set_ylabel(f"Integrated gradient for {self.feature_names[first]}")
        ax.figure.colorbar(artist, ax=ax, label=self.feature_names[second])
        ax.set_title(f"Interaction score: {scores[first, second]:.3f}")
        return ax
