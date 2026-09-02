"""Generate deterministic visualization examples for the MISTIC tutorials."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mistic.explanations import IntegratedGradientsResult

plt.switch_backend("Agg")


OUTPUT_DIRECTORY = Path(__file__).parent / "_static" / "figures"
FEATURE_NAMES = (
    "Concave points",
    "Worst texture",
    "Worst radius",
    "Mean area",
    "Worst perimeter",
    "Mean concavity",
    "Radius SE",
    "Mean smoothness",
)


def example_result():
    """Create structured illustrative inputs and attributions."""
    rng = np.random.default_rng(12)
    sample_count = 56
    inputs = rng.normal(size=(sample_count, len(FEATURE_NAMES)))
    target = np.repeat(["B", "M"], sample_count // 2)
    class_direction = np.where(target == "M", 1.0, -1.0)

    values = 0.08 * rng.normal(size=inputs.shape)
    strengths = np.array([0.42, 0.31, 0.25, 0.20, 0.16, 0.12, 0.08, 0.05])
    values += inputs * strengths
    values[:, :4] += class_direction[:, np.newaxis] * strengths[:4] * 0.65
    # Add an interaction-like pattern for the dependence example.
    values[:, 0] += 0.22 * inputs[:, 0] * inputs[:, 1]

    return IntegratedGradientsResult(
        values=values,
        inputs=inputs,
        feature_indices=np.arange(len(FEATURE_NAMES)),
        feature_names=FEATURE_NAMES,
        reference_points=np.zeros_like(inputs),
        model_indices=(0, 1, 2),
        num_steps=100,
        target=target,
    )


def save_selection_curve():
    """Draw an illustrative feature-selection performance trajectory."""
    feature_count = np.array([30, 24, 19, 15, 12, 10, 8, 6, 4, 2])
    auc = np.array([0.962, 0.966, 0.970, 0.974, 0.978, 0.981, 0.979, 0.974, 0.958, 0.910])
    f1 = np.array([0.918, 0.923, 0.930, 0.936, 0.941, 0.946, 0.943, 0.935, 0.912, 0.846])
    score = 0.5 * (auc + f1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.25), sharex=True)
    for ax, values, label, color in zip(
        axes,
        (auc, f1, score),
        ("ROC AUC", "F1", "Combined score"),
        ("#146b67", "#c95f48", "#7256a8"),
    ):
        ax.plot(feature_count, values, marker="o", color=color, linewidth=2)
        best = int(np.argmax(values))
        ax.scatter(
            feature_count[best],
            values[best],
            s=90,
            facecolors="none",
            edgecolors="#e4aa28",
            linewidths=2.5,
            zorder=4,
        )
        ax.axvline(10, color="#536b70", linestyle="--", linewidth=1, alpha=0.65)
        ax.set_title(label)
        ax.set_xlabel("Number of selected features")
        ax.grid(alpha=0.22)
        ax.invert_xaxis()
    axes[0].set_ylabel("Development performance")
    fig.suptitle("Illustrative backward-selection trajectory", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "selection-curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_explanation_figures(result):
    """Draw heatmap, summary, and interaction examples."""
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    result.heatmap(
        ax=ax,
        cluster=True,
        cmap="coolwarm",
        target_strip_width=0.18,
        strip_pad=0.18,
        colorbar_width=0.14,
        colorbar_pad=0.16,
        colorbar_gap=0.65,
    )
    ax.set_title("Illustrative integrated-gradient profiles", fontweight="bold")
    fig.savefig(OUTPUT_DIRECTORY / "ig-heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    result.summary_plot(
        ax=ax,
        max_features=8,
        cmap="coolwarm",
        random_state=7,
        scatter_kwargs={"s": 28, "alpha": 0.72, "edgecolors": "none"},
    )
    ax.axvline(0, color="#536b70", linewidth=1)
    ax.set_title("Illustrative attribution summary", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "ig-summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    result.interaction_plot(
        feature="Concave points",
        interaction_feature="Worst texture",
        ax=ax,
        cmap="viridis",
        scatter_kwargs={
            "s": 40,
            "alpha": 0.78,
            "edgecolors": "white",
            "linewidths": 0.25,
        },
    )
    ax.axhline(0, color="#536b70", linewidth=1)
    ax.set_title("Illustrative local dependence and interaction", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "ig-interaction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    """Generate every static tutorial figure."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    save_selection_curve()
    save_explanation_figures(example_result())


if __name__ == "__main__":
    main()
