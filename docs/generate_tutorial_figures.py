"""Generate deterministic visualization examples for the MISTIC tutorials."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mistic.explanations import IntegratedGradientsResult

plt.switch_backend("Agg")


OUTPUT_DIRECTORY = Path(__file__).parent / "_static" / "figures"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
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


def benchmark_results():
    """Load paired measured results for the five documented methods."""
    svm_results = pd.read_csv(
        REPOSITORY_ROOT / "validation" / "Synthetic100_rank09_vs_sklearn_10seeds_results.csv"
    )
    tree_results = pd.read_csv(
        REPOSITORY_ROOT / "validation" / "Synthetic100_tree_baselines_10seeds_results.csv"
    )
    labels = {
        "MiSTIC forward-knee (weight=0.90)": "MISTIC",
        "sklearn: all 100 features": "RBF SVC",
        "sklearn: linear-SVM RFE (20)": "SVM-RFE",
        "sklearn: random forest": "Random forest",
        "sklearn: histogram gradient boosting": "Boosted trees",
    }
    combined = pd.concat((svm_results, tree_results), ignore_index=True)
    combined = combined[combined["method"].isin(labels)].copy()
    combined["display_method"] = combined["method"].map(labels)
    combined["comparison_feature_count"] = combined["num_features"]

    # The original validation table stored MISTIC's mean per-member count in
    # ``num_features``. Its recovery statistics were calculated from the full
    # unified set (all sets contained fewer than the top-20 reporting cap), so
    # recovered-signal count divided by precision gives its true cardinality.
    mistic_rows = combined["display_method"].eq("MISTIC")
    recovered_signal_count = combined.loc[mistic_rows, "signal_recall"] * 20
    combined.loc[mistic_rows, "comparison_feature_count"] = np.rint(
        recovered_signal_count / combined.loc[mistic_rows, "signal_precision"]
    )
    return combined, list(labels.values())


def save_benchmark_figures():
    """Plot measured predictive and feature-recovery comparisons."""
    results, order = benchmark_results()
    colors = {
        "MISTIC": "#146b67",
        "RBF SVC": "#4979b8",
        "SVM-RFE": "#7256a8",
        "Random forest": "#c95f48",
        "Boosted trees": "#e4aa28",
    }
    positions = np.arange(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("roc_auc", "f1", "balanced_accuracy"),
        ("ROC AUC", "F1", "Balanced accuracy"),
    ):
        summary = results.groupby("display_method")[metric].agg(["mean", "std"]).reindex(order)
        for position, method in enumerate(order):
            ax.errorbar(
                summary.loc[method, "mean"],
                position,
                xerr=summary.loc[method, "std"],
                color=colors[method],
                marker="o",
                markersize=8,
                capsize=4,
                linewidth=2,
            )
        ax.set_title(title)
        ax.set_xlabel("Blind-set metric (mean ± SD)")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(positions, order)
    axes[0].invert_yaxis()
    fig.suptitle("Synthetic-100 predictive comparison across 10 inner seeds", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "synthetic-method-performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = (
        results.groupby("display_method")
        .agg(
            signal_recall=("signal_recall", "mean"),
            signal_precision=("signal_precision", "mean"),
            roc_auc=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            feature_count=("comparison_feature_count", "mean"),
            feature_count_sd=("comparison_feature_count", "std"),
        )
        .reindex(order)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        summary["signal_recall"],
        width,
        color="#146b67",
        label="Signal recall",
    )
    axes[0].bar(
        positions + width / 2,
        summary["signal_precision"],
        width,
        color="#7256a8",
        label="Signal precision",
    )
    axes[0].set_xticks(positions, order, rotation=24, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Known-signal recovery")
    axes[0].set_title("Selected or top-ranked feature set")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for method in order:
        axes[1].errorbar(
            summary.loc[method, "feature_count"],
            summary.loc[method, "roc_auc"],
            xerr=summary.loc[method, "feature_count_sd"],
            yerr=summary.loc[method, "roc_auc_sd"],
            color=colors[method],
            marker="o",
            markersize=9,
            capsize=4,
            linestyle="none",
            label=method,
        )
    axes[1].set_xlabel("Unified or model feature count (mean ± SD)")
    axes[1].set_ylabel("Blind ROC AUC (mean ± SD)")
    axes[1].set_title("Performance versus model size")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Predictive compactness and signal recovery", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "synthetic-method-recovery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def main():
    """Generate every static tutorial figure."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    save_selection_curve()
    save_explanation_figures(example_result())
    save_benchmark_figures()


if __name__ == "__main__":
    main()
