from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ======================================================================
# INTERNAL STYLE HELPERS
# ======================================================================

def _bold_ticklabels(
    ax,
    size: int = 12,
) -> None:
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(size)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(size)


def _thicken_frame(
    ax,
    linewidth: float = 2.4,
) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
        spine.set_color("black")


def _viridis_triplet() -> tuple:
    cmap = plt.get_cmap("viridis")
    return (
        cmap(0.72),  # mean stability
        cmap(0.48),  # worst-case stability
        cmap(0.22),  # cross-dimensional agreement
    )


def _double_outline_points(
    ax,
    x,
    y,
    color,
    size_outer: float = 320,
    size_mid: float = 245,
    size_inner: float = 170,
    alpha: float = 0.95,
    zorder: int = 5,
):
    ax.scatter(
        x,
        y,
        s=size_outer,
        facecolors="none",
        edgecolors="black",
        linewidths=2.0,
        zorder=zorder,
    )
    ax.scatter(
        x,
        y,
        s=size_mid,
        facecolors="none",
        edgecolors="white",
        linewidths=1.6,
        zorder=zorder + 1,
    )
    ax.scatter(
        x,
        y,
        s=size_inner,
        c=[color],
        edgecolors="none",
        alpha=alpha,
        zorder=zorder + 2,
    )


# ======================================================================
# PUBLIC API
# ======================================================================

def plot_pca_variance_explained(
    dimensionality,
    *,
    selected_dimension: int,
    output_path: str | Path,
    config,
    plausible_dimension_min: int | None = None,
    plausible_dimension_max: int | None = None,
) -> Path:
    """Plot cumulative PCA variance explained using MorphoGlia stability style."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(dimensionality, pd.DataFrame):
        required_columns = {"component", "cumulative_variance"}
        missing = required_columns - set(dimensionality.columns)
        if missing:
            raise ValueError(
                "PCA variance plot requires dimensionality columns: "
                f"{sorted(missing)}"
            )

        table = (
            dimensionality[["component", "cumulative_variance"]]
            .copy()
            .sort_values("component", kind="mergesort")
        )
        components = table["component"].to_numpy(dtype=int)
        cumulative = table["cumulative_variance"].to_numpy(dtype=float)

    else:
        cumulative = np.asarray(
            getattr(dimensionality, "cumulative_variance"),
            dtype=float,
        )
        components = np.arange(1, len(cumulative) + 1, dtype=int)

        if plausible_dimension_min is None:
            plausible_dimension_min = getattr(
                dimensionality,
                "plausible_dimension_min",
                None,
            )
        if plausible_dimension_max is None:
            plausible_dimension_max = getattr(
                dimensionality,
                "plausible_dimension_max",
                None,
            )

    if cumulative.ndim != 1 or len(cumulative) == 0:
        raise ValueError(
            "PCA cumulative variance must be a non-empty 1D array."
        )
    if not np.isfinite(cumulative).all():
        raise ValueError(
            "PCA cumulative variance contains non-finite values."
        )

    cumulative_percent = (
        cumulative * 100.0
        if float(np.nanmax(cumulative)) <= 1.000001
        else cumulative.copy()
    )

    selected_dimension = int(selected_dimension)
    n_components = len(components)

    if selected_dimension < 1 or selected_dimension > n_components:
        raise ValueError(
            "Selected PCA dimensionality is outside the fitted PCA range: "
            f"d={selected_dimension}, available=1..{n_components}."
        )

    selected_variance = float(
        cumulative_percent[selected_dimension - 1]
    )

    cmap = plt.get_cmap("viridis")
    curve_color = cmap(0.58)
    selected_color = cmap(0.82)
    interval_color = cmap(0.30)
    line_width = float(
        getattr(config, "effective_line_width", 3.2)
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.8))

    interval_is_valid = (
        plausible_dimension_min is not None
        and plausible_dimension_max is not None
    )

    if interval_is_valid:
        lower = max(1, int(plausible_dimension_min))
        upper = min(n_components, int(plausible_dimension_max))
        interval_is_valid = lower <= upper

        if interval_is_valid:
            ax.axvspan(
                lower - 0.35,
                upper + 0.35,
                color=interval_color,
                alpha=0.10,
                linewidth=0,
                zorder=0,
            )

    ax.plot(
        components,
        cumulative_percent,
        color=curve_color,
        linewidth=line_width,
        alpha=0.86,
        zorder=2,
    )

    _double_outline_points(
        ax=ax,
        x=components,
        y=cumulative_percent,
        color=curve_color,
        size_outer=155,
        size_mid=112,
        size_inner=74,
        alpha=0.94,
        zorder=4,
    )

    _double_outline_points(
        ax=ax,
        x=np.asarray([selected_dimension]),
        y=np.asarray([selected_variance]),
        color=selected_color,
        size_outer=430,
        size_mid=320,
        size_inner=215,
        alpha=1.0,
        zorder=9,
    )

    # Selected dimensionality annotation.
    # Place it clearly to the right of the selected point while keeping
    # it vertically aligned with the selected cumulative variance.
    annotation_x = (
        selected_dimension
        + max(
            6.0,
            0.10 * n_components,
        )
    )

    ax.annotate(
        (
            f"Selected d = {selected_dimension}\n"
            f"{selected_variance:.1f}% cumulative"
        ),
        xy=(selected_dimension, selected_variance),
        xytext=(annotation_x, selected_variance),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="black",
        zorder=12,
    )

    ax.set_xlabel(
        "Number of PCA Dimensions",
        fontsize=16,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel(
        "Cumulative Variance Explained (%)",
        fontsize=15,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_title(
        "PCA Cumulative Variance Explained",
        fontsize=22,
        fontweight="bold",
        pad=44,
    )

    ax.set_xlim(0.5, n_components + 0.5)
    ax.set_ylim(0.0, 102.0)

    if n_components <= 12:
        ticks = components
    else:
        ticks = np.unique(
            np.concatenate(
                [
                    np.linspace(1, n_components, 9, dtype=int),
                    np.asarray([selected_dimension], dtype=int),
                ]
            )
        )

    ax.set_xticks(ticks)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(False)

    _bold_ticklabels(ax, size=14)
    _thicken_frame(ax, linewidth=2.2)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=curve_color,
            lw=line_width,
            label="Cumulative variance",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=11,
            markerfacecolor=selected_color,
            markeredgecolor="black",
            markeredgewidth=1.8,
            label=f"Selected d = {selected_dimension}",
        ),
    ]

    if interval_is_valid:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=interval_color,
                lw=8,
                alpha=0.22,
                label=f"Plausible interval {lower}–{upper}",
            )
        )

    legend = ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.6, 1.012),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=13,
        handlelength=2.4,
        columnspacing=1.6,
        handletextpad=0.7,
    )

    for label in legend.get_texts():
        label.set_fontweight("bold")

    fig.tight_layout()
    
    
    fig.savefig(
        output_path,
        dpi=config.effective_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def plot_stability_heatmap(
    clustering,
    output_path: str | Path,
    config,
) -> Path:
    """
    Plot stability across PCA dimensions as a square heatmap.

    - No numbers inside the cells.
    - Colorbar height matches the heatmap automatically.
    - White grid between squares.
    - Thick black outer frame.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = getattr(
        clustering,
        "stability_by_dimension_and_k",
        None,
    )

    if table is None or table.empty:
        raise ValueError(
            "Clustering result does not contain "
            "'stability_by_dimension_and_k'."
        )

    pivot = (
        table.pivot(
            index="pca_dimensions",
            columns="k",
            values="mean_subsample_ari",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    n_rows, n_cols = pivot.shape

    fig_width = max(7.5, 0.95 * n_cols + 2.8)
    fig_height = max(5.0, 0.92 * n_rows + 2.0)

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
    )

    image = ax.imshow(
        pivot.to_numpy(),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks(
        np.arange(n_cols),
        labels=[str(x) for x in pivot.columns],
    )
    ax.set_yticks(
        np.arange(n_rows),
        labels=[str(x) for x in pivot.index],
    )

    ax.set_xlabel(
        "Number of Morphology States",
        fontsize=17,
        fontweight="bold",
    )
    ax.set_ylabel(
        "PCA Dimensions",
        fontsize=17,
        fontweight="bold",
    )
    ax.set_title(
        "Morphology-State Stability Across PCA Dimensions",
        fontsize=21,
        fontweight="bold",
        pad=14,
    )

    # White grid between squares
    ax.set_xticks(
        np.arange(-0.5, n_cols, 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, n_rows, 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.2,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    _bold_ticklabels(
        ax,
        size=13,
    )
    _thicken_frame(
        ax,
        linewidth=2.2,
    )

    # Colorbar with automatic height = heatmap height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right",
        size="4%",
        pad=0.14,
    )

    cbar = fig.colorbar(
        image,
        cax=cax,
    )
    cbar.set_label(
        "Mean Subsampling ARI",
        fontsize=16,
        fontweight="bold",
    )
    cbar.ax.tick_params(
        labelsize=12,
        width=1.6,
    )
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    cbar.outline.set_linewidth(1.8)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=config.effective_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def plot_multiresolution_stability(
    clustering,
    output_path: str | Path,
    config,
) -> Path:
    """
    Plot multiresolution stability summary using viridis-based colors.

    Curves:
    - mean_subsample_ari
    - minimum_subsample_ari
    - mean_dimension_agreement
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = getattr(
        clustering,
        "multiresolution_summary",
        None,
    )

    if summary is None or summary.empty:
        raise ValueError(
            "Clustering result does not contain "
            "'multiresolution_summary'."
        )

    summary = summary.sort_values("k").reset_index(drop=True)

    x = summary["k"].to_numpy()
    y_mean = summary["mean_subsample_ari"].to_numpy()
    y_min = summary["minimum_subsample_ari"].to_numpy()
    y_agree = summary["mean_dimension_agreement"].to_numpy()

    color_mean, color_min, color_agree = _viridis_triplet()

    fig, ax = plt.subplots(
        figsize=(10.8, 6.8),
    )

    line_width = 3.2
    alpha = 0.82

    ax.plot(
        x,
        y_mean,
        color=color_mean,
        linewidth=line_width,
        alpha=alpha,
        zorder=2,
    )
    ax.plot(
        x,
        y_min,
        color=color_min,
        linewidth=line_width,
        alpha=alpha,
        zorder=2,
    )
    ax.plot(
        x,
        y_agree,
        color=color_agree,
        linewidth=line_width,
        alpha=alpha,
        zorder=2,
    )

    _double_outline_points(
        ax=ax,
        x=x,
        y=y_mean,
        color=color_mean,
    )
    _double_outline_points(
        ax=ax,
        x=x,
        y=y_min,
        color=color_min,
    )
    _double_outline_points(
        ax=ax,
        x=x,
        y=y_agree,
        color=color_agree,
    )

    ax.set_xlabel(
        "Number of Morphology States",
        fontsize=20,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel(
        "Stability",
        fontsize=20,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_title(
        "Multiresolution Stability of Morphology States",
        fontsize=23,
        fontweight="bold",
        pad=44,
    )

    ax.set_xlim(
        x.min() - 0.45,
        x.max() + 0.45,
    )
    ax.set_ylim(
        0.0,
        1.02,
    )
    ax.set_xticks(x)

    _bold_ticklabels(
        ax,
        size=14,
    )
    _thicken_frame(
        ax,
        linewidth=2.2,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_mean,
            lw=line_width,
            label="Mean stability",
        ),
        Line2D(
            [0],
            [0],
            color=color_min,
            lw=line_width,
            label="Worst-case stability",
        ),
        Line2D(
            [0],
            [0],
            color=color_agree,
            lw=line_width,
            label="Cross-dimensional agreement",
        ),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.012),
        ncol=3,
        frameon=False,
        fontsize=13,
        handlelength=2.4,
        columnspacing=1.6,
        handletextpad=0.7,
    )


    
    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=config.effective_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path
