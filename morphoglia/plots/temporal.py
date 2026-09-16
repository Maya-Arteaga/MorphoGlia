from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

from ..display_order import (
    normalize_metadata_field,
    ordered_combinations,
    ordered_levels,
)

from .config import PlotConfig
from .style import cluster_display_order, mg_color


VINTAGE_BACKGROUND = "#FFFFFF"
VINTAGE_TEXT = "#202124"


def _timepoint_field(
    category_fields: Sequence[str],
) -> str | None:
    """Return the configured field carrying timepoint semantics."""

    matches = [
        str(field)
        for field in category_fields
        if normalize_metadata_field(field) == "timepoint"
    ]

    if len(matches) > 1:
        raise ValueError(
            "Category contains more than one field with timepoint semantics: "
            f"{matches}"
        )

    return matches[0] if matches else None


def _prepare_temporal_table(
    master: pd.DataFrame,
    clustering,
) -> tuple[pd.DataFrame, list[int]]:
    """Attach canonical human-facing morphology-state labels."""

    preferred_k = getattr(clustering, "preferred_k", None)
    if preferred_k is None:
        raise ValueError(
            "Timepoint Sankey requires preferred_k from the clustering decision layer."
        )

    preferred_k = int(preferred_k)
    solutions = getattr(clustering, "selected_solutions", None)

    if solutions is None or preferred_k not in solutions:
        raise ValueError(
            f"Preferred clustering solution K={preferred_k} is unavailable."
        )

    labels = np.asarray(
        solutions[preferred_k].labels,
        dtype=int,
    )

    table = master.copy().reset_index(drop=True)
    if labels.shape != (len(table),):
        raise ValueError(
            "Preferred clustering labels do not match the master-table row count."
        )

    ordered_raw, display_rank = cluster_display_order(
        master=table,
        labels=labels,
        complexity_column="Cell_area",
    )

    table["_temporal_state"] = np.asarray(
        [display_rank[int(raw)] for raw in labels],
        dtype=int,
    )

    states = list(range(1, len(ordered_raw) + 1))
    return table, states


def _group_combinations(
    table: pd.DataFrame,
    fields: tuple[str, ...],
    config: PlotConfig,
) -> list[tuple[str, ...]]:
    """Observed non-timepoint groups in the canonical display order."""

    if not fields:
        return [tuple()]

    observed = (
        table[list(fields)]
        .astype(str)
        .drop_duplicates()
        .to_numpy(dtype=str)
        .tolist()
    )

    return ordered_combinations(
        observed,
        field_names=fields,
        metadata_order=config.metadata_order,
        comparison_oriented=True,
    )


def _composition_matrix(
    table: pd.DataFrame,
    *,
    timepoint_field: str,
    timepoints: list[str],
    states: list[int],
) -> pd.DataFrame:
    """Percentage morphology-state composition at each sampled timepoint."""

    counts = pd.crosstab(
        table[timepoint_field].astype(str),
        table["_temporal_state"],
        dropna=False,
    )

    counts = counts.reindex(
        index=timepoints,
        columns=states,
        fill_value=0,
    )

    denominator = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(denominator, axis=0).fillna(0.0)


def _state_bounds(
    proportions: np.ndarray,
) -> list[tuple[float, float]]:
    """Stack C1..Cn from top to bottom on a normalized 0..1 axis."""

    proportions = np.asarray(proportions, dtype=float)
    total = float(np.nansum(proportions))

    if total <= 0:
        return [(0.0, 0.0) for _ in proportions]

    proportions = proportions / total
    top = 1.0
    bounds: list[tuple[float, float]] = []

    for value in proportions:
        bottom = top - float(value)
        bounds.append((bottom, top))
        top = bottom

    return bounds


def _ribbon_path(
    x0: float,
    x1: float,
    source: tuple[float, float],
    target: tuple[float, float],
) -> MplPath:
    """Smooth alluvial ribbon whose width may change between timepoints."""

    sy0, sy1 = source
    ty0, ty1 = target
    bend = (x1 - x0) * 0.42

    vertices = [
        (x0, sy1),
        (x0 + bend, sy1),
        (x1 - bend, ty1),
        (x1, ty1),
        (x1, ty0),
        (x1 - bend, ty0),
        (x0 + bend, sy0),
        (x0, sy0),
        (x0, sy1),
    ]

    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]

    return MplPath(vertices, codes)


def _draw_temporal_panel(
    axis,
    *,
    matrix: pd.DataFrame,
    timepoints: list[str],
    states: list[int],
    panel_title: str,
    config: PlotConfig,
) -> None:
    """
    Draw one Sankey-style temporal composition panel.

    The ribbons connect the SAME morphology state across sampled timepoints.
    Their changing width represents changing population composition. They are
    deliberately NOT interpreted as tracked single-cell state transitions.
    """

    axis.set_facecolor(VINTAGE_BACKGROUND)

    n_timepoints = len(timepoints)
    x_positions = np.arange(n_timepoints, dtype=float)
    node_width = 0.075

    bounds_by_time: list[list[tuple[float, float]]] = []

    for timepoint in timepoints:
        proportions = matrix.loc[timepoint, states].to_numpy(dtype=float)
        bounds_by_time.append(_state_bounds(proportions))

    # ------------------------------------------------------------------
    # RIBBONS
    # ------------------------------------------------------------------

    for time_index in range(n_timepoints - 1):
        x0 = x_positions[time_index] + node_width / 2
        x1 = x_positions[time_index + 1] - node_width / 2

        for state_index, state in enumerate(states):
            source = bounds_by_time[time_index][state_index]
            target = bounds_by_time[time_index + 1][state_index]

            if (source[1] - source[0]) <= 0 and (target[1] - target[0]) <= 0:
                continue

            path = _ribbon_path(x0, x1, source, target)
            color = mg_color(state - 1, n_colors=len(states))

            # Wide white under-stroke, then thin black outline.
            axis.add_patch(
                PathPatch(
                    path,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=5.5,
                    alpha=0.5,
                    zorder=1,
                    joinstyle="round",
                )
            )

            axis.add_patch(
                PathPatch(
                    path,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1.15,
                    alpha=0.92,
                    zorder=2,
                    joinstyle="round",
                )
            )

    # ------------------------------------------------------------------
    # TIMEPOINT NODES
    # ------------------------------------------------------------------

    for time_index, (x, timepoint) in enumerate(zip(x_positions, timepoints)):
        bounds = bounds_by_time[time_index]

        axis.text(
            x,
            1.045,
            str(timepoint).replace("_", "-"),
            ha="center",
            va="bottom",
            fontsize=config.effective_panel_title_size,
            fontweight="bold",
            color=VINTAGE_TEXT,
            zorder=10,
        )

        row_total = float(matrix.loc[timepoint, states].sum())
        if row_total <= 0:
            axis.text(
                x,
                0.5,
                "No data",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=VINTAGE_TEXT,
            )
            continue

        for state_index, state in enumerate(states):
            y0, y1 = bounds[state_index]
            height = y1 - y0

            if height <= 0:
                continue

            color = mg_color(state - 1, n_colors=len(states))

            axis.add_patch(
                Rectangle(
                    (x - node_width / 2, y0),
                    node_width,
                    height,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=2.0,
                    zorder=5,
                )
            )

            percentage = 100.0 * float(matrix.loc[timepoint, state])

            if height >= 0.0005:
                label = f"{percentage:.1f}%" if height >= 0.085 else f"{percentage:.0f}%"
                text = axis.text(
                    x,
                    (y0 + y1) / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="black",
                    zorder=9,
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=2.2, foreground="white"),
                        path_effects.Normal(),
                    ]
                )

    axis.set_xlim(-0.18, max(0.18, n_timepoints - 1 + 0.18))
    axis.set_ylim(-0.02, 1.10)
    axis.set_xticks([])
    axis.set_yticks([])

    for spine in axis.spines.values():
        spine.set_visible(False)

    if panel_title:
        axis.set_title(
            panel_title.replace("_", "-"),
            fontsize=config.effective_panel_title_size,
            fontweight="bold",
            color=VINTAGE_TEXT,
            pad=14,
        )


def plot_timepoint_sankey(
    master: pd.DataFrame,
    clustering,
    category_fields: Sequence[str],
    output_path: str | Path,
    config: PlotConfig,
) -> Path | None:
    """
    Generate a vintage Sankey-style alluvial whenever Category includes timepoint.

    Interpretation
    --------------
    This is a population-composition visualization. Ribbons connect the same
    morphology state across sampled timepoints and vary in width with that
    state's percentage. They do NOT claim that individual cells were tracked
    or transitioned between states.

    If additional Category fields are selected, one panel is produced for each
    observed non-timepoint combination (for example one panel per sex).
    """

    fields = tuple(str(field) for field in category_fields)
    timepoint_field = _timepoint_field(fields)

    if timepoint_field is None:
        return None

    if timepoint_field not in master.columns:
        raise ValueError(
            f"Configured timepoint field {timepoint_field!r} is missing from the master table."
        )

    table, states = _prepare_temporal_table(master, clustering)

    other_fields = tuple(
        field
        for field in fields
        if field != timepoint_field
    )

    missing = [
        field
        for field in other_fields
        if field not in table.columns
    ]
    if missing:
        raise ValueError(
            "Timepoint Sankey grouping fields are missing from the master table: "
            f"{missing}"
        )

    timepoints = ordered_levels(
        table[timepoint_field].astype(str).unique().tolist(),
        field_name=timepoint_field,
        metadata_order=config.metadata_order,
    )

    # A Sankey/alluvial needs at least two sampled timepoints.
    if len(timepoints) < 2:
        return None

    groups = _group_combinations(
        table,
        other_fields,
        config,
    )

    n_panels = max(1, len(groups))
    n_columns = min(3, n_panels)
    n_rows = int(ceil(n_panels / n_columns))

    fig_width = max(
        9.5,
        n_columns * max(7.0, 2.0 * len(timepoints) + 2.0),
    )
    fig_height = max(6.4, 5.7 * n_rows + 1.2)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    fig.patch.set_facecolor(VINTAGE_BACKGROUND)

    used = set()

    for panel_index, group in enumerate(groups):
        row = panel_index // n_columns
        column = panel_index % n_columns
        used.add((row, column))

        axis = axes[row, column]
        selection = np.ones(len(table), dtype=bool)

        for field, value in zip(other_fields, group):
            selection &= table[field].astype(str).eq(str(value)).to_numpy()

        subset = table.loc[selection].copy()
        matrix = _composition_matrix(
            subset,
            timepoint_field=timepoint_field,
            timepoints=timepoints,
            states=states,
        )

        panel_title = (
            "-".join(group)
            if group
            else ""
        )

        _draw_temporal_panel(
            axis,
            matrix=matrix,
            timepoints=timepoints,
            states=states,
            panel_title=panel_title,
            config=config,
        )

    for row in range(n_rows):
        for column in range(n_columns):
            if (row, column) not in used:
                axes[row, column].axis("off")
                axes[row, column].set_facecolor(VINTAGE_BACKGROUND)

    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=10,
            markerfacecolor=mg_color(state - 1, n_colors=len(states)),
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=f"C{state}",
        )
        for state in states
    ]

    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.91, 0.90),
        frameon=False,
        prop={
            "size": config.effective_legend_font_size,
            "weight": "bold",
        },
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.suptitle(
        "Morphology-State Composition Across Time Points",
        fontsize=config.effective_figure_title_size,
        fontweight="bold",
        color=VINTAGE_TEXT,
        y=0.985,
    )

    fig.text(
        0.5,
        0.955,
        (
            ""
        ),
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#6F6C67",
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.885,
        bottom=0.07,
        top=0.88,
        wspace=0.20,
        hspace=0.28,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=config.effective_dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)
    return output_path
