from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..category import (
    CATEGORY_COLUMN,
)

from ..display_order import (
    ordered_combinations,
    ordered_levels,
)

from .config import (
    PlotConfig,
)

from .style import (
    LIGHT_GRAY,
    cluster_display_order,
    mg_color,
    style_axis,
)


# ======================================================================
# VALIDATION
# ======================================================================

def _validate_category_fields(
    master: pd.DataFrame,
    category_fields: Sequence[str],
) -> tuple[str, ...]:
    """
    Validate the Category -> Plots contract.

    Category layout is defined from the original metadata fields selected
    upstream by CategoryConfig.

    The combined reserved 'category' string is never parsed to infer
    structure.
    """

    fields = tuple(
        str(field)
        for field in category_fields
    )


    if not fields:

        raise ValueError(
            "Category plotting requires at least "
            "one category field."
        )


    if CATEGORY_COLUMN not in master.columns:

        raise ValueError(
            "Category plots require the reserved "
            f"{CATEGORY_COLUMN!r} column produced "
            "by CategoryStage."
        )


    missing = [
        field
        for field in fields
        if field not in master.columns
    ]


    if missing:

        raise ValueError(
            "Category plotting fields are missing "
            f"from the master table: {missing}"
        )


    for field in fields:

        if master[
            field
        ].isna().any():

            raise ValueError(
                f"Category field {field!r} "
                "contains missing values."
            )


    return fields


# ======================================================================
# CLUSTER DISPLAY
# ======================================================================

def _prepare_plot_table(
    master: pd.DataFrame,
    clustering,
    umap_embedding: np.ndarray,
) -> tuple[
    pd.DataFrame,
    int,
    dict[int, object],
]:
    """
    Prepare one common plotting table.

    Analytical GMM labels are converted only for DISPLAY using the exact
    same median-Cell_area ordering used by Morphology States.

    Therefore C1, C2, ... have identical meaning and identical colors
    everywhere in MorphoGlia.
    """

    preferred_k = getattr(
        clustering,
        "preferred_k",
        None,
    )


    if preferred_k is None:

        raise ValueError(
            "No preferred clustering resolution exists."
        )


    preferred_k = int(
        preferred_k
    )


    selected_solutions = getattr(
        clustering,
        "selected_solutions",
        None,
    )


    if (
        selected_solutions is None
        or preferred_k
        not in selected_solutions
    ):

        raise ValueError(
            "Preferred selected clustering "
            "solution is unavailable."
        )


    solution = (
        selected_solutions[
            preferred_k
        ]
    )


    d = int(
        solution
        .pca_dimensions
    )


    labels = np.asarray(
        solution.labels,
        dtype=int,
    )


    if labels.shape != (
        len(master),
    ):

        raise ValueError(
            "Preferred-solution labels do not "
            "match master-table row count."
        )


    table = (
        master
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ==================================================================
    # SAME DISPLAY ORDER AS MORPHOLOGY STATES
    # ==================================================================

    (
        ordered_clusters,
        display_rank,
    ) = cluster_display_order(
        master=table,
        labels=labels,
        complexity_column="Cell_area",
    )


    table[
        "_display_cluster"
    ] = np.asarray(
        [
            display_rank[
                int(
                    raw_cluster
                )
            ]
            for raw_cluster in labels
        ],
        dtype=int,
    )


    # ==================================================================
    # SHARED UMAP
    # ==================================================================

    embedding = np.asarray(
        umap_embedding,
        dtype=float,
    )


    if embedding.shape != (
        len(table),
        2,
    ):

        raise ValueError(
            "Shared UMAP embedding must have shape "
            f"({len(table)}, 2)."
        )


    table[
        "_UMAP1"
    ] = embedding[
        :,
        0,
    ]


    table[
        "_UMAP2"
    ] = embedding[
        :,
        1,
    ]


    # ==================================================================
    # CLUSTER COLORS
    # ==================================================================

    cluster_colors = {
        display_cluster:
            mg_color(int(display_cluster) - 1, n_colors=len(ordered_clusters))

        for display_cluster
        in range(
            1,
            len(
                ordered_clusters
            )
            + 1,
        )
    }


    return (
        table,
        d,
        cluster_colors,
    )


# ======================================================================
# CATEGORY LEVELS
# ======================================================================


def _field_levels(
    table: pd.DataFrame,
    field: str,
    config: PlotConfig,
) -> list[str]:
    """
    Canonical display order for one category field.
    """

    return ordered_levels(
        table[
            field
        ]
        .astype(str)
        .tolist(),
        field_name=field,
        metadata_order=(
            config
            .effective_metadata_order
        ),
    )


# ======================================================================
# PANEL LAYOUT
# ======================================================================


def _build_category_layout(
    table: pd.DataFrame,
    category_fields: tuple[str, ...],
    config: PlotConfig,
) -> tuple[
    int,
    int,
    list[dict],
]:
    """
    Build the canonical category layout.

    Every field uses the shared MorphoGlia metadata-ordering contract:
        explicit user order
        -> semantic automatic order
        -> alphabetical fallback

    The reserved combined category string is never parsed.
    """

    # ==================================================================
    # ONE FIELD
    # ==================================================================

    if len(
        category_fields
    ) == 1:

        field = (
            category_fields[
                0
            ]
        )

        levels = _field_levels(
            table,
            field,
            config,
        )

        values = (
            table[
                field
            ]
            .astype(str)
        )

        panels = [
            {
                "row": 0,
                "column": column,
                "title": level,
                "mask": (
                    values
                    .eq(level)
                    .to_numpy()
                ),
            }
            for column, level
            in enumerate(
                levels
            )
        ]

        return (
            1,
            len(levels),
            panels,
        )


    # ==================================================================
    # TWO FIELDS
    #
    # First selected field = rows.
    # Second selected field = columns.
    # ==================================================================

    if len(
        category_fields
    ) == 2:

        row_field = (
            category_fields[
                0
            ]
        )

        column_field = (
            category_fields[
                1
            ]
        )

        row_levels = _field_levels(
            table,
            row_field,
            config,
        )

        column_levels = _field_levels(
            table,
            column_field,
            config,
        )

        row_values = (
            table[
                row_field
            ]
            .astype(str)
        )

        column_values = (
            table[
                column_field
            ]
            .astype(str)
        )

        panels = []

        for row, row_level in enumerate(
            row_levels
        ):

            for (
                column,
                column_level,
            ) in enumerate(
                column_levels
            ):

                mask = (
                    row_values.eq(
                        row_level
                    )
                    & column_values.eq(
                        column_level
                    )
                )

                panels.append(
                    {
                        "row": row,
                        "column": column,
                        "title": (
                            f"{row_level}-"
                            f"{column_level}"
                        ),
                        "mask": mask.to_numpy(),
                    }
                )

        return (
            len(row_levels),
            len(column_levels),
            panels,
        )


    # ==================================================================
    # MORE THAN TWO FIELDS
    #
    # Flatten observed combinations using the same comparison-oriented
    # ordering used by composition tables and profile tables.
    # ==================================================================

    raw_combinations = [
        tuple(
            str(value)
            for value in row
        )
        for row in (
            table[
                list(
                    category_fields
                )
            ]
            .astype(str)
            .drop_duplicates()
            .to_numpy(
                dtype=str
            )
        )
    ]

    combinations = ordered_combinations(
        raw_combinations,
        field_names=(
            category_fields
        ),
        metadata_order=(
            config
            .effective_metadata_order
        ),
        comparison_oriented=True,
    )

    n_panels = len(
        combinations
    )

    n_columns = min(
        4,
        max(
            1,
            n_panels,
        ),
    )

    n_rows = int(
        ceil(
            n_panels
            / n_columns
        )
    )

    panels = []

    for index, combination in enumerate(
        combinations
    ):

        mask = np.ones(
            len(
                table
            ),
            dtype=bool,
        )

        for field, value in zip(
            category_fields,
            combination,
        ):

            mask &= (
                table[
                    field
                ]
                .astype(str)
                .eq(
                    value
                )
                .to_numpy()
            )

        panels.append(
            {
                "row": (
                    index
                    // n_columns
                ),
                "column": (
                    index
                    % n_columns
                ),
                "title": "-".join(
                    combination
                ),
                "mask": mask,
            }
        )

    return (
        n_rows,
        n_columns,
        panels,
    )


# ======================================================================
# CLUSTER LEGEND
# ======================================================================

def _cluster_legend_handles(
    cluster_colors: dict[
        int,
        object,
    ],
) -> list[Line2D]:

    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=9,
            markerfacecolor=(
                cluster_colors[
                    cluster
                ]
            ),
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=f"C{cluster}",
        )

        for cluster
        in sorted(
            cluster_colors
        )
    ]


# ======================================================================
# CATEGORY PANEL
# ======================================================================

def _draw_category_panel(
    axis,
    *,
    table: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    highlight_mask: np.ndarray,
    title: str,
    cluster_colors: dict[
        int,
        object,
    ],
    config: PlotConfig,
) -> None:
    """
    Draw one category panel.

    IMPORTANT
    ---------
    First draw ALL cells in light gray.

    Then overlay only cells belonging to the selected category using
    canonical morphology-state colors.

    This produces the visual background/reference cloud rather than
    removing the selected cells from the gray layer.
    """

    # ==================================================================
    # COMPLETE REFERENCE POPULATION
    # ==================================================================

    axis.scatter(
        x,
        y,
        s=(
            config
            .effective_point_size
        ),
        color=LIGHT_GRAY,
        alpha=(
            config
            .effective_background_alpha
        ),
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )


    # ==================================================================
    # SELECTED CATEGORY ON TOP
    # ==================================================================

    selected_clusters = (
        table.loc[
            highlight_mask,
            "_display_cluster",
        ]
        .to_numpy(
            dtype=int
        )
    )


    selected_x = x[
        highlight_mask
    ]


    selected_y = y[
        highlight_mask
    ]


    for cluster in sorted(
        cluster_colors
    ):

        cluster_mask = (
            selected_clusters
            == cluster
        )


        if not np.any(
            cluster_mask
        ):

            continue


        axis.scatter(
            selected_x[
                cluster_mask
            ],
            selected_y[
                cluster_mask
            ],
            s=(
                config
                .effective_point_size
            ),
            color=(
                cluster_colors[
                    cluster
                ]
            ),
            alpha=(
                config
                .effective_highlight_alpha
            ),
            edgecolors="black",
            linewidths=(
                config
                .effective_point_edge_width
            ),
            rasterized=True,
            zorder=3,
        )


    style_axis(
        axis,
        title=title.replace(
            "_",
            "-",
        ),
        xlabel="",
        ylabel="",
        config=config,
        hide_ticks=True,
    )


# ======================================================================
# GENERIC FIGURE
# ======================================================================

def _save_category_figure(
    *,
    table: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    category_fields: tuple[str, ...],
    cluster_colors: dict[
        int,
        object,
    ],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    config: PlotConfig,
) -> Path:

    (
        n_rows,
        n_columns,
        panels,
    ) = _build_category_layout(
        table=table,
        category_fields=(
            category_fields
        ),
        config=config,
    )


    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            6.0 * n_columns
            + 1.4,
            5.3 * n_rows
            + 1.2,
        ),
        squeeze=False,
    )


    # ==================================================================
    # DRAW PANELS
    # ==================================================================

    used_positions = set()


    for panel in panels:

        row = int(
            panel[
                "row"
            ]
        )


        column = int(
            panel[
                "column"
            ]
        )


        used_positions.add(
            (
                row,
                column,
            )
        )


        _draw_category_panel(
            axes[
                row,
                column,
            ],
            table=table,
            x=x,
            y=y,
            highlight_mask=(
                panel[
                    "mask"
                ]
            ),
            title=(
                panel[
                    "title"
                ]
            ),
            cluster_colors=(
                cluster_colors
            ),
            config=config,
        )


    # ==================================================================
    # UNUSED AXES
    # ==================================================================

    for row in range(
        n_rows
    ):

        for column in range(
            n_columns
        ):

            if (
                row,
                column,
            ) not in used_positions:

                axes[
                    row,
                    column,
                ].axis(
                    "off"
                )


    # ==================================================================
    # SHARED TYPOGRAPHY
    # ==================================================================

    fig.suptitle(
        title,
        fontsize=(
            config
            .effective_figure_title_size
        ),
        fontweight="bold",
        y=0.97,
    )


    fig.supxlabel(
        xlabel,
        fontsize=(
            config
            .effective_axis_label_size
        ),
        fontweight="bold",
        y=0.045,
    )


    fig.supylabel(
        ylabel,
        fontsize=(
            config
            .effective_axis_label_size
        ),
        fontweight="bold",
        x=0.035,
    )


    # ==================================================================
    # SINGLE FIGURE-LEVEL LEGEND
    #
    # Dedicated margin = never overlays an axis.
    # ==================================================================

    legend = fig.legend(
        handles=(
            _cluster_legend_handles(
                cluster_colors
            )
        ),
        loc="upper left",
        bbox_to_anchor=(
            0.875,
            0.86,
        ),
        frameon=False,
        borderaxespad=0.0,
        prop={
            "size":
                config
                .effective_legend_font_size,

            "weight":
                "bold",
        },
    )


    for text in (
        legend
        .get_texts()
    ):

        text.set_fontweight(
            "bold"
        )


    # Reserve a real column on the right for the legend.
    fig.subplots_adjust(
        left=0.07,
        right=0.84,
        bottom=0.11,
        top=0.84,
        wspace=0.10,
        hspace=0.24,
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    fig.savefig(
        output_path,
        dpi=(
            config
            .effective_dpi
        ),
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    return output_path


# ======================================================================
# PUBLIC API
# ======================================================================

def plot_category_states(
    master: pd.DataFrame,
    clustering,
    umap_embedding: np.ndarray,
    output_dir: str | Path,
    category_fields: Sequence[str],
    config: PlotConfig,
) -> dict[str, Path]:
    """
    Generate the canonical category-highlighted morphology-space plots.

    Category semantics come entirely from CategoryConfig upstream.

    This renderer never knows biological field names such as:
        condition
        region
        layer
        genotype
        treatment

    Layout
    ------
    1 field:
        one horizontal panel per level

    2 fields:
        first field = rows
        second field = columns

    >2 fields:
        observed combinations in a compact grid
    """

    category_fields = (
        _validate_category_fields(
            master=master,
            category_fields=(
                category_fields
            ),
        )
    )


    (
        table,
        d,
        cluster_colors,
    ) = _prepare_plot_table(
        master=master,
        clustering=clustering,
        umap_embedding=(
            umap_embedding
        ),
    )


    output_dir = Path(
        output_dir
    )


    # ==================================================================
    # PCA
    # ==================================================================

    pca_path = (
        output_dir
        / "Category_PCA.png"
    )


    _save_category_figure(
        table=table,
        x=table[
            "PC1"
        ].to_numpy(
            dtype=float
        ),
        y=table[
            "PC2"
        ].to_numpy(
            dtype=float
        ),
        category_fields=(
            category_fields
        ),
        cluster_colors=(
            cluster_colors
        ),
        title=(
            f"Category | "
            f"PCA Morphology Space | "
            f"d = {d} PCs"
        ),
        xlabel="PC1",
        ylabel="PC2",
        output_path=(
            pca_path
        ),
        config=config,
    )


    # ==================================================================
    # UMAP
    # ==================================================================

    umap_path = (
        output_dir
        / "Category_UMAP.png"
    )


    _save_category_figure(
        table=table,
        x=table[
            "_UMAP1"
        ].to_numpy(
            dtype=float
        ),
        y=table[
            "_UMAP2"
        ].to_numpy(
            dtype=float
        ),
        category_fields=(
            category_fields
        ),
        cluster_colors=(
            cluster_colors
        ),
        title=(
            f"Category | "
            f"UMAP | "
            f"d = {d} PCs from PCA"
        ),
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        output_path=(
            umap_path
        ),
        config=config,
    )


    return {
        "pca":
            pca_path,

        "umap":
            umap_path,
    }
