from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as path_effects

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import chi2_contingency

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
    BAR_OUTLINE_WIDTH,
    MG_RESIDUAL_COLORS,
    cluster_display_order,
    mg_color,
)


# ======================================================================
# STATE TABLE
# ======================================================================

def _prepare_state_table(
    master: pd.DataFrame,
    clustering,
) -> tuple[
    pd.DataFrame,
    list[int],
    dict[int, object],
]:
    """
    Attach canonical display-state identity to the master table.

    Analytical GMM labels are NOT modified.

    Human-facing C1...Cn ordering uses the same median Cell_area rule
    as every other MorphoGlia plot.
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
            "Preferred selected clustering solution "
            "is unavailable."
        )


    solution = (
        selected_solutions[
            preferred_k
        ]
    )


    labels = np.asarray(
        solution.labels,
        dtype=int,
    )


    if labels.shape != (
        len(master),
    ):

        raise ValueError(
            "Preferred-solution labels do not match "
            "master-table row count."
        )


    table = (
        master
        .copy()
        .reset_index(
            drop=True
        )
    )


    (
        ordered_raw_clusters,
        display_rank,
    ) = cluster_display_order(
        master=table,
        labels=labels,
        complexity_column="Cell_area",
    )


    table[
        "_state"
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


    state_ids = list(
        range(
            1,
            len(
                ordered_raw_clusters
            )
            + 1,
        )
    )


    state_colors = {
        state:
            mg_color(state - 1, n_colors=len(state_ids))

        for state
        in state_ids
    }


    return (
        table,
        state_ids,
        state_colors,
    )


# ======================================================================
# CATEGORY CONTRACT
# ======================================================================

def _validate_category(
    table: pd.DataFrame,
    category_fields: Sequence[str],
) -> tuple[str, ...]:

    fields = tuple(
        str(field)
        for field
        in category_fields
    )


    if not fields:

        raise ValueError(
            "Category composition requires at least "
            "one configured category field."
        )


    if CATEGORY_COLUMN not in table.columns:

        raise ValueError(
            "Category composition requires the reserved "
            f"{CATEGORY_COLUMN!r} column."
        )


    missing = [
        field
        for field
        in fields
        if field not in table.columns
    ]


    if missing:

        raise ValueError(
            "Configured category fields are missing "
            f"from master table: {missing}"
        )


    return fields



def _category_display(
    table: pd.DataFrame,
    category_fields: tuple[str, ...],
    config: PlotConfig,
) -> tuple[
    pd.Series,
    list[str],
]:
    """
    Build human-facing category labels and their canonical display order.

    The ordering contract is shared with every other metadata/category plot.
    """

    values = (
        table[
            list(
                category_fields
            )
        ]
        .astype(str)
    )

    display = (
        values
        .agg(
            "-".join,
            axis=1,
        )
    )

    raw_combinations = [
        tuple(
            str(value)
            for value in row
        )
        for row in (
            values
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

    order = [
        "-".join(
            combination
        )
        for combination
        in combinations
    ]

    return (
        display,
        order,
    )


# ======================================================================
# TABLES
# ======================================================================

def build_composition_tables(
    master: pd.DataFrame,
    clustering,
    category_fields: Sequence[str],
    config: PlotConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build all category x morphology-state descriptive/statistical tables.
    """

    if config is None:
        config = PlotConfig()

    (
        table,
        state_ids,
        _,
    ) = _prepare_state_table(
        master=master,
        clustering=clustering,
    )


    fields = (
        _validate_category(
            table,
            category_fields,
        )
    )


    (
        category_display,
        category_order,
    ) = _category_display(
        table,
        fields,
        config,
    )


    table[
        "_category_display"
    ] = category_display


    # ==================================================================
    # COUNTS
    # ==================================================================

    counts = pd.crosstab(
        table[
            "_category_display"
        ],
        table[
            "_state"
        ],
        dropna=False,
    )


    counts = counts.reindex(
        index=(
            category_order
        ),
        columns=(
            state_ids
        ),
        fill_value=0,
    )


    counts.columns = [
        f"C{state}"
        for state
        in state_ids
    ]


    counts.index.name = (
        "category"
    )


    # ==================================================================
    # PERCENTAGES
    # ==================================================================

    row_denominator = (
        counts
        .sum(
            axis=1
        )
        .replace(
            0,
            np.nan,
        )
    )


    row_percentages = (
        counts
        .div(
            row_denominator,
            axis=0,
        )
        .mul(
            100.0
        )
        .fillna(
            0.0
        )
    )


    column_denominator = (
        counts
        .sum(
            axis=0
        )
        .replace(
            0,
            np.nan,
        )
    )


    column_percentages = (
        counts
        .div(
            column_denominator,
            axis=1,
        )
        .mul(
            100.0
        )
        .fillna(
            0.0
        )
    )


    # ==================================================================
    # GLOBAL STATE SIZES
    # ==================================================================

    state_sizes = pd.DataFrame(
        {
            "state":
                counts.columns,

            "count":
                counts.sum(
                    axis=0
                ).to_numpy(
                    dtype=int
                ),
        }
    )


    total_cells = int(
        state_sizes[
            "count"
        ].sum()
    )


    state_sizes[
        "percentage"
    ] = (
        state_sizes[
            "count"
        ]
        / max(
            total_cells,
            1,
        )
        * 100.0
    )


    # ==================================================================
    # CHI-SQUARE + RESIDUALS
    # ==================================================================

    valid_counts = (
        counts
        .loc[
            counts.sum(
                axis=1
            )
            > 0,
            counts.sum(
                axis=0
            )
            > 0,
        ]
    )


    chi_stats = pd.DataFrame(
        [
            {
                "valid":
                    False,

                "chi_square":
                    np.nan,

                "degrees_of_freedom":
                    np.nan,

                "p_value":
                    np.nan,

                "cramers_v":
                    np.nan,

                "n_cells":
                    int(
                        valid_counts
                        .to_numpy()
                        .sum()
                    ),

                "n_categories":
                    int(
                        valid_counts
                        .shape[
                            0
                        ]
                    ),

                "n_states":
                    int(
                        valid_counts
                        .shape[
                            1
                        ]
                    ),

                "minimum_expected_count":
                    np.nan,

                "fraction_expected_below_5":
                    np.nan,
            }
        ]
    )


    expected = pd.DataFrame()
    residuals = pd.DataFrame()


    if (
        valid_counts.shape[
            0
        ]
        >= 2
        and valid_counts.shape[
            1
        ]
        >= 2
    ):

        (
            chi_square,
            p_value,
            dof,
            expected_array,
        ) = chi2_contingency(
            valid_counts.to_numpy(),
            correction=False,
        )


        expected = pd.DataFrame(
            expected_array,
            index=(
                valid_counts.index
            ),
            columns=(
                valid_counts.columns
            ),
        )


        # Pearson standardized residuals.
        #
        # This intentionally preserves the definition used in the
        # previous MorphoGlia analysis scripts:
        #
        #     (Observed - Expected) / sqrt(Expected)
        #
        residuals = (
            valid_counts
            - expected
        ) / np.sqrt(
            expected
        )


        n_cells = float(
            valid_counts
            .to_numpy()
            .sum()
        )


        min_dimension = min(
            valid_counts.shape[
                0
            ]
            - 1,
            valid_counts.shape[
                1
            ]
            - 1,
        )


        if (
            n_cells > 0
            and min_dimension > 0
        ):

            cramers_v = float(
                np.sqrt(
                    chi_square
                    / (
                        n_cells
                        * min_dimension
                    )
                )
            )

        else:

            cramers_v = np.nan


        minimum_expected = float(
            np.min(
                expected_array
            )
        )


        fraction_expected_below_5 = float(
            np.mean(
                expected_array
                < 5.0
            )
        )


        chi_stats = pd.DataFrame(
            [
                {
                    "valid":
                        True,

                    "chi_square":
                        float(
                            chi_square
                        ),

                    "degrees_of_freedom":
                        int(
                            dof
                        ),

                    "p_value":
                        float(
                            p_value
                        ),

                    "cramers_v":
                        cramers_v,

                    "n_cells":
                        int(
                            n_cells
                        ),

                    "n_categories":
                        int(
                            valid_counts
                            .shape[
                                0
                            ]
                        ),

                    "n_states":
                        int(
                            valid_counts
                            .shape[
                                1
                            ]
                        ),

                    "minimum_expected_count":
                        minimum_expected,

                    "fraction_expected_below_5":
                        fraction_expected_below_5,
                }
            ]
        )


    return {
        "state_sizes":
            state_sizes,

        "counts":
            counts,

        "row_percentages":
            row_percentages,

        "column_percentages":
            column_percentages,

        "chi_square_stats":
            chi_stats,

        "expected_counts":
            expected,

        "standardized_residuals":
            residuals,
    }


# ======================================================================
# TABLE EXPORT
# ======================================================================

def _save_tables(
    tables: dict[
        str,
        pd.DataFrame,
    ],
    output_dir: Path,
) -> dict[str, Path]:

    table_dir = (
        output_dir
        / "Tables"
    )


    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    filenames = {
        "state_sizes":
            "State_Sizes.csv",

        "counts":
            "Category_State_Counts.csv",

        "row_percentages":
            "Category_State_Row_Percentages.csv",

        "column_percentages":
            "Category_State_Column_Percentages.csv",

        "chi_square_stats":
            "ChiSquare_Stats.csv",

        "expected_counts":
            "ChiSquare_Expected_Counts.csv",

        "standardized_residuals":
            "ChiSquare_Standardized_Residuals.csv",
    }


    paths = {}


    for (
        key,
        filename,
    ) in filenames.items():

        table = tables[
            key
        ]


        path = (
            table_dir
            / filename
        )


        table.to_csv(
            path,
            index=(
                key
                in {
                    "counts",
                    "row_percentages",
                    "column_percentages",
                    "expected_counts",
                    "standardized_residuals",
                }
            ),
        )


        paths[
            key
        ] = path


    return paths


# ======================================================================
# SHARED PLOT HELPERS
# ======================================================================


def _state_colors(
    columns: Sequence[str],
) -> list:

    n_colors = len(
        columns
    )

    colors = []

    for column in columns:

        state = int(
            str(
                column
            )
            .removeprefix(
                "C"
            )
        )

        colors.append(
            mg_color(
                state - 1,
                n_colors=n_colors,
            )
        )

    return colors



def _bold_ticks(
    axis,
) -> None:

    for label in (
        axis.get_xticklabels()
        + axis.get_yticklabels()
    ):

        label.set_fontweight(
            "bold"
        )


def _frame(
    axis,
    config: PlotConfig,
) -> None:

    for spine in (
        axis
        .spines
        .values()
    ):

        spine.set_linewidth(
            config
            .effective_frame_line_width
        )

        spine.set_color(
            "black"
        )



def _legend_handles(
    columns: Sequence[str],
) -> list[Line2D]:

    n_colors = len(
        columns
    )

    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=9,
            markerfacecolor=(
                mg_color(
                    int(
                        str(
                            column
                        )
                        .removeprefix(
                            "C"
                        )
                    )
                    - 1,
                    n_colors=n_colors,
                )
            ),
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=str(
                column
            ),
        )

        for column
        in columns
    ]



# ======================================================================
# STATE ABUNDANCE
# ======================================================================

def _plot_state_abundance(
    state_sizes: pd.DataFrame,
    output_dir: Path,
    config: PlotConfig,
) -> dict[str, Path]:

    plot_dir = (
        output_dir
        / "Plots"
    )


    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    states = (
        state_sizes[
            "state"
        ]
        .astype(str)
        .tolist()
    )


    colors = (
        _state_colors(
            states
        )
    )


    paths = {}


    for (
        value_column,
        ylabel,
        filename,
    ) in [
        (
            "count",
            "Number of Cells",
            "State_Abundance_Counts.png",
        ),
        (
            "percentage",
            "Percentage of Cells",
            "State_Abundance_Percentages.png",
        ),
    ]:

        fig, axis = plt.subplots(
            figsize=(
                max(
                    6.5,
                    len(
                        states
                    )
                    * 1.2,
                ),
                5.5,
            )
        )


        axis.bar(
            states,
            state_sizes[
                value_column
            ].to_numpy(
                dtype=float
            ),
            color=colors,
            edgecolor="black",
            linewidth=(
                BAR_OUTLINE_WIDTH
            ),
        )


        axis.set_title(
            "Morphology-State Abundance",
            fontsize=(
                config
                .effective_panel_title_size
            ),
            fontweight="bold",
            pad=12,
        )


        axis.set_xlabel(
            "Morphology State",
            fontsize=(
                config
                .effective_axis_label_size
            ),
            fontweight="bold",
        )


        axis.set_ylabel(
            ylabel,
            fontsize=(
                config
                .effective_axis_label_size
            ),
            fontweight="bold",
        )


        if (
            value_column
            == "percentage"
        ):

            axis.set_ylim(
                0,
                max(
                    100.0,
                    float(
                        state_sizes[
                            value_column
                        ].max()
                    )
                    * 1.12,
                ),
            )


        axis.grid(
            False
        )


        _bold_ticks(
            axis
        )


        _frame(
            axis,
            config,
        )


        path = (
            plot_dir
            / filename
        )


        fig.tight_layout()


        fig.savefig(
            path,
            dpi=(
                config
                .effective_dpi
            ),
            bbox_inches="tight",
        )


        plt.close(
            fig
        )


        paths[
            value_column
        ] = path


    return paths


# ======================================================================
# STACKED BARS
# ======================================================================

def _plot_stacked(
    values: pd.DataFrame,
    *,
    percentage: bool,
    output_path: Path,
    config: PlotConfig,
) -> Path:

    categories = (
        values
        .index
        .astype(str)
        .tolist()
    )


    columns = (
        values
        .columns
        .astype(str)
        .tolist()
    )


    colors = (
        _state_colors(
            columns
        )
    )


    fig_height = max(
        5.0,
        len(
            categories
        )
        * 0.72
        + 2.3,
    )


    fig, axis = plt.subplots(
        figsize=(
            10.0,
            fig_height,
        )
    )


    left = np.zeros(
        len(
            categories
        ),
        dtype=float,
    )


    y = np.arange(
        len(
            categories
        )
    )


    for (
        column,
        color,
    ) in zip(
        columns,
        colors,
    ):

        width = (
            values[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )


        axis.barh(
            y,
            width,
            left=left,
            color=color,
            edgecolor="black",
            linewidth=2.5,
            label=column,
        )


        left += width


    axis.set_yticks(
        y
    )


    axis.set_yticklabels(
        categories,
        fontweight="bold",
    )


    axis.invert_yaxis()


    axis.set_xlabel(
        (
            "Percentage of Cells"
            if percentage
            else "Number of Cells"
        ),
        fontsize=(
            config
            .effective_axis_label_size
        ),
        fontweight="bold",
    )


    axis.set_ylabel(
        "Category",
        fontsize=(
            config
            .effective_axis_label_size
        ),
        fontweight="bold",
    )


    axis.set_title(
        (
            "Morphology-State Composition"
            if percentage
            else "Morphology-State Counts"
        ),
        fontsize=(
            config
            .effective_panel_title_size
        ),
        fontweight="bold",
        pad=12,
    )


    if percentage:

        axis.set_xlim(
            0,
            100,
        )


        axis.xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda value, _:
                    f"{value:.0f}%"
            )
        )


    axis.grid(
        False
    )


    _bold_ticks(
        axis
    )


    _frame(
        axis,
        config,
    )


    legend = axis.legend(
        handles=(
            _legend_handles(
                columns
            )
        ),
        loc="upper left",
        bbox_to_anchor=(
            1.01,
            1.0,
        ),
        frameon=False,
        prop={
            "weight":
                "bold",

            "size":
                config
                .effective_legend_font_size,
        },
    )


    for text in (
        legend
        .get_texts()
    ):

        text.set_fontweight(
            "bold"
        )


    fig.tight_layout()


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
# PIE PLOTS
# ======================================================================

def _plot_pies(
    row_percentages: pd.DataFrame,
    master: pd.DataFrame,
    category_fields: Sequence[str],
    output_path: Path,
    config: PlotConfig,
) -> Path:
    """
    Plot morphology-state composition using the exact same category-layout
    rule as the Category PCA/UMAP figures.

    Layout
    ------
    1 category field:
        one horizontal row

    2 category fields:
        first field = rows
        second field = columns

    >2 category fields:
        observed combinations in a compact grid

    The reserved combined category string is never parsed.
    """

    fields = tuple(
        str(field)
        for field in category_fields
    )


    if not fields:

        raise ValueError(
            "Pie plots require at least one category field."
        )


    missing = [
        field
        for field in fields
        if field not in master.columns
    ]


    if missing:

        raise ValueError(
            "Pie-plot category fields are missing "
            f"from master table: {missing}"
        )


    table = (
        master
        .copy()
        .reset_index(
            drop=True
        )
    )


    state_columns = (
        row_percentages
        .columns
        .astype(str)
        .tolist()
    )


    colors = (
        _state_colors(
            state_columns
        )
    )


    # ==================================================================
    # ONE FIELD
    #
    # A | B | C
    # ==================================================================

    if len(
        fields
    ) == 1:

        field = fields[
            0
        ]


        levels = ordered_levels(
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


        n_rows = 1
        n_columns = len(
            levels
        )


        panels = [
            {
                "row":
                    0,

                "column":
                    column,

                "category":
                    str(
                        level
                    ),

                "title":
                    str(
                        level
                    ),
            }

            for column, level
            in enumerate(
                levels
            )
        ]


    # ==================================================================
    # TWO FIELDS
    #
    # EXACT SAME CONTRACT AS CATEGORY PCA / UMAP:
    #
    # first field  = rows
    # second field = columns
    #
    # Example:
    #
    #             CA1        CA3        SUB
    # SCOP     SCOP-CA1   SCOP-CA3   SCOP-SUB
    # SS       SS-CA1     SS-CA3     SS-SUB
    # ==================================================================

    elif len(
        fields
    ) == 2:

        row_field = fields[
            0
        ]


        column_field = fields[
            1
        ]


        row_levels = ordered_levels(
            table[
                row_field
            ]
            .astype(str)
            .tolist(),
            field_name=row_field,
            metadata_order=(
                config
                .effective_metadata_order
            ),
        )


        column_levels = ordered_levels(
            table[
                column_field
            ]
            .astype(str)
            .tolist(),
            field_name=column_field,
            metadata_order=(
                config
                .effective_metadata_order
            ),
        )


        n_rows = len(
            row_levels
        )


        n_columns = len(
            column_levels
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

                category = (
                    f"{row_level}-"
                    f"{column_level}"
                )


                panels.append(
                    {
                        "row":
                            row,

                        "column":
                            column,

                        "category":
                            category,

                        "title":
                            category,
                    }
                )


    # ==================================================================
    # MORE THAN TWO FIELDS
    #
    # Same generic fallback philosophy as Category plots.
    # ==================================================================

    else:

        raw_combinations = [
            tuple(
                str(value)
                for value in row
            )
            for row in (
                table[
                    list(
                        fields
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
            field_names=fields,
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
                n_panels
            ),
        )


        n_rows = int(
            ceil(
                n_panels
                / n_columns
            )
        )


        panels = []


        for (
            index,
            combination,
        ) in enumerate(
            combinations
        ):

            category = "-".join(
                combination
            )


            panels.append(
                {
                    "row":
                        index
                        // n_columns,

                    "column":
                        index
                        % n_columns,

                    "category":
                        category,

                    "title":
                        category,
                }
            )


    # ==================================================================
    # FIGURE
    # ==================================================================

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            4.2
            * n_columns,
            4.1
            * n_rows,
        ),
        squeeze=False,
    )


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


        axis = axes[
            row,
            column,
        ]


        category = str(
            panel[
                "category"
            ]
        )


        if category not in row_percentages.index:

            axis.axis(
                "off"
            )

            continue


        values = (
            row_percentages
            .loc[
                category,
                state_columns,
            ]
            .to_numpy(
                dtype=float
            )
        )


        # ==============================================================
        # PIE
        #
        # Clean wedges with manually drawn separators.
        #
        # This avoids the pointed artifact produced when several
        # wedge edge strokes converge at the center.
        # ==============================================================

        wedges, texts, autotexts = axis.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            
            wedgeprops={
                "edgecolor": "none",
                "linewidth": 0.0,
            },
            textprops={
                "fontweight": "bold",
                "fontsize": 9,
            },
        )
            
        # --------------------------------------------------------------
        # Percentage labels
        #
        # Black text with a white outline, rendered above all pie
        # separators and circumference strokes.
        # --------------------------------------------------------------
    
        for autotext in autotexts:
    
            autotext.set_color(
                "black"
            )
    
            autotext.set_fontweight(
                "bold"
            )
    
            autotext.set_zorder(
                20
            )
    
            autotext.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=2.1,
                        foreground="white",
                    ),
                    path_effects.Normal(),
                ]
            )
        
        
        


        # --------------------------------------------------------------
        # Slice separators
        #
        # Draw a wide white line first, then a thinner black line
        # directly over it.
        #
        # Starting very slightly away from the exact center prevents
        # the separator strokes from building a visible central spike.
        # --------------------------------------------------------------

        separator_angles = [
            wedge.theta1
            for wedge in wedges
        ]


        for angle in separator_angles:

            radians = np.deg2rad(
                angle
            )


            x_outer = np.cos(
                radians
            )

            y_outer = np.sin(
                radians
            )


            # Tiny offset from the mathematical center.
            center_offset = 0.001

            x_inner = (
                center_offset
                * x_outer
            )

            y_inner = (
                center_offset
                * y_outer
            )


            # White under-stroke.
            axis.plot(
                [
                    x_inner,
                    x_outer,
                ],
                [
                    y_inner,
                    y_outer,
                ],
                color="white",
                linewidth=4.0,
                solid_capstyle="butt",
                zorder=4,
            )


            # Thin black separator.
            axis.plot(
                [
                    x_inner,
                    x_outer,
                ],
                [
                    y_inner,
                    y_outer,
                ],
                color="black",
                linewidth=1.8,
                solid_capstyle="butt",
                zorder=5,
            )


        # --------------------------------------------------------------
        # Dominant outer circumference.
        # --------------------------------------------------------------

        outer_ring = plt.Circle(
            (0, 0),
            1.0,
            fill=False,
            edgecolor="black",
            linewidth=4.0,
            zorder=10,
        )


        axis.add_artist(
            outer_ring
        )
        
        
        


        axis.set_title(
            panel[
                "title"
            ],
            fontsize=(
                config
                .effective_panel_title_size
            ),
            fontweight="bold",
            y=0.95,
            pad=4,
        )


        axis.set_aspect(
            "equal"
        )
        
        

    # ==================================================================
    # UNUSED PANELS
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
    # FIGURE TITLE + LEGEND
    # ==================================================================

    fig.suptitle(
        "Morphology-State Composition",
        fontsize=(
            config
            .effective_figure_title_size
        ),
        fontweight="bold",
        y=0.98,
    )


    legend = fig.legend(
        handles=(
            _legend_handles(
                state_columns
            )
        ),
        loc="upper left",
        bbox_to_anchor=(
            0.90,
            0.86,
        ),
        frameon=False,
        borderaxespad=0.0,
        prop={
            "weight":
                "bold",

            "size":
                config
                .effective_legend_font_size,
        },
    )


    for text in (
        legend
        .get_texts()
    ):

        text.set_fontweight(
            "bold"
        )


    fig.subplots_adjust(
        left=0.06,
        right=0.87,
        bottom=0.06,
        top=0.88,
        wspace=0.10,
        hspace=0.22,
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
# STANDARDIZED RESIDUALS
# ======================================================================

def _plot_standardized_residuals(
    residuals: pd.DataFrame,
    output_path: Path,
    config: PlotConfig,
) -> Path | None:

    if residuals.empty:

        return None


    # Rows = morphology states.
    # Columns = categories.
    matrix = (
        residuals
        .T
    )


    values = matrix.to_numpy(
        dtype=float
    )


    minimum = float(
        np.nanmin(
            values
        )
    )


    maximum = float(
        np.nanmax(
            values
        )
    )


    lower = min(
        minimum,
        -3.5,
    )


    upper = max(
        maximum,
        3.5,
    )


    bounds = [
        lower,
        -3.0,
        -2.0,
        0.0,
        2.0,
        3.0,
        upper,
    ]


    cmap = mcolors.ListedColormap(
        MG_RESIDUAL_COLORS,
        name="morphoglia_residuals",
    )


    norm = mcolors.BoundaryNorm(
        boundaries=bounds,
        ncolors=cmap.N,
    )


    fig_width = max(
        7.0,
        matrix.shape[
            1
        ]
        * 1.25
        + 3.0,
    )


    fig_height = max(
        5.5,
        matrix.shape[
            0
        ]
        * 1.15
        + 2.5,
    )


    fig, axis = plt.subplots(
        figsize=(
            fig_width,
            fig_height,
        )
    )


    image = axis.imshow(
        values,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )


    axis.set_xticks(
        np.arange(
            matrix.shape[
                1
            ]
        )
    )


    axis.set_xticklabels(
        matrix.columns,
        rotation=0,
        ha="center",
        fontweight="bold",
    )


    axis.set_yticks(
        np.arange(
            matrix.shape[
                0
            ]
        )
    )


    axis.set_yticklabels(
        matrix.index,
        fontweight="bold",
    )


    # Thin internal white grid.
    axis.set_xticks(
        np.arange(
            -0.5,
            matrix.shape[
                1
            ],
            1,
        ),
        minor=True,
    )


    axis.set_yticks(
        np.arange(
            -0.5,
            matrix.shape[
                0
            ],
            1,
        ),
        minor=True,
    )


    axis.grid(
        which="minor",
        color="white",
        linewidth=0.8,
    )


    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )


    # Residual values remain visible because they are statistically
    # meaningful, unlike the stability heatmap values.
    for row in range(
        values.shape[
            0
        ]
    ):

        for column in range(
            values.shape[
                1
            ]
        ):

            value = values[
                row,
                column,
            ]


            if not np.isfinite(
                value
            ):

                continue


            text_color = (
                "white"
                if abs(
                    value
                )
                >= 2.0
                else "black"
            )


            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=text_color,
            )


    axis.set_title(
        "Standardized Residuals",
        fontsize=(
            config
            .effective_figure_title_size
        ),
        fontweight="bold",
        pad=14,
    )


    axis.set_xlabel(
        "Category",
        fontsize=(
            config
            .effective_axis_label_size
        ),
        fontweight="bold",
        labelpad=10,
    )


    axis.set_ylabel(
        "Morphology State",
        fontsize=(
            config
            .effective_axis_label_size
        ),
        fontweight="bold",
        labelpad=10,
    )


    # Strong outer frame.
    for spine in (
        axis
        .spines
        .values()
    ):

        spine.set_linewidth(
            config
            .effective_frame_line_width
            * 1.5
        )


    # Colorbar exactly matched to heatmap height.
    divider = make_axes_locatable(
        axis
    )


    cbar_axis = divider.append_axes(
        "right",
        size="4%",
        pad=0.16,
    )


    colorbar = fig.colorbar(
        image,
        cax=cbar_axis,
    )


    colorbar.set_ticks(
        [
            -3,
            -2,
            0,
            2,
            3,
        ]
    )


    colorbar.set_ticklabels(
        [
            "−3",
            "−2",
            "0",
            "+2",
            "+3",
        ]
    )


    for tick in (
        colorbar
        .ax
        .get_yticklabels()
    ):

        tick.set_fontweight(
            "bold"
        )


    colorbar.outline.set_linewidth(
        config
        .effective_frame_line_width
    )


    fig.tight_layout()


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

def plot_category_composition(
    master: pd.DataFrame,
    clustering,
    category_fields: Sequence[str],
    output_dir: str | Path,
    config: PlotConfig,
) -> dict[str, Path]:
    """
    Generate all canonical category x morphology-state descriptive outputs.

    The function is completely dataset-neutral.

    Category semantics come only from CategoryConfig/category_fields.
    """

    output_dir = Path(
        output_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    tables = (
        build_composition_tables(
            master=master,
            clustering=clustering,
            category_fields=(
                category_fields
            ),
            config=config,
        )
    )


    files = (
        _save_tables(
            tables,
            output_dir,
        )
    )


    # ==================================================================
    # GLOBAL ABUNDANCE
    # ==================================================================

    abundance_paths = (
        _plot_state_abundance(
            tables[
                "state_sizes"
            ],
            output_dir,
            config,
        )
    )


    files[
        "state_abundance_counts_plot"
    ] = abundance_paths[
        "count"
    ]


    files[
        "state_abundance_percentages_plot"
    ] = abundance_paths[
        "percentage"
    ]


    # ==================================================================
    # STACKED COUNTS
    # ==================================================================

    plot_dir = (
        output_dir
        / "Plots"
    )


    files[
        "stacked_counts_plot"
    ] = _plot_stacked(
        tables[
            "counts"
        ],
        percentage=False,
        output_path=(
            plot_dir
            / "Category_State_Stacked_Counts.png"
        ),
        config=config,
    )


    # ==================================================================
    # STACKED PERCENTAGES
    # ==================================================================

    files[
        "stacked_percentages_plot"
    ] = _plot_stacked(
        tables[
            "row_percentages"
        ],
        percentage=True,
        output_path=(
            plot_dir
            / "Category_State_Stacked_Percentages.png"
        ),
        config=config,
    )


    # ==================================================================
    # PIE PLOTS
    # ==================================================================

    files[
        "pies_plot"
    ] = _plot_pies(
        tables[
            "row_percentages"
        ],
        master=master,
        category_fields=(
            category_fields
        ),
        output_path=(
            plot_dir
            / "Category_State_Pies.png"
        ),
        config=config,
    )


    # ==================================================================
    # STANDARDIZED RESIDUALS
    # ==================================================================

    residual_path = (
        _plot_standardized_residuals(
            tables[
                "standardized_residuals"
            ],
            output_path=(
                plot_dir
                / "ChiSquare_Standardized_Residuals.png"
            ),
            config=config,
        )
    )


    if residual_path is not None:

        files[
            "standardized_residuals_plot"
        ] = residual_path


    return files
