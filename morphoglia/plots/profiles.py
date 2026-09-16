from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
    optimal_leaf_ordering,
)

from ..display_order import (
    ordered_combinations,
)

from .config import (
    PlotConfig,
)

from .style import (
    MG_DIVERGING_COLORS,
    cluster_display_order,
)


# ======================================================================
# CONTRACT VALIDATION
# ======================================================================

def _validate_analytical_features(
    master: pd.DataFrame,
    analytical_features: Sequence[str],
) -> list[str]:
    """
    Validate the canonical Feature Selection -> Profiles contract.

    analytical_features must come from:

        FeaturePreparationResult.analytical_features

    Profiles never rediscovers analytical features from the master table.
    """

    features = [
        str(feature)
        for feature
        in analytical_features
    ]


    if not features:

        raise ValueError(
            "Morphometric profiles require the canonical "
            "FeaturePreparationResult.analytical_features."
        )


    if len(
        features
    ) != len(
        set(
            features
        )
    ):

        raise ValueError(
            "analytical_features contains duplicates."
        )


    missing = [
        feature
        for feature
        in features
        if feature not in master.columns
    ]


    if missing:

        raise ValueError(
            "Selected morphometric features are missing "
            f"from the master table: {missing}"
        )


    non_numeric = [
        feature
        for feature
        in features
        if not pd.api.types.is_numeric_dtype(
            master[
                feature
            ]
        )
    ]


    if non_numeric:

        raise ValueError(
            "Morphometric profile features must be numeric: "
            f"{non_numeric}"
        )


    return features


def _validate_category_fields(
    master: pd.DataFrame,
    category_fields: Sequence[str],
) -> tuple[str, ...]:

    fields = tuple(
        str(field)
        for field
        in category_fields
    )


    if not fields:

        raise ValueError(
            "Morphometric category profiles require at least "
            "one configured category field."
        )


    missing = [
        field
        for field
        in fields
        if field not in master.columns
    ]


    if missing:

        raise ValueError(
            "Configured category fields are missing "
            f"from the master table: {missing}"
        )


    for field in fields:

        if master[
            field
        ].isna().any():

            raise ValueError(
                f"Category field {field!r} contains missing values."
            )


    return fields


# ======================================================================
# CATEGORY GROUPS
# ======================================================================


def _category_groups(
    master: pd.DataFrame,
    category_fields: tuple[str, ...],
    config: PlotConfig,
) -> tuple[
    pd.Series,
    list[str],
]:
    """
    Build category labels and the shared canonical metadata display order.

    For multiple fields, flattened outputs preserve the existing
    comparison-oriented convention: fields 2..N are outer grouping
    variables and field 1 varies within them.
    """

    values = (
        master[
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
# MORPHOLOGY-STATE GROUPS
# ======================================================================

def _state_labels(
    master: pd.DataFrame,
    clustering,
) -> tuple[
    np.ndarray,
    list[str],
]:
    """
    Convert raw analytical GMM labels to the canonical C1...Cn
    DISPLAY ordering.

    Analytical cluster identities are not modified.
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


    labels = np.asarray(
        selected_solutions[
            preferred_k
        ].labels,
        dtype=int,
    )


    if labels.shape != (
        len(
            master
        ),
    ):

        raise ValueError(
            "Preferred-solution labels do not match "
            "master-table row count."
        )


    (
        ordered_raw_clusters,
        display_rank,
    ) = cluster_display_order(
        master=master,
        labels=labels,
        complexity_column="Cell_area",
    )


    display_labels = np.asarray(
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


    state_order = [
        f"C{display_rank[raw_cluster]}"
        for raw_cluster
        in ordered_raw_clusters
    ]


    return (
        display_labels,
        state_order,
    )


# ======================================================================
# AGGREGATION
# ======================================================================

def build_morphometric_profile_tables(
    master: pd.DataFrame,
    clustering,
    analytical_features: Sequence[str],
    category_fields: Sequence[str],
    config: PlotConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build raw and normalized morphometric profile matrices.

    Raw matrices
    ------------
    Rows:
        selected analytical morphometric features

    Columns:
        categories
        or
        morphology states

    Values:
        arithmetic mean of the raw morphometric feature.

    Normalization
    -------------
    Each feature is independently min-max normalized to [-1, +1]
    using one shared range across the Category and Morphology State
    panels.

    Therefore identical colors have the same interpretation in both
    panels.
    """


    if config is None:
        config = PlotConfig()

    table = (
        master
        .copy()
        .reset_index(
            drop=True
        )
    )


    features = (
        _validate_analytical_features(
            table,
            analytical_features,
        )
    )


    fields = (
        _validate_category_fields(
            table,
            category_fields,
        )
    )


    (
        category_labels,
        category_order,
    ) = _category_groups(
        table,
        fields,
        config,
    )


    (
        display_states,
        state_order,
    ) = _state_labels(
        table,
        clustering,
    )


    table[
        "_profile_category"
    ] = category_labels


    table[
        "_profile_state"
    ] = [
        f"C{state}"
        for state
        in display_states
    ]


    # ==================================================================
    # CATEGORY MEANS
    # ==================================================================

    category_raw = (
        table
        .groupby(
            "_profile_category",
            sort=False,
        )[
            features
        ]
        .mean()
        .reindex(
            category_order
        )
        .T
    )


    category_raw.index.name = (
        "feature"
    )


    category_raw.columns.name = (
        "category"
    )


    # ==================================================================
    # STATE MEANS
    # ==================================================================

    state_raw = (
        table
        .groupby(
            "_profile_state",
            sort=False,
        )[
            features
        ]
        .mean()
        .reindex(
            state_order
        )
        .T
    )


    state_raw.index.name = (
        "feature"
    )


    state_raw.columns.name = (
        "state"
    )


    # ==================================================================
    # NORMALIZATION
    # ==================================================================

    (
        category_normalized,
        state_normalized,
    ) = _normalize_profile_pair_to_unit(
        category_raw=category_raw,
        state_raw=state_raw,
    )


    return {
        "category_raw":
            category_raw,

        "state_raw":
            state_raw,

        "category_normalized":
            category_normalized,

        "state_normalized":
            state_normalized,
    }


def _normalize_profile_pair_to_unit(
    category_raw: pd.DataFrame,
    state_raw: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Normalize every morphometric feature to [-1, +1] using one
    shared range across BOTH profile panels.

    For each feature:

        minimum = min(all category means, all state means)
        maximum = max(all category means, all state means)

    The same transformation is then applied to Category and Morphology
    State profiles.

    This makes the shared color scale directly comparable between panels.

    Features with zero range across both panels are represented as 0.
    """

    if not category_raw.index.equals(
        state_raw.index
    ):
        raise ValueError(
            "Category and state profile tables must contain "
            "the same morphometric features in the same order."
        )


    combined = pd.concat(
        [
            category_raw,
            state_raw,
        ],
        axis=1,
    )


    row_min = combined.min(
        axis=1
    )


    row_max = combined.max(
        axis=1
    )


    denominator = (
        row_max
        - row_min
    ).replace(
        0,
        np.nan,
    )


    def normalize(
        matrix: pd.DataFrame,
    ) -> pd.DataFrame:

        result = (
            matrix
            .sub(
                row_min,
                axis=0,
            )
            .div(
                denominator,
                axis=0,
            )
            .mul(
                2.0
            )
            .sub(
                1.0
            )
        )


        return result.fillna(
            0.0
        )


    return (
        normalize(
            category_raw
        ),
        normalize(
            state_raw
        ),
    )


# ======================================================================
# CSV EXPORT
# ======================================================================

def _save_profile_tables(
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
        "category_raw":
            "Morphometric_Category_Means.csv",

        "state_raw":
            "Morphometric_State_Means.csv",

        "category_normalized":
            "Morphometric_Category_Normalized.csv",

        "state_normalized":
            "Morphometric_State_Normalized.csv",
    }


    files = {}


    for (
        key,
        filename,
    ) in filenames.items():

        path = (
            table_dir
            / filename
        )


        tables[
            key
        ].to_csv(
            path,
            index=True,
        )


        files[
            key
        ] = path


    return files


# ======================================================================
# COLOR MAP
# ======================================================================

def _profile_colormap():

    return (
        mcolors
        .LinearSegmentedColormap
        .from_list(
            "morphoglia_morphometric_profiles",
            MG_DIVERGING_COLORS,
        )
    )


# ======================================================================
# DENDROGRAM
# ======================================================================

def _orient_linkage_to_preferred_order(
    linkage_matrix: np.ndarray,
    labels: Sequence[str],
    preferred_order: Sequence[str],
) -> np.ndarray:
    """
    Orient a fixed hierarchical tree toward a preferred left-to-right order.

    IMPORTANT
    ---------
    This does NOT modify:
        - cluster membership
        - linkage distances
        - merge topology

    It only swaps the left/right children of existing internal nodes.

    Among all leaf orders obtainable by valid branch flips, the orientation
    with the fewest inversions relative to preferred_order is selected.
    """

    oriented = np.asarray(
        linkage_matrix,
        dtype=float,
    ).copy()

    n_leaves = len(labels)

    preferred_rank = {
        str(label): rank
        for rank, label in enumerate(preferred_order)
    }

    leaf_rank = {
        index: preferred_rank[str(label)]
        for index, label in enumerate(labels)
    }

    def orient_node(
        node_id: int,
    ) -> list[int]:

        # Original observations are leaves.
        if node_id < n_leaves:
            return [leaf_rank[node_id]]

        row_index = (
            node_id
            - n_leaves
        )

        left_id = int(
            oriented[
                row_index,
                0,
            ]
        )

        right_id = int(
            oriented[
                row_index,
                1,
            ]
        )

        left_ranks = orient_node(
            left_id
        )

        right_ranks = orient_node(
            right_id
        )

        # Number of preferred-order inversions if:
        #
        #     LEFT -> RIGHT
        #
        inversions_left_right = sum(
            left_rank > right_rank
            for left_rank in left_ranks
            for right_rank in right_ranks
        )

        # Number of preferred-order inversions if:
        #
        #     RIGHT -> LEFT
        #
        inversions_right_left = sum(
            right_rank > left_rank
            for right_rank in right_ranks
            for left_rank in left_ranks
        )

        # Flipping children changes only visual orientation.
        if (
            inversions_right_left
            < inversions_left_right
        ):

            oriented[
                row_index,
                0,
            ] = right_id

            oriented[
                row_index,
                1,
            ] = left_id

            return (
                right_ranks
                + left_ranks
            )

        return (
            left_ranks
            + right_ranks
        )

    root_id = (
        n_leaves
        + oriented.shape[0]
        - 1
    )

    orient_node(
        root_id
    )

    return oriented


def _draw_top_dendrogram(
    axis,
    matrix: pd.DataFrame,
    preferred_order: Sequence[str] | None = None,
) -> list[int]:
    """
    Draw a column-similarity dendrogram and return its leaf order.

    The heatmap beneath the dendrogram MUST use this same order so that
    every dendrogram leaf is physically aligned with its corresponding
    displayed column.

    Column identities are never changed; only their visual order in this
    figure is determined by hierarchical similarity.
    """

    n_columns = int(
        matrix.shape[
            1
        ]
    )


    if n_columns < 2:

        axis.axis(
            "off"
        )

        return list(
            range(
                n_columns
            )
        )


    observations = (
        matrix
        .T
        .fillna(
            0.0
        )
        .to_numpy(
            dtype=float
        )
    )


    linkage_matrix = linkage(
        observations,
        method="average",
        metric="euclidean",
    )
    
    if preferred_order is not None:

        linkage_matrix = (
            _orient_linkage_to_preferred_order(
                linkage_matrix=linkage_matrix,
                labels=list(
                    matrix.columns
                ),
                preferred_order=preferred_order,
            )
        )
    

    
    result = dendrogram(
        linkage_matrix,
        ax=axis,
        orientation="top",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="black",
    )



    for collection in (
        axis
        .collections
    ):

        collection.set_linewidth(
            2.4
        )


    axis.set_xticks(
        []
    )


    axis.set_yticks(
        []
    )


    axis.set_frame_on(
        False
    )


    return [
        int(index)
        for index
        in result[
            "leaves"
        ]
    ]


# ======================================================================
# HEATMAP PANEL
# ======================================================================

def _draw_profile_heatmap(
    axis,
    matrix: pd.DataFrame,
    *,
    show_ylabels: bool,
    config: PlotConfig,
):
    """
    Draw one morphometric-profile heatmap.

    This is NOT:
        the clustering stability heatmap
        or the standardized-residual heatmap.
    """

    values = matrix.to_numpy(
        dtype=float
    )


    image = axis.imshow(
        values,
        cmap=(
            _profile_colormap()
        ),
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
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
        fontsize=11,
        fontweight="bold",
    )


    axis.set_yticks(
        np.arange(
            matrix.shape[
                0
            ]
        )
    )


    if show_ylabels:

        axis.set_yticklabels(
            [
                str(feature)
                .replace(
                    "_",
                    " "
                )
                for feature
                in matrix.index
            ],
            fontsize=8,
            fontweight="bold",
        )

    else:

        axis.set_yticklabels(
            []
        )


    # Thin white grid between heatmap cells.
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
        linewidth=(
            config
            .effective_heatmap_grid_width
        ),
    )


    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )


    # Clean black outer frame.
    for spine in (
        axis
        .spines
        .values()
    ):

        spine.set_visible(
            True
        )

        spine.set_color(
            "black"
        )

        spine.set_linewidth(
            config
            .effective_frame_line_width
            * 1.35
        )


    axis.set_xlabel(
        ""
    )


    axis.set_ylabel(
        ""
    )


    return image


# ======================================================================
# FIGURE
# ======================================================================

def _plot_profile_figure(
    category_matrix: pd.DataFrame,
    state_matrix: pd.DataFrame,
    output_path: Path,
    config: PlotConfig,
) -> Path:
    """
    Generate the canonical bi-panel morphometric-profile heatmap.
    """

    n_features = int(
        category_matrix.shape[
            0
        ]
    )


    figure_height = max(
        10.0,
        min(
            0.32
            * n_features
            + 3.5,
            30.0,
        ),
    )


    figure_width = max(
        15.0,
        (
            category_matrix.shape[
                1
            ]
            + state_matrix.shape[
                1
            ]
        )
        * 1.35
        + 6.0,
    )


    fig = plt.figure(
        figsize=(
            figure_width,
            figure_height,
        )
    )
    
    fig.subplots_adjust(
        top=0.94,
    )
    


    grid = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[
            max(
                1,
                category_matrix.shape[
                    1
                ]
            ),
            max(
                1,
                state_matrix.shape[
                    1
                ]
            ),
        ],
        height_ratios=[
            0.09,
            1.0,
        ],
        wspace=0.05,
        hspace=0.00,
    )


    category_dendrogram_axis = (
        fig.add_subplot(
            grid[
                0,
                0,
            ]
        )
    )


    state_dendrogram_axis = (
        fig.add_subplot(
            grid[
                0,
                1,
            ]
        )
    )


    category_axis = (
        fig.add_subplot(
            grid[
                1,
                0,
            ]
        )
    )


    state_axis = (
        fig.add_subplot(
            grid[
                1,
                1,
            ]
        )
    )


    # ==================================================================
    # DENDROGRAMS
    # ==================================================================

    category_order = (
        _draw_top_dendrogram(
            category_dendrogram_axis,
            category_matrix,
            preferred_order=list(
                category_matrix.columns
            ),
        )
    )


    state_order = (
        _draw_top_dendrogram(
            state_dendrogram_axis,
            state_matrix,
            preferred_order=list(
                state_matrix.columns
            ),
        )
    )


    category_plot_matrix = (
        category_matrix.iloc[
            :,
            category_order,
        ]
    )


    state_plot_matrix = (
        state_matrix.iloc[
            :,
            state_order,
        ]
    )


    # ==================================================================
    # HEATMAPS
    #
    # Display order exactly matches the dendrogram leaves above.
    # ==================================================================

    category_image = (
        _draw_profile_heatmap(
            category_axis,
            category_plot_matrix,
            show_ylabels=True,
            config=config,
        )
    )


    _draw_profile_heatmap(
        state_axis,
        state_plot_matrix,
        show_ylabels=False,
        config=config,
    )



    # ==================================================================
    # PANEL TITLES
    #
    # ==================================================================
    
    category_dendrogram_axis.set_title(
        "Category",
        fontsize=(
            config
            .effective_panel_title_size
            * 1.75
        ),
        fontweight="bold",
        pad=8,
    )
    
    
    state_dendrogram_axis.set_title(
        "Morphology States",
        fontsize=(
            config
            .effective_panel_title_size
            * 1.75
        ),
        fontweight="bold",
        pad=8,
    )
    
    
    fig.suptitle(
        "Morphometric Profiles",
        fontsize=(
            config
            .effective_figure_title_size
            * 1.70
        ),
        fontweight="bold",
        y=0.98,
    )




    # ==================================================================
    # SINGLE SHARED COLORBAR
    #
    # Height automatically matches the right heatmap.
    # ==================================================================

    divider = make_axes_locatable(
        state_axis
    )


    colorbar_axis = (
        divider.append_axes(
            "right",
            size="4%",
            pad=0.15,
        )
    )


    colorbar = fig.colorbar(
        category_image,
        cax=colorbar_axis,
    )


    colorbar.set_ticks(
        [
            -1.0,
            0.0,
            1.0,
        ]
    )


    colorbar.set_ticklabels(
        [
            "−1",
            "0",
            "+1",
        ]
    )


    colorbar.ax.tick_params(
        labelsize=11,
        width=1.2,
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

def plot_morphometric_profiles(
    master: pd.DataFrame,
    clustering,
    analytical_features: Sequence[str],
    category_fields: Sequence[str],
    output_dir: str | Path,
    config: PlotConfig,
) -> dict[str, Path]:
    """
    Generate the canonical MorphoGlia morphometric-profile output.

    Inputs
    ------
    analytical_features
        Must be FeaturePreparationResult.analytical_features.

    Outputs
    -------
    Tables/Morphometric_Category_Means.csv
    Tables/Morphometric_State_Means.csv
    Tables/Morphometric_Category_Normalized.csv
    Tables/Morphometric_State_Normalized.csv

    Morphometric_Profiles.png

    Important
    ---------
    This plot is completely separate from:

        Stability_Heatmap.png
        ChiSquare_Standardized_Residuals.png
    """

    output_dir = Path(
        output_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    tables = (
        build_morphometric_profile_tables(
            master=master,
            clustering=clustering,
            analytical_features=(
                analytical_features
            ),
            category_fields=(
                category_fields
            ),
            config=config,
        )
    )


    files = (
        _save_profile_tables(
            tables,
            output_dir,
        )
    )


    figure_path = (
        output_dir
        / "Morphometric_Profiles.png"
    )


    files[
        "plot"
    ] = _plot_profile_figure(
        category_matrix=(
            tables[
                "category_normalized"
            ]
        ),
        state_matrix=(
            tables[
                "state_normalized"
            ]
        ),
        output_path=(
            figure_path
        ),
        config=config,
    )


    return files
