from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from matplotlib.colors import to_rgb

from .config import PlotConfig

from .style import (
    mg_color,
    cluster_display_order,
    style_axis,
    add_cluster_legend,
    scatter_double_outline,
)


# ======================================================================
# PROTOTYPE MATCHING
# ======================================================================

def _prototype_indices(
    master: pd.DataFrame,
    prototypes: pd.DataFrame,
) -> np.ndarray:
    """
    Recover master-table row indices for the observed prototypes.
    """

    identity = (
        master[
            [
                "image_id",
                "cell_id",
            ]
        ]
        .copy()
    )


    identity[
        "_row_index"
    ] = np.arange(
        len(master),
        dtype=int,
    )


    matched = (
        prototypes[
            [
                "image_id",
                "cell_id",
                "cluster",
            ]
        ]
        .merge(
            identity,
            on=[
                "image_id",
                "cell_id",
            ],
            how="left",
            validate="one_to_one",
        )
    )


    if matched[
        "_row_index"
    ].isna().any():

        raise RuntimeError(
            "Could not match all prototypes "
            "to the master table."
        )


    return matched[
        "_row_index"
    ].to_numpy(
        dtype=int
    )


# ======================================================================
# UMAP
# ======================================================================

def compute_umap_from_pca(
    master: pd.DataFrame,
    pca_dimensions: int,
    config: PlotConfig,
) -> np.ndarray:
    """
    Compute a 2D UMAP visualization from the retained PCA space.

    If the selected clustering solution uses d PCA dimensions, UMAP
    receives PC1 ... PCd.

    UMAP remains visualization only.
    """

    if pca_dimensions < 2:

        raise ValueError(
            "pca_dimensions must be >= 2."
        )


    pc_columns = [
        f"PC{i}"
        for i in range(
            1,
            pca_dimensions + 1,
        )
    ]


    missing = [
        column
        for column in pc_columns
        if column not in master.columns
    ]


    if missing:

        raise ValueError(
            f"Missing PCA columns: {missing}"
        )


    X = master[
        pc_columns
    ].to_numpy(
        dtype=float
    )


    if not np.isfinite(
        X
    ).all():

        raise ValueError(
            "PCA scores contain non-finite values."
        )


    embedding = umap.UMAP(
        n_components=2,
        n_neighbors=(
            config
            .effective_umap_n_neighbors
        ),
        min_dist=(
            config
            .effective_umap_min_dist
        ),
        random_state=(
            config
            .effective_random_state
        ),
    ).fit_transform(
        X
    )


    return np.asarray(
        embedding,
        dtype=float,
    )


# ======================================================================
# PROTOTYPE MASKS
# ======================================================================

def _load_prototype_masks(
    mapping,
    *,
    strongest_k: int,
    pca_dimensions: int,
    ordered_clusters: list[int],
) -> dict[int, list[np.ndarray]]:
    """
    Load up to four empirical prototype masks per cluster.

    Preferred format
    ----------------
        cluster_X_rank_1.png
        cluster_X_rank_2.png
        cluster_X_rank_3.png
        cluster_X_rank_4.png

    Historical compatibility
    ------------------------
        cluster_X.png

    remains accepted as rank 1.

    Rank 1 is mandatory.
    Ranks 2-4 are optional.
    """

    masks: dict[
        int,
        list[np.ndarray],
    ] = {}


    for raw_cluster in ordered_clusters:

        roots = [
            (
                Path(
                    mapping.output_dir
                )
                / "_Prototypes"
                / "Prototype_Masks"
            ),

            (
                Path(
                    mapping.output_dir
                )
                / (
                    f"K{strongest_k}_"
                    f"d{pca_dimensions}"
                )
                / "Prototype_Masks"
            ),

            (
                Path(
                    mapping.output_dir
                )
                / (
                    f"K{strongest_k}_"
                    f"d{pca_dimensions}"
                )
                / "Selections"
                / "Prototype_Masks"
            ),
        ]


        cluster_masks = []


        for selection_rank in range(
            1,
            5,
        ):

            ranked_filename = (
                f"cluster_"
                f"{raw_cluster}"
                f"_rank_"
                f"{selection_rank}.png"
            )


            ranked_paths = [
                root
                / ranked_filename
                for root
                in roots
            ]


            mask_path = next(
                (
                    path
                    for path
                    in ranked_paths
                    if path.exists()
                ),
                None,
            )


            legacy_paths = []


            if (
                mask_path is None
                and selection_rank == 1
            ):

                legacy_filename = (
                    f"cluster_"
                    f"{raw_cluster}.png"
                )


                legacy_paths = [
                    root
                    / legacy_filename
                    for root
                    in roots
                ]


                mask_path = next(
                    (
                        path
                        for path
                        in legacy_paths
                        if path.exists()
                    ),
                    None,
                )


            if (
                mask_path is None
                and selection_rank == 1
            ):

                attempted = "\n".join(
                    [
                        *(
                            f"  - {path}"
                            for path
                            in ranked_paths
                        ),
                        *(
                            f"  - {path}"
                            for path
                            in legacy_paths
                        ),
                    ]
                )


                raise FileNotFoundError(
                    "Canonical prototype mask "
                    "not found. Attempted:\n"
                    f"{attempted}"
                )


            if mask_path is None:

                continue


            mask = cv2.imread(
                str(
                    mask_path
                ),
                cv2.IMREAD_GRAYSCALE,
            )


            if mask is None:

                raise FileNotFoundError(
                    "Prototype mask could not "
                    "be read: "
                    f"{mask_path}"
                )


            mask = (
                mask > 0
            ).astype(
                np.uint8
            )


            if mask.ndim != 2:

                raise RuntimeError(
                    "Prototype mask must be 2D: "
                    f"{mask_path}"
                )


            cluster_masks.append(
                mask
            )


        if not cluster_masks:

            raise RuntimeError(
                "No prototype masks recovered "
                f"for cluster {raw_cluster}."
            )


        masks[
            raw_cluster
        ] = cluster_masks


    return masks


# ======================================================================
# PROTOTYPE ROW
# ======================================================================

def _plot_prototype_row(
    fig,
    grid_slot,
    *,
    masks: dict[int, list[np.ndarray]],
    ordered_clusters: list[int],
    display_rank: dict[int, int],
) -> None:
    """
    Render up to four empirical prototypes per cluster in a 2 x 2 grid.

    Slot order
    ----------
        rank 1 | rank 2
        rank 3 | rank 4

    All cells share the same pixel canvas so their absolute morphology
    remains visually comparable.
    """

    all_masks = [
        mask
        for raw_cluster in ordered_clusters
        for mask in masks[
            raw_cluster
        ]
    ]


    if not all_masks:

        raise RuntimeError(
            "No prototype masks are available."
        )


    max_height = max(
        mask.shape[
            0
        ]
        for mask in all_masks
    )


    max_width = max(
        mask.shape[
            1
        ]
        for mask in all_masks
    )


    padding = 12


    canvas_height = (
        max_height
        + 2 * padding
    )


    canvas_width = (
        max_width
        + 2 * padding
    )


    cluster_grid = (
        grid_slot.subgridspec(
            1,
            len(
                ordered_clusters
            ),
            wspace=0.18,
        )
    )


    for (
        cluster_column,
        raw_cluster,
    ) in enumerate(
        ordered_clusters
    ):

        display_cluster = (
            display_rank[
                raw_cluster
            ]
        )


        color = mg_color(display_cluster - 1, n_colors=len(ordered_clusters))


        cluster_masks = masks[
            raw_cluster
        ]


        local_grid = (
            cluster_grid[
                0,
                cluster_column,
            ]
            .subgridspec(
                3,
                2,
                height_ratios=[
                    0.16,
                    1.0,
                    1.0,
                ],
                hspace=0.03,
                wspace=0.03,
            )
        )


        # ==============================================================
        # CLUSTER TITLE
        # ==============================================================

        title_axis = fig.add_subplot(
            local_grid[
                0,
                :,
            ]
        )


        title_axis.text(
            0.5,
            0.5,
            f"C{display_cluster}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            transform=(
                title_axis.transAxes
            ),
        )


        title_axis.axis(
            "off"
        )


        # ==============================================================
        # FOUR PROTOTYPE SLOTS
        # ==============================================================

        for slot_index in range(
            4
        ):

            row = (
                1
                + slot_index // 2
            )


            column = (
                slot_index
                % 2
            )


            axis = fig.add_subplot(
                local_grid[
                    row,
                    column,
                ]
            )


            axis.axis(
                "off"
            )


            if slot_index >= len(
                cluster_masks
            ):

                continue


            mask = cluster_masks[
                slot_index
            ]


            height, width = (
                mask.shape
            )


            canvas = np.zeros(
                (
                    canvas_height,
                    canvas_width,
                ),
                dtype=np.uint8,
            )


            y0 = (
                canvas_height
                - height
            ) // 2


            x0 = (
                canvas_width
                - width
            ) // 2


            canvas[
                y0:y0 + height,
                x0:x0 + width,
            ] = mask


            rgb = np.ones(
                (
                    canvas_height,
                    canvas_width,
                    3,
                ),
                dtype=float,
            )


            rgb[
                canvas > 0
            ] = to_rgb(
                color
            )


            axis.imshow(
                rgb,
                interpolation="nearest",
            )


# ======================================================================
# STATE SCATTER
# ======================================================================

def _plot_state_scatter(
    axis,
    *,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    ordered_clusters: list[int],
    display_rank: dict[int, int],
    config: PlotConfig,
) -> None:
    """
    Draw the morphology-state scatter using canonical display ordering.
    """

    for raw_cluster in ordered_clusters:

        display_cluster = (
            display_rank[
                raw_cluster
            ]
        )


        color = mg_color(display_cluster - 1, n_colors=len(ordered_clusters))


        selection = (
            labels
            == raw_cluster
        )


        axis.scatter(
            x[
                selection
            ],
            y[
                selection
            ],
            s=(
                config
                .effective_point_size
            ),
            alpha=(
                config
                .effective_point_alpha
            ),
            color=color,
            edgecolors="black",
            linewidths=(
                config
                .effective_point_edge_width
            ),
            label=(
                f"C{display_cluster}"
            ),
            zorder=2,
        )


# ======================================================================
# PROTOTYPE TRIANGLES
# ======================================================================

def _plot_prototype_triangles(
    axis,
    *,
    x: np.ndarray,
    y: np.ndarray,
    prototype_indices: np.ndarray,
    prototype_clusters: np.ndarray,
    display_rank: dict[int, int],
    config: PlotConfig,
) -> None:

    for (
        index,
        raw_cluster,
    ) in zip(
        prototype_indices,
        prototype_clusters,
    ):

        display_cluster = (
            display_rank[
                int(
                    raw_cluster
                )
            ]
        )


        color = mg_color(display_cluster - 1, n_colors=len(display_rank))


        scatter_double_outline(
            axis,
            x[
                index
            ],
            y[
                index
            ],
            marker="^",
            size=(
                config
                .effective_prototype_marker_size
            ),
            color=color,
        )


# ======================================================================
# SINGLE PUBLIC FIGURE
# ======================================================================

def _save_morphology_state_figure(
    *,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    prototypes: pd.DataFrame,
    prototype_indices: np.ndarray,
    prototype_clusters: np.ndarray,
    masks: dict[int, list[np.ndarray]],
    ordered_clusters: list[int],
    display_rank: dict[int, int],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    config: PlotConfig,
) -> Path:
    """
    Save one canonical morphology-state figure.

    Layout
    ------
    Top:
        PCA or UMAP morphology-state space.

    Bottom:
        one 2 x 2 empirical prototype block per cluster.
    """

    figure_width = max(
        10.5,
        2.8
        * len(
            ordered_clusters
        ),
    )


    fig = plt.figure(
        figsize=(
            figure_width,
            9.4,
        )
    )


    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[
            4.2,
            2.35,
        ],
        hspace=0.18,
    )


    axis = fig.add_subplot(
        grid[
            0,
            0,
        ]
    )


    # ==================================================================
    # SCATTER
    # ==================================================================

    _plot_state_scatter(
        axis,
        x=x,
        y=y,
        labels=labels,
        ordered_clusters=(
            ordered_clusters
        ),
        display_rank=(
            display_rank
        ),
        config=config,
    )


    # ==================================================================
    # CANONICAL PROTOTYPE TRIANGLES
    #
    # Only selection_rank == 1 remains the canonical point marker.
    # ==================================================================

    if config.show_prototypes:

        _plot_prototype_triangles(
            axis,
            x=x,
            y=y,
            prototype_indices=(
                prototype_indices
            ),
            prototype_clusters=(
                prototype_clusters
            ),
            display_rank=(
                display_rank
            ),
            config=config,
        )


    # ==================================================================
    # STYLE
    # ==================================================================

    style_axis(
        axis,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        config=config,
        hide_ticks=True,
    )


    add_cluster_legend(
        axis,
        config,
    )


    # ==================================================================
    # 2 x 2 PROTOTYPE BLOCKS
    # ==================================================================

    _plot_prototype_row(
        fig,
        grid[
            1,
            0,
        ],
        masks=masks,
        ordered_clusters=(
            ordered_clusters
        ),
        display_rank=(
            display_rank
        ),
    )


    fig.text(
        0.100,   #de izquierda a derecah
        0.230,   #de abajo a arriba
        "Prototypes",
        fontsize=16,
        fontweight="bold",
        rotation=90,
        va="center",
    )


    # ==================================================================
    # SAVE
    # ==================================================================

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
# PUBLIC MORPHOLOGY-STATE OUTPUT
# ======================================================================

def plot_morphology_states(
    master: pd.DataFrame,
    clustering,
    mapping,
    output_dir: str | Path,
    config: PlotConfig,
    umap_embedding: np.ndarray | None = None,
    umap_pca_dimensions: int | None = None,
) -> dict[str, Path]:
    """
    Generate the two canonical MorphoGlia morphology-state figures.

    Outputs
    -------
    Morphology_States_PCA.png
        PCA morphology space + observed prototype row.

    Morphology_States_UMAP.png
        UMAP visualization + observed prototype row.

    Both figures use:
        - preferred clustering resolution
        - identical cluster display ordering
        - identical MorphoGlia palette
        - identical observed prototypes

    UMAP is computed from the same d retained PCA components used by
    the preferred clustering solution.
    """

    master = (
        master
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ==================================================================
    # PREFERRED RESOLUTION
    # ==================================================================

    # ==================================================================

    # PREFERRED RESOLUTION

    #

    # Final morphology-state figures use the biological resolution

    # selected by the clustering decision layer.

    #

    # There is deliberately no fallback to strongest_k.

    # ==================================================================

    preferred_k = getattr(

        clustering,

        "preferred_k",

        None,

    )


    if preferred_k is None:

        raise ValueError(

            "Morphology-state plots require preferred_k "

            "from the clustering decision layer."

        )


    preferred_k = int(

        preferred_k

    )


    if (

        preferred_k

        not in clustering.selected_solutions

    ):

        raise RuntimeError(

            "Preferred clustering solution is missing: "

            f"K={preferred_k}."

        )


    solution = (

        clustering

        .selected_solutions[

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


    if len(
        labels
    ) != len(
        master
    ):

        raise RuntimeError(
            "Cluster-label count does not match "
            "the master table."
        )


    # ==================================================================
    # DISPLAY ORDER
    # ==================================================================

    (
        ordered_clusters,
        display_rank,
    ) = cluster_display_order(
        master=master,
        labels=labels,
        complexity_column="Cell_area",
    )


    # ==================================================================
    # PROTOTYPES
    # ==================================================================

    if (
        preferred_k
        not in mapping.selections
        or "prototype"
        not in mapping.selections[
            preferred_k
        ]
    ):

        raise RuntimeError(
            "Prototype Mapping results are missing."
        )


    prototypes = (
        mapping
        .selections[
            preferred_k
        ][
            "prototype"
        ]
    )


    prototype_indices = (
        _prototype_indices(
            master,
            prototypes,
        )
    )


    prototype_clusters = (
        prototypes[
            "cluster"
        ].to_numpy(
            dtype=int
        )
    )


    masks = (
        _load_prototype_masks(
            mapping,
            strongest_k=(
                preferred_k
            ),
            pca_dimensions=d,
            ordered_clusters=(
                ordered_clusters
            ),
        )
    )


    # ==================================================================
    # PCA GEOMETRY
    # ==================================================================

    pc1 = master[
        "PC1"
    ].to_numpy(
        dtype=float
    )


    pc2 = master[
        "PC2"
    ].to_numpy(
        dtype=float
    )


    # ==================================================================
    # UMAP GEOMETRY
    # ==================================================================

    if umap_embedding is None:

        umap_embedding = (
            compute_umap_from_pca(
                master=master,
                pca_dimensions=d,
                config=config,
            )
        )

        umap_pca_dimensions = int(d)

    else:

        umap_embedding = np.asarray(
            umap_embedding,
            dtype=float,
        )

        if umap_pca_dimensions is None:
            umap_pca_dimensions = int(d)


        if umap_embedding.shape != (
            len(master),
            2,
        ):

            raise ValueError(
                "umap_embedding must have shape "
                f"({len(master)}, 2)."
            )


    # ==================================================================
    # OUTPUT PATHS
    # ==================================================================

    output_dir = Path(
        output_dir
    )


    pca_path = (
        output_dir
        / "Morphology_States_PCA.png"
    )


    umap_path = (
        output_dir
        / "Morphology_States_UMAP.png"
    )


    # ==================================================================
    # PCA FIGURE
    # ==================================================================

    _save_morphology_state_figure(
        x=pc1,
        y=pc2,
        labels=labels,
        prototypes=prototypes,
        prototype_indices=(
            prototype_indices
        ),
        prototype_clusters=(
            prototype_clusters
        ),
        masks=masks,
        ordered_clusters=(
            ordered_clusters
        ),
        display_rank=(
            display_rank
        ),
        title=(
            f"PCA Morphology Space | "
            f"d = {d} PCs"
        ),
        xlabel="PC1",
        ylabel="PC2",
        output_path=pca_path,
        config=config,
    )


    # ==================================================================
    # UMAP FIGURE
    # ==================================================================

    _save_morphology_state_figure(
        x=umap_embedding[
            :,
            0,
        ],
        y=umap_embedding[
            :,
            1,
        ],
        labels=labels,
        prototypes=prototypes,
        prototype_indices=(
            prototype_indices
        ),
        prototype_clusters=(
            prototype_clusters
        ),
        masks=masks,
        ordered_clusters=(
            ordered_clusters
        ),
        display_rank=(
            display_rank
        ),
        title=(
            f"UMAP | "
            f"d = {int(umap_pca_dimensions)} PCs from PCA"
        ),
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        output_path=umap_path,
        config=config,
    )


    return {
        "pca":
            pca_path,

        "umap":
            umap_path,
    }





# ======================================================================
# MULTIRESOLUTION MORPHOLOGY-STATE QC
# ======================================================================

def plot_multiresolution_morphology_states(
    master: pd.DataFrame,
    clustering,
    output_path: str | Path,
    config: PlotConfig,
    umap_embedding: np.ndarray | None = None,
) -> Path:
    """
    Compare all robust morphology-state resolutions in one figure.

    Layout
    ------
                PCA             UMAP

        K1      partition       partition
        K2      partition       partition
        ...

    Important
    ---------
    PCA geometry is identical across rows:
        PC1 versus PC2.

    UMAP geometry is also identical across rows.

    A single shared UMAP embedding is calculated using the maximum
    representative PCA dimensionality among the robust K solutions.

    Therefore the geometry does NOT move when K changes.

    Only the clustering partition changes between rows.

    This figure is intended as analytical QC for deciding whether robust
    K values represent competing partitions or nested coarse-grainings.
    """

    master = (
        master
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ==================================================================
    # ROBUST RESOLUTIONS
    # ==================================================================

    decision_k_values = (
        getattr(
            clustering,
            "pareto_k_values",
            (),
        )
        or clustering.robust_k_values
    )


    robust_k = tuple(
        int(k)
        for k in decision_k_values
    )


    if not robust_k:

        raise ValueError(
            "No robust clustering resolutions are available."
        )


    missing_solutions = [
        k
        for k in robust_k
        if k not in clustering.selected_solutions
    ]


    if missing_solutions:

        raise ValueError(
            "Selected clustering solutions are missing for "
            f"K={missing_solutions}."
        )


    # ==================================================================
    # COMMON PCA GEOMETRY
    # ==================================================================

    required_pcs = [
        "PC1",
        "PC2",
    ]


    missing_pcs = [
        column
        for column in required_pcs
        if column not in master.columns
    ]


    if missing_pcs:

        raise ValueError(
            "Multiresolution morphology-state QC requires "
            f"PCA columns: {missing_pcs}"
        )


    pc1 = master[
        "PC1"
    ].to_numpy(
        dtype=float
    )


    pc2 = master[
        "PC2"
    ].to_numpy(
        dtype=float
    )


    # ==================================================================
    # COMMON UMAP GEOMETRY
    #
    # Do NOT calculate one UMAP independently for each K.
    #
    # Otherwise:
    #     K changes
    #     d changes
    #     geometry changes
    #
    # and the visual comparison becomes confounded.
    #
    # Use the largest representative d among the robust solutions so
    # every selected clustering representation is contained within the
    # common visualization space.
    # ==================================================================

    representative_dimensions = {
        k:
            int(
                clustering
                .selected_solutions[
                    k
                ]
                .pca_dimensions
            )
        for k in robust_k
    }


    common_umap_d = max(
        representative_dimensions.values()
    )


    if umap_embedding is None:
        umap_embedding = (
            compute_umap_from_pca(
                master=master,
                pca_dimensions=(
                    common_umap_d
                ),
                config=config,
            )
        )
    else:
        umap_embedding = np.asarray(
            umap_embedding,
            dtype=float,
        )
        if umap_embedding.shape != (len(master), 2):
            raise ValueError(
                "umap_embedding must have shape "
                f"({len(master)}, 2)."
            )


    umap_x = (
        umap_embedding[
            :,
            0,
        ]
    )


    umap_y = (
        umap_embedding[
            :,
            1,
        ]
    )


    # ==================================================================
    # STABILITY SUMMARY
    # ==================================================================

    summary = getattr(
        clustering,
        "multiresolution_summary",
        None,
    )


    if (
        isinstance(
            summary,
            pd.DataFrame,
        )
        and not summary.empty
    ):

        summary = (
            summary
            .copy()
            .set_index(
                "k",
                drop=False,
            )
        )

    else:

        summary = None


    # ==================================================================
    # COMMON AXIS LIMITS
    #
    # Explicitly fixed so rows are geometrically comparable.
    # ==================================================================

    def limits(
        values: np.ndarray,
        fraction: float = 0.04,
    ) -> tuple[float, float]:

        values = np.asarray(
            values,
            dtype=float,
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

        span = (
            maximum
            - minimum
        )


        if span <= 0:

            span = 1.0


        margin = (
            fraction
            * span
        )


        return (
            minimum - margin,
            maximum + margin,
        )


    pca_xlim = limits(
        pc1
    )

    pca_ylim = limits(
        pc2
    )

    umap_xlim = limits(
        umap_x
    )

    umap_ylim = limits(
        umap_y
    )


    # ==================================================================
    # FIGURE
    # ==================================================================

    n_rows = len(
        robust_k
    )


    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(
            15.5,
            5.0 * n_rows,
        ),
        squeeze=False,
    )


    # ==================================================================
    # ONE ROW PER ROBUST K
    # ==================================================================

    for (
        row_index,
        k,
    ) in enumerate(
        robust_k
    ):

        solution = (
            clustering
            .selected_solutions[
                k
            ]
        )


        selected_d = int(
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
                f"K={k} labels contain "
                f"{len(labels)} rows but master contains "
                f"{len(master)} cells."
            )


        # ==============================================================
        # SAME HUMAN-FACING STATE ORDERING USED EVERYWHERE ELSE
        # ==============================================================

        (
            ordered_clusters,
            display_rank,
        ) = cluster_display_order(
            master=master,
            labels=labels,
            complexity_column="Cell_area",
        )


        pca_axis = axes[
            row_index,
            0,
        ]


        umap_axis = axes[
            row_index,
            1,
        ]


        # ==============================================================
        # PCA
        # ==============================================================

        _plot_state_scatter(
            pca_axis,
            x=pc1,
            y=pc2,
            labels=labels,
            ordered_clusters=(
                ordered_clusters
            ),
            display_rank=(
                display_rank
            ),
            config=config,
        )


        # ==============================================================
        # UMAP
        # ==============================================================

        _plot_state_scatter(
            umap_axis,
            x=umap_x,
            y=umap_y,
            labels=labels,
            ordered_clusters=(
                ordered_clusters
            ),
            display_rank=(
                display_rank
            ),
            config=config,
        )


        # ==============================================================
        # STABILITY NUMBERS
        # ==============================================================

        metric_text = ""


        if (
            summary is not None
            and k in summary.index
        ):

            row = summary.loc[
                k
            ]


            mean_stability = float(
                row[
                    "mean_subsample_ari"
                ]
            )


            worst_stability = float(
                row[
                    "minimum_subsample_ari"
                ]
            )


            dimension_agreement = float(
                row[
                    "mean_dimension_agreement"
                ]
            )


            metric_text = (
                f"mean={mean_stability:.3f}  |  "
                f"worst={worst_stability:.3f}  |  "
                f"across-d={dimension_agreement:.3f}"
            )


        # ==============================================================
        # STYLE
        # ==============================================================

        pca_title = (
            f"K={k}  |  selected d={selected_d}"
        )


        if metric_text:

            pca_title = (
                f"{pca_title}\n"
                f"{metric_text}"
            )


        style_axis(
            pca_axis,
            title=pca_title,
            xlabel="PC1",
            ylabel="PC2",
            config=config,
            hide_ticks=True,
        )


        style_axis(
            umap_axis,
            title=(
                f"K={k}  |  "
                f"shared UMAP from d={common_umap_d}"
            ),
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            config=config,
            hide_ticks=True,
        )


        # ==============================================================
        # IDENTICAL GEOMETRIC LIMITS ACROSS K
        # ==============================================================

        pca_axis.set_xlim(
            pca_xlim
        )

        pca_axis.set_ylim(
            pca_ylim
        )


        umap_axis.set_xlim(
            umap_xlim
        )

        umap_axis.set_ylim(
            umap_ylim
        )


        # ==============================================================
        # LEGEND
        #
        # One per row because the number of states changes with K.
        # ==============================================================

        add_cluster_legend(
            pca_axis,
            config,
        )


    # ==================================================================
    # COLUMN HEADERS
    # ==================================================================

    axes[
        0,
        0,
    ].text(
        0.5,
        1.18,
        "PCA",
        transform=(
            axes[
                0,
                0,
            ].transAxes
        ),
        ha="center",
        va="bottom",
        fontsize=20,
        fontweight="bold",
    )


    axes[
        0,
        1,
    ].text(
        0.5,
        1.18,
        "UMAP",
        transform=(
            axes[
                0,
                1,
            ].transAxes
        ),
        ha="center",
        va="bottom",
        fontsize=20,
        fontweight="bold",
    )


    # ==================================================================
    # GLOBAL TITLE
    # ==================================================================

    fig.suptitle(
        (
            "Robust Morphology-State Resolutions\n"
            "Common geometry across K"
        ),
        fontsize=22,
        fontweight="bold",
        y=0.995,
    )


    fig.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.965,
        ]
    )


    # ==================================================================
    # SAVE
    # ==================================================================

    output_path = Path(
        output_path
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