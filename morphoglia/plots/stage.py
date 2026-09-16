from __future__ import annotations

import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from ..core.instance_map import validate_instance_map
from ..category import (
    CATEGORY_COLUMN,
)

from .config import (
    PlotConfig,
)

from ..mapping.rendering import (
    render_selection_montage,
    extract_component_crop,
)

from .morphospace import (
    compute_umap_from_pca,
    plot_morphology_states,
)

from .categories import (
    plot_category_states,
)

from .composition import (
    plot_category_composition,
)

from .profiles import (
    plot_morphometric_profiles,
)

from .stability import (
    plot_pca_variance_explained,
    plot_stability_heatmap,
    plot_multiresolution_stability,
)

from .temporal import (
    plot_timepoint_sankey,
)


# ======================================================================
# RESULT
# ======================================================================

@dataclass
class PlotsResult:
    """
    Result of the canonical MorphoGlia Plots stage.
    """

    files: dict[
        str,
        Path,
    ]

    category_available: bool

    output_dir: Path


# ======================================================================
# STAGE
# ======================================================================

class PlotsStage:
    """
    Generate the canonical MorphoGlia scientific figures.

    This stage is dataset-neutral.

    Category-dependent plots consume only the reserved 'category'
    column produced upstream by CategoryStage.

    UMAP is calculated once per run and reused by all plots.
    """

    def __init__(
        self,
        config: PlotConfig,
    ):

        self.config = config


    # ==================================================================
    # RUN
    # ==================================================================

    def run(
        self,
        master: pd.DataFrame,
        clustering,
        mapping,
        output_dir: str | Path,
        category_fields: Sequence[str] = (),
        analytical_features: Sequence[str] = (),
        instance_source=None,
    ) -> PlotsResult:

        output_dir = Path(
            output_dir
        )


        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        files: dict[
            str,
            Path,
        ] = {}


        category_fields = tuple(
            str(field)
            for field in category_fields
        )


        analytical_features = tuple(
            str(feature)
            for feature in analytical_features
        )


        if (
            category_fields
            and CATEGORY_COLUMN
            not in master.columns
        ):

            raise ValueError(
                "Category fields were supplied to PlotsStage, "
                "but CategoryStage did not create the reserved "
                f"{CATEGORY_COLUMN!r} column."
            )


        if (
            category_fields
            and not analytical_features
        ):

            raise ValueError(
                "Category-dependent morphometric profiles require "
                "FeaturePreparationResult.analytical_features."
            )


        # ==============================================================
        # STRONGEST MORPHOLOGY-STATE SOLUTION
        # ==============================================================

        # ==============================================================
        # PREFERRED MORPHOLOGY-STATE SOLUTION
        #
        # Final biological plots share the single resolution selected by
        # the explicit clustering decision layer.
        #
        # strongest_k remains an analytical maximin diagnostic and does
        # not control final plots.
        # ==============================================================

        preferred_k = getattr(
            clustering,
            "preferred_k",
            None,
        )


        if preferred_k is None:

            raise ValueError(
                "PlotsStage requires preferred_k from "
                "the clustering decision layer."
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
                f"solution K={preferred_k} is unavailable."
            )


        preferred_solution = (
            selected_solutions[
                preferred_k
            ]
        )


        pca_dimensions = int(
            preferred_solution
            .pca_dimensions
        )


        # ==============================================================
        # SHARED UMAP
        #
        # Calculated once.
        #
        # Every UMAP-based public figure therefore uses exactly the
        # same visualization geometry.
        # ==============================================================

        shared_umap = getattr(
            clustering,
            "shared_umap_embedding",
            None,
        )
        shared_umap_d = getattr(
            clustering,
            "shared_umap_pca_dimensions",
            None,
        )

        if shared_umap is not None:
            shared_umap = np.asarray(
                shared_umap,
                dtype=float,
            )

        if (
            shared_umap is not None
            and shared_umap.shape == (len(master), 2)
        ):
            umap_embedding = shared_umap
            if shared_umap_d is None:
                shared_umap_d = int(pca_dimensions)
        else:
            supported_k_values = tuple(
                int(k)
                for k in getattr(
                    clustering,
                    "supported_k_values",
                    getattr(clustering, "robust_k_values", ()),
                )
            )
            supported_dimensions = [
                int(selected_solutions[k].pca_dimensions)
                for k in supported_k_values
                if k in selected_solutions
            ]
            shared_umap_d = (
                max(supported_dimensions)
                if supported_dimensions
                else int(pca_dimensions)
            )
            umap_embedding = (
                compute_umap_from_pca(
                    master=master,
                    pca_dimensions=(shared_umap_d),
                    config=self.config,
                )
            )


        # ==============================================================
        # 1. MORPHOLOGY STATES
        # ==============================================================

        morphology_paths = (
            plot_morphology_states(
                master=master,
                clustering=clustering,
                mapping=mapping,
                output_dir=(
                    output_dir
                    / "Morphology_States"
                ),
                config=self.config,
                umap_embedding=(
                    umap_embedding
                ),
                umap_pca_dimensions=(
                    int(shared_umap_d)
                ),
            )
        )


        files[
            "morphology_states_pca"
        ] = morphology_paths[
            "pca"
        ]


        files[
            "morphology_states_umap"
        ] = morphology_paths[
            "umap"
        ]


        # ==============================================================
        # PUBLIC PROTOTYPE GRID
        #
        # Do NOT copy the already-colored DRC QC PNG.
        #
        # Prototype identity and masks are saved analytical results.
        # Palette is a presentation setting, therefore the public Plots
        # artifact is rebuilt from the saved neutral masks every time
        # Plots runs.
        # ==============================================================

        prototype_solution_dir = (
            Path(
                mapping.output_dir
            )
            / (
                f"K{preferred_k}_"
                f"d{pca_dimensions}"
            )
        )

        prototype_table_path = (
            prototype_solution_dir
            / "prototype_cells.csv"
        )

        prototype_masks_dir = (
            prototype_solution_dir
            / "Prototype_Masks"
        )

        if not prototype_table_path.is_file():

            raise FileNotFoundError(
                "Saved prototype-cell table is missing: "
                f"{prototype_table_path}"
            )

        if not prototype_masks_dir.is_dir():

            raise FileNotFoundError(
                "Saved prototype-mask directory is missing: "
                f"{prototype_masks_dir}"
            )


        prototype_table = pd.read_csv(
            prototype_table_path
        )


        required_prototype_columns = {
            "cluster",
            "selection_rank",
        }

        missing_prototype_columns = (
            required_prototype_columns
            - set(
                prototype_table.columns
            )
        )

        if missing_prototype_columns:

            raise ValueError(
                "Prototype table is missing required columns: "
                f"{sorted(missing_prototype_columns)}"
            )


        prototype_table = (
            prototype_table
            .copy()
            .reset_index(
                drop=True
            )
        )


        prototype_table[
            "cluster"
        ] = (
            prototype_table[
                "cluster"
            ]
            .astype(int)
        )

        prototype_table[
            "selection_rank"
        ] = (
            prototype_table[
                "selection_rank"
            ]
            .astype(int)
        )


        # Public prototype montage uses the same six representatives per
        # morphology state used by the DRC QC montage.
        prototype_table = (
            prototype_table[
                prototype_table[
                    "selection_rank"
                ]
                .between(
                    1,
                    6,
                )
            ]
            .copy()
        )


        # If DRC retained both analytical/raw and human-facing display
        # identities, use the display identity for presentation.
        if (
            "display_cluster"
            in prototype_table.columns
        ):

            raw_clusters = (
                prototype_table[
                    "cluster"
                ]
                .astype(int)
                .copy()
            )

            prototype_table[
                "cluster"
            ] = (
                prototype_table[
                    "display_cluster"
                ]
                .astype(int)
            )

        else:

            raw_clusters = (
                prototype_table[
                    "cluster"
                ]
                .astype(int)
                .copy()
            )


        precomputed_crops = {}

        missing_masks = []


        for (
            row_index,
            row,
        ) in prototype_table.iterrows():

            display_cluster = int(
                row[
                    "cluster"
                ]
            )

            raw_cluster = int(
                raw_clusters.loc[
                    row_index
                ]
            )

            rank = int(
                row[
                    "selection_rank"
                ]
            )


            candidates = [
                (
                    prototype_masks_dir
                    / (
                        f"cluster_{raw_cluster}"
                        f"_rank_{rank}.png"
                    )
                ),
                (
                    prototype_masks_dir
                    / (
                        f"cluster_{display_cluster}"
                        f"_rank_{rank}.png"
                    )
                ),
            ]


            # Historical rank-1 canonical mask.
            if rank == 1:

                candidates.extend(
                    [
                        (
                            prototype_masks_dir
                            / (
                                f"cluster_{raw_cluster}.png"
                            )
                        ),
                        (
                            prototype_masks_dir
                            / (
                                f"cluster_{display_cluster}.png"
                            )
                        ),
                    ]
                )


            mask_path = next(
                (
                    candidate
                    for candidate
                    in candidates
                    if candidate.is_file()
                ),
                None,
            )


            if mask_path is None:

                missing_masks.append(
                    (
                        raw_cluster,
                        display_cluster,
                        rank,
                    )
                )

                continue


            precomputed_crops[
                (
                    display_cluster,
                    rank,
                )
            ] = mask_path


        if missing_masks:

            # ----------------------------------------------------------
            # LEGACY PROTOTYPE-MASK MIGRATION
            #
            # Older MorphoGlia runs persisted neutral masks only for
            # prototype ranks 1-4, although prototype_cells.csv contains
            # ranks 1-6.
            #
            # Recover only the missing cells from the authoritative
            # effective instance maps, cache the neutral masks, and then
            # continue with normal presentation rendering.
            #
            # PCA, GMM clustering, state ordering, and prototype selection
            # are NOT recomputed.
            # ----------------------------------------------------------

            if instance_source is None:

                raise RuntimeError(
                    "Prototype ranks are missing from the legacy mask "
                    "cache, but no effective instance-map source was "
                    "provided to the Plots stage."
                )


            image_cache = {}


            for (
                raw_cluster,
                display_cluster,
                rank,
            ) in missing_masks:


                candidates = (
                    prototype_table[
                        prototype_table[
                            "cluster"
                        ]
                        .astype(int)
                        .eq(
                            display_cluster
                        )
                        & prototype_table[
                            "selection_rank"
                        ]
                        .astype(int)
                        .eq(
                            rank
                        )
                    ]
                )


                if len(
                    candidates
                ) != 1:

                    raise RuntimeError(
                        "Could not uniquely recover prototype cell for "
                        f"raw cluster {raw_cluster}, "
                        f"display cluster {display_cluster}, "
                        f"rank {rank}. "
                        f"Found {len(candidates)} rows."
                    )


                prototype_cell = (
                    candidates
                    .iloc[0]
                    .copy()
                )


                image_id = str(
                    prototype_cell[
                        "image_id"
                    ]
                )


                # ------------------------------------------------------
                # Lazy load only the source images actually needed for
                # the missing prototype ranks.
                # ------------------------------------------------------

                if image_id not in image_cache:

                    try:

                        instance_path = Path(
                            instance_source[
                                image_id
                            ]
                        )

                    except (
                        TypeError,
                        KeyError,
                    ):

                        instance_path = (
                            Path(
                                instance_source
                            )
                            / f"{image_id}.tif"
                        )


                    if not instance_path.is_file():

                        raise FileNotFoundError(
                            "Effective instance map required to recover "
                            "a legacy prototype mask was not found: "
                            f"{instance_path}"
                        )


                    instance_map = (
                        validate_instance_map(
                            np.asarray(
                                tifffile.imread(
                                    str(
                                        instance_path
                                    )
                                )
                            )
                        )
                    )


                    if instance_map.ndim != 2:

                        raise ValueError(
                            "Prototype reconstruction currently requires "
                            "2D instance maps, but "
                            f"{instance_path.name!r} has shape "
                            f"{instance_map.shape}."
                        )


                    image_cache[
                        image_id
                    ] = {
                        "instance_map":
                            instance_map,
                    }


                mask = (
                    extract_component_crop(
                        cell=(
                            prototype_cell
                        ),
                        image_cache=(
                            image_cache
                        ),
                    )
                )


                if mask is None:

                    raise RuntimeError(
                        "Could not recover exact component for "
                        f"image_id={image_id}, "
                        f"raw cluster={raw_cluster}, "
                        f"display cluster={display_cluster}, "
                        f"rank={rank}."
                    )


                mask = np.asarray(
                    mask,
                    dtype=np.uint8,
                )


                mask_path = (
                    prototype_masks_dir
                    / (
                        f"cluster_{raw_cluster}"
                        f"_rank_{rank}.png"
                    )
                )


                mask_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )


                Image.fromarray(
                    mask
                ).save(
                    mask_path
                )


                # Immediately make the newly cached mask available to
                # this same Plots run.
                precomputed_crops[
                    (
                        display_cluster,
                        rank,
                    )
                ] = mask_path


                print(
                    "Cached legacy prototype mask: "
                    f"C{display_cluster} rank {rank} "
                    f"<- {image_id}"
                )


        final_prototype_grid = (
            output_dir
            / "Morphology_States"
            / "Prototype_Cells.png"
        )


        final_prototype_grid.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        render_selection_montage(
            selection=(
                prototype_table
            ),
            image_cache={},
            output_path=(
                final_prototype_grid
            ),
            title=(
                "Prototype Cells | "
                f"K = {preferred_k}, "
                f"d = {pca_dimensions}"
            ),
            label_columns=(),
            ncols=3,
            max_cells_per_cluster=6,
            show_full_id=False,
            precomputed_crops=(
                precomputed_crops
            ),
            palette=(
                self.config
                .effective_morphology_state_palette
            ),
            dpi=(
                self.config
                .effective_dpi
            ),
        )


        files[
            "morphology_states_prototypes"
        ] = final_prototype_grid


        # ==============================================================
        # 2. CATEGORY
        #
        # Automatically enabled when CategoryStage created the
        # canonical reserved category column.
        # ==============================================================

        category_available = (
            bool(
                category_fields
            )
            and CATEGORY_COLUMN
            in master.columns
        )


        if category_available:

            category_paths = (
                plot_category_states(
                    master=master,
                    clustering=clustering,
                    output_dir=(
                        output_dir
                        / "Category"
                    ),
                    config=self.config,
                    umap_embedding=(
                        umap_embedding
                    ),
                    category_fields=(
                        category_fields
                    ),
                )
            )


            files[
                "category_pca"
            ] = category_paths[
                "pca"
            ]


            files[
                "category_umap"
            ] = category_paths[
                "umap"
            ]


            # ==========================================================
            # CATEGORY × MORPHOLOGY-STATE COMPOSITION
            # ==========================================================

            composition_files = (
                plot_category_composition(
                    master=master,
                    clustering=clustering,
                    category_fields=(
                        category_fields
                    ),
                    output_dir=(
                        output_dir
                        / "Composition"
                    ),
                    config=self.config,
                )
            )


            for (
                composition_key,
                composition_path,
            ) in composition_files.items():

                files[
                    f"composition_{composition_key}"
                ] = composition_path


            # ==========================================================
            # TIMEPOINT SANKEY / ALLUVIAL
            #
            # Automatically generated when one selected Category field has
            # timepoint semantics. This is a descriptive composition plot,
            # not an inferred single-cell transition model.
            # ==========================================================

            sankey_path = (
                plot_timepoint_sankey(
                    master=master,
                    clustering=clustering,
                    category_fields=(
                        category_fields
                    ),
                    output_path=(
                        output_dir
                        / "Composition"
                        / "Plots"
                        / "Category_State_Timepoint_Sankey.png"
                    ),
                    config=self.config,
                )
            )

            if sankey_path is not None:
                files[
                    "composition_timepoint_sankey_plot"
                ] = sankey_path


            # ==========================================================
            # MORPHOMETRIC PROFILES
            #
            # Uses exactly the analytical feature set produced upstream
            # by Feature Preparation.
            #
            # No numeric columns are rediscovered from master.
            # ==========================================================

            profile_files = (
                plot_morphometric_profiles(
                    master=master,
                    clustering=clustering,
                    analytical_features=(
                        analytical_features
                    ),
                    category_fields=(
                        category_fields
                    ),
                    output_dir=(
                        output_dir
                        / "Profiles"
                    ),
                    config=self.config,
                )
            )


            for (
                profile_key,
                profile_path,
            ) in profile_files.items():

                files[
                    f"profiles_{profile_key}"
                ] = profile_path


        # ==============================================================
        # 3. PCA CUMULATIVE VARIANCE EXPLAINED
        #
        # Reuse the exact DRC QC artifact. If older saved DRC output
        # predates this figure, rebuild it from dimensionality.csv and
        # backfill the QC location first.
        # ==============================================================

        drc_root = (
            output_dir
            .parent
            / "Dimensionality_Reduction_Clustering"
        )

        dimensionality_path = (
            drc_root
            / "Dimensionality_Reduction"
            / "dimensionality.csv"
        )

        dimension_summary_path = (
            drc_root
            / "Dimensionality_Reduction"
            / "dimension_summary.csv"
        )

        pca_variance_qc_path = (
            drc_root
            / "QC"
            / "Stability"
            / "PCA_Variance_Explained.png"
        )

        pca_variance_plot_path = (
            output_dir
            / "Stability"
            / "PCA_Variance_Explained.png"
        )

        # Public Plots output is rendered from the saved analytical
        # dimensionality table rather than copied from the DRC QC PNG.
        #
        # This deliberately separates:
        #
        #   DRC/QC     -> artifact from the analytical DRC run
        #   Plots_*    -> current presentation rendering
        #
        # Re-rendering here ensures that changes to plotting style and
        # user-selected DPI take effect without rerunning PCA/clustering.
        if dimensionality_path.is_file():

            dimensionality_table = pd.read_csv(
                dimensionality_path
            )

            plausible_min = None
            plausible_max = None

            if dimension_summary_path.is_file():

                dimension_summary = pd.read_csv(
                    dimension_summary_path
                )

                if not dimension_summary.empty:

                    summary_row = (
                        dimension_summary.iloc[0]
                    )

                    lower = summary_row.get(
                        "plausible_dimension_min"
                    )

                    upper = summary_row.get(
                        "plausible_dimension_max"
                    )

                    if pd.notna(lower):
                        plausible_min = int(
                            lower
                        )

                    if pd.notna(upper):
                        plausible_max = int(
                            upper
                        )

            plot_pca_variance_explained(
                dimensionality_table,
                selected_dimension=pca_dimensions,
                plausible_dimension_min=plausible_min,
                plausible_dimension_max=plausible_max,
                output_path=pca_variance_plot_path,
                config=self.config,
            )

            files[
                "pca_variance_explained"
            ] = pca_variance_plot_path

        elif pca_variance_qc_path.is_file():

            # Backward-compatible fallback for historical output
            # folders that contain the QC figure but not the saved
            # dimensionality table required for re-rendering.
            pca_variance_plot_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                pca_variance_qc_path,
                pca_variance_plot_path,
            )

            files[
                "pca_variance_explained"
            ] = pca_variance_plot_path


        # 4. STABILITY HEATMAP
        # ==============================================================

        stability_table = getattr(
            clustering,
            "stability_by_dimension_and_k",
            None,
        )


        if (
            isinstance(
                stability_table,
                pd.DataFrame,
            )
            and not stability_table.empty
        ):

            heatmap_path = (
                output_dir
                / "Stability"
                / "Stability_Heatmap.png"
            )


            plot_stability_heatmap(
                clustering=clustering,
                output_path=(
                    heatmap_path
                ),
                config=self.config,
            )


            files[
                "stability_heatmap"
            ] = heatmap_path


        # ==============================================================
        # 5. MULTIRESOLUTION STABILITY
        # ==============================================================

        summary_table = getattr(
            clustering,
            "multiresolution_summary",
            None,
        )


        if (
            isinstance(
                summary_table,
                pd.DataFrame,
            )
            and not summary_table.empty
        ):

            multiresolution_path = (
                output_dir
                / "Stability"
                / "Multiresolution_Stability.png"
            )


            plot_multiresolution_stability(
                clustering=clustering,
                output_path=(
                    multiresolution_path
                ),
                config=self.config,
            )


            files[
                "multiresolution_stability"
            ] = (
                multiresolution_path
            )


        # ==============================================================
        # RESULT
        # ==============================================================

        return PlotsResult(
            files=files,
            category_available=(
                category_available
            ),
            output_dir=(
                output_dir
            ),
        )
