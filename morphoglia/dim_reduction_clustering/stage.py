from __future__ import annotations

# MG_RESUME_DRC_V1
import hashlib
import pickle
from dataclasses import fields as dataclass_fields, is_dataclass

from ..checkpoint import (
    CheckpointJournal,
    atomic_write_bytes,
    fingerprint,
    source_digest,
)

from ..plots.style import morphology_state_rgb

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import tifffile

from .clustering import (
    ClusteringEngine,
)

from .decision import (
    evaluate_resolution_decision,
    build_preferred_cluster_interpretation,
)
from .dimensionality import (
    DimensionalityReductionEngine,
)
from .feature_preparation import (
    FeaturePreparationEngine,
)
from ..core.instance_map import (
    validate_instance_map,
)
from ..mapping.rendering import (
    extract_component_crop,
    render_selection_montage,
)
from ..mapping.selection import (
    select_cluster_prototypes,
)
from ..morphometrics import (
    METRIC_DESCRIPTIONS,
)
from ..plots.morphospace import (
    compute_umap_from_pca,
    plot_morphology_states,
    plot_multiresolution_morphology_states,
)


from ..plots.stability import (
    plot_multiresolution_stability,
    plot_pca_variance_explained,
    plot_stability_heatmap,
)


from ..plots.style import (
    cluster_display_order,
)

# ======================================================================
# RESULT
# ======================================================================

@dataclass
class DimReductionClusteringResult:
    """
    Complete result of the combined analytical morphology-state stage.

    Feature preparation, dimensionality reduction, clustering, and their
    analytical QC remain separate internal result objects, but are exposed
    to the researcher as one pipeline stage.
    """

    master: pd.DataFrame
    feature_preparation: object
    dimensionality_reduction: object
    clustering: object
    prototype_selections: dict[int, pd.DataFrame]
    files: dict[str, Path]
    output_dir: Path


# ======================================================================
# PCA SCORES
# ======================================================================

def _attach_pca_scores(
    master: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:

    output = (
        master
        .copy()
        .reset_index(
            drop=True
        )
    )

    scores = np.asarray(
        scores
    )

    if scores.ndim != 2:
        raise ValueError(
            "PCA scores must be a 2D array."
        )

    if scores.shape[0] != len(output):
        raise ValueError(
            "PCA score rows do not match master-table rows: "
            f"{scores.shape[0]} versus {len(output)}."
        )

    stale_pc_columns = [
        column
        for column in output.columns
        if (
            column.startswith("PC")
            and column[2:].isdigit()
        )
    ]

    if stale_pc_columns:
        output = output.drop(
            columns=stale_pc_columns
        )

    for index in range(
        scores.shape[1]
    ):
        output[
            f"PC{index + 1}"
        ] = scores[
            :,
            index,
        ]

    return output


# ======================================================================
# MORPHOMETRIC CONTRACT
# ======================================================================

def _morphometric_columns(
    master: pd.DataFrame,
) -> list[str]:
    """
    Recover the canonical Morphometrics output contract.

    Columns are selected from MorphoGlia's registered metric vocabulary,
    never by generic numeric-column discovery.
    """

    columns = [
        feature
        for feature in METRIC_DESCRIPTIONS
        if feature in master.columns
    ]

    if not columns:
        raise ValueError(
            "No registered MorphoGlia morphometric columns were found "
            "in the master table."
        )

    return columns


# ======================================================================
# CLUSTER SOLUTION ATTACHMENT
# ======================================================================

def _attach_solution(
    master: pd.DataFrame,
    solution,
) -> pd.DataFrame:

    n_cells = len(
        master
    )

    labels = np.asarray(
        solution.labels
    )

    probability = np.asarray(
        solution.cluster_probability
    )

    reliability = np.asarray(
        solution.consensus_reliability
    )

    for name, values in {
        "labels": labels,
        "cluster_probability": probability,
        "consensus_reliability": reliability,
    }.items():

        if len(values) != n_cells:
            raise ValueError(
                f"Clustering solution {name} contains "
                f"{len(values)} rows but master contains "
                f"{n_cells} cells."
            )

    data = master.copy()

    data["cluster"] = (
        labels.astype(int)
    )

    data["cluster_probability"] = (
        probability.astype(float)
    )

    data["consensus_reliability"] = (
        reliability.astype(float)
    )

    data["pca_dimensions"] = int(
        solution.pca_dimensions
    )

    data["covariance_type"] = str(
        solution.covariance_type
    )

    return data


# ======================================================================
# INSTANCE-MAP CACHE
# ======================================================================

class _LazyInstanceCache(dict):
    """Load only the effective instance map requested by prototype/cluster QC."""

    def __init__(
        self,
        master: pd.DataFrame,
        instance_dir,
    ):
        super().__init__()

        self.allowed = set(
            master[
                "image_id"
            ].astype(str).unique()
        )

        if isinstance(
            instance_dir,
            Mapping,
        ):

            self.instance_paths_by_image = {
                str(
                    image_id
                ): Path(
                    path
                )
                for image_id, path in instance_dir.items()
            }

            self.instance_dir = None

        else:

            self.instance_dir = Path(
                instance_dir
            )

            if not self.instance_dir.exists():
                raise FileNotFoundError(
                    "Prototype QC requires canonical instance maps, "
                    f"but the directory does not exist: {self.instance_dir}"
                )

            self.instance_paths_by_image = None

    def __missing__(
        self,
        image_id,
    ):
        image_id = str(
            image_id
        )

        if image_id not in self.allowed:
            raise KeyError(
                "Unknown image_id requested by prototype QC: "
                f"{image_id}"
            )

        if self.instance_paths_by_image is not None:

            if image_id not in self.instance_paths_by_image:
                raise FileNotFoundError(
                    "Effective instance resolver has no path for "
                    f"image_id={image_id!r}."
                )

            path = (
                self.instance_paths_by_image[
                    image_id
                ]
            )

        else:

            path = (
                self.instance_dir
                / f"{image_id}.tif"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing canonical instance map: {path}"
            )

        instance_map = validate_instance_map(
            np.asarray(
                tifffile.imread(
                    str(
                        path
                    )
                )
            )
        )

        if instance_map.ndim != 2:
            raise ValueError(
                "Current prototype rendering requires 2D instance maps, "
                f"but {path.name!r} has shape {instance_map.shape}."
            )

        value = {
            "instance_map": (
                instance_map
            )
        }

        self[
            image_id
        ] = value

        return value


def _build_instance_cache(master: pd.DataFrame, instance_dir) -> dict:
    return _LazyInstanceCache(master=master, instance_dir=instance_dir)





# ======================================================================
# PROTOTYPE QC
# ======================================================================

def _save_prototype_qc(
    master: pd.DataFrame,
    clustering,
    instance_dir: Path,
    output_dir: Path,
    mapping_config,
    k_values: tuple[int, ...] | None = None,
) -> tuple[
    dict[int, pd.DataFrame],
    object,
    dict[str, Path],
]:
    """
    Save empirical prototype-cell QC for every robust clustering
    resolution.

    Output policy
    -------------
    preferred_k
        Canonical biological resolution.

        Saves:
            prototype_cells.csv
            Prototype_Cells.png
            rank-1 Prototype_Masks/

    other robust K
        Alternative supported analytical resolutions.

        Save:
            prototype_cells.csv
            Prototype_Cells.png

        These alternatives remain QC only and do not receive canonical
        prototype masks.

    All resolutions use six observed prototype cells per cluster and the
    same native-scale 3 x 2 renderer.
    """

    prototype_root = (
        output_dir
        / "QC"
        / "Prototype_Cells"
    )


    prototype_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Lazy cache: no TIFF is loaded here. Each required prototype source
    # image is loaded on first access by _LazyInstanceCache.__missing__().
    image_cache = _build_instance_cache(
        master=master,
        instance_dir=instance_dir,
    )


    selections: dict[
        int,
        pd.DataFrame,
    ] = {}


    files: dict[
        str,
        Path,
    ] = {}


    # ==================================================================
    # CANONICAL DECISION
    # ==================================================================

    preferred_k = getattr(
        clustering,
        "preferred_k",
        None,
    )


    if preferred_k is None:

        raise ValueError(
            "Prototype QC requires preferred_k from "
            "the clustering decision layer."
        )


    preferred_k = int(
        preferred_k
    )


    robust_k_values = tuple(
        int(k)
        for k in getattr(
            clustering,
            "robust_k_values",
            (),
        )
    )


    if not robust_k_values:

        raise ValueError(
            "Prototype QC requires at least one "
            "robust clustering resolution."
        )


    export_k_values = (
        robust_k_values
        if k_values is None
        else tuple(dict.fromkeys(int(k) for k in k_values))
    )

    unsupported = tuple(k for k in export_k_values if k not in robust_k_values)
    if unsupported:
        raise ValueError(
            "Prototype QC requested unsupported morphology-state counts: "
            + ", ".join(str(k) for k in unsupported)
        )


    if (
        preferred_k
        not in robust_k_values
    ):

        raise RuntimeError(
            "Preferred clustering resolution "
            f"K={preferred_k} is not present in "
            f"robust_k_values={robust_k_values}."
        )


    selected_solutions = getattr(
        clustering,
        "selected_solutions",
        None,
    )


    if selected_solutions is None:

        raise RuntimeError(
            "Prototype QC requires selected "
            "clustering solutions."
        )


    for k in export_k_values:

        if (
            k
            not in selected_solutions
        ):

            raise RuntimeError(
                "Prototype QC is missing selected "
                f"solution K={k}."
            )


    # ==================================================================
    # TEMPORARY MAPPING-LIKE VIEW
    #
    # plot_morphology_states() consumes one canonical rank-1 observed
    # prototype per cluster.
    # ==================================================================

    mapping_view = SimpleNamespace(
        selections={},
        output_dir=prototype_root,
    )


    # ==================================================================
    # EVERY ROBUST RESOLUTION
    # ==================================================================

    for k in export_k_values:

        solution = (
            selected_solutions[
                k
            ]
        )


        d = int(
            solution.pca_dimensions
        )


        data = _attach_solution(
            master=master,
            solution=solution,
        )


        # --------------------------------------------------------------
        # HUMAN-FACING C1 ... CK ORDER
        # --------------------------------------------------------------

        (
            _,
            display_rank,
        ) = cluster_display_order(
            master=master,
            labels=np.asarray(
                solution.labels,
                dtype=int,
            ),
            complexity_column="Cell_area",
        )


        # --------------------------------------------------------------
        # SIX EMPIRICAL PROTOTYPES / CLUSTER
        # --------------------------------------------------------------

        prototype_cells = (
            select_cluster_prototypes(
                data,
                pca_dimensions=d,
                n_per_cluster=6,
            )
        )


        prototype_cells[
            "display_cluster"
        ] = (
            prototype_cells[
                "cluster"
            ]
            .astype(int)
            .map(
                display_rank
            )
            .astype(int)
        )




        # --------------------------------------------------------------
        # RANK-1 CANONICAL PROTOTYPES
        # --------------------------------------------------------------

        canonical_prototypes = (
            prototype_cells[
                prototype_cells[
                    "selection_rank"
                ]
                == 1
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        selections[
            k
        ] = canonical_prototypes


        mapping_view.selections[
            k
        ] = {
            "prototype":
                canonical_prototypes,
        }


        # ==============================================================
        # OUTPUT LOCATION
        # ==============================================================

        solution_dir = (
            prototype_root
            / f"K{k}_d{d}"
        )


        solution_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ==============================================================
        # SIX-PROTOTYPE TABLE
        # ==============================================================

        table_path = (
            solution_dir
            / "prototype_cells.csv"
        )


        prototype_cells.to_csv(
            table_path,
            index=False,
        )


        files[
            f"prototype_table_K{k}"
        ] = table_path


        # ==============================================================
        # PROTOTYPE MASKS — PREFERRED K ONLY
        #
        # Export ranks 1-4 for the 2 x 2 morphology-state panel.
        #
        # cluster_X.png remains the canonical rank-1 alias.
        # ==============================================================

        if True:

            masks_dir = (
                solution_dir
                / "Prototype_Masks"
            )


            masks_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            figure_prototypes = (
                prototype_cells[
                    prototype_cells[
                        "selection_rank"
                    ]
                    .astype(int)
                    .le(6)
                ]
                .copy()
                .sort_values(
                    [
                        "cluster",
                        "selection_rank",
                    ],
                    kind="mergesort",
                )
                .reset_index(
                    drop=True
                )
            )


            for (
                _,
                prototype_cell,
            ) in figure_prototypes.iterrows():

                mask = (
                    extract_component_crop(
                        prototype_cell,
                        image_cache,
                    )
                )


                if mask is None:

                    raise RuntimeError(
                        "Could not recover exact "
                        "prototype component."
                    )


                raw_cluster = int(
                    prototype_cell[
                        "cluster"
                    ]
                )


                selection_rank = int(
                    prototype_cell[
                        "selection_rank"
                    ]
                )


                ranked_mask_path = (
                    masks_dir
                    / (
                        f"cluster_"
                        f"{raw_cluster}"
                        f"_rank_"
                        f"{selection_rank}.png"
                    )
                )


                success = cv2.imwrite(
                    str(
                        ranked_mask_path
                    ),
                    mask,
                )


                if not success:

                    raise RuntimeError(
                        "Could not save ranked "
                        "prototype mask: "
                        f"{ranked_mask_path}"
                    )


                files[
                    (
                        f"prototype_mask_"
                        f"K{k}_cluster_"
                        f"{raw_cluster}_rank_"
                        f"{selection_rank}"
                    )
                ] = ranked_mask_path


                if selection_rank == 1:

                    canonical_mask_path = (
                        masks_dir
                        / (
                            f"cluster_"
                            f"{raw_cluster}.png"
                        )
                    )


                    success = cv2.imwrite(
                        str(
                            canonical_mask_path
                        ),
                        mask,
                    )


                    if not success:

                        raise RuntimeError(
                            "Could not save canonical "
                            "prototype mask: "
                            f"{canonical_mask_path}"
                        )


                    files[
                        (
                            f"prototype_mask_"
                            f"K{k}_cluster_"
                            f"{raw_cluster}"
                        )
                    ] = canonical_mask_path


        # ==============================================================
        # CANONICAL 3 × 2 MONTAGE
        # ==============================================================

        montage_path = (
            solution_dir
            / "Prototype_Cells.png"
        )


        render_selection_montage(
            prototype_cells,
            image_cache=image_cache,
            output_path=montage_path,
            title=(
                f"Prototype cells — d={d}, {k} morphology states"
            ),
            label_columns=(),
            padding=(
                mapping_config
                .effective_montage_padding
            ),
            ncols=3,
            max_cells_per_cluster=6,
            show_full_id=False,
        )


        files[
            f"prototype_montage_K{k}"
        ] = montage_path


    return (
        selections,
        mapping_view,
        files,
    )


# ======================================================================
# TABLE OUTPUTS
# ======================================================================

def _save_analytical_tables(
    feature_result,
    dimensionality_result,
    clustering_result,
    output_dir: Path,
) -> dict[str, Path]:

    files: dict[
        str,
        Path,
    ] = {}

    feature_dir = (
        output_dir
        / "Feature_Preparation"
    )

    dr_dir = (
        output_dir
        / "Dimensionality_Reduction"
    )

    clustering_dir = (
        output_dir
        / "Clustering"
    )

    feature_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dr_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clustering_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    path = (
        feature_dir
        / "feature_qc.csv"
    )

    feature_result.qc_report.to_csv(
        path,
        index=False,
    )

    files[
        "feature_qc"
    ] = path

    if (
        feature_result
        .correlation_matrix
        is not None
    ):

        path = (
            feature_dir
            / "correlation_matrix.csv"
        )

        (
            feature_result
            .correlation_matrix
            .to_csv(
                path
            )
        )

        files[
            "correlation_matrix"
        ] = path

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------

    path = (
        dr_dir
        / "loadings.csv"
    )

    dimensionality_result.loadings.to_csv(
        path
    )

    files[
        "pca_loadings"
    ] = path

    path = (
        dr_dir
        / "scaling_report.csv"
    )

    (
        dimensionality_result
        .scaling_report
        .to_csv(
            path,
            index=False,
        )
    )

    files[
        "scaling_report"
    ] = path

    n_components = len(
        dimensionality_result.eigenvalues
    )

    dimensionality_table = pd.DataFrame(
        {
            "component":
                np.arange(
                    1,
                    n_components + 1,
                ),
            "eigenvalue":
                dimensionality_result.eigenvalues,
            "explained_variance_ratio":
                dimensionality_result
                .explained_variance_ratio,
            "cumulative_variance":
                dimensionality_result
                .cumulative_variance,
            "parallel_threshold":
                dimensionality_result
                .parallel_threshold,
            "parallel_supported":
                dimensionality_result
                .parallel_supported,
            "broken_stick_expected":
                dimensionality_result
                .broken_stick_expected,
            "broken_stick_supported":
                dimensionality_result
                .broken_stick_supported,
        }
    )

    path = (
        dr_dir
        / "dimensionality.csv"
    )

    dimensionality_table.to_csv(
        path,
        index=False,
    )

    files[
        "dimensionality"
    ] = path

    summary_path = (
        dr_dir
        / "dimension_summary.csv"
    )

    pd.DataFrame(
        [
            {
                "parallel_dimension":
                    dimensionality_result
                    .parallel_dimension,
                "broken_stick_dimension":
                    dimensionality_result
                    .broken_stick_dimension,
                "two_nn_dimension":
                    dimensionality_result
                    .two_nn_dimension,
                "plausible_dimension_min":
                    dimensionality_result
                    .plausible_dimension_min,
                "plausible_dimension_max":
                    dimensionality_result
                    .plausible_dimension_max,
            }
        ]
    ).to_csv(
        summary_path,
        index=False,
    )

    files[
        "dimension_summary"
    ] = summary_path

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    clustering_tables = {
        "model_scan":
            clustering_result.model_scan,
        "candidate_assignments":
            clustering_result.candidate_assignments,
        "stability":
            clustering_result
            .stability_by_dimension_and_k,
        "dimension_agreement":
            clustering_result.dimension_agreement,
        "multiresolution_summary":
            clustering_result.multiresolution_summary,
    }

    for name, table in clustering_tables.items():

        path = (
            clustering_dir
            / f"{name}.csv"
        )

        table.to_csv(
            path,
            index=False,
        )

        files[
            name
        ] = path

    # One explicit cell-level table containing every robust resolution.
    selected_rows = []

    for k in clustering_result.robust_k_values:

        k = int(
            k
        )

        solution = (
            clustering_result
            .selected_solutions[
                k
            ]
        )

        for index in range(
            len(
                solution.labels
            )
        ):

            selected_rows.append(
                {
                    "row_index":
                        index,
                    "k":
                        k,
                    "pca_dimensions":
                        int(
                            solution
                            .pca_dimensions
                        ),
                    "covariance_type":
                        str(
                            solution
                            .covariance_type
                        ),
                    "cluster":
                        int(
                            solution
                            .labels[
                                index
                            ]
                        ),
                    "cluster_probability":
                        float(
                            solution
                            .cluster_probability[
                                index
                            ]
                        ),
                    "consensus_reliability":
                        float(
                            solution
                            .consensus_reliability[
                                index
                            ]
                        ),
                    "mean_subsample_ari":
                        float(
                            solution
                            .mean_subsample_ari
                        ),
                    "std_subsample_ari":
                        float(
                            solution
                            .std_subsample_ari
                        ),
                }
            )

    selected_path = (
        clustering_dir
        / "selected_solution_cell_labels.csv"
    )

    pd.DataFrame(
        selected_rows
    ).to_csv(
        selected_path,
        index=False,
    )

    files[
        "selected_solution_cell_labels"
    ] = selected_path

    return files


# ======================================================================
# PUBLIC STAGE
# ======================================================================


def _save_cluster_map_sample_qc(
    master: pd.DataFrame,
    clustering,
    prototype_files: dict[str, Path],
    instance_dir: Path,
    output_dir: Path,
    random_seed: int,
    max_images: int = 20,
) -> dict[str, Path]:
    """Render the same deterministic source-image sample for every supported count."""

    state_counts = tuple(
        int(k)
        for k in getattr(clustering, "supported_k_values", ())
    )
    if not state_counts:
        state_counts = (int(getattr(clustering, "automatic_k", clustering.preferred_k)),)

    candidate_ids = set()
    for k in state_counts:
        table_path = prototype_files.get(f"prototype_table_K{k}")
        if table_path is None:
            continue
        table = pd.read_csv(table_path)
        if "image_id" in table.columns:
            candidate_ids.update(table["image_id"].dropna().astype(str).tolist())

    candidate_ids = sorted(candidate_ids)
    if not candidate_ids:
        return {}

    max_images = max(1, int(max_images))
    if len(candidate_ids) > max_images:
        rng = np.random.default_rng(int(random_seed))
        chosen = sorted(
            rng.choice(
                np.asarray(candidate_ids, dtype=object),
                size=max_images,
                replace=False,
            ).tolist()
        )
    else:
        chosen = candidate_ids

    root = output_dir / "QC" / "Cluster_Map_Sample"
    root.mkdir(parents=True, exist_ok=True)

    sample_path = root / "sample_images.csv"
    pd.DataFrame({"image_id": chosen}).to_csv(sample_path, index=False)
    files = {"cluster_map_sample_table": sample_path}

    sample_master = master[master["image_id"].astype(str).isin(chosen)].copy()
    image_cache = _build_instance_cache(master=sample_master, instance_dir=instance_dir)

    for k in state_counts:
        if k not in clustering.selected_solutions:
            raise RuntimeError(
                f"Supported state count {k} is missing from selected_solutions."
            )

        solution = clustering.selected_solutions[k]
        data = _attach_solution(master=master, solution=solution)
        _, display_rank = cluster_display_order(
            master=master,
            labels=np.asarray(solution.labels, dtype=int),
            complexity_column="Cell_area",
        )
        data["display_cluster"] = (
            data["cluster"].astype(int).map(display_rank).astype(int)
        )

        k_dir = root / f"K{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        for image_id in chosen:
            cells = data[data["image_id"].astype(str).eq(image_id)]
            if cells.empty:
                continue

            instance_map = image_cache[image_id]["instance_map"]
            h, w = instance_map.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)

            for _, row in cells.iterrows():
                value = row.get("map_label")
                if pd.isna(value):
                    continue
                mask = instance_map == int(value)
                if not np.any(mask):
                    continue
                rgb = morphology_state_rgb(int(row["display_cluster"]), n_colors=k)
                rgba[mask, 0] = rgb[0]
                rgba[mask, 1] = rgb[1]
                rgba[mask, 2] = rgb[2]
                rgba[mask, 3] = 255

            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            ok, encoded = cv2.imencode(".png", bgra)
            if not ok:
                raise RuntimeError(
                    f"Could not encode DRC QC cluster map for image_id={image_id}, K={k}."
                )

            map_path = k_dir / f"{image_id}.png"
            map_path.write_bytes(encoded.tobytes())
            files[f"cluster_map_sample_K{k}_{image_id}"] = map_path

    return files


# ======================================================================
# DRC CRASH-RECOVERY CHECKPOINTS
# ======================================================================

_DRC_CHECKPOINT_VERSION = 1


def _drc_input_digest(
    master: pd.DataFrame,
    morphometric_columns: list[str],
) -> str:
    """
    Hash only the true computational inputs to DRC.

    Old downstream PC/cluster columns are deliberately excluded so a saved
    analytical master table cannot invalidate its own upstream DRC cache.
    """

    columns = [
        str(column)
        for column in morphometric_columns
    ]

    if "image_id" in master.columns:
        columns = [
            "image_id",
            *columns,
        ]

    data = (
        master[
            columns
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    row_hashes = (
        pd.util.hash_pandas_object(
            data,
            index=True,
            categorize=True,
        )
        .to_numpy(
            dtype=np.uint64,
            copy=False,
        )
    )

    digest = hashlib.sha256()
    digest.update(
        row_hashes.tobytes()
    )

    digest.update(
        repr(
            [
                (
                    str(column),
                    str(data[column].dtype),
                )
                for column in data.columns
            ]
        ).encode(
            "utf-8"
        )
    )

    return digest.hexdigest()


def _drc_clustering_config_payload(
    config,
) -> dict:
    """
    Return clustering settings that affect the expensive fit itself.

    The researcher-selected morphology-state count is excluded because that
    choice is applied only by the later resolution-decision layer.
    """

    excluded = {
        "number_of_morphology_states",
        "automatic_number_of_morphology_states",
        "number_of_morphology_states_used_downstream",
        "morphology_state_selection_source",
    }

    if is_dataclass(config):

        names = [
            field_info.name
            for field_info in dataclass_fields(
                config
            )
        ]

    else:

        names = []

        for cls in reversed(
            type(config).__mro__
        ):

            for name in getattr(
                cls,
                "__annotations__",
                {},
            ):

                name = str(name)

                if name not in names:
                    names.append(name)

        for name in vars(config):

            name = str(name)

            if name not in names:
                names.append(name)

    return {
        name:
            getattr(
                config,
                name,
            )
        for name in names
        if (
            name not in excluded
            and hasattr(
                config,
                name,
            )
        )
    }


def _drc_checkpoint_path(
    config,
    name: str,
) -> Path:

    return (
        Path(
            config.output_dir
        )
        / "Technical_Record"
        / "Checkpoints"
        / "DRC"
        / f"{name}.pkl"
    )


def _drc_pickle_write(
    path: Path,
    value,
) -> Path:
    """
    Atomically persist one trusted local MorphoGlia checkpoint.
    """

    payload = pickle.dumps(
        value,
        protocol=(
            pickle.HIGHEST_PROTOCOL
        ),
    )

    return atomic_write_bytes(
        path,
        payload,
    )


def _drc_pickle_read(
    path: Path,
):

    with Path(path).open(
        "rb"
    ) as file:

        return pickle.load(
            file
        )


def _validate_drc_feature_result(
    value,
    *,
    expected_rows: int,
) -> None:

    data = getattr(
        value,
        "data",
        None,
    )

    analytical_features = list(
        getattr(
            value,
            "analytical_features",
            (),
        )
    )

    qc_report = getattr(
        value,
        "qc_report",
        None,
    )

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise ValueError(
            "DRC Feature Preparation checkpoint has no DataFrame data."
        )

    if len(data) != int(expected_rows):
        raise ValueError(
            "DRC Feature Preparation checkpoint row count does not match "
            "the current master table."
        )

    if not analytical_features:
        raise ValueError(
            "DRC Feature Preparation checkpoint has no analytical features."
        )

    if list(data.columns) != analytical_features:
        raise ValueError(
            "DRC Feature Preparation checkpoint feature order is inconsistent."
        )

    if not isinstance(
        qc_report,
        pd.DataFrame,
    ):
        raise ValueError(
            "DRC Feature Preparation checkpoint has no QC DataFrame."
        )


def _validate_drc_dimensionality_result(
    value,
    *,
    expected_rows: int,
    expected_features,
) -> None:

    scores = np.asarray(
        getattr(
            value,
            "scores",
            None,
        )
    )

    transformed = np.asarray(
        getattr(
            value,
            "transformed_features",
            None,
        )
    )

    scaled = np.asarray(
        getattr(
            value,
            "scaled_features",
            None,
        )
    )

    feature_names = list(
        getattr(
            value,
            "feature_names",
            (),
        )
    )

    expected_features = [
        str(feature)
        for feature in expected_features
    ]

    if (
        scores.ndim != 2
        or scores.shape[0] != int(expected_rows)
    ):
        raise ValueError(
            "DRC dimensionality checkpoint PCA scores do not match "
            "the current cell population."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "DRC dimensionality checkpoint contains non-finite PCA scores."
        )

    expected_shape = (
        int(expected_rows),
        len(expected_features),
    )

    if (
        transformed.shape != expected_shape
        or scaled.shape != expected_shape
    ):
        raise ValueError(
            "DRC dimensionality checkpoint transformed/scaled feature "
            "matrices have incompatible shape."
        )

    if feature_names != expected_features:
        raise ValueError(
            "DRC dimensionality checkpoint feature identity/order changed."
        )

    plausible_min = int(
        getattr(
            value,
            "plausible_dimension_min"
        )
    )

    plausible_max = int(
        getattr(
            value,
            "plausible_dimension_max"
        )
    )

    if not (
        1
        <= plausible_min
        <= plausible_max
        <= scores.shape[1]
    ):
        raise ValueError(
            "DRC dimensionality checkpoint has an invalid plausible "
            "PCA interval."
        )

    if not isinstance(
        getattr(
            value,
            "loadings",
            None,
        ),
        pd.DataFrame,
    ):
        raise ValueError(
            "DRC dimensionality checkpoint has no PCA loadings DataFrame."
        )

    if not isinstance(
        getattr(
            value,
            "scaling_report",
            None,
        ),
        pd.DataFrame,
    ):
        raise ValueError(
            "DRC dimensionality checkpoint has no scaling-report DataFrame."
        )


def _validate_drc_clustering_result(
    value,
    *,
    expected_rows: int,
) -> None:

    selected_solutions = getattr(
        value,
        "selected_solutions",
        None,
    )

    robust_k_values = tuple(
        int(k)
        for k in getattr(
            value,
            "robust_k_values",
            (),
        )
    )

    if not isinstance(
        selected_solutions,
        dict,
    ):
        raise ValueError(
            "DRC clustering checkpoint has no selected_solutions mapping."
        )

    if not robust_k_values:
        raise ValueError(
            "DRC clustering checkpoint has no robust morphology-state counts."
        )

    for k in robust_k_values:

        if k not in selected_solutions:
            raise ValueError(
                "DRC clustering checkpoint is missing selected solution "
                f"K={k}."
            )

        solution = selected_solutions[k]

        for name in (
            "labels",
            "cluster_probability",
            "consensus_reliability",
        ):

            values = np.asarray(
                getattr(
                    solution,
                    name,
                    None,
                )
            )

            if (
                values.ndim != 1
                or len(values) != int(expected_rows)
            ):
                raise ValueError(
                    "DRC clustering checkpoint selected solution "
                    f"K={k} has incompatible {name}."
                )

    for name in (
        "model_scan",
        "candidate_assignments",
        "stability_by_dimension_and_k",
        "dimension_agreement",
        "multiresolution_summary",
    ):

        if not isinstance(
            getattr(
                value,
                name,
                None,
            ),
            pd.DataFrame,
        ):
            raise ValueError(
                "DRC clustering checkpoint is missing DataFrame "
                f"{name!r}."
            )


def _drc_checkpoint_value(
    *,
    journal: CheckpointJournal,
    phase: str,
    path: Path,
    item_signature: str,
    compute,
    validate,
):
    """
    Reuse one completed global DRC substage or compute/commit it atomically.
    """

    reusable = journal.reusable_record(
        item_id=phase,
        item_signature=item_signature,
        outputs=[
            path
        ],
    )

    if reusable is not None:

        try:

            value = _drc_pickle_read(
                path
            )

            validate(
                value
            )

        except Exception as exc:

            print(
                "DRC resume checkpoint rejected for "
                f"{phase}: {exc}"
            )

        else:

            print(
                "DRC RESUME:",
                phase.replace(
                    "_",
                    " ",
                ),
            )

            return value

    value = compute()

    validate(
        value
    )

    _drc_pickle_write(
        path,
        value,
    )

    # Commit only after the on-disk object can be restored and validated.
    restored = _drc_pickle_read(
        path
    )

    validate(
        restored
    )

    journal.commit(
        item_id=phase,
        item_signature=item_signature,
        outputs=[
            path
        ],
    )

    return restored


def run_dim_reduction_clustering(
    master: pd.DataFrame,
    config,
    instance_dir: str | Path,
) -> DimReductionClusteringResult:
    """
    Run MorphoGlia's complete morphology-state discovery analysis.

    Scientific sequence
    -------------------

    Morphometrics
        ↓
    Feature Preparation
        ↓
    Scaling + PCA
        ↓
    Dimensionality diagnostics
        ↓
    Multiresolution clustering
        ↓
    Stability / consensus
        ↓
    Analytical QC
    """

    master = (
        master
        .copy()
        .reset_index(
            drop=True
        )
    )

    output_dir = (
        Path(
            config.output_dir
        )
        / "Dimensionality_Reduction_Clustering"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("DIMENSIONALITY REDUCTION + CLUSTERING")
    print("=" * 72)
    print()

    # ==================================================================
    # FEATURE PREPARATION
    # ==================================================================

    morphometric_columns = (
        _morphometric_columns(
            master
        )
    )

    print(
        "Morphometric inputs:",
        len(
            morphometric_columns
        ),
    )

    morphometric_table = (
        master[
            morphometric_columns
        ]
        .copy()
    )

    # ==================================================================
    # GLOBAL DRC CHECKPOINT CONTEXT
    # ==================================================================

    drc_input_digest = _drc_input_digest(
        master,
        morphometric_columns,
    )

    drc_journal = CheckpointJournal(
        output_dir=(
            Path(
                config.output_dir
            )
        ),
        stage="dim_reduction_clustering",
        resume=bool(
            getattr(
                config,
                "resume",
                False,
            )
        ),
        stage_signature=fingerprint(
            "drc_resume_v1",
            drc_input_digest,
        ),
    )

    drc_package_dir = Path(
        __file__
    ).resolve().parent

    drc_config_source = (
        drc_package_dir
        / "config.py"
    )

    feature_signature = fingerprint(
        "drc_feature_preparation_v1",
        drc_input_digest,
        config.feature_preparation,
        source_digest(
            drc_package_dir
            / "feature_preparation.py"
        ),
        source_digest(
            drc_config_source
        ),
    )

    dimensionality_signature = fingerprint(
        "drc_dimensionality_v1",
        feature_signature,
        config.dimensionality_reduction,
        int(
            config.random_seed
        ),
        source_digest(
            drc_package_dir
            / "dimensionality.py"
        ),
        source_digest(
            drc_config_source
        ),
    )

    clustering_signature = fingerprint(
        "drc_clustering_v1",
        dimensionality_signature,
        _drc_clustering_config_payload(
            config.clustering
        ),
        int(
            config.random_seed
        ),
        source_digest(
            drc_package_dir
            / "clustering.py"
        ),
        source_digest(
            drc_package_dir
            / "resolution_support.py"
        ),
        source_digest(
            drc_config_source
        ),
    )

    feature_checkpoint_path = _drc_checkpoint_path(
        config,
        "feature_preparation",
    )

    dimensionality_checkpoint_path = _drc_checkpoint_path(
        config,
        "dimensionality_reduction",
    )

    clustering_checkpoint_path = _drc_checkpoint_path(
        config,
        "clustering",
    )

    feature_result = _drc_checkpoint_value(
        journal=drc_journal,
        phase="feature_preparation",
        path=feature_checkpoint_path,
        item_signature=feature_signature,
        compute=lambda: FeaturePreparationEngine(config.feature_preparation).fit_transform(morphometric_table),
        validate=lambda value: _validate_drc_feature_result(
            value,
            expected_rows=len(master),
        ),
    )

    print(
        "Analytical features:",
        len(
            feature_result
            .analytical_features
        ),
    )
    


    # ==================================================================
    # DIMENSIONALITY REDUCTION
    # ==================================================================

    dimensionality_result = _drc_checkpoint_value(
        journal=drc_journal,
        phase="dimensionality_reduction",
        path=dimensionality_checkpoint_path,
        item_signature=dimensionality_signature,
        compute=lambda: DimensionalityReductionEngine(config.dimensionality_reduction, random_state=config.random_seed).fit_transform(feature_result.data),
        validate=lambda value: _validate_drc_dimensionality_result(
            value,
            expected_rows=len(master),
            expected_features=feature_result.analytical_features,
        ),
    )

    master = _attach_pca_scores(
        master=master,
        scores=(
            dimensionality_result
            .scores
        ),
    )

    print(
        "Parallel-analysis dimension:",
        dimensionality_result
        .parallel_dimension,
    )

    print(
        "Broken-stick dimension:",
        dimensionality_result
        .broken_stick_dimension,
    )

    print(
        "Two-NN dimension:",
        dimensionality_result
        .two_nn_dimension,
    )

    print(
        "Plausible PCA interval:",
        (
            dimensionality_result
            .plausible_dimension_min,
            dimensionality_result
            .plausible_dimension_max,
        ),
    )

    # ==================================================================
    # CLUSTERING
    # ==================================================================

    if "image_id" not in master.columns:
        raise ValueError(
            "Clustering requires master['image_id'] for structured "
            "resampling."
        )

    clustering_result = _drc_checkpoint_value(
        journal=drc_journal,
        phase="clustering",
        path=clustering_checkpoint_path,
        item_signature=clustering_signature,
        compute=lambda: ClusteringEngine(config.clustering, random_state=config.random_seed, n_jobs=config.compute.resolve_cpu_budget()).fit(dimensionality=dimensionality_result, strata=master['image_id'].astype(str)),
        validate=lambda value: _validate_drc_clustering_result(
            value,
            expected_rows=len(master),
        ),
    )

    # ==================================================================
    # RESOLUTION DECISION
    # ==================================================================

    display_maps = {}


    for k in clustering_result.robust_k_values:

        k = int(
            k
        )


        solution = (
            clustering_result
            .selected_solutions[
                k
            ]
        )


        (
            _,
            display_rank,
        ) = cluster_display_order(
            master=master,
            labels=np.asarray(
                solution.labels,
                dtype=int,
            ),
            complexity_column="Cell_area",
        )


        display_maps[
            k
        ] = {
            int(
                raw_cluster
            ):
                int(
                    display_cluster
                )
            for (
                raw_cluster,
                display_cluster,
            ) in display_rank.items()
        }


    resolution_decision = (
        evaluate_resolution_decision(
            clustering=(
                clustering_result
            ),
            display_maps=(
                display_maps
            ),
        )
    )


    clustering_result.maximin_k = resolution_decision.maximin_k
    clustering_result.pareto_k_values = resolution_decision.pareto_k_values

    supported_counts = tuple(
        int(value) for value in resolution_decision.supported_k_values
    )
    automatic_count = int(resolution_decision.automatic_k)
    requested_count = config.clustering.requested_number_of_morphology_states

    requested_is_available = (
        requested_count is None
        or int(requested_count) in supported_counts
    )

    if (
        requested_count is not None
        and not requested_is_available
    ):
        print()
        print(
            "Requested morphology-state count "
            f"K={int(requested_count)} is not available for this dataset."
        )
        print(
            "Using automatic selection "
            f"K={automatic_count}."
        )

    if requested_count is None or not requested_is_available:
        effective_count = automatic_count
        selection_source = "automatic_data_driven"
    else:
        effective_count = int(requested_count)
        selection_source = "researcher_selection"

    for decision_object in (resolution_decision, clustering_result):
        decision_object.automatic_k = automatic_count
        decision_object.effective_k = effective_count
        decision_object.requested_k = requested_count
        decision_object.selection_source = selection_source
        decision_object.preferred_k = effective_count
        # Internal compatibility only; no user-facing output uses this name.
        decision_object.default_k = automatic_count

    resolution_decision.metrics["effective_selection"] = (
        resolution_decision.metrics["k"].astype(int).eq(effective_count)
    )
    resolution_decision.metrics["selection_source"] = selection_source
    resolution_decision.metrics["requested_number_of_morphology_states"] = (
        pd.NA if requested_count is None else int(requested_count)
    )

    fallback_used = False

    if (
        "automatic_fallback"
        in resolution_decision.metrics.columns
    ):
        fallback_used = bool(
            resolution_decision
            .metrics[
                "automatic_fallback"
            ]
            .fillna(False)
            .astype(bool)
            .any()
        )

    if fallback_used:
        print(
            "Strictly reproducible morphology-state counts: none"
        )
        print(
            "Automatic fallback:",
            f"K={automatic_count}",
        )
    else:
        print(
            "Stable/reproducible numbers of morphology states:",
            ", ".join(
                str(value)
                for value in supported_counts
            ),
        )
    print("Stable bands:", resolution_decision.stable_bands)
    print("Band representatives:", resolution_decision.band_representatives)
    print(
        "Global Pareto-leading counts (diagnostic only):",
        ", ".join(str(value) for value in resolution_decision.pareto_k_values),
    )
    if resolution_decision.maximin_k is not None:
        print(
            "Historical maximin diagnostic:",
            f"{int(resolution_decision.maximin_k)} states",
        )

    criteria = resolution_decision.support_criteria
    print("Automatic selection criterion (data-driven):")
    print("  1. Use the first/coarsest reproducible stable band.")
    print("  2. Within that band, maximize cross-dimensional agreement.")
    print("  3. Then compare worst-case and mean resampling stability,")
    print("     membership reliability, state size, and the lower-count tie-breaker.")
    print("Reproducibility gates:")
    print(
        "  mean/worst resampling ARI >= "
        f"{criteria['mean_subsample_ari_min']:.2f}/"
        f"{criteria['worst_subsample_ari_min']:.2f}"
    )
    print(
        "  membership probability >= "
        f"{criteria['membership_probability_min']:.2f}; "
        "cross-dimensional agreement >= "
        f"{criteria['dimension_agreement_floor']:.3f}; "
        "minimum state size >= "
        f"{int(criteria['minimum_state_size'])}"
    )
    print(
        "  isolated peaks additionally require local non-dominance and "
        "mean/worst ARI >= "
        f"{criteria['isolated_mean_subsample_ari_min']:.2f}/"
        f"{criteria['isolated_worst_subsample_ari_min']:.2f}, "
        "membership >= "
        f"{criteria['isolated_membership_probability_min']:.2f}, and "
        "cross-dimensional agreement >= "
        f"{criteria['isolated_dimension_agreement_floor']:.3f}"
    )
    print("Automatic number of morphology states:", automatic_count)
    print(
        "Researcher selection:",
        "not supplied" if requested_count is None else f"{requested_count} states",
    )
    print(
        "Number of morphology states used downstream:",
        f"{effective_count} ({'automatic data-driven selection' if requested_count is None else 'researcher selection'})",
    )
    print()

    print("Evidence for each selectable number of morphology states:")
    candidate_metrics = resolution_decision.metrics[
        resolution_decision.metrics["supported_resolution"].fillna(False)
    ]
    for _, row in candidate_metrics.iterrows():
        count = int(row["k"])
        mode = str(row.get("support_mode", "stable_band"))
        label = "isolated stable peak" if mode == "isolated_stable_peak" else "stable band"
        print(f"  {count} states ({label})")
        print(f"    mean subsample ARI:     {row['mean_subsample_ari']:.3f}")
        print(f"    worst subsample ARI:    {row['minimum_subsample_ari']:.3f}")
        print(f"    mean across-d:          {row['mean_dimension_agreement']:.3f}")
        print(f"    minimum across-d:       {row['minimum_dimension_agreement']:.3f}")
        print(f"    membership probability: {row['mean_membership_probability']:.3f}")
        print(f"    min state size:         {int(row['minimum_cluster_size_across_d'])}")
    print()

    globally_dominated = candidate_metrics[~candidate_metrics["pareto_optimal"]]
    if not globally_dominated.empty:
        print("Global Pareto diagnostic:")
        for _, row in globally_dominated.iterrows():
            print(
                f"  {int(row['k'])} states are dominated by "
                f"{row['dominated_by']} in the global comparison, but remain "
                "selectable after passing reproducibility and local-support gates."
            )
        print()

    decision_nesting = resolution_decision.nesting[
        resolution_decision.nesting["decision_pair"]
    ]
    if not decision_nesting.empty:
        print("Population-identity stability between supported interpretations:")
        for _, row in decision_nesting.iterrows():
            print(
                f"  {int(row['coarse_k'])} -> {int(row['fine_k'])} states "
                f"({row['transition_scope'].replace('_', ' ')})"
            )
            print(
                "    weighted parent purity: "
                f"{row['weighted_parent_purity']:.3f}"
            )
            print(
                "    minimum parent purity:  "
                f"{row['minimum_parent_purity']:.3f}"
            )
            print(
                "    split entropy:          "
                f"{row['weighted_split_entropy']:.3f}"
            )
        print()


    # ==================================================================
    # TABLE OUTPUTS
    # ==================================================================

    files = (
        _save_analytical_tables(
            feature_result=(
                feature_result
            ),
            dimensionality_result=(
                dimensionality_result
            ),
            clustering_result=(
                clustering_result
            ),
            output_dir=(
                output_dir
            ),
        )
    )
    # ==================================================================
    # RESOLUTION DECISION OUTPUTS
    # ==================================================================

    decision_dir = (
        output_dir
        / "Decision"
    )


    decision_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    decision_metrics_path = (
        decision_dir
        / "resolution_decision.csv"
    )


    resolution_decision.metrics.to_csv(
        decision_metrics_path,
        index=False,
    )


    files[
        "resolution_decision"
    ] = decision_metrics_path


    transition_path = (
        decision_dir
        / "resolution_transitions.csv"
    )


    resolution_decision.transitions.to_csv(
        transition_path,
        index=False,
    )


    files[
        "resolution_transitions"
    ] = transition_path


    nesting_path = (
        decision_dir
        / "resolution_nesting.csv"
    )


    resolution_decision.nesting.to_csv(
        nesting_path,
        index=False,
    )


    files[
        "resolution_nesting"
    ] = nesting_path


    # ==================================================================
    # PREFERRED-K BIOLOGICAL INTERPRETATION
    # ==================================================================

    preferred_k = int(
        resolution_decision
        .preferred_k
    )


    preferred_solution = (
        clustering_result
        .selected_solutions[
            preferred_k
        ]
    )


    (
        preferred_cluster_summary,
        preferred_morphology_profiles,
        preferred_category_composition,
    ) = build_preferred_cluster_interpretation(
        master=master,
        analytical_features=(
            feature_result
            .analytical_features
        ),
        solution=(
            preferred_solution
        ),
        display_map=(
            display_maps[
                preferred_k
            ]
        ),
    )


    preferred_summary_path = (
        decision_dir
        / "preferred_k_cluster_summary.csv"
    )


    preferred_cluster_summary.to_csv(
        preferred_summary_path,
        index=False,
    )


    files[
        "preferred_k_cluster_summary"
    ] = preferred_summary_path


    preferred_profiles_path = (
        decision_dir
        / "preferred_k_morphology_profiles.csv"
    )


    preferred_morphology_profiles.to_csv(
        preferred_profiles_path,
        index=False,
    )


    files[
        "preferred_k_morphology_profiles"
    ] = preferred_profiles_path


    if not preferred_category_composition.empty:

        preferred_category_path = (
            decision_dir
            / "preferred_k_category_composition.csv"
        )


        preferred_category_composition.to_csv(
            preferred_category_path,
            index=False,
        )


        files[
            "preferred_k_category_composition"
        ] = preferred_category_path


    print(
        f"Morphology-state interpretation used downstream "
        f"({preferred_k} states):"
    )


    for (
        _,
        row,
    ) in preferred_cluster_summary.iterrows():

        print(
            f"  {row['cluster']}: "
            f"{int(row['n_cells'])} cells "
            f"({100 * row['fraction']:.1f}%)"
        )


        print(
            "    membership: "
            f"{row['mean_membership_probability']:.3f}"
        )


        print(
            "    reliability: "
            f"{row['mean_consensus_reliability']:.3f}"
        )


        print(
            "    positive morphology: "
            f"{row['top_positive_features']}"
        )


        print(
            "    negative morphology: "
            f"{row['top_negative_features']}"
        )


    print()


    # ==================================================================
    # PROTOTYPE QC
    # ==================================================================

    (
        prototype_selections,
        prototype_mapping_view,
        prototype_files,
    ) = _save_prototype_qc(
        master=master,
        clustering=(
            clustering_result
        ),
        instance_dir=(
            instance_dir
        ),
        output_dir=(
            output_dir
        ),
        mapping_config=(
            config.mapping
        ),
    )

    files.update(
        prototype_files
    )


    files.update(
        _save_cluster_map_sample_qc(
            master=master,
            clustering=clustering_result,
            prototype_files=prototype_files,
            instance_dir=instance_dir,
            output_dir=output_dir,
            random_seed=int(config.random_seed),
            max_images=20,
        )
    )


    # ==================================================================
    # PREFERRED MORPHOLOGY-STATE SOLUTION
    #
    # This is the canonical single-resolution representation used for
    # analytical morphology-state visualization.
    #
    # preferred_k is selected by the explicit resolution-decision layer.
    #
    # strongest_k is retained only as the historical maximin diagnostic
    # and is used here solely as a backward-compatible fallback.
    # ==================================================================

    preferred_k = (
        getattr(
            clustering_result,
            "preferred_k",
            None,
        )
        or clustering_result
        .strongest_k
    )


    if preferred_k is None:

        raise RuntimeError(
            "Analytical QC requires a preferred "
            "clustering solution."
        )


    preferred_k = int(
        preferred_k
    )


    if (
        preferred_k
        not in clustering_result
        .selected_solutions
    ):

        raise RuntimeError(
            "Preferred clustering solution is missing."
        )


    preferred_solution = (
        clustering_result
        .selected_solutions[
            preferred_k
        ]
    )


    d = int(
        preferred_solution
        .pca_dimensions
    )


    # ==================================================================
    # CANONICAL SINGLE-RESOLUTION UMAP
    #
    # UMAP is visualization only.
    #
    # It is calculated from the same PCA dimensionality selected for
    # the strongest clustering solution.
    # ==================================================================

    shared_umap_d = max(
        int(
            clustering_result
            .selected_solutions[int(k)]
            .pca_dimensions
        )
        for k in clustering_result.robust_k_values
    )

    umap_embedding = (
        compute_umap_from_pca(
            master=master,
            pca_dimensions=(
                shared_umap_d
            ),
            config=(
                config.plots
            ),
        )
    )

    clustering_result.shared_umap_embedding = np.asarray(
        umap_embedding,
        dtype=float,
    )
    clustering_result.shared_umap_pca_dimensions = int(shared_umap_d)


    # ==================================================================
    # CANONICAL MORPHOLOGY-STATE PCA + UMAP
    #
    # These figures show the current strongest/default resolution.
    # ==================================================================

    morphology_states_dir = (
        output_dir
        / "QC"
        / "Morphology_States"
    )


    morphology_states_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    morphology_paths = (
        plot_morphology_states(
            master=master,
            clustering=(
                clustering_result
            ),
            mapping=(
                prototype_mapping_view
            ),
            output_dir=(
                morphology_states_dir
            ),
            config=(
                config.plots
            ),
            umap_embedding=(
                umap_embedding
            ),
            umap_pca_dimensions=(
                shared_umap_d
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


    # ==================================================================
    # MULTIRESOLUTION MORPHOLOGY-STATE QC
    #
    # One row per robust K:
    #
    #       PCA        UMAP
    #
    # K=2   ...        ...
    # K=3   ...        ...
    # K=4   ...        ...
    #
    # PCA geometry is identical between rows.
    #
    # UMAP geometry is also shared between rows so only the clustering
    # partition changes. This makes K-to-K comparison interpretable.
    # ==================================================================

    multiresolution_morphology_path = (
        morphology_states_dir
        / "Morphology_States_Multiresolution.png"
    )


    plot_multiresolution_morphology_states(
        master=master,
        clustering=(
            clustering_result
        ),
        output_path=(
            multiresolution_morphology_path
        ),
        config=(
            config.plots
        ),
        umap_embedding=(
            umap_embedding
        ),
    )


    files[
        "morphology_states_multiresolution"
    ] = (
        multiresolution_morphology_path
    )


    # ==================================================================
    # STABILITY QC
    # ==================================================================

    stability_dir = (
        output_dir
        / "QC"
        / "Stability"
    )


    stability_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ==================================================================
    # PCA CUMULATIVE VARIANCE EXPLAINED
    # ==================================================================

    pca_variance_path = (
        stability_dir
        / "PCA_Variance_Explained.png"
    )

    plot_pca_variance_explained(
        dimensionality_result,
        selected_dimension=d,
        output_path=pca_variance_path,
        config=config.plots,
    )

    files[
        "pca_variance_explained"
    ] = pca_variance_path


    # ==================================================================
    # STABILITY HEATMAP
    # ==================================================================

    if (
        not clustering_result
        .stability_by_dimension_and_k
        .empty
    ):

        stability_path = (
            stability_dir
            / "Stability_Heatmap.png"
        )


        plot_stability_heatmap(
            clustering=(
                clustering_result
            ),
            output_path=(
                stability_path
            ),
            config=(
                config.plots
            ),
        )


        files[
            "stability_heatmap"
        ] = stability_path


    # ==================================================================
    # MULTIRESOLUTION STABILITY
    # ==================================================================

    if (
        not clustering_result
        .multiresolution_summary
        .empty
    ):

        multiresolution_path = (
            stability_dir
            / "Multiresolution_Stability.png"
        )


        plot_multiresolution_stability(
            clustering=(
                clustering_result
            ),
            output_path=(
                multiresolution_path
            ),
            config=(
                config.plots
            ),
        )


        files[
            "multiresolution_stability"
        ] = multiresolution_path


    # ==================================================================
    # COMPLETE
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "DIMENSIONALITY REDUCTION + CLUSTERING COMPLETE"
    )

    print(
        "=" * 72
    )

    print()


    return DimReductionClusteringResult(
        master=master,
        feature_preparation=(
            feature_result
        ),
        dimensionality_reduction=(
            dimensionality_result
        ),
        clustering=(
            clustering_result
        ),
        prototype_selections=(
            prototype_selections
        ),
        files=files,
        output_dir=(
            output_dir
        ),
    )