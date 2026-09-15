from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any
import json

import numpy as np
import pandas as pd

from .config import (
    PipelineConfig,
    PIPELINE_STAGES,
)

from .compute import (
    with_compute_limits,
)

from .metadata.stage import (
    run_metadata,
)
from .metadata.files import (
    FilenameInterpretation,
    NomenclatureScan,
)

from .preprocessing.stage import (
    run_preprocessing,
)
from .segmentation.stage import (
    run_segmentation,
)

from .instance_refinement import (
    load_saved_instance_refinement,
    passthrough_instance_refinement,
    run_instance_refinement,
)

from .morphometrics import (
    run_morphometrics,
)

from .category import (
    apply_category,
)



from .dim_reduction_clustering import (
    run_dim_reduction_clustering,
)

from .mapping import (
    MappingStage,
)

from .plots import (
    PlotsStage,
)

from .plots.style import (
    set_morphology_state_palette,
)

from .plots.config import (
    plot_output_directory_name,
)

from .technical_record.summary import (
    save_analysis_summary,
)
from .technical_record.timings import (
    save_timings_csv,
)
from .technical_record.errors import (
    log_error,
)


# ======================================================================
# RESULT
# ======================================================================

@dataclass
class PipelineResult:
    """
    Result of one canonical MorphoGlia pipeline execution.

    Stage-specific result objects are retained so advanced users can
    inspect diagnostics without rerunning the analysis.

    master
        Latest canonical cell-level table.

        Depending on enabled stages, this may contain:

            metadata
            morphometrics
            category
            PC1 ... PCn

    master_path
        Path to the latest saved master_table.csv.
    """

    master: pd.DataFrame | None = None
    master_path: Path | None = None

    metadata: Any | None = None
    preprocessing: Any | None = None
    segmentation: Any | None = None
    instance_refinement: Any | None = None
    morphometrics: Any | None = None

    category_applied: bool = False

    feature_preparation: Any | None = None
    dimensionality_reduction: Any | None = None
    clustering: Any | None = None
    mapping: Any | None = None
    plots: Any | None = None


# ======================================================================
# CONSOLE
# ======================================================================

def _section(
    title: str,
) -> None:

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print()


# ======================================================================
# DEPENDENCIES
# ======================================================================

_STAGE_DEPENDENCIES = {
    "metadata": (),
    "preprocessing": (
        "metadata",
    ),
    "segmentation": (
        "metadata",
        "preprocessing",
    ),
    "instance_refinement": (
        "segmentation",
    ),
    "morphometrics": (
        "metadata",
        "instance_refinement",
    ),
    "category": (),
    "dim_reduction_clustering": (),
    "mapping": (),
    "plots": (),
}


def _validate_stage_dependencies(
    config: PipelineConfig,
) -> None:
    """Validate only globally unsupported stage requests.

    True/False controls execution only. Required inputs for an ON stage are
    resolved at runtime from current in-memory results or existing canonical
    files.
    """

    if config.run.is_enabled(
        "spatial_analysis"
    ):
        raise NotImplementedError(
            "Spatial Analysis is part of the canonical MorphoGlia "
            "pipeline sequence but is not implemented yet. "
            "Set config.run.spatial_analysis = False."
        )


# ======================================================================
# MASTER TABLE
# ======================================================================

def _load_master(
    output_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    """
    Load the canonical master table from a previous pipeline run.
    """

    path = (
        output_dir
        / "Data"
        / "master_table.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Required Morphometrics output was not found: "
            f"{path}"
        )

    master = pd.read_csv(
        path
    )

    if master.empty:
        raise ValueError(
            f"Existing master table is empty: {path}"
        )

    print()
    print("Existing Morphometrics output detected")
    print("-" * 72)
    print(f"Master: {path}")
    print(f"Cells:  {len(master)}")
    print()

    return (
        master,
        path,
    )


def _save_master(
    master: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Save the latest canonical master table.

    The same path is overwritten as the table gains Category and PCA
    information, so master_table.csv remains the cell-level source of
    truth for the current run.
    """

    data_dir = (
        output_dir
        / "Data"
    )


    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    path = (
        data_dir
        / "master_table.csv"
    )


    master.to_csv(
        path,
        index=False,
    )


    return path


def _attach_pca_scores(
    master: pd.DataFrame,
    scores,
) -> pd.DataFrame:
    """
    Attach full analytical PCA coordinates to the master.

    Mapping and Plots consume PC1 ... PCd directly from the master.

    PCA is fitted only once upstream by Dimensionality Reduction.
    """

    output = (
        master
        .copy()
        .reset_index(
            drop=True
        )
    )


    if scores.ndim != 2:

        raise ValueError(
            "PCA scores must be a 2D array."
        )


    if scores.shape[0] != len(
        output
    ):

        raise ValueError(
            "PCA score rows do not match master-table rows: "
            f"{scores.shape[0]} versus {len(output)}."
        )


    # Prevent stale PCA columns if the helper is ever applied more than
    # once to the same table.
    stale_pc_columns = [
        column
        for column in output.columns
        if (
            column.startswith(
                "PC"
            )
            and column[
                2:
            ].isdigit()
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
# CLUSTERING STRATA
# ======================================================================

def _clustering_strata(
    master: pd.DataFrame,
) -> pd.Series:
    """
    Return the canonical structured-resampling strata.

    Cells are subsampled within source images rather than treating all
    cells as IID observations.
    """

    if "image_id" not in master.columns:

        raise ValueError(
            "Clustering requires master['image_id'] "
            "for structured resampling."
        )


    return (
        master[
            "image_id"
        ]
        .astype(str)
    )


# ======================================================================
# INSTANCE-MAP DIRECTORY
# ======================================================================

def _instance_directory(
    segmentation_result,
) -> Path:
    """
    Resolve the single canonical instance-map directory generated by
    Segmentation.
    """

    instance_paths = [
        Path(
            path
        )
        for path in (
            segmentation_result
            .instance_paths
        )
    ]


    if not instance_paths:

        raise ValueError(
            "Mapping requires canonical instance maps, but "
            "SegmentationResult.instance_paths is empty."
        )


    parents = {
        path.parent.resolve()
        for path in instance_paths
    }


    if len(
        parents
    ) != 1:

        raise ValueError(
            "Canonical preprocessing binaries are expected "
            "to share one directory. Found: "
            f"{sorted(str(path) for path in parents)}"
        )


    return next(
        iter(
            parents
        )
    )


# ======================================================================
# SAVED INPUT LOADERS
# ======================================================================

def _read_csv_required(
    path: Path,
    *,
    label: str,
) -> pd.DataFrame:
    """Read one required canonical CSV output."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required saved {label} was not found: {path}"
        )

    table = pd.read_csv(path)

    if table.empty:
        raise ValueError(
            f"Saved {label} is empty: {path}"
        )

    return table


def _truthy_series(
    series: pd.Series,
) -> pd.Series:
    """Interpret a persisted boolean column robustly."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({
            "true",
            "1",
            "yes",
            "y",
        })
    )


def _load_saved_metadata(
    config: PipelineConfig,
    output_dir: Path,
):
    """
    Reconstruct the Metadata result from the saved nomenclature record.

    Metadata being OFF means its current configuration is not re-applied.
    The saved declaration is read exactly as data produced by the previous
    Metadata execution. Only missing or structurally unreadable data is an
    error.
    """

    path = (
        output_dir
        / "Technical_Record"
        / "nomenclature.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Required Metadata output does not exist: "
            f"{path}"
        )

    table = pd.read_csv(
        path,
        keep_default_na=False,
    )

    if table.empty:
        raise ValueError(
            f"Saved nomenclature record is empty: {path}"
        )

    required = {
        "original_filename",
        "canonical_filename",
        "status",
        "issue",
        "field_positions",
    }

    missing = required - set(table.columns)

    if missing:
        raise ValueError(
            "Saved nomenclature record cannot be read. Missing columns: "
            f"{sorted(missing)}"
        )

    declarations = {
        str(value)
        for value in table["field_positions"].unique()
    }

    if len(declarations) != 1:
        raise ValueError(
            "Saved nomenclature record contains more than one field mapping."
        )

    declaration = next(iter(declarations))

    field_order = []

    if declaration.strip():
        for item in declaration.split(";"):
            item = item.strip()
            if not item:
                continue
            field_name = item.split("=", 1)[0].strip()
            if field_name and field_name not in field_order:
                field_order.append(field_name)

    interpretations = []

    for _, row in table.iterrows():

        metadata = {
            field_name: str(row[field_name])
            for field_name in field_order
            if (
                field_name in table.columns
                and str(row[field_name]) != ""
            )
        }

        canonical_filename = str(
            row["canonical_filename"]
        ).strip()

        issue = str(
            row["issue"]
        ).strip()

        interpretations.append(
            FilenameInterpretation(
                original_filename=str(
                    row["original_filename"]
                ),
                status=str(
                    row["status"]
                ),
                metadata=metadata,
                canonical_filename=(
                    canonical_filename
                    if canonical_filename
                    else None
                ),
                issue=(
                    issue
                    if issue
                    else None
                ),
            )
        )

    scan = NomenclatureScan(
        input_dir=Path(config.input_dir),
        interpretations=interpretations,
        field_order=tuple(field_order),
    )

    if scan.recognized_count == 0:
        raise ValueError(
            "Saved Metadata output contains no recognized images."
        )

    return SimpleNamespace(
        scan=scan,
        microns_per_pixel=(
            config.metadata.effective_microns_per_pixel
        ),
        calibration_source="loaded",
        configuration_csv=(
            output_dir
            / "Technical_Record"
            / "configuration.csv"
        ),
        nomenclature_csv=path,
        loaded=True,
    )


def _load_saved_preprocessing(
    config: PipelineConfig,
    output_dir: Path,
    metadata_result,
):
    """Reconstruct the minimal PreprocessingResult needed by Segmentation."""

    mode = config.preprocessing.effective_input_mode
    prepared_paths = []

    for interpretation in metadata_result.scan.recognized:

        if mode in {
            "binary",
            "labels",
        }:
            path = (
                Path(config.input_dir)
                / interpretation.original_filename
            )

        else:
            canonical_filename = (
                interpretation.canonical_filename
            )

            if canonical_filename is None:
                raise ValueError(
                    "Saved recognized Metadata entry has no canonical filename."
                )

            path = (
                output_dir
                / "Preprocessing"
                / "Binary"
                / f"{Path(canonical_filename).stem}.tif"
            )

        if not path.is_file():
            raise FileNotFoundError(
                "Preprocessing is disabled, but a required prepared image is "
                f"missing: {path}. Enable config.run.preprocessing = True."
            )

        prepared_paths.append(path)

    return SimpleNamespace(
        prepared_paths=prepared_paths,
        input_mode=mode,
        skipped_others=(
            metadata_result.scan.others_count
        ),
        processed_count=len(prepared_paths),
        reused=True,
    )


def _load_saved_segmentation(
    output_dir: Path,
    metadata_result,
    input_mode: str,
):
    """
    Reconstruct SegmentationResult from the run-level Segmentation manifest.

    New runs avoid touching every TIFF merely to prove that Segmentation has
    not changed. Older runs without a manifest retain the historical fallback.
    """

    segmentation_root = (
        output_dir
        / "Segmentation"
    )

    instance_dir = (
        segmentation_root
        / "_Instance_Maps"
    )

    manifest_path = (
        segmentation_root
        / "segmentation_manifest.json"
    )

    expected_image_ids = []

    for interpretation in metadata_result.scan.recognized:
        canonical_filename = (
            interpretation.canonical_filename
        )

        if canonical_filename is None:
            raise ValueError(
                "Saved recognized Metadata entry has no canonical filename."
            )

        expected_image_ids.append(
            Path(
                canonical_filename
            ).stem
        )

    if not expected_image_ids:
        raise ValueError(
            "No canonical instance maps were recovered for Segmentation reuse."
        )

    generation_id = None

    if manifest_path.is_file():

        try:
            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                manifest = json.load(
                    file
                )
        except Exception as exc:
            raise ValueError(
                "Saved Segmentation manifest cannot be read: "
                f"{manifest_path}"
            ) from exc

        manifest_image_ids = [
            str(
                value
            )
            for value in manifest.get(
                "image_ids",
                [],
            )
        ]

        if manifest_image_ids != expected_image_ids:
            raise ValueError(
                "Saved Segmentation manifest does not match the current "
                "Metadata image set/order. Re-run "
                "config.run.segmentation = True."
            )

        generation_id = str(
            manifest.get(
                "generation_id",
                "",
            )
        ).strip() or None

        instance_paths = [
            instance_dir
            / f"{image_id}.tif"
            for image_id in expected_image_ids
        ]

    else:

        # Compatibility fallback for Segmentation results created before the
        # generation manifest existed.
        instance_paths = []

        for image_id in expected_image_ids:

            path = (
                instance_dir
                / f"{image_id}.tif"
            )

            if not path.is_file():
                raise FileNotFoundError(
                    "Segmentation is disabled, but a required canonical "
                    f"instance map is missing: {path}. Enable "
                    "config.run.segmentation = True."
                )

            instance_paths.append(
                path
            )

    return SimpleNamespace(
        instance_paths=instance_paths,
        input_mode=str(input_mode),
        skipped_others=(
            metadata_result.scan.others_count
        ),
        processed_count=len(instance_paths),
        generation_id=generation_id,
        manifest_path=(
            manifest_path
            if manifest_path.is_file()
            else None
        ),
        reused=True,
    )


def _canonical_instance_dir(
    output_dir: Path,
) -> Path:
    """Return the canonical instance-map directory after validating it."""

    path = (
        output_dir
        / "Segmentation"
        / "_Instance_Maps"
    )

    if not path.is_dir():
        raise FileNotFoundError(
            "Canonical instance-map directory was not found: "
            f"{path}"
        )

    if not any(
        child.is_file()
        and child.suffix.lower() in {
            ".tif",
            ".tiff",
        }
        for child in path.iterdir()
    ):
        raise FileNotFoundError(
            f"Canonical instance-map directory is empty: {path}"
        )

    return path


def _load_saved_feature_preparation(
    master: pd.DataFrame,
    output_dir: Path,
):
    """Reconstruct the analytical feature list used by saved DRC output."""

    root = (
        output_dir
        / "Dimensionality_Reduction_Clustering"
        / "Feature_Preparation"
    )

    qc_path = (
        root
        / "feature_qc.csv"
    )

    qc = _read_csv_required(
        qc_path,
        label="Feature Preparation QC",
    )

    if not {
        "feature",
        "kept",
    }.issubset(qc.columns):
        raise ValueError(
            "Saved feature_qc.csv is incompatible: expected 'feature' and "
            "'kept' columns."
        )

    keep = _truthy_series(
        qc["kept"]
    )

    analytical_features = [
        str(feature)
        for feature in qc.loc[
            keep,
            "feature",
        ].tolist()
    ]

    if not analytical_features:
        raise ValueError(
            "Saved Feature Preparation contains no analytical features."
        )

    missing = [
        feature
        for feature in analytical_features
        if feature not in master.columns
    ]

    if missing:
        raise ValueError(
            "Saved Feature Preparation is incompatible with master_table.csv. "
            f"Missing analytical features: {missing[:10]}"
        )

    excluded_features = [
        str(feature)
        for feature in qc.loc[
            ~keep,
            "feature",
        ].tolist()
    ]

    correlation_path = (
        root
        / "correlation_matrix.csv"
    )

    correlation_matrix = (
        pd.read_csv(
            correlation_path,
            index_col=0,
        )
        if correlation_path.is_file()
        else None
    )

    return SimpleNamespace(
        data=master[
            analytical_features
        ].copy(),
        analytical_features=(
            analytical_features
        ),
        excluded_features=(
            excluded_features
        ),
        qc_report=qc,
        correlation_matrix=(
            correlation_matrix
        ),
        reused=True,
    )


def _load_optional_csv(
    path: Path,
) -> pd.DataFrame:
    """Read an optional CSV, returning an empty DataFrame if absent."""

    return (
        pd.read_csv(path)
        if path.is_file()
        else pd.DataFrame()
    )


def _load_saved_clustering(
    master: pd.DataFrame,
    output_dir: Path,
):
    """
    Reconstruct the clustering interface required by Mapping and Plots from the
    canonical analytical CSV outputs.
    """

    root = (
        output_dir
        / "Dimensionality_Reduction_Clustering"
    )

    clustering_dir = (
        root
        / "Clustering"
    )

    selected_path = (
        clustering_dir
        / "selected_solution_cell_labels.csv"
    )

    selected = _read_csv_required(
        selected_path,
        label="selected clustering cell labels",
    )

    required = {
        "row_index",
        "k",
        "pca_dimensions",
        "covariance_type",
        "cluster",
        "cluster_probability",
        "consensus_reliability",
        "mean_subsample_ari",
        "std_subsample_ari",
    }

    missing = required - set(
        selected.columns
    )

    if missing:
        raise ValueError(
            "Saved selected_solution_cell_labels.csv is incompatible. "
            f"Missing columns: {sorted(missing)}"
        )

    selected_solutions = {}

    for raw_k, group in selected.groupby(
        "k",
        sort=True,
    ):

        k = int(raw_k)

        group = (
            group
            .copy()
            .sort_values(
                "row_index",
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        expected_indices = np.arange(
            len(master),
            dtype=int,
        )

        observed_indices = group[
            "row_index"
        ].to_numpy(
            dtype=int
        )

        if (
            len(group) != len(master)
            or not np.array_equal(
                observed_indices,
                expected_indices,
            )
        ):
            raise ValueError(
                "Saved clustering labels do not align with the current "
                f"master table for K={k}. Rerun "
                "config.run.dim_reduction_clustering = True."
            )

        dimensions = group[
            "pca_dimensions"
        ].astype(int).unique()

        covariance_types = group[
            "covariance_type"
        ].astype(str).unique()

        if len(dimensions) != 1:
            raise ValueError(
                f"Saved K={k} has inconsistent PCA dimensions."
            )

        if len(covariance_types) != 1:
            raise ValueError(
                f"Saved K={k} has inconsistent covariance types."
            )

        ari_values = group[
            "mean_subsample_ari"
        ].astype(float).unique()

        ari_sd_values = group[
            "std_subsample_ari"
        ].astype(float).unique()

        selected_solutions[k] = SimpleNamespace(
            labels=group[
                "cluster"
            ].to_numpy(
                dtype=int
            ),
            pca_dimensions=int(
                dimensions[0]
            ),
            covariance_type=str(
                covariance_types[0]
            ),
            cluster_probability=group[
                "cluster_probability"
            ].to_numpy(
                dtype=float
            ),
            consensus_reliability=group[
                "consensus_reliability"
            ].to_numpy(
                dtype=float
            ),
            mean_subsample_ari=float(
                ari_values[0]
            ),
            std_subsample_ari=float(
                ari_sd_values[0]
            ),
            reused=True,
        )

    robust_k_values = tuple(
        sorted(
            int(k)
            for k in selected_solutions
        )
    )

    if not robust_k_values:
        raise ValueError(
            "Saved clustering contains no robust K values."
        )

    decision_path = (
        root
        / "Decision"
        / "resolution_decision.csv"
    )

    decision = _read_csv_required(
        decision_path,
        label="resolution decision",
    )

    if not {
        "k",
        "pareto_optimal",
        "decision_role",
    }.issubset(decision.columns):
        raise ValueError(
            "Saved resolution_decision.csv is incompatible."
        )

    preferred_rows = decision[
        decision[
            "decision_role"
        ].astype(str)
        == "preferred"
    ]

    if len(preferred_rows) != 1:
        raise ValueError(
            "Saved resolution decision must contain exactly one preferred K."
        )

    preferred_k = int(
        preferred_rows.iloc[0][
            "k"
        ]
    )

    if preferred_k not in selected_solutions:
        raise ValueError(
            "Saved preferred K is absent from selected clustering solutions: "
            f"K={preferred_k}."
        )

    pareto_k_values = tuple(
        sorted(
            int(k)
            for k in decision.loc[
                _truthy_series(
                    decision[
                        "pareto_optimal"
                    ]
                ),
                "k",
            ].tolist()
        )
    )

    coarse_rows = decision[
        decision[
            "decision_role"
        ].astype(str)
        == "coarse"
    ]

    coarse_k = (
        int(
            coarse_rows.iloc[0][
                "k"
            ]
        )
        if len(coarse_rows) == 1
        else None
    )

    dimension_summary = _load_optional_csv(
        root
        / "Dimensionality_Reduction"
        / "dimension_summary.csv"
    )

    plausible_dimensions = ()

    if not dimension_summary.empty:
        row = dimension_summary.iloc[0]
        lower = row.get(
            "plausible_dimension_min"
        )
        upper = row.get(
            "plausible_dimension_max"
        )

        if pd.notna(lower) and pd.notna(upper):
            plausible_dimensions = tuple(
                range(
                    int(lower),
                    int(upper) + 1,
                )
            )

    multiresolution_summary = (
        _load_optional_csv(
            clustering_dir
            / "multiresolution_summary.csv"
        )
    )

    strongest_k = None

    strongest_columns = {
        "k",
        "minimum_subsample_ari",
        "mean_dimension_agreement",
        "mean_subsample_ari",
    }

    if (
        not multiresolution_summary.empty
        and strongest_columns.issubset(
            multiresolution_summary.columns
        )
    ):
        strongest_candidates = (
            multiresolution_summary[
                multiresolution_summary[
                    "k"
                ].astype(int)
                .isin(
                    robust_k_values
                )
            ]
            .sort_values(
                [
                    "minimum_subsample_ari",
                    "mean_dimension_agreement",
                    "mean_subsample_ari",
                    "k",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        if not strongest_candidates.empty:
            strongest_k = int(
                strongest_candidates.iloc[
                    0
                ][
                    "k"
                ]
            )

    # strongest_k is the historical maximin resolution. If an older saved
    # result lacks the summary columns needed to reconstruct it, preferred_k
    # is the compatibility fallback because current downstream plots prefer
    # preferred_k whenever it is available.
    if strongest_k is None:
        strongest_k = preferred_k

    representative_dimensions = {
        int(k): int(
            solution.pca_dimensions
        )
        for k, solution in (
            selected_solutions.items()
        )
    }

    return SimpleNamespace(
        robust_k_values=robust_k_values,
        pareto_k_values=pareto_k_values,
        preferred_k=preferred_k,
        coarse_k=coarse_k,
        strongest_k=strongest_k,
        maximin_k=strongest_k,
        representative_dimensions=(
            representative_dimensions
        ),
        by_solution={},
        selected_solutions=selected_solutions,
        stability_by_dimension_and_k=(
            _load_optional_csv(
                clustering_dir
                / "stability.csv"
            )
        ),
        multiresolution_summary=(
            multiresolution_summary
        ),
        dimension_agreement=(
            _load_optional_csv(
                clustering_dir
                / "dimension_agreement.csv"
            )
        ),
        model_scan=(
            _load_optional_csv(
                clustering_dir
                / "model_scan.csv"
            )
        ),
        candidate_assignments=(
            _load_optional_csv(
                clustering_dir
                / "candidate_assignments.csv"
            )
        ),
        plausible_dimensions=(
            plausible_dimensions
        ),
        resolution_decision=decision,
        reused=True,
    )


_mg_load_saved_clustering_legacy = _load_saved_clustering


def _load_saved_mapping(
    config: PipelineConfig,
    output_dir: Path,
    clustering_result,
):
    """Reconstruct the Mapping interface required by Plots."""

    preferred_k = int(
        clustering_result.preferred_k
    )

    root = (
        output_dir
        / "Mapping"
    )

    label_path = (
        root
        / "cell_labels.csv"
    )

    prototype_path = (
        root
        / "_Prototypes"
        / "prototype_cells.csv"
    )

    prototype_grid = (
        root
        / "_Prototypes"
        / "Prototype_Cells.png"
    )

    labels = _read_csv_required(
        label_path,
        label="Mapping cell labels",
    )

    prototype_cells = _read_csv_required(
        prototype_path,
        label="Mapping prototype cells",
    )

    if not prototype_grid.is_file():
        raise FileNotFoundError(
            "Mapping is disabled, but the canonical prototype grid is "
            f"missing: {prototype_grid}. Enable config.run.mapping = True."
        )

    if "selection_rank" not in prototype_cells.columns:
        raise ValueError(
            "Saved Mapping prototype table has no selection_rank column."
        )

    canonical = (
        prototype_cells[
            prototype_cells[
                "selection_rank"
            ].astype(int)
            == 1
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if canonical.empty:
        raise ValueError(
            "Saved Mapping contains no canonical rank-1 prototypes."
        )

    if "pca_dimensions" in labels.columns:
        observed_dimensions = (
            labels[
                "pca_dimensions"
            ]
            .dropna()
            .astype(int)
            .unique()
        )

        expected_dimensions = int(
            clustering_result
            .selected_solutions[
                preferred_k
            ]
            .pca_dimensions
        )

        if (
            len(observed_dimensions) != 1
            or int(observed_dimensions[0])
            != expected_dimensions
        ):
            raise ValueError(
                "Saved Mapping was generated from a different preferred "
                "clustering solution. Enable config.run.mapping = True."
            )

    if len(labels) == 0:
        raise ValueError(
            "Saved Mapping cell_labels.csv is empty."
        )

    manifest_name = getattr(
        config.mapping,
        "mapping_manifest",
        "mapping.csv",
    )

    manifest = _load_optional_csv(
        root
        / str(manifest_name)
    )

    component_errors = _load_optional_csv(
        root
        / "component_match_errors.csv"
    )

    return SimpleNamespace(
        manifest=manifest,
        cell_labels={
            preferred_k:
                labels
        },
        selections={
            preferred_k: {
                "prototype":
                    canonical,
            }
        },
        component_errors=(
            component_errors
        ),
        preferred_k=preferred_k,
        output_dir=root,
        reused=True,
    )


def _category_available(
    master: pd.DataFrame | None,
    config: PipelineConfig,
) -> bool:
    """Return whether a Category result is expected and present."""

    return bool(
        master is not None
        and config.category.category_fields
        and "category" in master.columns
    )


def _validate_saved_category(
    master: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    """Validate the persisted Category column against current fields."""

    fields = tuple(
        config.category.category_fields
    )

    if not fields:
        return

    if "category" not in master.columns:
        raise ValueError(
            "Category is disabled, but master_table.csv has no 'category' "
            "column. Enable config.run.category = True."
        )

    missing = [
        field
        for field in fields
        if field not in master.columns
    ]

    if missing:
        raise ValueError(
            "The current Category declaration requires metadata columns that "
            f"are absent from master_table.csv: {missing}. Enable the required "
            "upstream stages."
        )

    expected = apply_category(
        master,
        config.category,
    )[
        "category"
    ].astype(str)

    observed = (
        master[
            "category"
        ]
        .astype(str)
    )

    if not observed.equals(expected):
        raise ValueError(
            "Category is disabled, but the saved 'category' column does not "
            "match the current config.category.category_fields. Enable "
            "config.run.category = True."
        )




def _existing_analysis_summary(
    output_dir: Path,
) -> dict:
    """Load the existing analysis summary so execution audit updates are non-destructive."""

    path = (
        output_dir
        / "Technical_Record"
        / "analysis_summary.json"
    )

    if not path.is_file():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            value = json.load(file)
    except Exception:
        return {}

    return (
        value
        if isinstance(value, dict)
        else {}
    )


def _save_execution_audit(
    *,
    config: PipelineConfig,
    output_dir: Path,
    actions: dict[str, str],
    seconds: dict[str, float],
) -> None:
    """
    Persist the simple public execution contract.

    True  -> the stage was requested and is executed now.
    False -> the stage is OFF and is not executed now.

    Reading an existing output as input for another stage does not change the
    OFF stage into a separate execution state.
    """

    summary = _existing_analysis_summary(
        output_dir
    )

    summary[
        "pipeline_execution"
    ] = {
        "requested": {
            stage: config.run.is_enabled(stage)
            for stage in PIPELINE_STAGES
        },
        "stage_actions": {
            stage: actions.get(
                stage,
                "OFF",
            )
            for stage in PIPELINE_STAGES
        },
        "public_controls": {
            "resume": bool(
                config.resume
            ),
            "microns_per_pixel": (
                config.metadata
                .effective_microns_per_pixel
            ),
            "input_mode": (
                config.preprocessing
                .effective_input_mode
            ),
            "invert": (
                config.preprocessing
                .effective_invert
                if config.preprocessing
                .effective_input_mode
                == "binary"
                else False
            ),
            "category_fields": list(
                config.category
                .category_fields
            ),
        },
    }

    save_analysis_summary(
        output_dir=output_dir,
        summary=summary,
    )

    save_timings_csv(
        output_dir=output_dir,
        rows=[
            {
                "stage": stage,
                "operation": actions.get(
                    stage,
                    "OFF",
                ),
                "ran": (
                    actions.get(stage)
                    == "RUN"
                ),
                "seconds": (
                    seconds.get(stage, 0.0)
                    if actions.get(stage) == "RUN"
                    else 0.0
                ),
            }
            for stage in PIPELINE_STAGES
        ],
    )


def _print_execution_plan(
    config: PipelineConfig,
    actions: dict[str, str],
) -> None:
    """Print the simple True / False stage execution result."""

    print()
    print("Stage execution")
    print("-" * 72)

    for stage in PIPELINE_STAGES:
        requested = config.run.is_enabled(stage)
        status = actions.get(stage, "OFF")

        print(
            f"{stage:<34} "
            f"{str(requested):<6} "
            f"{status}"
        )

    print()


# ======================================================================
# CANONICAL PIPELINE
# ======================================================================

# MG_STAGE_SKIP_UNAVAILABLE_V1
class _StagePrerequisiteUnavailable(
    RuntimeError
):
    # Internal control-flow signal for an unavailable saved prerequisite.

    def __init__(
        self,
        consumer_stage: str,
        label: str,
        original: Exception,
    ) -> None:

        self.consumer_stage = str(
            consumer_stage
        )

        self.label = str(
            label
        )

        self.original = original

        super().__init__(
            f"{self.consumer_stage}: unavailable prerequisite "
            f"{self.label}: {original}"
        )

@with_compute_limits


def run_pipeline(
    config: PipelineConfig,
) -> PipelineResult:
    """
    Execute the canonical MorphoGlia pipeline.

    Public stage flags have one simple meaning:

        True
            execute the stage now.

        False
            do not execute the stage now.

    An enabled stage may read existing canonical outputs produced by an OFF
    stage when it needs them as inputs. Reading those files is data loading,
    not execution of the OFF stage.

    If that saved prerequisite is absent, incompatible, or unreadable, the
    enabled consumer is SKIPPED. This is not an ERROR. ERROR is reserved for
    failures that occur while an enabled stage is actually executing.
    """

    if not isinstance(
        config,
        PipelineConfig,
    ):
        raise TypeError(
            "run_pipeline expects a PipelineConfig."
        )

    # Apply the presentation-only morphology-state identity palette.
    #
    # This changes colors only where mg_color() / morphology_state_rgb()
    # are already used. It does not change clustering or quantitative
    # statistical color semantics.
    set_morphology_state_palette(
        config.plots.effective_morphology_state_palette
    )

    output_dir = Path(
        config.output_dir
    )

    # A requested morphology-state count is allowed even on the first
    # analytical run. DRC first estimates the supported counts from the data
    # and then validates the requested interpretation against that evidence.
    requested_state_count = (
        config.clustering.requested_number_of_morphology_states
    )

    result = PipelineResult()

    actions = {
        stage: "OFF"
        for stage in PIPELINE_STAGES
    }

    seconds = {
        stage: 0.0
        for stage in PIPELINE_STAGES
    }

    current_stage = "pipeline"

    def operate(
        stage: str,
        action: str,
        function,
    ):
        nonlocal current_stage

        current_stage = stage
        start = perf_counter()

        try:
            value = function()
        except Exception as exc:
            seconds[stage] += (
                perf_counter()
                - start
            )
            actions[stage] = "ERROR"

            log_error(
                output_dir=output_dir,
                stage=stage,
                operation=action,
                message=(
                    f"Pipeline stage {stage!r} failed during {action}."
                ),
                exception=exc,
            )

            _save_execution_audit(
                config=config,
                output_dir=output_dir,
                actions=actions,
                seconds=seconds,
            )

            raise

        seconds[stage] += (
            perf_counter()
            - start
        )
        actions[stage] = action

        return value

    def mark_skipped(
        stage: str,
        label: str,
        exception: Exception,
    ) -> None:
        # Mark one requested stage as skipped; do not write an error log.

        actions[
            stage
        ] = "SKIPPED"

        print()
        print(
            f"{stage.upper()}: SKIPPED"
        )
        print(
            f"Prerequisite unavailable: {label}"
        )
        print(
            "Reason: "
            f"{type(exception).__name__}: {exception}"
        )
        print()


    def load_input(
        consumer_stage: str,
        label: str,
        function,
    ):
        # Load an existing prerequisite from an OFF stage.
        #
        # Missing/incompatible saved data skips the consumer. Real failures
        # inside operate(..., "RUN", ...) still remain ERROR.

        nonlocal current_stage

        current_stage = (
            consumer_stage
        )

        try:
            return function()

        except Exception as exc:

            mark_skipped(
                consumer_stage,
                label,
                exc,
            )

            raise _StagePrerequisiteUnavailable(
                consumer_stage=consumer_stage,
                label=label,
                original=exc,
            ) from None


    # Runtime objects. Some are generated in this execution and some are
    # loaded from canonical saved outputs only when needed.
    metadata_result = None
    preprocessing_result = None
    segmentation_result = None
    instance_refinement_result = None
    morphometrics_result = None

    master = None

    feature_result = None
    dimensionality_result = None
    clustering_result = None
    mapping_result = None
    plots_mapping_result = None

    # ==================================================================
    # PIPELINE HEADER
    # ==================================================================

    _section(
        "MORPHOGLIA PIPELINE"
    )

    # MG_RESUME_PIPELINE_V1
    print(
        f"Resume:           {bool(config.resume)}"
    )
    print()

    print(
        "Requested stages:"
    )

    for stage in PIPELINE_STAGES:
        print(
            f"  {stage:<32} "
            f"{config.run.is_enabled(stage)}"
        )

    print()

    try:
        # ==============================================================
        # RESERVED STAGE
        # ==============================================================

        if config.run.is_enabled(
            "spatial_analysis"
        ):
            current_stage = "spatial_analysis"
            actions[
                "spatial_analysis"
            ] = "ERROR"

            raise NotImplementedError(
                "Spatial Analysis is part of the canonical MorphoGlia "
                "pipeline sequence but is not implemented yet. Set "
                "config.run.spatial_analysis = False."
            )

        # ==============================================================
        # 1. METADATA
        # ==============================================================

        if config.run.is_enabled(
            "metadata"
        ):

            if actions.get("metadata") != "SKIPPED":
                try:
                    metadata_result = operate(
                        "metadata",
                        "RUN",
                        lambda: run_metadata(
                            input_dir=(
                                config.input_dir
                            ),
                            output_dir=(
                                output_dir
                            ),
                            config=(
                                config.metadata
                            ),
                        ),
                    )

                    result.metadata = (
                        metadata_result
                    )
                except _StagePrerequisiteUnavailable:
                    pass

        # ==============================================================
        # 2. PREPROCESSING
        # ==============================================================

        if config.run.is_enabled(
            "preprocessing"
        ):

            if actions.get("preprocessing") != "SKIPPED":
                try:
                    if metadata_result is None:
                        metadata_result = load_input(
                            "preprocessing",
                            "metadata",
                            lambda: _load_saved_metadata(
                                config,
                                output_dir,
                            ),
                        )
                        result.metadata = metadata_result

                    preprocessing_result = operate(
                        "preprocessing",
                        "RUN",
                        lambda: run_preprocessing(
                            input_dir=(
                                config.input_dir
                            ),
                            output_dir=(
                                output_dir
                            ),
                            metadata_scan=(
                                metadata_result.scan
                            ),
                            config=(
                                config.preprocessing
                            ),
                            resume=(
                                config.resume
                            ),
                        ),
                    )

                    result.preprocessing = (
                        preprocessing_result
                    )
                except _StagePrerequisiteUnavailable:
                    pass

        # ==============================================================
        # 3. SEGMENTATION
        # ==============================================================

        if config.run.is_enabled(
            "segmentation"
        ):

            if actions.get("segmentation") != "SKIPPED":
                try:
                    if metadata_result is None:
                        metadata_result = load_input(
                            "segmentation",
                            "metadata",
                            lambda: _load_saved_metadata(
                                config,
                                output_dir,
                            ),
                        )
                        result.metadata = metadata_result

                    if preprocessing_result is None:
                        preprocessing_result = load_input(
                            "segmentation",
                            "preprocessing",
                            lambda: _load_saved_preprocessing(
                                config,
                                output_dir,
                                metadata_result,
                            ),
                        )
                        result.preprocessing = (
                            preprocessing_result
                        )

                    segmentation_result = operate(
                        "segmentation",
                        "RUN",
                        lambda: run_segmentation(
                            preprocessing_result=(
                                preprocessing_result
                            ),
                            metadata_scan=(
                                metadata_result.scan
                            ),
                            output_dir=(
                                output_dir
                            ),
                            preprocessing_config=(
                                config.preprocessing
                            ),
                            config=(
                                config.segmentation
                            ),
                            resume=(
                                config.resume
                            ),
                        ),
                    )

                    result.segmentation = (
                        segmentation_result
                    )
                except _StagePrerequisiteUnavailable:
                    pass

        # ==============================================================
        # 4. INSTANCE REFINEMENT
        # ==============================================================

        if config.run.is_enabled(
            "instance_refinement"
        ):

            if actions.get("instance_refinement") != "SKIPPED":
                try:
                    if metadata_result is None:
                        metadata_result = load_input(
                            "instance_refinement",
                            "metadata",
                            lambda: _load_saved_metadata(
                                config,
                                output_dir,
                            ),
                        )
                        result.metadata = metadata_result

                    if segmentation_result is None:
                        segmentation_result = load_input(
                            "instance_refinement",
                            "segmentation",
                            lambda: _load_saved_segmentation(
                                output_dir=output_dir,
                                metadata_result=metadata_result,
                                input_mode=(
                                    config.preprocessing
                                    .effective_input_mode
                                ),
                            ),
                        )
                        result.segmentation = segmentation_result

                    instance_refinement_result = operate(
                        "instance_refinement",
                        "RUN",
                        lambda: run_instance_refinement(
                            segmentation_result=segmentation_result,
                            output_dir=output_dir,
                            config=config.instance_refinement,
                            resume=config.resume,
                        ),
                    )

                    result.instance_refinement = instance_refinement_result
                except _StagePrerequisiteUnavailable:
                    pass

        # ==============================================================
        # 5. MORPHOMETRICS
        # ==============================================================

        if config.run.is_enabled(
            "morphometrics"
        ):

            if actions.get("morphometrics") != "SKIPPED":
                try:
                    if metadata_result is None:
                        metadata_result = load_input(
                            "morphometrics",
                            "metadata",
                            lambda: _load_saved_metadata(
                                config,
                                output_dir,
                            ),
                        )
                        result.metadata = metadata_result

                    if segmentation_result is None:
                        segmentation_result = load_input(
                            "morphometrics",
                            "segmentation",
                            lambda: _load_saved_segmentation(
                                output_dir=output_dir,
                                metadata_result=metadata_result,
                                input_mode=(
                                    config.preprocessing
                                    .effective_input_mode
                                ),
                            ),
                        )
                        result.segmentation = segmentation_result

                    if instance_refinement_result is None:
                        if actions.get("segmentation") == "RUN":
                            instance_refinement_result = (
                                passthrough_instance_refinement(
                                    segmentation_result
                                )
                            )
                        else:
                            saved_refinement = load_input(
                                "morphometrics",
                                "instance_refinement",
                                lambda: load_saved_instance_refinement(
                                    segmentation_result=segmentation_result,
                                    output_dir=output_dir,
                                    allow_missing=True,
                                ),
                            )
                            instance_refinement_result = (
                                saved_refinement
                                if saved_refinement is not None
                                else passthrough_instance_refinement(
                                    segmentation_result
                                )
                            )

                        result.instance_refinement = (
                            instance_refinement_result
                        )

                    morphometrics_result = operate(
                        "morphometrics",
                        "RUN",
                        lambda: run_morphometrics(
                            instance_paths=(
                                instance_refinement_result
                                .effective_instance_paths
                            ),
                            metadata_scan=(
                                metadata_result.scan
                            ),
                            output_dir=(
                                output_dir
                            ),
                            config=config,
                        ),
                    )

                    result.morphometrics = (
                        morphometrics_result
                    )

                    master = (
                        morphometrics_result
                        .master
                        .copy()
                        .reset_index(
                            drop=True
                        )
                    )

                    result.master_path = (
                        _save_master(
                            master,
                            output_dir,
                        )
                    )
                except _StagePrerequisiteUnavailable:
                    pass

        if (
            instance_refinement_result is not None
            and not getattr(instance_refinement_result, "reused", False)
            and int(getattr(instance_refinement_result, "modified_count", 0)) > 0
            and not config.run.is_enabled("morphometrics")
            and any(
                config.run.is_enabled(stage)
                for stage in (
                    "category",
                    "dim_reduction_clustering",
                    "mapping",
                    "plots",
                )
            )
        ):
            raise ValueError(
                "Instance Refinement modified canonical object identities, "
                "but Morphometrics is OFF while downstream analytical stages "
                "are ON. Enable config.run.morphometrics = True so the master "
                "table is rebuilt from the refined instance maps."
            )

        # ==============================================================
        # LOAD MASTER ONLY IF A LATER REQUEST NEEDS IT
        # ==============================================================

        later_needs_master = any(
            config.run.is_enabled(stage)
            for stage in (
                "category",
                "dim_reduction_clustering",
                "mapping",
                "plots",
            )
        )

        if (
            master is None
            and later_needs_master
        ):

            def load_master_operation():
                loaded_master, loaded_path = (
                    _load_master(
                        output_dir
                    )
                )
                return (
                    loaded_master,
                    loaded_path,
                )

            master_consumer = next(
                stage
                for stage in (
                    "category",
                    "dim_reduction_clustering",
                    "mapping",
                    "plots",
                )
                if config.run.is_enabled(stage)
            )

            (
                master,
                result.master_path,
            ) = load_input(
                master_consumer,
                "master_table",
                load_master_operation,
            )

            result.morphometrics = (
                SimpleNamespace(
                    master=master,
                    reused=True,
                )
            )

        # ==============================================================
        # 5. CATEGORY
        # ==============================================================

        if config.run.is_enabled(
            "category"
        ):

            if actions.get("category") != "SKIPPED":
                try:
                    master = operate(
                        "category",
                        "RUN",
                        lambda: apply_category(
                            master,
                            config.category,
                        ),
                    )

                    result.category_applied = True

                    result.master_path = (
                        _save_master(
                            master,
                            output_dir,
                        )
                    )
                except _StagePrerequisiteUnavailable:
                    pass

        elif (
            later_needs_master
            and _category_available(
                master,
                config,
            )
        ):
            result.category_applied = True

        # ==============================================================
        # EFFECTIVE INSTANCE MAPS REQUIRED BY DRC OR MAPPING
        # ==============================================================

        needs_instance_maps = any(
            config.run.is_enabled(stage)
            for stage in (
                "dim_reduction_clustering",
                "mapping",
                "plots",
            )
        )

        instance_source = None

        if needs_instance_maps:
            instance_consumer = (
                "dim_reduction_clustering"
                if config.run.is_enabled(
                    "dim_reduction_clustering"
                )
                else (
                    "mapping"
                    if config.run.is_enabled(
                        "mapping"
                    )
                    else "plots"
                )
            )

            if segmentation_result is None:
                if metadata_result is None:
                    metadata_result = load_input(
                        instance_consumer,
                        "metadata",
                        lambda: _load_saved_metadata(
                            config,
                            output_dir,
                        ),
                    )
                    result.metadata = metadata_result

                segmentation_result = load_input(
                    instance_consumer,
                    "segmentation",
                    lambda: _load_saved_segmentation(
                        output_dir=output_dir,
                        metadata_result=metadata_result,
                        input_mode=(
                            config.preprocessing
                            .effective_input_mode
                        ),
                    ),
                )
                result.segmentation = segmentation_result

            if instance_refinement_result is None:
                if actions.get("segmentation") == "RUN":
                    instance_refinement_result = (
                        passthrough_instance_refinement(
                            segmentation_result
                        )
                    )
                else:
                    saved_refinement = load_input(
                        instance_consumer,
                        "instance_refinement",
                        lambda: load_saved_instance_refinement(
                            segmentation_result=segmentation_result,
                            output_dir=output_dir,
                            allow_missing=True,
                        ),
                    )
                    instance_refinement_result = (
                        saved_refinement
                        if saved_refinement is not None
                        else passthrough_instance_refinement(
                            segmentation_result
                        )
                    )
                result.instance_refinement = instance_refinement_result

            instance_source = (
                instance_refinement_result
                .effective_instance_paths_by_image
            )

        # ==============================================================
        # 6. DIMENSIONALITY REDUCTION + CLUSTERING
        # ==============================================================

        if config.run.is_enabled(
            "dim_reduction_clustering"
        ):

            if actions.get("dim_reduction_clustering") != "SKIPPED":
                try:
                    analytical_result = operate(
                        "dim_reduction_clustering",
                        "RUN",
                        lambda: run_dim_reduction_clustering(
                            master=master,
                            config=config,
                            instance_dir=(
                                instance_source
                            ),
                        ),
                    )

                    master = (
                        analytical_result.master
                    )

                    feature_result = (
                        analytical_result
                        .feature_preparation
                    )

                    dimensionality_result = (
                        analytical_result
                        .dimensionality_reduction
                    )

                    clustering_result = (
                        analytical_result
                        .clustering
                    )

                    result.feature_preparation = (
                        feature_result
                    )

                    result.dimensionality_reduction = (
                        dimensionality_result
                    )

                    result.clustering = (
                        clustering_result
                    )

                    result.master_path = (
                        _save_master(
                            master,
                            output_dir,
                        )
                    )
                except _StagePrerequisiteUnavailable:
                    pass

        # ==============================================================
        # LOAD SAVED DRC INPUT ONLY IF MAPPING OR PLOTS NEEDS IT
        # ==============================================================

        needs_clustering = any(
            config.run.is_enabled(stage)
            for stage in (
                "mapping",
                "plots",
            )
        )

        if (
            clustering_result is None
            and needs_clustering
        ):

            def load_drc_operation():
                loaded_feature = (
                    _load_saved_feature_preparation(
                        master,
                        output_dir,
                    )
                )

                loaded_clustering = (
                    _load_saved_clustering(
                        master,
                        output_dir,
                        requested_number_of_morphology_states=(
                            requested_state_count
                        ),
                    )
                )

                return (
                    loaded_feature,
                    loaded_clustering,
                )

            drc_consumer = (
                "mapping"
                if config.run.is_enabled(
                    "mapping"
                )
                else "plots"
            )

            (
                feature_result,
                clustering_result,
            ) = load_input(
                drc_consumer,
                "dimensionality reduction + clustering",
                load_drc_operation,
            )

            result.feature_preparation = (
                feature_result
            )
            result.clustering = (
                clustering_result
            )

        if clustering_result is not None:
            config.clustering.automatic_number_of_morphology_states = int(
                clustering_result.automatic_k
            )
            config.clustering.number_of_morphology_states_used_downstream = int(
                clustering_result.preferred_k
            )
            config.clustering.morphology_state_selection_source = str(
                clustering_result.selection_source
            )

        # ==============================================================
        # 7. MAPPING
        # ==============================================================

        if config.run.is_enabled(
            "mapping"
        ):

            if actions.get("mapping") != "SKIPPED":
                try:
                    category_fields = (
                        tuple(
                            config.category
                            .category_fields
                        )
                        if _category_available(
                            master,
                            config,
                        )
                        else ()
                    )

                    mapping_result = operate(
                        "mapping",
                        "RUN",
                        lambda: (
                            MappingStage(
                                config.mapping
                            )
                            .run(
                                master=master,
                                clustering=(
                                    clustering_result
                                ),
                                instance_dir=(
                                    instance_source
                                ),
                                output_dir=(
                                    output_dir
                                    / "Mapping"
                                ),
                                label_columns=(
                                    category_fields
                                ),
                                resume=(
                                    config.resume
                                ),
                            )
                        ),
                    )

                    result.mapping = (
                        mapping_result
                    )
                except _StagePrerequisiteUnavailable:
                    pass

        # ==============================================================
        # LOAD SAVED MAPPING INPUT ONLY IF PLOTS NEEDS IT
        # ==============================================================

        if config.run.is_enabled(
            "plots"
        ):

            if actions.get("plots") != "SKIPPED":
                try:
                    plots_mapping_result = load_input(
                        "plots",
                        "DRC prototype QC for the effective morphology-state count",
                        lambda: _load_drc_prototype_mapping(
                            output_dir=output_dir,
                            clustering_result=clustering_result,
                            master=master,
                            instance_dir=instance_source,
                            mapping_config=config.mapping,
                        ),
                    )
                except _StagePrerequisiteUnavailable:
                    pass

        # ==============================================================
        # 8. SPATIAL ANALYSIS
        #
        # Reserved. False remains OFF.
        # ==============================================================

        # ==============================================================
        # 9. PLOTS
        # ==============================================================

        if config.run.is_enabled(
            "plots"
        ):

            if actions.get("plots") != "SKIPPED":
                try:
                    category_fields = (
                        tuple(
                            config.category
                            .category_fields
                        )
                        if _category_available(
                            master,
                            config,
                        )
                        else ()
                    )

                    if (
                        category_fields
                        and feature_result is None
                    ):
                        raise RuntimeError(
                            "Plots requires the saved Feature Preparation result when "
                            "Category is available. Enable "
                            "config.run.dim_reduction_clustering = True or restore "
                            "its canonical Feature_Preparation outputs."
                        )

                    plots_result = operate(
                        "plots",
                        "RUN",
                        lambda: (
                            PlotsStage(
                                config.plots
                            )
                            .run(
                                master=master,
                                clustering=(
                                    clustering_result
                                ),
                                mapping=(
                                    plots_mapping_result
                                ),
                                output_dir=(
                                    output_dir
                                    / plot_output_directory_name(
                                        config.category.category_fields
                                    )
                                ),
                                category_fields=(
                                    category_fields
                                ),
                                analytical_features=(
                                    feature_result
                                    .analytical_features
                                    if feature_result
                                    is not None
                                    else ()
                                ),
                                instance_source=(
                                    instance_source
                                ),
                            )
                        ),
                    )

                    result.plots = (
                        plots_result
                    )
                except _StagePrerequisiteUnavailable:
                    pass

        # ==============================================================
        # FINAL RESULT
        # ==============================================================

        if master is not None:
            result.master = master

            if result.master_path is None:
                existing_master_path = (
                    output_dir
                    / "Data"
                    / "master_table.csv"
                )

                if existing_master_path.is_file():
                    result.master_path = (
                        existing_master_path
                    )

        _save_execution_audit(
            config=config,
            output_dir=output_dir,
            actions=actions,
            seconds=seconds,
        )

    except _StagePrerequisiteUnavailable as exc:

        if exc.consumer_stage in PIPELINE_STAGES:

            start_index = PIPELINE_STAGES.index(
                exc.consumer_stage
            )

            for downstream_stage in PIPELINE_STAGES[
                start_index:
            ]:

                if (
                    config.run.is_enabled(
                        downstream_stage
                    )
                    and actions.get(
                        downstream_stage
                    ) == "OFF"
                ):

                    actions[
                        downstream_stage
                    ] = "SKIPPED"

        if master is not None:

            result.master = master

            if result.master_path is None:

                existing_master_path = (
                    output_dir
                    / "Data"
                    / "master_table.csv"
                )

                if existing_master_path.is_file():

                    result.master_path = (
                        existing_master_path
                    )

        _save_execution_audit(
            config=config,
            output_dir=output_dir,
            actions=actions,
            seconds=seconds,
        )

    except Exception as exc:

        if current_stage in PIPELINE_STAGES:
            if actions.get(
                current_stage
            ) != "ERROR":
                actions[
                    current_stage
                ] = "ERROR"

                log_error(
                    output_dir=output_dir,
                    stage=current_stage,
                    operation="PIPELINE",
                    message=(
                        f"Pipeline failed while resolving stage "
                        f"{current_stage!r}."
                    ),
                    exception=exc,
                )

        _save_execution_audit(
            config=config,
            output_dir=output_dir,
            actions=actions,
            seconds=seconds,
        )

        raise

    # ==================================================================
    # COMPLETE
    # ==================================================================

    _section(
        "PIPELINE COMPLETE"
    )

    _print_execution_plan(
        config,
        actions,
    )

    if result.master is not None:

        print(
            "Cells:",
            len(
                result.master
            ),
        )

        print(
            "Master:",
            result.master_path,
        )

    if result.clustering is not None:

        print(
            "Robust K:",
            getattr(
                result.clustering,
                "robust_k_values",
                None,
            ),
        )

        print(
            "Pareto-optimal K:",
            getattr(
                result.clustering,
                "pareto_k_values",
                None,
            ),
        )

        print(
            "Coarse K:",
            getattr(
                result.clustering,
                "coarse_k",
                None,
            ),
        )

        print(
            "Preferred K:",
            getattr(
                result.clustering,
                "preferred_k",
                None,
            ),
        )

    if result.plots is not None:

        print(
            "Plot files:",
            len(
                result.plots.files
            ),
        )

    print()

    return result


# ======================================================================
# CLASS COMPATIBILITY WRAPPER
# ======================================================================

class MorphogliaPipeline:
    """
    Thin compatibility wrapper around run_pipeline().

    Existing code using:

        MorphogliaPipeline(config).run()

    continues to work.

    New code should preferably use:

        run_pipeline(config)
    """

    def __init__(
        self,
        config: PipelineConfig,
    ):

        if not isinstance(
            config,
            PipelineConfig,
        ):

            raise TypeError(
                "MorphogliaPipeline expects a PipelineConfig."
            )


        self.config = config


    def run(
        self,
    ) -> PipelineResult:

        return run_pipeline(
            self.config
        )


__all__ = [
    "PipelineResult",
    "run_pipeline",
    "MorphogliaPipeline",
]



# MG_MORPHOLOGY_STATE_SAVED_SELECTION_V2
def _load_saved_clustering(
    master: pd.DataFrame,
    output_dir: Path,
    requested_number_of_morphology_states=None,
):
    """Load cached analytical results and apply one shared state-count choice."""

    result = _mg_load_saved_clustering_legacy(master, output_dir)
    decision = result.resolution_decision.copy()

    if "supported_resolution" in decision.columns:
        supported = tuple(
            sorted(
                int(k)
                for k in decision.loc[
                    _truthy_series(decision["supported_resolution"]), "k"
                ].tolist()
            )
        )
    else:
        supported = tuple(sorted(int(k) for k in result.selected_solutions))

    if not supported:
        raise ValueError(
            "Saved analytical results contain no reproducible number of "
            "morphology states. Rerun Dimensionality Reduction and Clustering."
        )

    automatic_rows = pd.DataFrame()
    if "automatic_selection" in decision.columns:
        automatic_rows = decision[_truthy_series(decision["automatic_selection"])]
    elif "default_resolution" in decision.columns:
        # Compatibility with results generated before the terminology update.
        automatic_rows = decision[_truthy_series(decision["default_resolution"])]
    elif "decision_role" in decision.columns:
        automatic_rows = decision[decision["decision_role"].astype(str).eq("preferred")]

    if len(automatic_rows) != 1:
        raise ValueError(
            "Saved analytical results do not identify exactly one automatic "
            "number of morphology states. Rerun Dimensionality Reduction and Clustering."
        )
    automatic_count = int(automatic_rows.iloc[0]["k"])

    stable_bands = []
    if "band_id" in decision.columns and "supported_resolution" in decision.columns:
        band_rows = decision[_truthy_series(decision["supported_resolution"])].copy()
        band_rows = band_rows[band_rows["band_id"].notna()]
        for _, group in band_rows.groupby("band_id", sort=True):
            stable_bands.append(tuple(sorted(int(k) for k in group["k"].tolist())))
    if not stable_bands:
        stable_bands = [tuple(supported)]

    if "band_representative" in decision.columns:
        representatives = tuple(
            int(k)
            for k in decision.loc[
                _truthy_series(decision["band_representative"]), "k"
            ].tolist()
        )
    else:
        representatives = (automatic_count,)

    requested = (
        None
        if requested_number_of_morphology_states is None
        else int(requested_number_of_morphology_states)
    )

    requested_is_available = (
        requested is None
        or requested in supported
    )

    if requested is not None and not requested_is_available:
        print()
        print(
            "Requested morphology-state count "
            f"K={requested} is not available in the saved clustering results."
        )
        print(
            "Using automatic selection "
            f"K={automatic_count}."
        )

    if requested is None or not requested_is_available:
        effective_count = automatic_count
        selection_source = "automatic_data_driven"
    else:
        effective_count = requested
        selection_source = "researcher_selection"

    result.resolution_decision = decision
    result.robust_k_values = supported
    result.supported_k_values = supported
    result.stable_bands = tuple(stable_bands)
    result.band_representatives = representatives
    result.automatic_k = automatic_count
    result.effective_k = effective_count
    result.requested_k = requested
    result.selection_source = selection_source
    result.automatic_selection_rule = (
        "first_stable_band_then_cross_dimension_agreement_then_"
        "worst_case_and_mean_resampling_stability_then_membership_reliability"
    )
    # Internal compatibility attributes used by existing stages.
    result.default_k = automatic_count
    result.preferred_k = effective_count
    result.coarse_k = automatic_count

    result.resolution_decision["effective_selection"] = (
        result.resolution_decision["k"].astype(int).eq(effective_count)
    )
    result.resolution_decision["selection_source"] = selection_source

    print(
        "Stable/reproducible numbers of morphology states (saved evidence):",
        ", ".join(str(k) for k in supported),
    )
    print("Automatic selection criterion (data-driven):")
    print("  first/coarsest stable band; within it, highest cross-dimensional")
    print("  agreement, then worst-case/mean resampling stability and membership.")
    dimension_floor = (
        float(decision["support_dimension_floor"].dropna().iloc[0])
        if "support_dimension_floor" in decision.columns
        and not decision["support_dimension_floor"].dropna().empty
        else 0.25
    )
    isolated_floor = (
        float(decision["isolated_dimension_floor"].dropna().iloc[0])
        if "isolated_dimension_floor" in decision.columns
        and not decision["isolated_dimension_floor"].dropna().empty
        else 0.30
    )
    print("Reproducibility gates:")
    print(
        "  mean/worst resampling ARI >= 0.75/0.60; membership >= 0.70; "
        f"cross-dimensional agreement >= {dimension_floor:.3f}; state size >= 5"
    )
    print(
        "  isolated peaks: local non-dominance; mean/worst ARI >= 0.85/0.70; "
        f"membership >= 0.75; agreement >= {isolated_floor:.3f}"
    )
    print("Automatic number of morphology states:", automatic_count)
    print(
        "Researcher selection:",
        "not supplied" if requested is None else f"{requested} states",
    )
    print(
        "Number of morphology states used downstream:",
        f"{effective_count} ({'automatic data-driven selection' if requested is None else 'researcher selection'})",
    )
    print()

    return result


def _load_drc_prototype_mapping(
    output_dir: Path,
    clustering_result,
    master: pd.DataFrame,
    instance_dir,
    mapping_config,
):
    """Load or cheaply materialize prototype QC for the effective state count."""

    supported = tuple(int(k) for k in clustering_result.supported_k_values)
    state_count = int(clustering_result.preferred_k)
    if state_count not in supported:
        raise ValueError(
            f"{state_count} morphology states are unavailable. Supported counts: "
            + ", ".join(str(value) for value in supported)
        )

    dimensions = int(
        clustering_result.selected_solutions[state_count].pca_dimensions
    )
    drc_root = output_dir / "Dimensionality_Reduction_Clustering"
    prototype_root = drc_root / "QC" / "Prototype_Cells"
    solution_dir = prototype_root / f"K{state_count}_d{dimensions}"
    table_path = solution_dir / "prototype_cells.csv"
    grid_path = solution_dir / "Prototype_Cells.png"
    masks_dir = solution_dir / "Prototype_Masks"
    canonical_masks = (
        [path for path in masks_dir.glob("cluster_*.png") if "_rank_" not in path.name]
        if masks_dir.is_dir()
        else []
    )

    generated = False
    if (
        not table_path.is_file()
        or not grid_path.is_file()
        or len(canonical_masks) < state_count
    ):
        if instance_dir is None:
            instance_dir = _canonical_instance_dir(output_dir)
        from .dim_reduction_clustering.stage import _save_prototype_qc

        _save_prototype_qc(
            master=master,
            clustering=clustering_result,
            instance_dir=instance_dir,
            output_dir=drc_root,
            mapping_config=mapping_config,
            k_values=(state_count,),
        )
        generated = True

    table = _read_csv_required(
        table_path,
        label=f"prototype cells for {state_count} morphology states",
    )
    if "selection_rank" not in table.columns:
        raise ValueError(f"Prototype table has no selection_rank column: {table_path}")
    canonical = table[table["selection_rank"].astype(int).eq(1)].copy()
    if len(canonical) != state_count:
        raise ValueError(
            f"Expected {state_count} rank-1 prototypes, found {len(canonical)}: "
            f"{table_path}"
        )

    return SimpleNamespace(
        selections={state_count: {"prototype": canonical.reset_index(drop=True)}},
        preferred_k=state_count,
        output_dir=prototype_root,
        reused=not generated,
        source="Dimensionality_Reduction_Clustering/QC",
    )
