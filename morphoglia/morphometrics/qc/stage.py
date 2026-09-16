from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ObjectQCConfig
from .grid import (
    plot_qc_grid,
    select_qc_examples,
)
from .size import score_size_outliers
from .tubular import score_tubular_objects


# ======================================================================
# RESULT
# ======================================================================

@dataclass
class ObjectQCResult:
    """
    Result of object-level QC.

    table
        Complete object-level QC table.

        The transient ROI column is retained in memory so downstream
        code may reuse exact component masks, but it is not written to
        CSV.

    included_keys
        Set of:

            (image_id, cell_id)

        identifying objects accepted for downstream morphometrics.

    csv_path
        Saved object_qc.csv.

    *_grid_path
        QC grid path when suspicious objects were detected.

        None when no objects of that class were detected.
    """

    table: pd.DataFrame

    included_keys: set[
        tuple[str, str]
    ]

    detected_count: int
    included_count: int

    small_count: int
    large_count: int

    tubular_candidate_count: int
    tubular_count: int

    csv_path: Path

    small_grid_path: Path | None
    large_grid_path: Path | None
    tubular_grid_path: Path | None


# ======================================================================
# HELPERS
# ======================================================================

def _validate_object_table(
    table: pd.DataFrame,
) -> None:

    required = {
        "image_id",
        "cell_id",
        "component_area",
        "roi",
    }


    missing = (
        required
        - set(
            table.columns
        )
    )


    if missing:

        raise ValueError(
            "Object QC requires columns: "
            f"{sorted(required)}. "
            f"Missing: {sorted(missing)}"
        )


    if table.empty:

        raise ValueError(
            "Object QC received an empty object table."
        )


    key_columns = [
        "image_id",
        "cell_id",
    ]


    if table.duplicated(
        key_columns
    ).any():

        duplicates = (
            table.loc[
                table.duplicated(
                    key_columns,
                    keep=False,
                ),
                key_columns,
            ]
            .head(
                10
            )
            .to_dict(
                orient="records"
            )
        )


        raise ValueError(
            "Object identities are not unique. "
            f"Examples: {duplicates}"
        )


def _qc_reason(
    row: pd.Series,
) -> str:

    reasons = []


    if bool(
        row[
            "suspicious_small"
        ]
    ):

        reasons.append(
            "small"
        )


    if bool(
        row[
            "suspicious_large"
        ]
    ):

        reasons.append(
            "large"
        )


    if bool(
        row[
            "suspicious_tubular"
        ]
    ):

        reasons.append(
            "tubular"
        )


    return ";".join(
        reasons
    )


# ======================================================================
# STAGE
# ======================================================================

def run_object_qc(
    object_table: pd.DataFrame,
    output_dir: str | Path,
    config: ObjectQCConfig,
) -> ObjectQCResult:
    """
    Run object-level QC.

    Sequence
    --------

    detected connected objects
        ↓
    robust log-area QC
        ├── suspicious small
        └── suspicious large
        ↓
    tubular candidate selection
        based on robust size-z
        ↓
    tubular geometry
        elongation
        thickness uniformity
        body prominence
        compact-body rescue
        ↓
    suspicious tubular objects
        ↓
    True / False filtering decisions
        ↓
    accepted object population

    Notes
    -----
    QC detection and QC exclusion are deliberately separate.

    For example:

        config.tubular = False

    still calculates tubular scores, flags suspicious tubular objects,
    saves them to object_qc.csv, and generates a QC grid when needed.

    It simply does not exclude them from the downstream population.
    """

    _validate_object_table(
        object_table
    )


    output_dir = Path(
        output_dir
    )


    qc_dir = (
        output_dir
        / "_QC_Segmentation"
    )


    qc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    csv_path = (
        qc_dir
        / "object_qc.csv"
    )


    small_grid_path = (
        qc_dir
        / "Small_Objects.png"
    )


    large_grid_path = (
        qc_dir
        / "Large_Objects.png"
    )


    tubular_grid_path = (
        qc_dir
        / "Tubular_Objects.png"
    )


    table = (
        object_table
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ==================================================================
    # SIZE QC
    # ==================================================================

    size_result = score_size_outliers(
        table[
            "component_area"
        ].to_numpy(),
        z_threshold=(
            config
            .size_z_threshold
        ),
    )


    for column in (
        "log_component_area",
        "size_robust_z",
        "suspicious_small",
        "suspicious_large",
    ):

        table[
            column
        ] = (
            size_result
            .table[
                column
            ]
            .to_numpy()
        )


    # ==================================================================
    # SMALL GRID
    # ==================================================================

    small_selected = (
        select_qc_examples(
            table,
            flag_column=(
                "suspicious_small"
            ),
            score_column=(
                "size_robust_z"
            ),
            grid_n=(
                config
                .grid_n
            ),
            extreme="low",
        )
    )


    small_grid = plot_qc_grid(
        small_selected,
        small_grid_path,
        grid_n=(
            config
            .grid_n
        ),
        title=(
            "Suspicious Small Objects"
        ),
        score_column=(
            "size_robust_z"
        ),
    )


    # ==================================================================
    # LARGE GRID
    # ==================================================================

    large_selected = (
        select_qc_examples(
            table,
            flag_column=(
                "suspicious_large"
            ),
            score_column=(
                "size_robust_z"
            ),
            grid_n=(
                config
                .grid_n
            ),
            extreme="high",
        )
    )


    large_grid = plot_qc_grid(
        large_selected,
        large_grid_path,
        grid_n=(
            config
            .grid_n
        ),
        title=(
            "Suspicious Large Objects"
        ),
        score_column=(
            "size_robust_z"
        ),
    )


    # ==================================================================
    # TUBULAR QC
    #
    # Statistical small-object outliers do not define the tubular
    # reference population.
    #
    # This is independent from whether config.small_objects is True.
    #
    # A fragment may therefore be retained by user choice while still
    # being prevented from distorting tubular ranks.
    # ==================================================================

    tubular_eligible = ~(
        table[
            "suspicious_small"
        ].to_numpy()
    )


    tubular_result = (
        score_tubular_objects(
            table,
            candidate_z_threshold=(
                config
                .tubular_candidate_z
            ),
            score_threshold=(
                config
                .tubular_score_threshold
            ),
            eligible_mask=(
                tubular_eligible
            ),
        )
    )


    tubular_columns = [
        "tubular_eligible",
        "tubular_candidate",
        "radius_median",
        "radius_max",
        "thickness_variation",
        "body_prominence",
        "elongation",
        "circularity",
        "solidity",
        "tube_score",
        "compact_body_rescue",
        "suspicious_tubular",
    ]


    for column in tubular_columns:

        table[
            column
        ] = (
            tubular_result
            .table[
                column
            ]
            .to_numpy()
        )


    # ==================================================================
    # TUBULAR GRID
    # ==================================================================

    tubular_selected = (
        select_qc_examples(
            table,
            flag_column=(
                "suspicious_tubular"
            ),
            score_column=(
                "tube_score"
            ),
            grid_n=(
                config
                .grid_n
            ),
            extreme="high",
        )
    )


    tubular_grid = plot_qc_grid(
        tubular_selected,
        tubular_grid_path,
        grid_n=(
            config
            .grid_n
        ),
        title=(
            "Suspicious Tubular Objects"
        ),
        score_column=(
            "tube_score"
        ),
    )


    # ==================================================================
    # TRUE / FALSE FILTER DECISIONS
    # ==================================================================

    table[
        "excluded_small"
    ] = (
        table[
            "suspicious_small"
        ]
        & bool(
            config
            .small_objects
        )
    )


    table[
        "excluded_large"
    ] = (
        table[
            "suspicious_large"
        ]
        & bool(
            config
            .large_objects
        )
    )


    table[
        "excluded_tubular"
    ] = (
        table[
            "suspicious_tubular"
        ]
        & bool(
            config
            .tubular
        )
    )


    table[
        "included"
    ] = ~(
        table[
            "excluded_small"
        ]
        |
        table[
            "excluded_large"
        ]
        |
        table[
            "excluded_tubular"
        ]
    )


    # ==================================================================
    # AUDIT REASON
    # ==================================================================

    table[
        "qc_reasons"
    ] = table.apply(
        _qc_reason,
        axis=1,
    )


    # ==================================================================
    # ACCEPTED OBJECT IDENTITIES
    # ==================================================================

    included = table.loc[
        table[
            "included"
        ],
        [
            "image_id",
            "cell_id",
        ],
    ]


    included_keys = set(
        zip(
            included[
                "image_id"
            ].astype(str),
            included[
                "cell_id"
            ].astype(str),
        )
    )


    # ==================================================================
    # SAVE AUDIT TABLE
    #
    # ROI arrays are transient and are never serialized into CSV.
    # ==================================================================

    table.drop(
        columns=[
            "roi",
        ],
    ).to_csv(
        csv_path,
        index=False,
    )


    # ==================================================================
    # RESULT
    # ==================================================================

    return ObjectQCResult(
        table=table,

        included_keys=(
            included_keys
        ),

        detected_count=(
            len(table)
        ),

        included_count=int(
            table[
                "included"
            ].sum()
        ),

        small_count=int(
            table[
                "suspicious_small"
            ].sum()
        ),

        large_count=int(
            table[
                "suspicious_large"
            ].sum()
        ),

        tubular_candidate_count=(
            tubular_result
            .candidate_count
        ),

        tubular_count=int(
            table[
                "suspicious_tubular"
            ].sum()
        ),

        csv_path=(
            csv_path
        ),

        small_grid_path=(
            small_grid
        ),

        large_grid_path=(
            large_grid
        ),

        tubular_grid_path=(
            tubular_grid
        ),
    )
