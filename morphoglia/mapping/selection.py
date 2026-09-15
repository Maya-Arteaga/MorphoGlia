from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ======================================================================
# DEFAULT COLUMN NAMES
# ======================================================================

DEFAULT_CLUSTER_COLUMN = "cluster"
DEFAULT_PROBABILITY_COLUMN = "cluster_probability"
DEFAULT_RELIABILITY_COLUMN = "consensus_reliability"


# ======================================================================
# HELPERS
# ======================================================================

def _pca_columns(
    pca_dimensions: int,
) -> list[str]:
    """
    Return the analytical PCA columns PC1 ... PCd.
    """

    if pca_dimensions < 1:
        raise ValueError(
            "pca_dimensions must be >= 1."
        )

    return [
        f"PC{i}"
        for i in range(
            1,
            pca_dimensions + 1,
        )
    ]


def _require_columns(
    data: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    """
    Validate required dataframe columns.
    """

    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )


def _attach_selection_metadata(
    data: pd.DataFrame,
    selection_type: str,
) -> pd.DataFrame:
    """
    Add canonical Mapping selection metadata.
    """

    output = data.copy()

    output[
        "selection_type"
    ] = selection_type

    output[
        "selection_rank"
    ] = (
        output.groupby(
            DEFAULT_CLUSTER_COLUMN,
            sort=True,
        )
        .cumcount()
        + 1
    )

    return output


# ======================================================================
# PROTOTYPE
# ======================================================================

def select_cluster_prototypes(
    data: pd.DataFrame,
    pca_dimensions: int,
    cluster_column: str = DEFAULT_CLUSTER_COLUMN,
    reliability_column: str = DEFAULT_RELIABILITY_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
    n_per_cluster: int = 1,
) -> pd.DataFrame:
    """
    Select empirical prototype cells for each cluster.

    Definition
    ----------
    For each cluster:

    1. Use the analytical PCA space PC1 ... PCd.
    2. Calculate the coordinate-wise median of the cluster.
    3. Calculate every observed cell's Euclidean distance to that
       robust center.
    4. Rank actual observed cells by distance to that center.
    5. Return the nearest n_per_cluster observed cells.

    All returned cells are empirical prototype cells.

    selection_rank = 1
        Canonical prototype: the observed cell nearest the robust
        cluster center.

    selection_rank = 2 ... N
        Additional empirical prototype cells ordered by increasing
        distance from the same robust center.

    Reliability and cluster probability are used only as deterministic
    tie-breakers.

    No synthetic centroid is returned.

    UMAP is never used for prototype selection.
    """

    n_per_cluster = int(
        n_per_cluster
    )

    if n_per_cluster < 1:
        raise ValueError(
            "n_per_cluster must be >= 1."
        )


    pca_columns = _pca_columns(
        pca_dimensions
    )


    _require_columns(
        data,
        [
            cluster_column,
            reliability_column,
            probability_column,
            *pca_columns,
        ],
    )


    selected = []


    for cluster in sorted(
        data[
            cluster_column
        ]
        .dropna()
        .unique()
    ):

        cluster_data = (
            data[
                data[
                    cluster_column
                ]
                == cluster
            ]
            .copy()
        )


        X = (
            cluster_data[
                pca_columns
            ]
            .to_numpy(
                dtype=float
            )
        )


        if not np.isfinite(
            X
        ).all():

            raise ValueError(
                f"Cluster {cluster} contains "
                "non-finite PCA values."
            )


        # --------------------------------------------------------------
        # ROBUST CLUSTER CENTER
        # --------------------------------------------------------------

        center = np.median(
            X,
            axis=0,
        )


        # --------------------------------------------------------------
        # DISTANCE TO ROBUST CENTER
        # --------------------------------------------------------------

        cluster_data[
            "prototype_distance"
        ] = np.linalg.norm(
            X - center,
            axis=1,
        )


        # --------------------------------------------------------------
        # EMPIRICAL PROTOTYPE CELLS
        #
        # Distance defines centrality.
        #
        # Reliability and probability are deterministic tie-breakers.
        # --------------------------------------------------------------

        prototypes = (
            cluster_data
            .sort_values(
                [
                    "prototype_distance",
                    reliability_column,
                    probability_column,
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
                na_position="last",
                kind="mergesort",
            )
            .head(
                n_per_cluster
            )
            .copy()
        )


        prototypes[
            "selection_rank"
        ] = np.arange(
            1,
            len(
                prototypes
            ) + 1,
            dtype=int,
        )


        prototypes[
            "selection_type"
        ] = "prototype"


        prototypes[
            "prototype_center_method"
        ] = "coordinate_median"


        prototypes[
            "prototype_pca_dimensions"
        ] = int(
            pca_dimensions
        )


        selected.append(
            prototypes
        )


    if not selected:

        return pd.DataFrame()


    return pd.concat(
        selected,
        ignore_index=True,
    )


# ======================================================================
# HIGH-CONFIDENCE EXEMPLARS
# ======================================================================

def select_high_confidence_examples(
    data: pd.DataFrame,
    n_per_cluster: int = 4,
    cluster_column: str = DEFAULT_CLUSTER_COLUMN,
    reliability_column: str = DEFAULT_RELIABILITY_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> pd.DataFrame:
    """
    Select the most confidently assigned observed cells per cluster.

    Ranking:
        1. consensus reliability descending
        2. cluster probability descending

    These are exemplars, NOT cluster prototypes.
    """

    if n_per_cluster < 1:
        raise ValueError(
            "n_per_cluster must be >= 1."
        )


    _require_columns(
        data,
        [
            cluster_column,
            reliability_column,
            probability_column,
        ],
    )


    selected = []


    for cluster in sorted(
        data[
            cluster_column
        ].dropna().unique()
    ):

        cluster_data = (
            data[
                data[
                    cluster_column
                ]
                == cluster
            ]
            .sort_values(
                [
                    reliability_column,
                    probability_column,
                ],
                ascending=[
                    False,
                    False,
                ],
                na_position="last",
                kind="mergesort",
            )
            .head(
                n_per_cluster
            )
            .copy()
        )


        selected.append(
            cluster_data
        )


    if not selected:

        return pd.DataFrame()


    output = pd.concat(
        selected,
        ignore_index=True,
    )


    output[
        "selection_type"
    ] = "high_confidence"


    output[
        "selection_rank"
    ] = (
        output.groupby(
            cluster_column,
            sort=True,
        )
        .cumcount()
        + 1
    )


    return output


# ======================================================================
# BOUNDARY EXEMPLARS
# ======================================================================

def select_boundary_examples(
    data: pd.DataFrame,
    n_per_cluster: int = 4,
    cluster_column: str = DEFAULT_CLUSTER_COLUMN,
    reliability_column: str = DEFAULT_RELIABILITY_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> pd.DataFrame:
    """
    Select the least reliable / most ambiguous observed cells.

    Ranking:
        1. consensus reliability ascending
        2. cluster probability ascending

    These cells characterize the uncertain boundary of the
    coarse-graining.
    """

    if n_per_cluster < 1:
        raise ValueError(
            "n_per_cluster must be >= 1."
        )


    _require_columns(
        data,
        [
            cluster_column,
            reliability_column,
            probability_column,
        ],
    )


    selected = []


    for cluster in sorted(
        data[
            cluster_column
        ].dropna().unique()
    ):

        cluster_data = (
            data[
                data[
                    cluster_column
                ]
                == cluster
            ]
            .sort_values(
                [
                    reliability_column,
                    probability_column,
                ],
                ascending=[
                    True,
                    True,
                ],
                na_position="last",
                kind="mergesort",
            )
            .head(
                n_per_cluster
            )
            .copy()
        )


        selected.append(
            cluster_data
        )


    if not selected:

        return pd.DataFrame()


    output = pd.concat(
        selected,
        ignore_index=True,
    )


    output[
        "selection_type"
    ] = "boundary"


    output[
        "selection_rank"
    ] = (
        output.groupby(
            cluster_column,
            sort=True,
        )
        .cumcount()
        + 1
    )


    return output