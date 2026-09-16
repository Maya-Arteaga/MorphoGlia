from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ======================================================================
# RESULT
# ======================================================================

@dataclass
class ResolutionDecisionResult:
    """
    Decision layer for supported clustering resolutions.

    robust_k_values
        Resolutions supported by the clustering robustness machinery.

    pareto_k_values
        Robust resolutions that are not dominated by another robust K.

    maximin_k
        Resolution selected by the historical maximin/worst-case rule.

    preferred_k
        Preferred decision resolution after accounting explicitly for
        uncertainty in PCA dimensionality.

    coarse_k
        Nearest simpler Pareto-supported resolution below preferred_k.

    metrics
        Publication-ready resolution decision table.

    transitions
        Cell-level overlap between consecutive robust resolutions.

    nesting
        Pair-level population-identity continuity metrics.
    """

    robust_k_values: tuple[int, ...]
    pareto_k_values: tuple[int, ...]
    maximin_k: int | None
    preferred_k: int | None
    coarse_k: int | None

    metrics: pd.DataFrame
    transitions: pd.DataFrame
    nesting: pd.DataFrame


# ======================================================================
# PARETO DOMINANCE
# ======================================================================

_MAXIMIZE_METRICS = (
    "mean_subsample_ari",
    "minimum_subsample_ari",
    "mean_dimension_agreement",
    "minimum_dimension_agreement",
    "mean_membership_probability",
    "minimum_cluster_size_across_d",
)

_MINIMIZE_METRICS = (
    "maximum_singletons_across_d",
)


def _dominates(
    candidate: pd.Series,
    target: pd.Series,
) -> bool:
    """
    Return True when candidate is no worse on every available criterion
    and strictly better on at least one.

    Cluster size is used here only as a robustness/sanity criterion.
    It is NOT combined into an arbitrary weighted score.
    """

    at_least_one_better = False


    for metric in _MAXIMIZE_METRICS:

        if (
            metric not in candidate.index
            or metric not in target.index
        ):
            continue


        candidate_value = float(
            candidate[
                metric
            ]
        )

        target_value = float(
            target[
                metric
            ]
        )


        if not (
            np.isfinite(candidate_value)
            and np.isfinite(target_value)
        ):
            continue


        if candidate_value < target_value:

            return False


        if candidate_value > target_value:

            at_least_one_better = True


    for metric in _MINIMIZE_METRICS:

        if (
            metric not in candidate.index
            or metric not in target.index
        ):
            continue


        candidate_value = float(
            candidate[
                metric
            ]
        )

        target_value = float(
            target[
                metric
            ]
        )


        if not (
            np.isfinite(candidate_value)
            and np.isfinite(target_value)
        ):
            continue


        if candidate_value > target_value:

            return False


        if candidate_value < target_value:

            at_least_one_better = True


    return at_least_one_better


# ======================================================================
# TRANSITION / NESTING ANALYSIS
# ======================================================================

def _transition_analysis(
    clustering,
    robust_k_values: tuple[int, ...],
    display_maps: dict[
        int,
        dict[int, int],
    ] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    transition_rows = []
    nesting_rows = []


    for (
        coarse_k,
        fine_k,
    ) in zip(
        robust_k_values[:-1],
        robust_k_values[1:],
    ):

        coarse_solution = (
            clustering
            .selected_solutions[
                coarse_k
            ]
        )


        fine_solution = (
            clustering
            .selected_solutions[
                fine_k
            ]
        )


        coarse_labels = np.asarray(
            coarse_solution.labels,
            dtype=int,
        )


        fine_labels = np.asarray(
            fine_solution.labels,
            dtype=int,
        )


        if len(
            coarse_labels
        ) != len(
            fine_labels
        ):

            raise ValueError(
                f"K={coarse_k} and K={fine_k} "
                "contain different numbers of cells."
            )


        table = pd.crosstab(
            coarse_labels,
            fine_labels,
        )


        coarse_totals = (
            table.sum(
                axis=1
            )
        )


        fine_totals = (
            table.sum(
                axis=0
            )
        )


        coarse_map = (
            {}
            if display_maps is None
            else display_maps.get(
                coarse_k,
                {},
            )
        )


        fine_map = (
            {}
            if display_maps is None
            else display_maps.get(
                fine_k,
                {},
            )
        )


        # --------------------------------------------------------------
        # TIDY TRANSITION TABLE
        # --------------------------------------------------------------

        for coarse_cluster in (
            table.index
        ):

            for fine_cluster in (
                table.columns
            ):

                count = int(
                    table.loc[
                        coarse_cluster,
                        fine_cluster,
                    ]
                )


                if count == 0:

                    continue


                coarse_fraction = float(
                    count
                    / coarse_totals.loc[
                        coarse_cluster
                    ]
                )


                fine_parent_fraction = float(
                    count
                    / fine_totals.loc[
                        fine_cluster
                    ]
                )


                transition_rows.append(
                    {
                        "coarse_k":
                            int(
                                coarse_k
                            ),

                        "fine_k":
                            int(
                                fine_k
                            ),

                        "coarse_raw_cluster":
                            int(
                                coarse_cluster
                            ),

                        "fine_raw_cluster":
                            int(
                                fine_cluster
                            ),

                        "coarse_display_cluster":
                            int(
                                coarse_map.get(
                                    int(
                                        coarse_cluster
                                    ),
                                    int(
                                        coarse_cluster
                                    ),
                                )
                            ),

                        "fine_display_cluster":
                            int(
                                fine_map.get(
                                    int(
                                        fine_cluster
                                    ),
                                    int(
                                        fine_cluster
                                    ),
                                )
                            ),

                        "count":
                            count,

                        "coarse_fraction":
                            coarse_fraction,

                        "fine_parent_fraction":
                            fine_parent_fraction,
                    }
                )


        # --------------------------------------------------------------
        # FINE → COARSE PARENT PURITY
        #
        # If every fine state comes almost entirely from one coarse
        # population, the finer partition is hierarchically nested.
        # --------------------------------------------------------------

        parent_purities = []


        for fine_cluster in (
            table.columns
        ):

            fine_size = int(
                fine_totals.loc[
                    fine_cluster
                ]
            )


            dominant_parent_count = int(
                table[
                    fine_cluster
                ].max()
            )


            purity = float(
                dominant_parent_count
                / fine_size
            )


            parent_purities.append(
                (
                    fine_size,
                    purity,
                )
            )


        weighted_parent_purity = float(
            sum(
                size * purity
                for (
                    size,
                    purity,
                ) in parent_purities
            )
            / sum(
                size
                for (
                    size,
                    _,
                ) in parent_purities
            )
        )


        mean_parent_purity = float(
            np.mean(
                [
                    purity
                    for (
                        _,
                        purity,
                    ) in parent_purities
                ]
            )
        )


        minimum_parent_purity = float(
            np.min(
                [
                    purity
                    for (
                        _,
                        purity,
                    ) in parent_purities
                ]
            )
        )


        # --------------------------------------------------------------
        # COARSE → FINE SPLIT ENTROPY
        #
        # 0
        #     one coherent child state
        #
        # 1
        #     maximally dispersed among fine states
        #
        # This measures how much population identity is reorganized when
        # the resolution is increased.
        # --------------------------------------------------------------

        entropy_rows = []


        for coarse_cluster in (
            table.index
        ):

            counts = (
                table.loc[
                    coarse_cluster
                ]
                .to_numpy(
                    dtype=float
                )
            )


            total = counts.sum()


            probabilities = (
                counts[
                    counts > 0
                ]
                / total
            )


            if len(
                probabilities
            ) <= 1:

                entropy = 0.0

            else:

                entropy = float(
                    -np.sum(
                        probabilities
                        * np.log(
                            probabilities
                        )
                    )
                    / np.log(
                        len(
                            table.columns
                        )
                    )
                )


            entropy_rows.append(
                (
                    int(
                        total
                    ),
                    entropy,
                )
            )


        weighted_split_entropy = float(
            sum(
                size * entropy
                for (
                    size,
                    entropy,
                ) in entropy_rows
            )
            / sum(
                size
                for (
                    size,
                    _,
                ) in entropy_rows
            )
        )


        nesting_rows.append(
            {
                "coarse_k":
                    int(
                        coarse_k
                    ),

                "fine_k":
                    int(
                        fine_k
                    ),

                "weighted_parent_purity":
                    weighted_parent_purity,

                "mean_parent_purity":
                    mean_parent_purity,

                "minimum_parent_purity":
                    minimum_parent_purity,

                "weighted_split_entropy":
                    weighted_split_entropy,
            }
        )


    transition_columns = [
        "coarse_k",
        "fine_k",
        "coarse_raw_cluster",
        "fine_raw_cluster",
        "coarse_display_cluster",
        "fine_display_cluster",
        "count",
        "coarse_fraction",
        "fine_parent_fraction",
    ]


    nesting_columns = [
        "coarse_k",
        "fine_k",
        "weighted_parent_purity",
        "mean_parent_purity",
        "minimum_parent_purity",
        "weighted_split_entropy",
    ]


    transitions = pd.DataFrame(
        transition_rows,
        columns=transition_columns,
    )


    nesting = pd.DataFrame(
        nesting_rows,
        columns=nesting_columns,
    )


    return (
        transitions,
        nesting,
    )


# ======================================================================
# DECISION
# ======================================================================

def evaluate_resolution_decision(
    clustering,
    display_maps: dict[
        int,
        dict[int, int],
    ] | None = None,
) -> ResolutionDecisionResult:
    """
    Make an explicit, auditable resolution decision.

    Decision hierarchy
    ------------------

    1. Start from robust K values identified by the clustering engine.

    2. Remove Pareto-dominated resolutions.

    3. Among the remaining candidates, prioritize invariance to the
       uncertain PCA dimensionality:

            mean dimension agreement
            minimum dimension agreement

       followed by:

            mean subsample stability
            minimum subsample stability
            mean membership probability

       and finally lower K as a deterministic simplicity tie-breaker.

    This deliberately avoids inventing arbitrary weighted sums.
    """

    robust_k_values = tuple(
        int(k)
        for k in clustering.robust_k_values
    )


    if not robust_k_values:

        raise ValueError(
            "Resolution decision requires at least "
            "one robust K."
        )


    metrics = (
        clustering
        .multiresolution_summary
        .copy()
    )


    metrics = (
        metrics[
            metrics[
                "k"
            ]
            .astype(int)
            .isin(
                robust_k_values
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    metrics[
        "k"
    ] = (
        metrics[
            "k"
        ]
        .astype(int)
    )


    dominated_by: dict[
        int,
        tuple[int, ...],
    ] = {}


    for (
        _,
        target,
    ) in metrics.iterrows():

        target_k = int(
            target[
                "k"
            ]
        )


        dominators = []


        for (
            _,
            candidate,
        ) in metrics.iterrows():

            candidate_k = int(
                candidate[
                    "k"
                ]
            )


            if candidate_k == target_k:

                continue


            if _dominates(
                candidate,
                target,
            ):

                dominators.append(
                    candidate_k
                )


        dominated_by[
            target_k
        ] = tuple(
            sorted(
                dominators
            )
        )


    pareto_k_values = tuple(
        int(k)
        for k in robust_k_values
        if not dominated_by[
            int(k)
        ]
    )


    metrics[
        "pareto_optimal"
    ] = (
        metrics[
            "k"
        ]
        .isin(
            pareto_k_values
        )
    )


    metrics[
        "dominated_by"
    ] = (
        metrics[
            "k"
        ]
        .map(
            lambda k:
                ",".join(
                    str(value)
                    for value in dominated_by[
                        int(k)
                    ]
                )
        )
    )


    maximin_k = getattr(
        clustering,
        "strongest_k",
        None,
    )


    if maximin_k is not None:

        maximin_k = int(
            maximin_k
        )


    # ==================================================================
    # PREFERRED RESOLUTION
    #
    # Dimensionality uncertainty is explicit upstream, therefore
    # cross-dimensional reproducibility is the primary discriminator
    # among non-dominated robust solutions.
    # ==================================================================

    candidate_table = (
        metrics[
            metrics[
                "k"
            ]
            .isin(
                pareto_k_values
            )
        ]
        .copy()
    )


    sort_columns = [
        column
        for column in (
            "mean_dimension_agreement",
            "minimum_dimension_agreement",
            "mean_subsample_ari",
            "minimum_subsample_ari",
            "mean_membership_probability",
        )
        if column
        in candidate_table.columns
    ]


    candidate_table = (
        candidate_table
        .sort_values(
            [
                *sort_columns,
                "k",
            ],
            ascending=[
                *(
                    [False]
                    * len(
                        sort_columns
                    )
                ),
                True,
            ],
            kind="mergesort",
        )
    )


    preferred_k = int(
        candidate_table.iloc[
            0
        ][
            "k"
        ]
    )


    coarser_candidates = [
        int(k)
        for k in pareto_k_values
        if int(k) < preferred_k
    ]


    coarse_k = (
        max(
            coarser_candidates
        )
        if coarser_candidates
        else None
    )


    # ==================================================================
    # ROLE
    # ==================================================================

    def role(
        k: int,
    ) -> str:

        k = int(
            k
        )


        if k == preferred_k:

            return "preferred"


        if (
            coarse_k is not None
            and k == coarse_k
        ):

            return "coarse"


        if k in pareto_k_values:

            return "pareto_candidate"


        return "dominated"


    metrics[
        "decision_role"
    ] = (
        metrics[
            "k"
        ]
        .map(
            role
        )
    )


    # ==================================================================
    # POPULATION-IDENTITY STABILITY
    # ==================================================================

    (
        transitions,
        nesting,
    ) = _transition_analysis(
        clustering=clustering,
        robust_k_values=(
            robust_k_values
        ),
        display_maps=(
            display_maps
        ),
    )


    # ``decision_pair`` is part of the permanent nesting-table schema.
    #
    # With a single robust K there are legitimately no consecutive
    # resolutions to compare. The table is therefore empty, but must
    # remain structurally valid for downstream reporting and CSV output.

    nesting[
        "decision_pair"
    ] = (
        nesting[
            "coarse_k"
        ]
        .isin(
            pareto_k_values
        )
        & nesting[
            "fine_k"
        ]
        .isin(
            pareto_k_values
        )
    )


    return ResolutionDecisionResult(
        robust_k_values=(
            robust_k_values
        ),
        pareto_k_values=(
            pareto_k_values
        ),
        maximin_k=(
            maximin_k
        ),
        preferred_k=(
            preferred_k
        ),
        coarse_k=(
            coarse_k
        ),
        metrics=(
            metrics
        ),
        transitions=(
            transitions
        ),
        nesting=(
            nesting
        ),
    )


# ======================================================================
# PREFERRED-K MORPHOLOGICAL INTERPRETATION
# ======================================================================

def build_preferred_cluster_interpretation(
    master: pd.DataFrame,
    analytical_features: list[str],
    solution,
    display_map: dict[int, int],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Characterize the preferred clustering solution morphologically.

    Returns
    -------
    cluster_summary
        Population size, membership confidence, reliability, and top
        positive/negative morphometric signals.

    morphology_profiles
        Tidy per-cluster × feature table with raw means/medians and
        standardized effect-like summaries.

    category_composition
        Category composition when the reserved 'category' column exists.
    """

    data = (
        master
        .copy()
        .reset_index(
            drop=True
        )
    )


    labels = np.asarray(
        solution.labels,
        dtype=int,
    )


    probability = np.asarray(
        solution.cluster_probability,
        dtype=float,
    )


    reliability = np.asarray(
        solution.consensus_reliability,
        dtype=float,
    )


    if len(
        labels
    ) != len(
        data
    ):

        raise ValueError(
            "Preferred clustering labels do not "
            "match master-table rows."
        )


    data[
        "raw_cluster"
    ] = labels


    data[
        "display_cluster"
    ] = [
        int(
            display_map[
                int(label)
            ]
        )
        for label in labels
    ]


    data[
        "cluster_probability"
    ] = probability


    data[
        "consensus_reliability"
    ] = reliability


    # ==================================================================
    # FEATURES
    # ==================================================================

    features = [
        feature
        for feature in analytical_features
        if (
            feature in data.columns
            and pd.api.types.is_numeric_dtype(
                data[
                    feature
                ]
            )
        )
    ]


    raw = (
        data[
            features
        ]
        .astype(float)
    )


    global_mean = (
        raw.mean(
            axis=0
        )
    )


    global_std = (
        raw.std(
            axis=0,
            ddof=0,
        )
        .replace(
            0.0,
            np.nan,
        )
    )


    standardized = (
        raw
        .subtract(
            global_mean,
            axis=1,
        )
        .divide(
            global_std,
            axis=1,
        )
    )


    profile_rows = []


    for display_cluster in sorted(
        data[
            "display_cluster"
        ]
        .unique()
    ):

        mask = (
            data[
                "display_cluster"
            ]
            == display_cluster
        )


        for feature in features:

            raw_values = (
                raw.loc[
                    mask,
                    feature,
                ]
            )


            z_values = (
                standardized.loc[
                    mask,
                    feature,
                ]
            )


            profile_rows.append(
                {
                    "display_cluster":
                        int(
                            display_cluster
                        ),

                    "cluster":
                        f"C{int(display_cluster)}",

                    "feature":
                        feature,

                    "raw_mean":
                        float(
                            raw_values.mean()
                        ),

                    "raw_median":
                        float(
                            raw_values.median()
                        ),

                    "mean_z":
                        float(
                            z_values.mean()
                        ),

                    "median_z":
                        float(
                            z_values.median()
                        ),

                    "abs_mean_z":
                        float(
                            abs(
                                z_values.mean()
                            )
                        ),
                }
            )


    profiles = pd.DataFrame(
        profile_rows
    )


    # ==================================================================
    # CLUSTER SUMMARY
    # ==================================================================

    summary_rows = []


    for display_cluster in sorted(
        data[
            "display_cluster"
        ]
        .unique()
    ):

        cluster_data = (
            data[
                data[
                    "display_cluster"
                ]
                == display_cluster
            ]
        )


        cluster_profiles = (
            profiles[
                profiles[
                    "display_cluster"
                ]
                == display_cluster
            ]
        )


        positive = (
            cluster_profiles
            .sort_values(
                "mean_z",
                ascending=False,
            )
            .head(
                5
            )
        )


        negative = (
            cluster_profiles
            .sort_values(
                "mean_z",
                ascending=True,
            )
            .head(
                5
            )
        )


        positive_text = "; ".join(
            (
                f"{row.feature} "
                f"({row.mean_z:+.2f})"
            )
            for row in (
                positive.itertuples()
            )
        )


        negative_text = "; ".join(
            (
                f"{row.feature} "
                f"({row.mean_z:+.2f})"
            )
            for row in (
                negative.itertuples()
            )
        )


        summary_rows.append(
            {
                "display_cluster":
                    int(
                        display_cluster
                    ),

                "cluster":
                    f"C{int(display_cluster)}",

                "n_cells":
                    int(
                        len(
                            cluster_data
                        )
                    ),

                "fraction":
                    float(
                        len(
                            cluster_data
                        )
                        / len(
                            data
                        )
                    ),

                "mean_membership_probability":
                    float(
                        cluster_data[
                            "cluster_probability"
                        ]
                        .mean()
                    ),

                "mean_consensus_reliability":
                    float(
                        cluster_data[
                            "consensus_reliability"
                        ]
                        .mean()
                    ),

                "top_positive_features":
                    positive_text,

                "top_negative_features":
                    negative_text,
            }
        )


    cluster_summary = pd.DataFrame(
        summary_rows
    )


    # ==================================================================
    # CATEGORY COMPOSITION
    # ==================================================================

    category_rows = []


    if (
        "category"
        in data.columns
    ):

        for (
            display_cluster,
            cluster_data,
        ) in data.groupby(
            "display_cluster",
            sort=True,
        ):

            counts = (
                cluster_data[
                    "category"
                ]
                .astype(str)
                .value_counts(
                    dropna=False
                )
            )


            for (
                category,
                count,
            ) in counts.items():

                category_rows.append(
                    {
                        "display_cluster":
                            int(
                                display_cluster
                            ),

                        "cluster":
                            f"C{int(display_cluster)}",

                        "category":
                            str(
                                category
                            ),

                        "count":
                            int(
                                count
                            ),

                        "within_cluster_fraction":
                            float(
                                count
                                / len(
                                    cluster_data
                                )
                            ),
                    }
                )


    category_composition = pd.DataFrame(
        category_rows
    )


    return (
        cluster_summary,
        profiles,
        category_composition,
    )


# MG_STABLE_BAND_DECISION_OVERRIDE
from .resolution_support import evaluate_resolution_support as _mg_resolution_support_v2

_mg_legacy_evaluate_resolution_decision = evaluate_resolution_decision


def evaluate_resolution_decision(clustering, display_maps):
    support = _mg_resolution_support_v2(clustering.multiresolution_summary)
    supported = tuple(int(k) for k in support.supported_k_values)
    observed = tuple(int(k) for k in clustering.robust_k_values)

    if supported != observed:
        raise RuntimeError(
            f"Resolution support mismatch: clustering={observed}, decision={supported}"
        )

    result = _mg_legacy_evaluate_resolution_decision(
        clustering=clustering,
        display_maps=display_maps,
    )

    default_k = int(support.default_k)

    # Compatibility aliases while Mapping/current loaders still use preferred_k.
    result.robust_k_values = supported
    result.preferred_k = default_k
    result.coarse_k = default_k

    # Canonical multiresolution vocabulary.
    result.supported_k_values = supported
    result.stable_bands = support.stable_bands
    result.band_representatives = support.band_representatives
    result.default_k = default_k

    interpretation = support.table[
        [
            "k",
            "absolute_support",
            "supported_resolution",
            "isolated_candidate",
            "band_id",
            "band_role",
            "band_representative",
            "default_resolution",
            "support_dimension_floor",
            "isolated_dimension_floor",
        ]
    ].copy()

    result.metrics = result.metrics.merge(
        interpretation,
        on="k",
        how="left",
        validate="one_to_one",
    )

    clustering.supported_k_values = supported
    clustering.stable_bands = support.stable_bands
    clustering.band_representatives = support.band_representatives
    clustering.default_k = default_k
    clustering.preferred_k = default_k
    clustering.coarse_k = default_k
    clustering.resolution_support_table = support.table.copy()

    return result


# MG_MORPHOLOGY_STATE_DECISION_V4
def evaluate_resolution_decision(clustering, display_maps):
    """Select reproducible morphology-state counts without a larger-K reward."""

    support = _mg_resolution_support_v2(clustering.multiresolution_summary)
    supported = tuple(int(k) for k in support.supported_k_values)
    observed = tuple(int(k) for k in clustering.robust_k_values)

    if supported != observed:
        raise RuntimeError(
            f"Resolution support mismatch: clustering={observed}, decision={supported}"
        )

    result = _mg_legacy_evaluate_resolution_decision(
        clustering=clustering,
        display_maps=display_maps,
    )

    automatic_k = int(support.automatic_k)
    representatives = tuple(int(k) for k in support.band_representatives)

    result.robust_k_values = supported
    result.supported_k_values = supported
    result.stable_bands = support.stable_bands
    result.band_representatives = representatives
    result.automatic_k = automatic_k
    result.automatic_selection_rule = support.automatic_selection_rule
    result.support_criteria = dict(support.support_criteria)
    result.effective_k = automatic_k
    result.selection_source = "automatic_data_driven"

    # Internal compatibility attributes used by existing downstream stages.
    # They are implementation details, not scientific vocabulary.
    result.default_k = automatic_k
    result.preferred_k = automatic_k
    result.coarse_k = automatic_k

    interpretation = support.table[
        [
            "k",
            "absolute_support",
            "local_support",
            "supported_resolution",
            "isolated_candidate",
            "support_mode",
            "band_id",
            "band_role",
            "band_representative",
            "automatic_selection",
            "support_dimension_floor",
            "isolated_dimension_floor",
        ]
    ].copy()

    result.metrics = result.metrics.merge(
        interpretation,
        on="k",
        how="left",
        validate="one_to_one",
    )
    result.metrics["pareto_is_diagnostic_only"] = True
    result.metrics["effective_selection"] = result.metrics["k"].eq(automatic_k)
    result.metrics["selection_source"] = "automatic_data_driven"
    result.metrics["decision_role"] = "supported"
    result.metrics.loc[
        result.metrics["k"].eq(automatic_k),
        "decision_role",
    ] = "preferred"

    transitions, nesting = _transition_analysis(
        clustering=clustering,
        robust_k_values=supported,
        display_maps=display_maps,
    )
    nesting["decision_pair"] = True
    band_by_k = {
        int(k): int(index)
        for index, band in enumerate(support.stable_bands, start=1)
        for k in band
    }
    if nesting.empty:
        nesting["transition_scope"] = pd.Series(dtype="object")
    else:
        nesting["transition_scope"] = nesting.apply(
            lambda row: (
                "within_stable_band"
                if band_by_k.get(int(row["coarse_k"]))
                == band_by_k.get(int(row["fine_k"]))
                else "between_stable_bands"
            ),
            axis=1,
        )
    result.transitions = transitions
    result.nesting = nesting

    clustering.supported_k_values = supported
    clustering.robust_k_values = supported
    clustering.stable_bands = support.stable_bands
    clustering.band_representatives = representatives
    clustering.automatic_k = automatic_k
    clustering.automatic_selection_rule = support.automatic_selection_rule
    clustering.support_criteria = dict(support.support_criteria)
    clustering.effective_k = automatic_k
    clustering.selection_source = "automatic_data_driven"
    clustering.default_k = automatic_k
    clustering.preferred_k = automatic_k
    clustering.coarse_k = automatic_k
    clustering.resolution_support_table = support.table.copy()

    return result
