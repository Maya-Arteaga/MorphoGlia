from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

# Absolute support gate. No reward for larger K.
MEAN_ARI_MIN = 0.70
WORST_ARI_MIN = 0.50
MEMBERSHIP_MIN = 0.70
DIM_REL_MIN = 0.70
DIM_ABS_MIN = 0.25
MIN_CLUSTER_SIZE = 5

# An isolated K has no persistence support and must therefore be stronger.
ISOLATED_MEAN_ARI_MIN = 0.70
ISOLATED_WORST_ARI_MIN = 0.50
ISOLATED_MEMBERSHIP_MIN = 0.70
ISOLATED_DIM_REL_MIN = 0.70
ISOLATED_DIM_ABS_MIN = 0.25

AUTOMATIC_SELECTION_RULE = (
    "first_stable_band_then_cross_dimension_agreement_"
    "then_worst_case_and_mean_resampling_stability_"
    "then_membership_reliability"
)


@dataclass(frozen=True)
class ResolutionSupport:
    supported_k_values: tuple[int, ...]
    stable_bands: tuple[tuple[int, ...], ...]
    band_representatives: tuple[int, ...]
    automatic_k: int
    automatic_selection_rule: str
    support_criteria: dict[str, float | int | str]
    table: pd.DataFrame

    @property
    def default_k(self) -> int:
        """Internal compatibility alias; never use as scientific language."""
        return self.automatic_k


def _bands(values):
    values = sorted(set(int(v) for v in values))
    if not values:
        return ()
    groups = [[values[0]]]
    for value in values[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return tuple(tuple(group) for group in groups)


def _locally_nondominated(table, k):
    """True when neither adjacent state count dominates K on stability."""
    metrics = (
        "mean_subsample_ari",
        "minimum_subsample_ari",
        "mean_dimension_agreement",
    )
    row = table.loc[table["k"].eq(int(k))].iloc[0]
    neighbors = table[table["k"].isin((int(k) - 1, int(k) + 1))]

    for _, neighbor in neighbors.iterrows():
        no_worse = all(float(neighbor[name]) >= float(row[name]) for name in metrics)
        strictly_better = any(float(neighbor[name]) > float(row[name]) for name in metrics)
        if no_worse and strictly_better:
            return False

    return True


def _representative(table, band):
    subset = table[table["k"].astype(int).isin(band)].copy()
    centre = float(np.mean(band))
    subset["_centre_distance"] = (subset["k"].astype(float) - centre).abs()

    columns = ["mean_dimension_agreement"]
    ascending = [False]
    if "minimum_dimension_agreement" in subset.columns:
        columns.append("minimum_dimension_agreement")
        ascending.append(False)

    columns += [
        "minimum_subsample_ari",
        "mean_subsample_ari",
        "mean_membership_probability",
        "minimum_cluster_size_across_d",
        "_centre_distance",
        "k",
    ]
    ascending += [False, False, False, False, True, True]

    ranked = subset.sort_values(
        columns,
        ascending=ascending,
        kind="mergesort",
    )
    return int(ranked.iloc[0]["k"])


def evaluate_resolution_support(summary):
    if summary is None or summary.empty:
        raise ValueError("The multiresolution summary is empty.")

    required = {
        "k",
        "mean_subsample_ari",
        "minimum_subsample_ari",
        "mean_dimension_agreement",
        "mean_membership_probability",
        "minimum_cluster_size_across_d",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Resolution support is missing columns: {missing}")

    table = summary.copy().sort_values("k", kind="mergesort").reset_index(drop=True)
    table["k"] = table["k"].astype(int)

    best_dimension = float(
        pd.to_numeric(table["mean_dimension_agreement"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .max()
    )
    dimension_floor = max(DIM_ABS_MIN, DIM_REL_MIN * best_dimension)
    isolated_dimension_floor = max(
        ISOLATED_DIM_ABS_MIN,
        ISOLATED_DIM_REL_MIN * best_dimension,
    )

    base = (
        table["mean_subsample_ari"].astype(float).ge(MEAN_ARI_MIN)
        & table["minimum_subsample_ari"].astype(float).ge(WORST_ARI_MIN)
        & table["mean_membership_probability"].astype(float).ge(MEMBERSHIP_MIN)
        & table["mean_dimension_agreement"].astype(float).ge(dimension_floor)
        & table["minimum_cluster_size_across_d"].astype(float).ge(MIN_CLUSTER_SIZE)
    )

    initial_bands = _bands(table.loc[base, "k"].tolist())
    isolated = {band[0] for band in initial_bands if len(band) == 1}
    local_support = pd.Series(False, index=table.index, dtype=bool)

    for band in initial_bands:
        if len(band) > 1:
            local_support.loc[table["k"].isin(band)] = True
        else:
            k = int(band[0])
            local_support.loc[table["k"].eq(k)] = _locally_nondominated(table, k)

    keep = base & local_support

    for k in isolated:
        row = table.loc[table["k"].eq(k)].iloc[0]
        strict = (
            float(row["mean_subsample_ari"]) >= ISOLATED_MEAN_ARI_MIN
            and float(row["minimum_subsample_ari"]) >= ISOLATED_WORST_ARI_MIN
            and float(row["mean_membership_probability"]) >= ISOLATED_MEMBERSHIP_MIN
            and float(row["mean_dimension_agreement"]) >= isolated_dimension_floor
            and float(row["minimum_cluster_size_across_d"]) >= MIN_CLUSTER_SIZE
        )
        if not strict:
            keep.loc[table["k"].eq(k)] = False

    strictly_supported = tuple(
        int(k)
        for k in table.loc[keep, "k"].tolist()
    )

    fallback_used = not strictly_supported

    if fallback_used:
        # No candidate passed every strict reproducibility gate.
        #
        # Continue with the strongest available evaluated candidate using
        # the same deterministic ranking already used for representatives.
        fallback_k = _representative(
            table,
            tuple(
                table["k"]
                .astype(int)
                .tolist()
            ),
        )

        supported = (
            int(fallback_k),
        )

    else:
        supported = strictly_supported

    stable_bands = _bands(supported)
    representatives = tuple(
        _representative(table, band)
        for band in stable_bands
    )
    automatic_k = int(representatives[0])

    support_criteria = {
        "mean_subsample_ari_min": float(MEAN_ARI_MIN),
        "worst_subsample_ari_min": float(WORST_ARI_MIN),
        "membership_probability_min": float(MEMBERSHIP_MIN),
        "dimension_agreement_floor": float(dimension_floor),
        "minimum_state_size": int(MIN_CLUSTER_SIZE),
        "isolated_mean_subsample_ari_min": float(ISOLATED_MEAN_ARI_MIN),
        "isolated_worst_subsample_ari_min": float(ISOLATED_WORST_ARI_MIN),
        "isolated_membership_probability_min": float(ISOLATED_MEMBERSHIP_MIN),
        "isolated_dimension_agreement_floor": float(isolated_dimension_floor),
        "automatic_selection_rule": AUTOMATIC_SELECTION_RULE,
    }

    table["absolute_support"] = base.astype(bool)
    table["local_support"] = local_support.astype(bool)
    # supported_resolution remains the operational downstream set.
    # strict_reproducibility_support preserves whether the count actually
    # passed the scientific support gate.
    table["strict_reproducibility_support"] = (
        table["k"].isin(strictly_supported)
    )
    table["supported_resolution"] = table["k"].isin(supported)
    table["automatic_fallback"] = False
    table["isolated_candidate"] = table["k"].isin(isolated)
    table["band_id"] = pd.Series([pd.NA] * len(table), dtype="Int64")
    table["band_role"] = ""
    table["band_representative"] = False
    table["automatic_selection"] = False
    table["support_mode"] = ""
    table["support_dimension_floor"] = float(dimension_floor)
    table["isolated_dimension_floor"] = float(isolated_dimension_floor)

    for index, band in enumerate(stable_bands, start=1):
        if index == 1:
            role = "coarse"
        elif index == len(stable_bands):
            role = "fine"
        else:
            role = "intermediate"

        mask = table["k"].isin(band)
        table.loc[mask, "band_id"] = index
        table.loc[mask, "band_role"] = role
        table.loc[
            table["k"].eq(representatives[index - 1]),
            "band_representative",
        ] = True

    supported_mask = table["k"].isin(supported)

    if fallback_used:
        fallback_mask = table["k"].eq(automatic_k)

        table.loc[
            fallback_mask,
            "support_mode",
        ] = "automatic_fallback"

        table.loc[
            fallback_mask,
            "automatic_fallback",
        ] = True

    else:
        table.loc[
            supported_mask & table["k"].isin(isolated),
            "support_mode",
        ] = "isolated_stable_peak"

        table.loc[
            supported_mask & ~table["k"].isin(isolated),
            "support_mode",
        ] = "stable_band"

    table.loc[
        table["k"].eq(automatic_k),
        "automatic_selection",
    ] = True

    return ResolutionSupport(
        supported_k_values=supported,
        stable_bands=stable_bands,
        band_representatives=representatives,
        automatic_k=automatic_k,
        automatic_selection_rule=AUTOMATIC_SELECTION_RULE,
        support_criteria=support_criteria,
        table=table,
    )
