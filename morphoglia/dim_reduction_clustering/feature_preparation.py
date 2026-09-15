from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FeaturePreparationConfig


@dataclass
class FeaturePreparationResult:
    """
    Output of the unsupervised MorphoGlia Feature Selection stage.
    """

    data: pd.DataFrame
    analytical_features: list[str]
    excluded_features: list[str]

    qc_report: pd.DataFrame
    correlation_matrix: pd.DataFrame | None


class FeaturePreparationEngine:
    """
    Automatic unsupervised QC for morphological features.

    This stage does NOT use category labels.

    It removes features that should not enter morphological discovery:
    - structural/spatial exclusions
    - explicit user exclusions
    - non-numeric features
    - all-missing features
    - constant features

    Correlation is reported but, by default, correlated features are not
    automatically removed because PCA is intended to model covariance.
    """

    def __init__(
        self,
        config: FeaturePreparationConfig,
    ):
        self.config = config

    def fit_transform(
        self,
        data: pd.DataFrame,
    ) -> FeaturePreparationResult:

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "FeaturePreparationEngine expects a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Feature-preparation input is empty."
            )

        structural_exclusions = set(
            self.config.structural_exclusions
        )

        user_exclusions = set(
            self.config.exclude_features
        )

        selected = []
        excluded = []
        rows = []

        for feature in data.columns:

            series = data[feature]

            reason = None

            # ----------------------------------------------------------
            # Structural exclusions
            # ----------------------------------------------------------

            if feature in structural_exclusions:
                reason = "structural_exclusion"

            # ----------------------------------------------------------
            # User exclusions
            # ----------------------------------------------------------

            elif feature in user_exclusions:
                reason = "user_exclusion"

            # ----------------------------------------------------------
            # Numeric-only
            # ----------------------------------------------------------

            elif (
                self.config.numeric_only
                and not pd.api.types.is_numeric_dtype(series)
            ):
                reason = "non_numeric"

            # ----------------------------------------------------------
            # All missing
            # ----------------------------------------------------------

            elif (
                self.config.drop_all_missing
                and series.notna().sum() == 0
            ):
                reason = "all_missing"

            # ----------------------------------------------------------
            # Constant
            # ----------------------------------------------------------

            elif (
                self.config.drop_constant
                and series.nunique(dropna=True) <= 1
            ):
                reason = "constant"

            keep = reason is None

            if keep:
                selected.append(feature)
            else:
                excluded.append(feature)

            numeric_series = (
                pd.to_numeric(
                    series,
                    errors="coerce",
                )
                if pd.api.types.is_numeric_dtype(series)
                else None
            )

            rows.append(
                {
                    "feature": feature,
                    "kept": keep,
                    "reason": (
                        "kept"
                        if keep
                        else reason
                    ),
                    "dtype": str(series.dtype),
                    "missing_values": int(
                        series.isna().sum()
                    ),
                    "unique_values": int(
                        series.nunique(
                            dropna=True
                        )
                    ),
                    "minimum": (
                        float(
                            numeric_series.min()
                        )
                        if (
                            numeric_series is not None
                            and numeric_series.notna().any()
                        )
                        else np.nan
                    ),
                    "maximum": (
                        float(
                            numeric_series.max()
                        )
                        if (
                            numeric_series is not None
                            and numeric_series.notna().any()
                        )
                        else np.nan
                    ),
                }
            )

        if not selected:
            raise ValueError(
                "Feature Preparation removed every analytical feature."
            )

        selected_data = (
            data[selected]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        # --------------------------------------------------------------
        # Important:
        #
        # Missing values are NOT silently imputed here.
        # The stage reports them so the downstream missing-value policy
        # can be explicit.
        # --------------------------------------------------------------

        qc_report = pd.DataFrame(
            rows
        )

        correlation_matrix = None

        if self.config.redundancy_report:

            correlation_matrix = (
                selected_data.corr(
                    method="pearson"
                )
            )

        if self.config.prune_correlated:
            raise NotImplementedError(
                "Automatic correlation pruning is intentionally "
                "not implemented yet. MorphoGlia currently reports "
                "feature redundancy without imposing an arbitrary "
                "correlation cutoff."
            )

        return FeaturePreparationResult(
            data=selected_data,
            analytical_features=selected,
            excluded_features=excluded,
            qc_report=qc_report,
            correlation_matrix=correlation_matrix,
        )
