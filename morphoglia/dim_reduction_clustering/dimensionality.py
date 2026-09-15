from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import DimensionalityReductionConfig


@dataclass
class DimensionalityReductionResult:
    """
    Results of the MorphoGlia dimensionality-reduction stage.
    """

    feature_names: list[str]

    transformed_features: np.ndarray
    scaled_features: np.ndarray

    scores: np.ndarray
    loadings: pd.DataFrame

    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray

    parallel_threshold: np.ndarray
    parallel_supported: np.ndarray
    parallel_dimension: int

    broken_stick_expected: np.ndarray
    broken_stick_supported: np.ndarray
    broken_stick_dimension: int

    two_nn_dimension: float

    plausible_dimension_min: int
    plausible_dimension_max: int

    scaling_report: pd.DataFrame


class DimensionalityReductionEngine:
    """
    Automatic dimensionality analysis for morphometric data.

    Workflow
    --------
    1. Apply configured scaling.
    2. Fit full PCA.
    3. Estimate retained dimensionality with Parallel Analysis.
    4. Estimate retained dimensionality with Broken Stick.
    5. Estimate local intrinsic dimensionality with Two-NN.
    6. Define the plausible PCA interval from PA and Broken Stick.

    Two-NN is diagnostic only. It does not vote on the retained
    PCA dimensionality.
    """

    def __init__(
        self,
        config: DimensionalityReductionConfig,
        random_state: int = 24,
    ):
        self.config = config
        self.random_state = int(
            random_state
        )

        self.scaler = None
        self.pca = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        data: pd.DataFrame,
    ) -> DimensionalityReductionResult:

        self._validate_input(
            data
        )

        feature_names = list(
            data.columns
        )

        X = data.to_numpy(
            dtype=float
        )

        (
            X_transformed,
            X_scaled,
            scaling_report,
        ) = self._scale(
            X=X,
            feature_names=feature_names,
        )

        # --------------------------------------------------------------
        # Full PCA
        # --------------------------------------------------------------

        self.pca = PCA(
            svd_solver="full"
        )

        scores = self.pca.fit_transform(
            X_scaled
        )

        n_components = (
            scores.shape[1]
        )

        pc_names = [
            f"PC{i}"
            for i in range(
                1,
                n_components + 1,
            )
        ]

        loadings = pd.DataFrame(
            self.pca.components_.T,
            index=feature_names,
            columns=pc_names,
        )

        loadings.index.name = (
            "feature"
        )

        eigenvalues = (
            self.pca.explained_variance_.copy()
        )

        explained_variance_ratio = (
            self.pca
            .explained_variance_ratio_
            .copy()
        )

        cumulative_variance = (
            np.cumsum(
                explained_variance_ratio
            )
        )

        # --------------------------------------------------------------
        # Parallel Analysis
        # --------------------------------------------------------------

        (
            parallel_threshold,
            parallel_supported,
            parallel_dimension,
        ) = self._parallel_analysis(
            X_scaled=X_scaled,
            observed_eigenvalues=eigenvalues,
        )

        # --------------------------------------------------------------
        # Broken Stick
        # --------------------------------------------------------------

        broken_stick_expected = (
            self._broken_stick_expectation(
                n_components
            )
        )

        broken_stick_supported = (
            explained_variance_ratio
            > broken_stick_expected
        )

        broken_stick_dimension = (
            self._leading_true_count(
                broken_stick_supported
            )
        )

        # --------------------------------------------------------------
        # Two-NN
        # --------------------------------------------------------------

        two_nn_dimension = (
            self._two_nn_dimension(
                X_scaled
            )
        )

        # --------------------------------------------------------------
        # Plausible PCA interval
        #
        # PA and Broken Stick define the interval.
        # Two-NN remains an independent diagnostic.
        # --------------------------------------------------------------

        plausible_dimension_min = max(
            1,
            min(
                parallel_dimension,
                broken_stick_dimension,
            ),
        )

        plausible_dimension_max = max(
            1,
            max(
                parallel_dimension,
                broken_stick_dimension,
            ),
        )

        plausible_dimension_max = min(
            plausible_dimension_max,
            n_components,
        )

        return DimensionalityReductionResult(
            feature_names=feature_names,
            transformed_features=X_transformed,
            scaled_features=X_scaled,
            scores=scores,
            loadings=loadings,
            eigenvalues=eigenvalues,
            explained_variance_ratio=explained_variance_ratio,
            cumulative_variance=cumulative_variance,
            parallel_threshold=parallel_threshold,
            parallel_supported=parallel_supported,
            parallel_dimension=parallel_dimension,
            broken_stick_expected=broken_stick_expected,
            broken_stick_supported=broken_stick_supported,
            broken_stick_dimension=broken_stick_dimension,
            two_nn_dimension=two_nn_dimension,
            plausible_dimension_min=plausible_dimension_min,
            plausible_dimension_max=plausible_dimension_max,
            scaling_report=scaling_report,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(
        data: pd.DataFrame,
    ) -> None:

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "DimensionalityReductionEngine expects "
                "a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Dimensionality-reduction input is empty."
            )

        non_numeric = [
            column
            for column in data.columns
            if not pd.api.types.is_numeric_dtype(
                data[column]
            )
        ]

        if non_numeric:
            raise ValueError(
                "Dimensionality-reduction input contains "
                "non-numeric features: "
                f"{non_numeric}"
            )

        values = data.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "Dimensionality-reduction input contains "
                "NaN or infinite values."
            )

        if data.shape[0] < 3:
            raise ValueError(
                "At least 3 observations are required."
            )

        if data.shape[1] < 1:
            raise ValueError(
                "At least 1 morphometric feature is required."
            )

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def _scale(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
    ]:

        mode = (
            self.config.effective_scaling
        )

        X_transformed = X.copy()

        rows = []

        # --------------------------------------------------------------
        # RAW
        # --------------------------------------------------------------

        if mode == "raw":

            self.scaler = None

            X_scaled = (
                X_transformed.copy()
            )

            for index, feature in enumerate(
                feature_names
            ):

                rows.append(
                    {
                        "feature": feature,
                        "scaling": "raw",
                        "skew_before": float(
                            stats.skew(
                                X[:, index]
                            )
                        ),
                        "non_negative": bool(
                            np.min(
                                X[:, index]
                            )
                            >= 0
                        ),
                        "log1p_applied": False,
                        "scaler_center": np.nan,
                        "scaler_scale": np.nan,
                    }
                )

            return (
                X_transformed,
                X_scaled,
                pd.DataFrame(rows),
            )

        # --------------------------------------------------------------
        # Z-SCORE
        # --------------------------------------------------------------

        if mode == "zscore":

            self.scaler = (
                StandardScaler()
            )

            X_scaled = (
                self.scaler.fit_transform(
                    X_transformed
                )
            )

            for index, feature in enumerate(
                feature_names
            ):

                rows.append(
                    {
                        "feature": feature,
                        "scaling": "zscore",
                        "skew_before": float(
                            stats.skew(
                                X[:, index]
                            )
                        ),
                        "non_negative": bool(
                            np.min(
                                X[:, index]
                            )
                            >= 0
                        ),
                        "log1p_applied": False,
                        "scaler_center": float(
                            self.scaler.mean_[
                                index
                            ]
                        ),
                        "scaler_scale": float(
                            self.scaler.scale_[
                                index
                            ]
                        ),
                    }
                )

            return (
                X_transformed,
                X_scaled,
                pd.DataFrame(rows),
            )

        # --------------------------------------------------------------
        # ROBUST
        # --------------------------------------------------------------

        if mode == "robust":

            self.scaler = (
                RobustScaler()
            )

            X_scaled = (
                self.scaler.fit_transform(
                    X_transformed
                )
            )

            for index, feature in enumerate(
                feature_names
            ):

                rows.append(
                    {
                        "feature": feature,
                        "scaling": "robust",
                        "skew_before": float(
                            stats.skew(
                                X[:, index]
                            )
                        ),
                        "non_negative": bool(
                            np.min(
                                X[:, index]
                            )
                            >= 0
                        ),
                        "log1p_applied": False,
                        "scaler_center": float(
                            self.scaler.center_[
                                index
                            ]
                        ),
                        "scaler_scale": float(
                            self.scaler.scale_[
                                index
                            ]
                        ),
                    }
                )

            return (
                X_transformed,
                X_scaled,
                pd.DataFrame(rows),
            )

        # --------------------------------------------------------------
        # LOG
        #
        # Exact current MorphoGlia rule:
        #
        # 1. inspect each raw feature
        # 2. if non-negative and positively skewed beyond threshold,
        #    apply log1p
        # 3. z-score ALL features afterward
        # --------------------------------------------------------------

        if mode == "log":

            threshold = (
                self.config
                .effective_log_skew_threshold
            )

            log_flags = []

            skew_values = []

            non_negative_values = []

            for index in range(
                X.shape[1]
            ):

                values = X[
                    :,
                    index,
                ]

                skewness = float(
                    stats.skew(
                        values
                    )
                )

                non_negative = bool(
                    np.min(
                        values
                    )
                    >= 0
                )

                apply_log = bool(
                    non_negative
                    and np.isfinite(
                        skewness
                    )
                    and skewness
                    > threshold
                )

                if apply_log:

                    X_transformed[
                        :,
                        index,
                    ] = np.log1p(
                        values
                    )

                log_flags.append(
                    apply_log
                )

                skew_values.append(
                    skewness
                )

                non_negative_values.append(
                    non_negative
                )

            self.scaler = (
                StandardScaler()
            )

            X_scaled = (
                self.scaler.fit_transform(
                    X_transformed
                )
            )

            for index, feature in enumerate(
                feature_names
            ):

                rows.append(
                    {
                        "feature": feature,
                        "scaling": "log",
                        "skew_before": skew_values[
                            index
                        ],
                        "non_negative": (
                            non_negative_values[
                                index
                            ]
                        ),
                        "log1p_applied": (
                            log_flags[
                                index
                            ]
                        ),
                        "scaler_center": float(
                            self.scaler.mean_[
                                index
                            ]
                        ),
                        "scaler_scale": float(
                            self.scaler.scale_[
                                index
                            ]
                        ),
                    }
                )

            return (
                X_transformed,
                X_scaled,
                pd.DataFrame(rows),
            )

        raise ValueError(
            f"Unsupported scaling mode: {mode}"
        )

    # ------------------------------------------------------------------
    # Parallel Analysis
    # ------------------------------------------------------------------

    def _parallel_analysis(
        self,
        X_scaled: np.ndarray,
        observed_eigenvalues: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        int,
    ]:

        iterations = (
            self.config
            .effective_parallel_analysis_iterations
        )

        percentile = (
            self.config
            .effective_parallel_analysis_percentile
        )

        rng = np.random.default_rng(
            self.random_state
        )

        n_components = len(
            observed_eigenvalues
        )

        null_eigenvalues = np.zeros(
            (
                iterations,
                n_components,
            ),
            dtype=float,
        )

        for iteration in range(
            iterations
        ):

            X_null = np.empty_like(
                X_scaled
            )

            for feature_index in range(
                X_scaled.shape[1]
            ):

                X_null[
                    :,
                    feature_index,
                ] = rng.permutation(
                    X_scaled[
                        :,
                        feature_index,
                    ]
                )

            null_pca = PCA(
                svd_solver="full"
            )

            null_pca.fit(
                X_null
            )

            null_eigenvalues[
                iteration,
                :,
            ] = (
                null_pca
                .explained_variance_
            )

        threshold = np.percentile(
            null_eigenvalues,
            percentile,
            axis=0,
        )

        supported = (
            observed_eigenvalues
            > threshold
        )

        dimension = (
            self._leading_true_count(
                supported
            )
        )

        return (
            threshold,
            supported,
            dimension,
        )

    # ------------------------------------------------------------------
    # Broken Stick
    # ------------------------------------------------------------------

    @staticmethod
    def _broken_stick_expectation(
        n_components: int,
    ) -> np.ndarray:

        expected = np.zeros(
            n_components,
            dtype=float,
        )

        for component_index in range(
            1,
            n_components + 1,
        ):

            expected[
                component_index - 1
            ] = (
                sum(
                    1.0 / index
                    for index in range(
                        component_index,
                        n_components + 1,
                    )
                )
                / n_components
            )

        return expected

    # ------------------------------------------------------------------
    # Two-NN intrinsic dimensionality
    # ------------------------------------------------------------------

    @staticmethod
    def _two_nn_dimension(
        X_scaled: np.ndarray,
    ) -> float:

        if len(
            X_scaled
        ) < 3:

            return np.nan

        neighbors = (
            NearestNeighbors(
                n_neighbors=3
            )
        )

        neighbors.fit(
            X_scaled
        )

        distances, _ = (
            neighbors.kneighbors(
                X_scaled
            )
        )

        r1 = distances[
            :,
            1,
        ]

        r2 = distances[
            :,
            2,
        ]

        valid = (
            (r1 > 0)
            & (r2 > r1)
            & np.isfinite(r1)
            & np.isfinite(r2)
        )

        if valid.sum() < 3:

            return np.nan

        mu = (
            r2[
                valid
            ]
            / r1[
                valid
            ]
        )

        denominator = np.sum(
            np.log(
                mu
            )
        )

        if denominator <= 0:

            return np.nan

        return float(
            valid.sum()
            / denominator
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _leading_true_count(
        mask: np.ndarray,
    ) -> int:

        count = 0

        for value in mask:

            if not bool(
                value
            ):
                break

            count += 1

        return count
