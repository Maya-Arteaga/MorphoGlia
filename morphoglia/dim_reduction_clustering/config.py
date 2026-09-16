from __future__ import annotations

"""
Configuration for MorphoGlia's combined analytical stage.

Internal sequence:

    feature preparation
        -> dimensionality reduction
        -> multiresolution clustering
        -> resolution decision
"""



# ====================================================================
# FEATURE PREPARATION
# ====================================================================

from dataclasses import dataclass, field



SPATIAL_COORDINATE_FEATURES = (
    "Soma_centroid",
    "Soma_centroid_x",
    "Soma_centroid_y",
    "CH_centroid",
    "CH_centroid_x",
    "CH_centroid_y",
)



@dataclass
class FeaturePreparationConfig:
    """
    Configuration for automatic morphometric feature QC.

    This stage does NOT perform supervised feature selection.

    Its purpose is only to construct a valid morphometric matrix before
    dimensionality reduction.

    Default behavior:
      - use scalar numeric morphometric features
      - remove all-missing features
      - remove constant / zero-variance features
      - report feature redundancy
      - do not automatically prune correlated features
      - never use biological category labels for selection

    Correlated variables are retained by default because PCA is responsible
    for resolving covariance structure in the following stage.
    """

    numeric_only: bool = True

    drop_all_missing: bool = True

    drop_constant: bool = True

    redundancy_report: bool = True

    prune_correlated: bool = False

    exclude_features: list[str] = field(
        default_factory=list
    )
    
    
    
    @property
    def structural_exclusions(self) -> tuple[str, ...]:
        """
        Features stored for mapping/spatial analysis but excluded from
        morphological feature analysis.
        """
        return SPATIAL_COORDINATE_FEATURES
    
        
        

    def configuration_rows(
        self,
    ) -> list[dict]:
        """
        Return Technical Record rows.
        """

        rows = [
            {
                "section": "Feature Preparation",
                "parameter": "method",
                "value": "automatic_unsupervised_qc",
                "source": "default",
            },
            {
                "section": "Feature Preparation",
                "parameter": "numeric_only",
                "value": self.numeric_only,
                "source": "default",
            },
            {
                "section": "Feature Preparation",
                "parameter": "drop_all_missing",
                "value": self.drop_all_missing,
                "source": "default",
            },
            {
                "section": "Feature Preparation",
                "parameter": "drop_constant",
                "value": self.drop_constant,
                "source": "default",
            },
            {
                "section": "Feature Preparation",
                "parameter": "redundancy_report",
                "value": self.redundancy_report,
                "source": "default",
            },
            
            {
                "section": "Feature Preparation",
                "parameter": "structural_exclusions",
                "value": list(self.structural_exclusions),
                "source": "default",
            },
            
            
            
            {
                "section": "Feature Preparation",
                "parameter": "prune_correlated",
                "value": self.prune_correlated,
                "source": "default",
            },
        ]

        rows.append(
            {
                "section": "Feature Preparation",
                "parameter": "exclude_features",
                "value": list(self.exclude_features),
                "source": (
                    "user"
                    if self.exclude_features
                    else "default"
                ),
            }
        )
        
        
        

        


        

        return rows



# ====================================================================
# DIMENSIONALITY REDUCTION
# ====================================================================

from dataclasses import dataclass


# ======================================================================
# Canonical MorphoGlia dimensionality-reduction defaults
# ======================================================================

DEFAULT_SCALING = "log"

# In log mode:
#   non-negative feature
#   + skewness > threshold
#   -> log1p
# Then all features are standardized before PCA.
DEFAULT_LOG_SKEW_THRESHOLD = 1.0

# Parallel Analysis:
# independently permute each feature to preserve its marginal distribution
# while destroying cross-feature covariance.
DEFAULT_PA_ITERATIONS = 500
DEFAULT_PA_PERCENTILE = 95.0

ALLOWED_SCALING = (
    "raw",
    "zscore",
    "log",
    "robust",
)


@dataclass
class DimensionalityReductionConfig:
    """
    Configuration for the canonical MorphoGlia latent morphospace.

    Scientific design
    -----------------
    1. Scale the morphometric matrix.
    2. Fit the full PCA spectrum.
    3. Estimate supported linear dimensionality with:
         - Parallel Analysis
         - Broken-Stick criterion
    4. Independently estimate intrinsic dimensionality with Two-NN.
    5. If Parallel Analysis and Broken Stick disagree, retain the
       resulting plausible dimensionality interval.

    The final operational dimensionality is resolved downstream using
    clustering stability across that plausible interval.

    There is therefore no user-selected:
      - number of PCs
      - explained-variance cutoff
      - PCA dimensionality threshold
    """

    scaling: str | None = None

    log_skew_threshold: float | None = None

    parallel_analysis_iterations: int | None = None

    parallel_analysis_percentile: float | None = None


    # ------------------------------------------------------------------
    # Effective values
    # ------------------------------------------------------------------

    @property
    def effective_scaling(self) -> str:
        value = (
            DEFAULT_SCALING
            if self.scaling is None
            else self.scaling
        )

        if value not in ALLOWED_SCALING:
            raise ValueError(
                f"Unknown scaling mode: {value!r}. "
                f"Allowed: {ALLOWED_SCALING}"
            )

        return value


    @property
    def effective_log_skew_threshold(self) -> float:
        return (
            DEFAULT_LOG_SKEW_THRESHOLD
            if self.log_skew_threshold is None
            else float(self.log_skew_threshold)
        )


    @property
    def effective_parallel_analysis_iterations(self) -> int:
        value = (
            DEFAULT_PA_ITERATIONS
            if self.parallel_analysis_iterations is None
            else int(self.parallel_analysis_iterations)
        )

        if value < 1:
            raise ValueError(
                "parallel_analysis_iterations must be >= 1"
            )

        return value


    @property
    def effective_parallel_analysis_percentile(self) -> float:
        value = (
            DEFAULT_PA_PERCENTILE
            if self.parallel_analysis_percentile is None
            else float(self.parallel_analysis_percentile)
        )

        if not 0.0 < value < 100.0:
            raise ValueError(
                "parallel_analysis_percentile must be between 0 and 100"
            )

        return value


    # ------------------------------------------------------------------
    # Fixed analytical design
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        return "pca"


    @property
    def pca_mode(self) -> str:
        return "full"


    @property
    def dimensionality_criteria(self) -> tuple[str, str]:
        return (
            "parallel_analysis",
            "broken_stick",
        )


    @property
    def intrinsic_dimension_method(self) -> str:
        return "two_nn"


    @property
    def dimension_resolution(self) -> str:
        return "clustering_stability_across_plausible_interval"


    # ------------------------------------------------------------------
    # Technical Record
    # ------------------------------------------------------------------

    def configuration_rows(self) -> list[dict]:
        return [
            {
                "section": "Dimensionality Reduction",
                "parameter": "method",
                "value": self.method,
                "source": "default",
            },
            {
                "section": "Dimensionality Reduction",
                "parameter": "pca_mode",
                "value": self.pca_mode,
                "source": "default",
            },
            {
                "section": "Dimensionality Reduction",
                "parameter": "scaling",
                "value": self.effective_scaling,
                "source": (
                    "default"
                    if self.scaling is None
                    else "user"
                ),
            },
            {
                "section": "Dimensionality Reduction",
                "parameter": "log_skew_threshold",
                "value": self.effective_log_skew_threshold,
                "source": (
                    "default"
                    if self.log_skew_threshold is None
                    else "user"
                ),
            },
            {
                "section": "Dimensionality Reduction",
                "parameter": "parallel_analysis_iterations",
                "value": self.effective_parallel_analysis_iterations,
                "source": (
                    "default"
                    if self.parallel_analysis_iterations is None
                    else "user"
                ),
            },
            {
                "section": "Dimensionality Reduction",
                "parameter": "parallel_analysis_percentile",
                "value": self.effective_parallel_analysis_percentile,
                "source": (
                    "default"
                    if self.parallel_analysis_percentile is None
                    else "user"
                ),
            },
            {
                "section": "Dimensionality Reduction",
                "parameter": "dimensionality_criteria",
                "value": list(self.dimensionality_criteria),
                "source": "default",
            },
            {
                "section": "Dimensionality Reduction",
                "parameter": "intrinsic_dimension_method",
                "value": self.intrinsic_dimension_method,
                "source": "default",
            },
            {
                "section": "Dimensionality Reduction",
                "parameter": "dimension_resolution",
                "value": self.dimension_resolution,
                "source": "default",
            },
        ]



# ====================================================================
# CLUSTERING
# ====================================================================

from dataclasses import dataclass


# ======================================================================
# CANONICAL DEFAULTS
# ======================================================================

DEFAULT_COVARIANCE_TYPES = (
    "full",
    "diag",
    "tied",
    "spherical",
)

# Computational search ceiling.
#
# This is NOT a hypothesis that K=12 is biologically meaningful.
# MorphoGlia scans candidate coarse-grainings automatically and the
# stability landscape determines which resolutions are supported.
DEFAULT_K_MAX = 12

# Fast landscape estimate for every d × K.
DEFAULT_STABILITY_ITERATIONS = 50

# More detailed per-cell consensus for the selected robust resolutions.
DEFAULT_CONSENSUS_ITERATIONS = 100

# Validated perturbation:
# retain 80% of cells from each image / supplied stratum without replacement.
DEFAULT_SUBSAMPLE_FRACTION = 0.80


@dataclass
class ClusteringConfig:
    """
    Configuration for multiresolution morphology-state coarse-graining.

    Scientific design
    -----------------
    Clustering operates only in PCA space.

    For every plausible PCA dimensionality d:

        1. Scan candidate K values.
        2. For every exact (d, K), compare GMM covariance structures.
        3. Select covariance structure by minimum BIC.
        4. Keep the resulting partition; do NOT use BIC across dimensions.
        5. Estimate partition stability using structured subsampling
           without replacement.

    Then, for every K:

        6. Compare the same K across plausible PCA dimensionalities.
        7. Summarize its multiresolution stability.
        8. Choose the consensus-representative PCA dimensionality.
        9. Identify robust coarse-graining resolutions.
       10. Calculate detailed cell-level consensus reliability only for
           those selected resolutions.

    Important
    ---------
    - K=1 is included as a density-model null/reference.
    - K=1 is NOT treated as a meaningful stability winner because its
      partition is trivially stable.
    - UMAP is visualization only.
    - Category labels are never used for discovery.
    """

    # None -> estimate stable choices and use the automatic criterion.
    # Integer -> use one state count supported by a previous estimation.
    number_of_morphology_states: int | None = None

    k_max: int | None = None

    stability_iterations: int | None = None
    consensus_iterations: int | None = None

    subsample_fraction: float | None = None


    # ==================================================================
    # FIXED SCIENTIFIC DESIGN
    # ==================================================================

    @property
    def method(
        self,
    ) -> str:

        return "gmm_multiresolution_consensus"


    @property
    def k_min_model(
        self,
    ) -> int:

        return 1


    @property
    def k_min_coarse_graining(
        self,
    ) -> int:

        return 2


    @property
    def covariance_types(
        self,
    ) -> tuple[str, ...]:

        return DEFAULT_COVARIANCE_TYPES


    @property
    def model_selection(
        self,
    ) -> str:

        return "bic_within_exact_pca_dimension_and_k"


    @property
    def stability_method(
        self,
    ) -> str:

        return "stratified_subsampling_without_replacement"


    @property
    def dimension_agreement_method(
        self,
    ) -> str:

        return "same_k_adjusted_rand_index"


    @property
    def representative_dimension_rule(
        self,
    ) -> str:

        return "highest_mean_same_k_cross_dimension_ari"


    @property
    def robust_resolution_rule(
        self,
    ) -> str:

        return "stability_plateau_before_largest_drop"


    @property
    def strongest_resolution_rule(
        self,
    ) -> str:

        return (
            "maximin_stability_across_dimensions_then_"
            "dimension_agreement_then_mean_stability"
        )


    @property
    def automatic_morphology_state_rule(
        self,
    ) -> str:

        return (
            "first_stable_band_then_cross_dimension_agreement_"
            "then_worst_case_and_mean_resampling_stability_"
            "then_membership_reliability"
        )


    @property
    def requested_number_of_morphology_states(
        self,
    ) -> int | None:

        if self.number_of_morphology_states is None:
            return None

        raw_value = self.number_of_morphology_states
        if isinstance(raw_value, bool):
            raise ValueError(
                "number_of_morphology_states must be a whole number, not a boolean."
            )

        try:
            value = int(raw_value)
            is_whole = float(raw_value) == float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "number_of_morphology_states must be a whole number."
            ) from exc

        if not is_whole:
            raise ValueError(
                "number_of_morphology_states must be a whole number."
            )

        if value < self.k_min_coarse_graining:
            raise ValueError(
                "number_of_morphology_states must be at least 2."
            )

        return value


    # ==================================================================
    # EFFECTIVE VALUES
    # ==================================================================

    @property
    def effective_k_max(
        self,
    ) -> int:

        value = (
            DEFAULT_K_MAX
            if self.k_max is None
            else int(
                self.k_max
            )
        )

        if value < 2:

            raise ValueError(
                "k_max must be >= 2."
            )

        return value


    @property
    def effective_stability_iterations(
        self,
    ) -> int:

        value = (
            DEFAULT_STABILITY_ITERATIONS
            if self.stability_iterations is None
            else int(
                self.stability_iterations
            )
        )

        if value < 1:

            raise ValueError(
                "stability_iterations must be >= 1."
            )

        return value


    @property
    def effective_consensus_iterations(
        self,
    ) -> int:

        value = (
            DEFAULT_CONSENSUS_ITERATIONS
            if self.consensus_iterations is None
            else int(
                self.consensus_iterations
            )
        )

        if value < 1:

            raise ValueError(
                "consensus_iterations must be >= 1."
            )

        return value


    @property
    def effective_subsample_fraction(
        self,
    ) -> float:

        value = (
            DEFAULT_SUBSAMPLE_FRACTION
            if self.subsample_fraction is None
            else float(
                self.subsample_fraction
            )
        )

        if not 0.0 < value < 1.0:

            raise ValueError(
                "subsample_fraction must be between 0 and 1."
            )

        return value


    # ==================================================================
    # TECHNICAL RECORD
    # ==================================================================

    def configuration_rows(
        self,
    ) -> list[dict]:

        return [
            {
                "section": "Clustering",
                "parameter": "method",
                "value": self.method,
                "source": "default",
            },
            {
                "section": "Clustering",
                "parameter": "k_min_model",
                "value": self.k_min_model,
                "source": "default",
            },
            {
                "section": "Clustering",
                "parameter": "k_min_coarse_graining",
                "value": self.k_min_coarse_graining,
                "source": "default",
            },
            {
                "section": "Clustering",
                "parameter": "number_of_morphology_states",
                "value": (
                    "automatic"
                    if self.requested_number_of_morphology_states is None
                    else self.requested_number_of_morphology_states
                ),
                "source": (
                    "automatic"
                    if self.requested_number_of_morphology_states is None
                    else "user"
                ),
            },
            {
                "section": "Clustering",
                "parameter": "automatic_morphology_state_rule",
                "value": self.automatic_morphology_state_rule,
                "source": "automatic",
            },
            *(
                [
                    {
                        "section": "Clustering",
                        "parameter": "automatic_number_of_morphology_states",
                        "value": self.automatic_number_of_morphology_states,
                        "source": "automatic",
                    },
                    {
                        "section": "Clustering",
                        "parameter": "number_of_morphology_states_used_downstream",
                        "value": self.number_of_morphology_states_used_downstream,
                        "source": (
                            "automatic"
                            if self.morphology_state_selection_source
                            == "automatic_data_driven"
                            else "user"
                        ),
                    },
                    {
                        "section": "Clustering",
                        "parameter": "morphology_state_selection_source",
                        "value": self.morphology_state_selection_source,
                        "source": (
                            "automatic"
                            if self.morphology_state_selection_source
                            == "automatic_data_driven"
                            else "user"
                        ),
                    },
                ]
                if getattr(
                    self,
                    "automatic_number_of_morphology_states",
                    None,
                ) is not None
                else []
            ),
            {
                "section": "Clustering",
                "parameter": "k_max",
                "value": self.effective_k_max,
                "source": (
                    "default"
                    if self.k_max is None
                    else "user"
                ),
            },
            {
                "section": "Clustering",
                "parameter": "covariance_types",
                "value": list(
                    self.covariance_types
                ),
                "source": "default",
            },
            {
                "section": "Clustering",
                "parameter": "model_selection",
                "value": self.model_selection,
                "source": "default",
            },
            {
                "section": "Clustering",
                "parameter": "stability_method",
                "value": self.stability_method,
                "source": "default",
            },
            {
                "section": "Clustering",
                "parameter": "stability_iterations",
                "value": self.effective_stability_iterations,
                "source": (
                    "default"
                    if self.stability_iterations is None
                    else "user"
                ),
            },
            {
                "section": "Clustering",
                "parameter": "consensus_iterations",
                "value": self.effective_consensus_iterations,
                "source": (
                    "default"
                    if self.consensus_iterations is None
                    else "user"
                ),
            },
            {
                "section": "Clustering",
                "parameter": "subsample_fraction",
                "value": self.effective_subsample_fraction,
                "source": (
                    "default"
                    if self.subsample_fraction is None
                    else "user"
                ),
            },
            {
                "section": "Clustering",
                "parameter": "dimension_agreement_method",
                "value": self.dimension_agreement_method,
                "source": "default",
            },
            {
                "section": "Clustering",
                "parameter": "representative_dimension_rule",
                "value": self.representative_dimension_rule,
                "source": "default",
            },
            {
                "section": "Clustering",
                "parameter": "robust_resolution_rule",
                "value": self.robust_resolution_rule,
                "source": "default",
            },
            {
                "section": "Clustering",
                "parameter": "strongest_resolution_rule",
                "value": self.strongest_resolution_rule,
                "source": "default",
            },
        ]
