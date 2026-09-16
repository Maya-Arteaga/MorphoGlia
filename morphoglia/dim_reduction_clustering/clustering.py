from __future__ import annotations

from dataclasses import dataclass

from joblib import (
    Parallel,
    delayed,
    parallel_backend,
)

import numpy as np
import pandas as pd

from threadpoolctl import (
    threadpool_limits,
)

from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from .config import ClusteringConfig

from .dimensionality import (
    DimensionalityReductionResult,
)


# ======================================================================
# NUMERICAL FITTING SETTINGS
#
# These are optimization settings, not biological parameters.
# ======================================================================

FULL_DATA_N_INIT = 10
SUBSAMPLE_N_INIT = 3

GMM_MAX_ITER = 500
GMM_REG_COVAR = 1e-6


# ======================================================================
# RESULT OBJECTS
# ======================================================================

@dataclass
class SolutionClusteringResult:
    """
    Full-data solution for one exact PCA dimensionality × K.
    """

    pca_dimensions: int
    k: int

    covariance_type: str

    bic: float
    icl: float

    labels: np.ndarray
    cluster_probability: np.ndarray


@dataclass
class SelectedResolutionResult:
    """
    Final detailed result for one robust K at its representative d.
    """

    k: int
    pca_dimensions: int

    covariance_type: str

    labels: np.ndarray
    cluster_probability: np.ndarray
    consensus_reliability: np.ndarray

    mean_subsample_ari: float
    std_subsample_ari: float


@dataclass
class ClusteringResult:
    """
    Complete MorphoGlia multiresolution clustering result.
    """

    model_scan: pd.DataFrame

    candidate_assignments: pd.DataFrame

    stability_by_dimension_and_k: pd.DataFrame

    dimension_agreement: pd.DataFrame

    multiresolution_summary: pd.DataFrame

    representative_dimensions: dict[int, int]

    robust_k_values: tuple[int, ...]

    strongest_k: int | None

    by_solution: dict[
        tuple[int, int],
        SolutionClusteringResult,
    ]

    selected_solutions: dict[
        int,
        SelectedResolutionResult,
    ]


    # Decision layer.
    #
    # strongest_k is retained as the historical maximin solution for
    # backward compatibility.
    pareto_k_values: tuple[int, ...] = ()
    preferred_k: int | None = None
    coarse_k: int | None = None
    maximin_k: int | None = None


# ======================================================================
# ENGINE
# ======================================================================

class ClusteringEngine:
    """
    GMM multiresolution consensus clustering.

    This engine implements the analytical logic validated during the
    MorphoGlia astrocyte/microglia development tests.

    Workflow
    --------
    A. Full-data landscape
       For every plausible PCA dimensionality d and every candidate K:
           - fit all supported GMM covariance structures
           - choose covariance by BIC
           - retain labels and membership probability

    B. Stability landscape
       For every exact d,K with K >= 2:
           - subsample WITHOUT replacement
           - preserve supplied strata independently
           - refit the selected covariance architecture
           - predict all original cells
           - calculate ARI against the reference partition

    C. Cross-dimensional agreement
       Compare the same K across all plausible PCA dimensions.

    D. Multiresolution coarse-graining
       Summarize each K across dimensions and identify the stable
       coarse-graining range.

    E. Detailed selected solutions
       For each robust K:
           - choose its consensus-representative dimensionality
           - calculate per-cell consensus reliability
    """

    def __init__(
        self,
        config: ClusteringConfig,
        random_state: int = 24,
        n_jobs: int = 1,
    ):

        self.config = config

        self.random_state = int(
            random_state
        )

        self.n_jobs = max(
            1,
            int(
                n_jobs
            ),
        )


    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def fit(
        self,
        dimensionality: DimensionalityReductionResult,
        strata: np.ndarray | pd.Series | list | None = None,
    ) -> ClusteringResult:

        scores = np.asarray(
            dimensionality.scores,
            dtype=float,
        )


        if scores.ndim != 2:

            raise ValueError(
                "PCA scores must be a 2D array."
            )


        if scores.shape[0] < 3:

            raise ValueError(
                "At least 3 cells are required for clustering."
            )


        if not np.isfinite(
            scores
        ).all():

            raise ValueError(
                "PCA scores contain non-finite values."
            )


        d_min = int(
            dimensionality.plausible_dimension_min
        )

        d_max = int(
            dimensionality.plausible_dimension_max
        )


        if d_min < 1:

            raise ValueError(
                "Plausible PCA dimensionality must start at >= 1."
            )


        if d_max > scores.shape[1]:

            raise ValueError(
                "Plausible PCA dimensionality exceeds "
                "available PCA components."
            )


        dimensions = list(
            range(
                d_min,
                d_max + 1,
            )
        )


        n_cells = scores.shape[0]


        effective_k_max = min(
            self.config.effective_k_max,
            max(
                1,
                n_cells - 1,
            ),
        )


        k_values = list(
            range(
                self.config.k_min_model,
                effective_k_max + 1,
            )
        )


        strata_array = (
            self._validate_strata(
                strata,
                n_cells,
            )
        )


        # ==============================================================
        # A. FULL-DATA d × K LANDSCAPE
        #
        # Every exact (dimension, K) solution is independent.
        # Parallelize at this level while keeping each numerical worker
        # single-threaded.
        # ==============================================================

        full_jobs = [
            (
                scores,
                dimension,
                k,
            )
            for dimension in dimensions
            for k in k_values
        ]


        full_results = (
            self._parallel_execute(
                self._fit_solution_job,
                full_jobs,
            )
        )


        model_scan_rows = []

        by_solution = {}


        for (
            dimension,
            k,
            scan_rows,
            solution,
        ) in full_results:

            model_scan_rows.extend(
                scan_rows
            )

            by_solution[
                (
                    dimension,
                    k,
                )
            ] = solution


        model_scan = pd.DataFrame(
            model_scan_rows
        )


        if not model_scan.empty:

            model_scan = (
                model_scan
                .sort_values(
                    [
                        "pca_dimensions",
                        "k",
                        "covariance_type",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )


        # ==============================================================
        # B. STABILITY LANDSCAPE
        #
        # One (dimension, K) stability experiment remains internally
        # identical and deterministic. Independent experiments run in
        # parallel.
        # ==============================================================

        stability_jobs = []


        for dimension in dimensions:

            for k in k_values:

                # K=1 is a null density model.
                # Its ARI would be trivially 1 and therefore misleading.
                if (
                    k
                    < self.config.k_min_coarse_graining
                ):

                    continue


                solution = by_solution[
                    (
                        dimension,
                        k,
                    )
                ]


                stability_jobs.append(
                    (
                        scores,
                        dimension,
                        k,
                        solution.covariance_type,
                        solution.labels,
                        solution.cluster_probability,
                        strata_array,
                        self.config
                        .effective_stability_iterations,
                    )
                )


        stability_rows = (
            self._parallel_execute(
                self._stability_job,
                stability_jobs,
            )
        )


        stability = pd.DataFrame(
            stability_rows
        )


        if not stability.empty:

            stability = (
                stability
                .sort_values(
                    [
                        "k",
                        "pca_dimensions",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )


        # ==============================================================
        # C. SAME-K CROSS-DIMENSION AGREEMENT
        # ==============================================================

        dimension_agreement = (
            self._dimension_agreement(
                by_solution=by_solution,
                dimensions=dimensions,
                k_values=[
                    k
                    for k in k_values
                    if (
                        k
                        >= self.config
                        .k_min_coarse_graining
                    )
                ],
            )
        )


        # ==============================================================
        # D. MULTIRESOLUTION SUMMARY
        # ==============================================================

        multiresolution_summary = (
            self._multiresolution_summary(
                stability=stability,
                dimension_agreement=(
                    dimension_agreement
                ),
                by_solution=by_solution,
                dimensions=dimensions,
            )
        )


        representative_dimensions = (
            self._representative_dimensions(
                dimension_agreement=(
                    dimension_agreement
                ),
                dimensions=dimensions,
                k_values=(
                    multiresolution_summary[
                        "k"
                    ].astype(
                        int
                    ).tolist()
                ),
            )
        )


        robust_k_values = (
            self._robust_k_values(
                multiresolution_summary
            )
        )


        strongest_k = (
            self._strongest_k(
                multiresolution_summary,
                robust_k_values,
            )
        )


        # ==============================================================
        # E. DETAILED CONSENSUS FOR ROBUST RESOLUTIONS ONLY
        #
        # Robust K solutions are independent and can be evaluated
        # concurrently.
        # ==============================================================

        consensus_jobs = []


        for k in robust_k_values:

            dimension = int(
                representative_dimensions[
                    k
                ]
            )


            solution = by_solution[
                (
                    dimension,
                    k,
                )
            ]


            consensus_jobs.append(
                (
                    scores,
                    k,
                    dimension,
                    solution.covariance_type,
                    solution.labels,
                    solution.cluster_probability,
                    strata_array,
                    self.config
                    .effective_consensus_iterations,
                )
            )


        consensus_results = (
            self._parallel_execute(
                self._consensus_job,
                consensus_jobs,
            )
        )


        selected_solutions = {
            int(k): solution
            for k, solution
            in consensus_results
        }


        # ==============================================================
        # CANDIDATE CELL ASSIGNMENTS
        # ==============================================================

        candidate_assignments = (
            self._candidate_assignments(
                by_solution=by_solution,
            )
        )


        return ClusteringResult(
            model_scan=model_scan,
            candidate_assignments=(
                candidate_assignments
            ),
            stability_by_dimension_and_k=(
                stability
            ),
            dimension_agreement=(
                dimension_agreement
            ),
            multiresolution_summary=(
                multiresolution_summary
            ),
            representative_dimensions=(
                representative_dimensions
            ),
            robust_k_values=tuple(
                robust_k_values
            ),
            strongest_k=strongest_k,
            by_solution=by_solution,
            selected_solutions=(
                selected_solutions
            ),
        )


    # ==================================================================
    # PARALLEL EXECUTION
    # ==================================================================

    def _parallel_execute(
        self,
        function,
        jobs,
    ):
        """
        Execute independent clustering jobs.

        The scientific unit of parallelism is one independent analysis
        job, while BLAS/OpenMP inside each worker is constrained to one
        native thread.

        This prevents oversubscription from small GMM fits.
        """

        jobs = list(
            jobs
        )


        if not jobs:

            return []


        workers = min(
            self.n_jobs,
            len(
                jobs
            ),
        )


        # --------------------------------------------------------------
        # Deterministic serial reference path.
        # --------------------------------------------------------------

        if workers <= 1:

            with threadpool_limits(
                limits=1
            ):

                return [
                    function(
                        *job
                    )
                    for job in jobs
                ]


        # --------------------------------------------------------------
        # Process-level parallelism.
        #
        # loky isolates independent GMM jobs. inner_max_num_threads=1
        # prevents each worker from starting its own multi-threaded
        # BLAS/OpenMP pool.
        # --------------------------------------------------------------

        with parallel_backend(
            "loky",
            inner_max_num_threads=1,
        ):

            return Parallel(
                n_jobs=workers,
                batch_size=1,
            )(
                delayed(
                    function
                )(
                    *job
                )
                for job in jobs
            )


    # ==================================================================
    # ONE FULL-DATA (d, K) JOB
    # ==================================================================

    def _fit_solution_job(
        self,
        scores: np.ndarray,
        dimension: int,
        k: int,
    ):

        X = scores[
            :,
            :dimension,
        ]


        (
            best_model,
            scan_rows,
        ) = self._fit_best_covariance(
            X=X,
            dimension=dimension,
            k=k,
        )


        raw_labels = (
            best_model.predict(
                X
            )
        )


        labels = (
            self._canonicalize_labels(
                model=best_model,
                labels=raw_labels,
            )
        )


        probabilities = (
            best_model
            .predict_proba(
                X
            )
            .max(
                axis=1
            )
        )


        bic = float(
            best_model.bic(
                X
            )
        )


        icl = float(
            self._icl(
                best_model,
                X,
            )
        )


        solution = (
            SolutionClusteringResult(
                pca_dimensions=dimension,
                k=k,
                covariance_type=str(
                    best_model.covariance_type
                ),
                bic=bic,
                icl=icl,
                labels=labels,
                cluster_probability=probabilities,
            )
        )


        return (
            dimension,
            k,
            scan_rows,
            solution,
        )


    # ==================================================================
    # ONE STABILITY (d, K) JOB
    # ==================================================================

    def _stability_job(
        self,
        scores: np.ndarray,
        dimension: int,
        k: int,
        covariance_type: str,
        reference_labels: np.ndarray,
        cluster_probability: np.ndarray,
        strata: np.ndarray | None,
        iterations: int,
    ) -> dict:

        X = scores[
            :,
            :dimension,
        ]


        ari_values = (
            self._subsample_ari(
                X=X,
                k=k,
                covariance_type=(
                    covariance_type
                ),
                reference_labels=(
                    reference_labels
                ),
                strata=strata,
                iterations=iterations,
                seed_offset=(
                    dimension
                    * 10_000
                    + k
                    * 100
                ),
            )
        )


        cluster_sizes = (
            pd.Series(
                reference_labels
            )
            .value_counts()
        )


        return {
            "pca_dimensions":
                dimension,

            "k":
                k,

            "covariance_type":
                covariance_type,

            "mean_subsample_ari":
                float(
                    np.mean(
                        ari_values
                    )
                ),

            "std_subsample_ari":
                float(
                    np.std(
                        ari_values,
                        ddof=0,
                    )
                ),

            "minimum_subsample_ari":
                float(
                    np.min(
                        ari_values
                    )
                ),

            "maximum_subsample_ari":
                float(
                    np.max(
                        ari_values
                    )
                ),

            "valid_iterations":
                int(
                    len(
                        ari_values
                    )
                ),

            "minimum_cluster_size":
                int(
                    cluster_sizes.min()
                ),

            "singleton_count":
                int(
                    (
                        cluster_sizes
                        == 1
                    )
                    .sum()
                ),

            "mean_membership_probability":
                float(
                    np.mean(
                        cluster_probability
                    )
                ),
        }


    # ==================================================================
    # ONE DETAILED CONSENSUS JOB
    # ==================================================================

    def _consensus_job(
        self,
        scores: np.ndarray,
        k: int,
        dimension: int,
        covariance_type: str,
        reference_labels: np.ndarray,
        cluster_probability: np.ndarray,
        strata: np.ndarray | None,
        iterations: int,
    ):

        X = scores[
            :,
            :dimension,
        ]


        (
            reliability,
            ari_values,
        ) = self._detailed_consensus(
            X=X,
            k=k,
            covariance_type=(
                covariance_type
            ),
            reference_labels=(
                reference_labels
            ),
            strata=strata,
            iterations=iterations,
            seed_offset=(
                1_000_000
                + dimension
                * 10_000
                + k
                * 100
            ),
        )


        solution = (
            SelectedResolutionResult(
                k=k,
                pca_dimensions=dimension,
                covariance_type=(
                    covariance_type
                ),
                labels=(
                    reference_labels.copy()
                ),
                cluster_probability=(
                    cluster_probability.copy()
                ),
                consensus_reliability=(
                    reliability
                ),
                mean_subsample_ari=float(
                    np.mean(
                        ari_values
                    )
                ),
                std_subsample_ari=float(
                    np.std(
                        ari_values,
                        ddof=0,
                    )
                ),
            )
        )


        return (
            k,
            solution,
        )


    # ==================================================================
    # FULL-DATA MODEL FITTING
    # ==================================================================

    def _fit_best_covariance(
        self,
        X: np.ndarray,
        dimension: int,
        k: int,
    ) -> tuple[
        GaussianMixture,
        list[dict],
    ]:

        rows = []

        fitted = []


        covariance_rank = {
            name: index
            for index, name in enumerate(
                self.config.covariance_types
            )
        }


        for covariance_type in (
            self.config.covariance_types
        ):

            try:

                model = GaussianMixture(
                    n_components=k,
                    covariance_type=(
                        covariance_type
                    ),
                    random_state=(
                        self.random_state
                    ),
                    n_init=FULL_DATA_N_INIT,
                    reg_covar=GMM_REG_COVAR,
                    max_iter=GMM_MAX_ITER,
                )


                model.fit(
                    X
                )


                bic = float(
                    model.bic(
                        X
                    )
                )


                icl = float(
                    self._icl(
                        model,
                        X,
                    )
                )


                rows.append(
                    {
                        "pca_dimensions": (
                            dimension
                        ),
                        "k": k,
                        "covariance_type": (
                            covariance_type
                        ),
                        "bic": bic,
                        "icl": icl,
                        "converged": bool(
                            model.converged_
                        ),
                        "error": "",
                    }
                )


                if bool(
                    model.converged_
                ):

                    fitted.append(
                        (
                            bic,
                            covariance_rank[
                                covariance_type
                            ],
                            model,
                        )
                    )


            except Exception as exc:

                rows.append(
                    {
                        "pca_dimensions": (
                            dimension
                        ),
                        "k": k,
                        "covariance_type": (
                            covariance_type
                        ),
                        "bic": np.nan,
                        "icl": np.nan,
                        "converged": False,
                        "error": str(
                            exc
                        ),
                    }
                )


        if not fitted:

            raise RuntimeError(
                f"No valid GMM converged for "
                f"d={dimension}, K={k}."
            )


        fitted.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )


        best_model = fitted[
            0
        ][
            2
        ]


        return (
            best_model,
            rows,
        )


    # ==================================================================
    # ICL
    # ==================================================================

    @staticmethod
    def _icl(
        model: GaussianMixture,
        X: np.ndarray,
    ) -> float:
        """
        Approximate Integrated Completed Likelihood.

        Lower is better.

        ICL = BIC + 2 * classification entropy
        """

        probabilities = (
            model.predict_proba(
                X
            )
        )


        epsilon = (
            np.finfo(
                float
            ).eps
        )


        entropy = float(
            -np.sum(
                probabilities
                * np.log(
                    probabilities
                    + epsilon
                )
            )
        )


        return float(
            model.bic(
                X
            )
            + 2.0
            * entropy
        )


    # ==================================================================
    # STRATA
    # ==================================================================

    @staticmethod
    def _validate_strata(
        strata,
        n_cells: int,
    ) -> np.ndarray | None:

        if strata is None:

            return None


        values = np.asarray(
            strata
        )


        if values.ndim != 1:

            raise ValueError(
                "strata must be one-dimensional."
            )


        if len(
            values
        ) != n_cells:

            raise ValueError(
                "strata length must equal "
                "the number of cells."
            )


        return values.astype(
            str
        )


    # ==================================================================
    # SUBSAMPLING
    # ==================================================================

    def _draw_subsample_indices(
        self,
        n_cells: int,
        strata: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:

        fraction = (
            self.config
            .effective_subsample_fraction
        )


        # --------------------------------------------------------------
        # No strata supplied:
        # one global subsample without replacement.
        # --------------------------------------------------------------

        if strata is None:

            sample_size = max(
                1,
                int(
                    round(
                        fraction
                        * n_cells
                    )
                ),
            )


            return np.sort(
                rng.choice(
                    np.arange(
                        n_cells
                    ),
                    size=sample_size,
                    replace=False,
                )
            )


        # --------------------------------------------------------------
        # Structured subsampling:
        # independently preserve every supplied stratum.
        # --------------------------------------------------------------

        selected_parts = []


        for stratum in np.unique(
            strata
        ):

            indices = np.flatnonzero(
                strata
                == stratum
            )


            sample_size = max(
                1,
                int(
                    round(
                        fraction
                        * len(
                            indices
                        )
                    )
                ),
            )


            selected = rng.choice(
                indices,
                size=sample_size,
                replace=False,
            )


            selected_parts.append(
                selected
            )


        return np.sort(
            np.concatenate(
                selected_parts
            )
        )


    # ==================================================================
    # LANDSCAPE STABILITY
    # ==================================================================

    def _subsample_ari(
        self,
        X: np.ndarray,
        k: int,
        covariance_type: str,
        reference_labels: np.ndarray,
        strata: np.ndarray | None,
        iterations: int,
        seed_offset: int,
    ) -> np.ndarray:

        rng = np.random.default_rng(
            self.random_state
            + int(
                seed_offset
            )
        )


        ari_values = []


        for _ in range(
            iterations
        ):

            indices = (
                self._draw_subsample_indices(
                    n_cells=X.shape[0],
                    strata=strata,
                    rng=rng,
                )
            )


            if len(
                indices
            ) <= k:

                continue


            try:

                model = GaussianMixture(
                    n_components=k,
                    covariance_type=(
                        covariance_type
                    ),
                    random_state=int(
                        rng.integers(
                            0,
                            2**31 - 1,
                        )
                    ),
                    n_init=SUBSAMPLE_N_INIT,
                    reg_covar=GMM_REG_COVAR,
                    max_iter=GMM_MAX_ITER,
                )


                model.fit(
                    X[
                        indices
                    ]
                )


                if not bool(
                    model.converged_
                ):

                    continue


                predicted = model.predict(
                    X
                )


                ari_values.append(
                    adjusted_rand_score(
                        reference_labels,
                        predicted,
                    )
                )


            except Exception:

                continue


        if not ari_values:

            raise RuntimeError(
                f"All structured subsampling fits failed "
                f"for K={k}."
            )


        return np.asarray(
            ari_values,
            dtype=float,
        )


    # ==================================================================
    # DETAILED CELL CONSENSUS
    # ==================================================================

    def _detailed_consensus(
        self,
        X: np.ndarray,
        k: int,
        covariance_type: str,
        reference_labels: np.ndarray,
        strata: np.ndarray | None,
        iterations: int,
        seed_offset: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Calculate detailed per-cell consensus reliability.

        Reliability for one cell is its mean co-assignment with the
        other members of its reference cluster across repeated
        structured subsamples.

        A full N×N matrix is deliberately avoided.
        """

        n_cells = X.shape[0]


        reliability_sum = np.zeros(
            n_cells,
            dtype=float,
        )


        reliability_count = np.zeros(
            n_cells,
            dtype=int,
        )


        ari_values = []


        rng = np.random.default_rng(
            self.random_state
            + int(
                seed_offset
            )
        )


        reference_clusters = np.unique(
            reference_labels
        )


        for _ in range(
            iterations
        ):

            indices = (
                self._draw_subsample_indices(
                    n_cells=n_cells,
                    strata=strata,
                    rng=rng,
                )
            )


            if len(
                indices
            ) <= k:

                continue


            try:

                model = GaussianMixture(
                    n_components=k,
                    covariance_type=(
                        covariance_type
                    ),
                    random_state=int(
                        rng.integers(
                            0,
                            2**31 - 1,
                        )
                    ),
                    n_init=SUBSAMPLE_N_INIT,
                    reg_covar=GMM_REG_COVAR,
                    max_iter=GMM_MAX_ITER,
                )


                model.fit(
                    X[
                        indices
                    ]
                )


                if not bool(
                    model.converged_
                ):

                    continue


                predicted = model.predict(
                    X
                )


                ari_values.append(
                    adjusted_rand_score(
                        reference_labels,
                        predicted,
                    )
                )


                # ------------------------------------------------------
                # Cell-level within-reference-cluster co-assignment
                # ------------------------------------------------------

                for reference_cluster in (
                    reference_clusters
                ):

                    members = np.flatnonzero(
                        reference_labels
                        == reference_cluster
                    )


                    if len(
                        members
                    ) <= 1:

                        continue


                    predicted_members = (
                        predicted[
                            members
                        ]
                    )


                    (
                        predicted_groups,
                        counts,
                    ) = np.unique(
                        predicted_members,
                        return_counts=True,
                    )


                    count_lookup = {
                        int(group): int(
                            count
                        )
                        for group, count in zip(
                            predicted_groups,
                            counts,
                        )
                    }


                    for predicted_group in (
                        predicted_groups
                    ):

                        local_members = members[
                            predicted_members
                            == predicted_group
                        ]


                        same_group_count = (
                            count_lookup[
                                int(
                                    predicted_group
                                )
                            ]
                            - 1
                        )


                        reliability_value = (
                            same_group_count
                            / (
                                len(
                                    members
                                )
                                - 1
                            )
                        )


                        reliability_sum[
                            local_members
                        ] += (
                            reliability_value
                        )


                        reliability_count[
                            local_members
                        ] += 1


            except Exception:

                continue


        if not ari_values:

            raise RuntimeError(
                f"All detailed consensus fits failed "
                f"for K={k}."
            )


        reliability = np.full(
            n_cells,
            np.nan,
            dtype=float,
        )


        valid = (
            reliability_count
            > 0
        )


        reliability[
            valid
        ] = (
            reliability_sum[
                valid
            ]
            / reliability_count[
                valid
            ]
        )


        return (
            reliability,
            np.asarray(
                ari_values,
                dtype=float,
            ),
        )


    # ==================================================================
    # SAME-K CROSS-DIMENSION AGREEMENT
    # ==================================================================

    @staticmethod
    def _dimension_agreement(
        by_solution: dict,
        dimensions: list[int],
        k_values: list[int],
    ) -> pd.DataFrame:

        rows = []


        for k in k_values:

            for d1 in dimensions:

                labels_1 = (
                    by_solution[
                        (
                            d1,
                            k,
                        )
                    ]
                    .labels
                )


                for d2 in dimensions:

                    labels_2 = (
                        by_solution[
                            (
                                d2,
                                k,
                            )
                        ]
                        .labels
                    )


                    rows.append(
                        {
                            "k": k,
                            "dimension_1": d1,
                            "dimension_2": d2,
                            "adjusted_rand_index": float(
                                adjusted_rand_score(
                                    labels_1,
                                    labels_2,
                                )
                            ),
                        }
                    )


        return (
            pd.DataFrame(
                rows
            )
            .sort_values(
                [
                    "k",
                    "dimension_1",
                    "dimension_2",
                ]
            )
            .reset_index(
                drop=True
            )
        )


    # ==================================================================
    # MULTIRESOLUTION SUMMARY
    # ==================================================================

    @staticmethod
    def _multiresolution_summary(
        stability: pd.DataFrame,
        dimension_agreement: pd.DataFrame,
        by_solution: dict,
        dimensions: list[int],
    ) -> pd.DataFrame:

        rows = []


        if stability.empty:

            return pd.DataFrame()


        for k in sorted(
            stability[
                "k"
            ].unique()
        ):

            stability_k = stability[
                stability[
                    "k"
                ]
                == k
            ]


            agreement_k = (
                dimension_agreement[
                    (
                        dimension_agreement[
                            "k"
                        ]
                        == k
                    )
                    &
                    (
                        dimension_agreement[
                            "dimension_1"
                        ]
                        != dimension_agreement[
                            "dimension_2"
                        ]
                    )
                ]
            )


            probabilities = []


            bic_wins = 0
            icl_wins = 0


            for dimension in dimensions:

                solution = by_solution[
                    (
                        dimension,
                        int(
                            k
                        ),
                    )
                ]


                probabilities.append(
                    float(
                        np.mean(
                            solution
                            .cluster_probability
                        )
                    )
                )


                same_dimension = [
                    result
                    for (
                        d,
                        candidate_k
                    ),
                    result
                    in by_solution.items()
                    if d == dimension
                ]


                bic_winner = min(
                    same_dimension,
                    key=lambda result: (
                        result.bic,
                        result.k,
                    ),
                )


                icl_winner = min(
                    same_dimension,
                    key=lambda result: (
                        result.icl,
                        result.k,
                    ),
                )


                if (
                    bic_winner.k
                    == k
                ):

                    bic_wins += 1


                if (
                    icl_winner.k
                    == k
                ):

                    icl_wins += 1


            rows.append(
                {
                    "k": int(
                        k
                    ),
                    "mean_subsample_ari": float(
                        stability_k[
                            "mean_subsample_ari"
                        ].mean()
                    ),
                    "minimum_subsample_ari": float(
                        stability_k[
                            "mean_subsample_ari"
                        ].min()
                    ),
                    "mean_dimension_agreement": float(
                        agreement_k[
                            "adjusted_rand_index"
                        ].mean()
                    ),
                    "minimum_dimension_agreement": float(
                        agreement_k[
                            "adjusted_rand_index"
                        ].min()
                    ),
                    "minimum_cluster_size_across_d": int(
                        stability_k[
                            "minimum_cluster_size"
                        ].min()
                    ),
                    "maximum_singletons_across_d": int(
                        stability_k[
                            "singleton_count"
                        ].max()
                    ),
                    "mean_membership_probability": float(
                        np.mean(
                            probabilities
                        )
                    ),
                    "bic_winner_dimensions": int(
                        bic_wins
                    ),
                    "icl_winner_dimensions": int(
                        icl_wins
                    ),
                }
            )


        return pd.DataFrame(
            rows
        )


    # ==================================================================
    # REPRESENTATIVE DIMENSION
    # ==================================================================

    @staticmethod
    def _representative_dimensions(
        dimension_agreement: pd.DataFrame,
        dimensions: list[int],
        k_values: list[int],
    ) -> dict[int, int]:

        result = {}


        for k in k_values:

            subset = dimension_agreement[
                (
                    dimension_agreement[
                        "k"
                    ]
                    == k
                )
                &
                (
                    dimension_agreement[
                        "dimension_1"
                    ]
                    != dimension_agreement[
                        "dimension_2"
                    ]
                )
            ]


            if subset.empty:

                result[
                    int(
                        k
                    )
                ] = int(
                    min(
                        dimensions
                    )
                )

                continue


            scores = (
                subset
                .groupby(
                    "dimension_1"
                )[
                    "adjusted_rand_index"
                ]
                .mean()
                .reset_index()
                .sort_values(
                    [
                        "adjusted_rand_index",
                        "dimension_1",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
            )


            result[
                int(
                    k
                )
            ] = int(
                scores.iloc[
                    0
                ][
                    "dimension_1"
                ]
            )


        return result


    # ==================================================================
    # ROBUST K FAMILY
    # ==================================================================

    @staticmethod
    def _robust_k_values(
        summary: pd.DataFrame,
    ) -> list[int]:
        """
        Identify the contiguous low-K stability plateau.

        Rule
        ----
        Starting at the smallest biological coarse-graining K, locate
        the largest decrease in mean subsampling stability between
        consecutive K values.

        The robust family ends immediately before that drop.

        Example:
            K2  0.82
            K3  0.79
            K4  0.83
            K5  0.56  <- largest drop

        robust family = K2, K3, K4
        """

        if summary.empty:

            return []


        ranked = (
            summary
            .sort_values(
                "k"
            )
            .reset_index(
                drop=True
            )
        )


        if len(
            ranked
        ) == 1:

            return [
                int(
                    ranked.iloc[
                        0
                    ][
                        "k"
                    ]
                )
            ]


        stability = ranked[
            "mean_subsample_ari"
        ].to_numpy(
            dtype=float
        )


        drops = (
            stability[
                :-1
            ]
            - stability[
                1:
            ]
        )


        largest_drop_index = int(
            np.argmax(
                drops
            )
        )


        # If no positive drop exists, retain all scanned resolutions.
        if (
            drops[
                largest_drop_index
            ]
            <= 0
        ):

            cutoff_index = (
                len(
                    ranked
                )
                - 1
            )

        else:

            cutoff_index = (
                largest_drop_index
            )


        return [
            int(
                value
            )
            for value in ranked.loc[
                :cutoff_index,
                "k",
            ].tolist()
        ]


    # ==================================================================
    # STRONGEST RESOLUTION
    # ==================================================================

    @staticmethod
    def _strongest_k(
        summary: pd.DataFrame,
        robust_k_values: list[int],
    ) -> int | None:

        if not robust_k_values:

            return None


        candidates = (
            summary[
                summary[
                    "k"
                ].isin(
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
            )
            .reset_index(
                drop=True
            )
        )


        return int(
            candidates.iloc[
                0
            ][
                "k"
            ]
        )


    # ==================================================================
    # CANDIDATE ASSIGNMENT TABLE
    # ==================================================================

    @staticmethod
    def _candidate_assignments(
        by_solution: dict[
            tuple[int, int],
            SolutionClusteringResult,
        ],
    ) -> pd.DataFrame:

        tables = []


        for (
            dimension,
            k,
        ) in sorted(
            by_solution
        ):

            result = by_solution[
                (
                    dimension,
                    k,
                )
            ]


            n_cells = len(
                result.labels
            )


            tables.append(
                pd.DataFrame(
                    {
                        "row_index": np.arange(
                            n_cells,
                            dtype=int,
                        ),
                        "pca_dimensions": (
                            dimension
                        ),
                        "k": k,
                        "cluster": (
                            result.labels
                        ),
                        "cluster_probability": (
                            result
                            .cluster_probability
                        ),
                        "covariance_type": (
                            result
                            .covariance_type
                        ),
                    }
                )
            )


        if not tables:

            return pd.DataFrame()


        return pd.concat(
            tables,
            ignore_index=True,
        )


    # ==================================================================
    # DETERMINISTIC HUMAN-FACING CLUSTER IDs
    # ==================================================================

    @staticmethod
    def _canonicalize_labels(
        model: GaussianMixture,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Order GMM components by their mean position along PC1.

        Output IDs are 1..K.
        """

        order = np.argsort(
            model.means_[
                :,
                0,
            ]
        )


        mapping = {
            int(
                component
            ): int(
                rank
                + 1
            )
            for rank, component in enumerate(
                order
            )
        }


        return np.asarray(
            [
                mapping[
                    int(
                        label
                    )
                ]
                for label in labels
            ],
            dtype=int,
        )


# MG_SUPPORTED_RESOLUTION_OVERRIDE
from .resolution_support import evaluate_resolution_support as _mg_resolution_support


def _mg_supported_k_values(self, summary):
    return list(_mg_resolution_support(summary).supported_k_values)


ClusteringEngine._robust_k_values = _mg_supported_k_values
