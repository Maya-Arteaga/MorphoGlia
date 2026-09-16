from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from skimage.morphology import skeletonize


# ======================================================================
# RESULT
# ======================================================================

@dataclass
class TubularQCResult:
    table: pd.DataFrame

    candidate_z_threshold: float
    candidate_count: int

    score_threshold: float
    suspicious_count: int


# ======================================================================
# MASK GEOMETRY
# ======================================================================

def _binary_mask(
    roi: np.ndarray,
) -> np.ndarray:
    """
    Convert a clean MorphoGlia ROI into a boolean foreground mask.

    Canonical clean ROI convention:

        foreground > 0
        background == 0
    """

    array = np.asarray(
        roi
    )

    if array.ndim == 3:
        array = cv2.cvtColor(
            array,
            cv2.COLOR_BGR2GRAY,
        )

    return (
        array > 0
    )


def _elongation_from_mask(
    mask: np.ndarray,
) -> float:
    """
    PCA elongation of foreground pixel coordinates.

    1.0:
        approximately isotropic

    larger:
        increasingly elongated
    """

    points = np.column_stack(
        np.nonzero(
            mask
        )
    ).astype(
        float
    )

    if len(points) < 3:
        return np.nan


    covariance = np.cov(
        points,
        rowvar=False,
    )


    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    eigenvalues = np.sort(
        eigenvalues
    )


    if (
        len(eigenvalues) < 2
        or eigenvalues[0] <= 0
    ):
        return np.inf


    return float(
        np.sqrt(
            eigenvalues[-1]
            / eigenvalues[0]
        )
    )


def _circularity(
    mask: np.ndarray,
) -> float:
    """
    Circularity:

        4 * pi * area / perimeter^2

    Larger values indicate more compact shapes.
    """

    image = (
        mask.astype(
            np.uint8
        )
        * 255
    )


    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )


    if not contours:
        return 0.0


    contour = max(
        contours,
        key=cv2.contourArea,
    )


    perimeter = cv2.arcLength(
        contour,
        True,
    )


    if perimeter <= 0:
        return 0.0


    area = float(
        mask.sum()
    )


    return float(
        4.0
        * np.pi
        * area
        / (
            perimeter ** 2
        )
    )


def _solidity(
    mask: np.ndarray,
) -> float:
    """
    Foreground area divided by convex-hull area.
    """

    points = np.column_stack(
        np.nonzero(
            mask
        )
    )


    if len(points) < 3:
        return 0.0


    xy = (
        points[
            :,
            ::-1
        ]
        .astype(
            np.int32
        )
    )


    hull = cv2.convexHull(
        xy
    )


    hull_area = cv2.contourArea(
        hull
    )


    if hull_area <= 0:
        return 0.0


    return float(
        mask.sum()
        / hull_area
    )


def _local_thickness_geometry(
    mask: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Measure local radius along the skeleton.

    Returns
    -------
    radius_median
    radius_max
    thickness_variation
    body_prominence

    thickness_variation
        MAD(radius) / median(radius)

        Smaller values indicate more uniform thickness.

    body_prominence
        max(radius) / median(radius)

        Smaller values indicate absence of a prominent local body.
    """

    distance = cv2.distanceTransform(
        mask.astype(
            np.uint8
        ),
        cv2.DIST_L2,
        5,
    )


    skeleton = skeletonize(
        mask
    )


    radii = distance[
        skeleton
    ]


    radii = radii[
        radii > 0
    ]


    if len(radii) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )


    radius_median = float(
        np.median(
            radii
        )
    )


    radius_max = float(
        np.max(
            radii
        )
    )


    mad = float(
        np.median(
            np.abs(
                radii
                - radius_median
            )
        )
    )


    if radius_median <= 0:
        return (
            radius_median,
            radius_max,
            np.nan,
            np.nan,
        )


    thickness_variation = float(
        mad
        / radius_median
    )


    body_prominence = float(
        radius_max
        / radius_median
    )


    return (
        radius_median,
        radius_max,
        thickness_variation,
        body_prominence,
    )


# ======================================================================
# TUBULAR QC
# ======================================================================

def score_tubular_objects(
    table: pd.DataFrame,
    *,
    candidate_z_threshold: float = -1.5,
    score_threshold: float = 0.80,
    eligible_mask=None,
) -> TubularQCResult:
    """
    Score tube-like connected objects.

    Tubular morphology is characterized by:

        high elongation
        +
        uniform local thickness
        +
        absence of a prominent local body

    Candidate selection uses the same robust size coordinate already
    produced by size QC.

    Canonical default interpretation:

        size_robust_z < -3.5
            suspicious small object

        -3.5 <= size_robust_z <= -1.5
            low-area plausible object
            evaluated for tubular morphology

        size_robust_z > -1.5
            ordinary-size object
            not evaluated for tubular morphology

    The lower boundary is supplied through eligible_mask. In the
    canonical Object QC workflow, suspicious-small objects are excluded
    from the tubular candidate population.

    The final tube score preserves the historical MorphoGlia logic:

        elongation rank
        +
        thickness-uniformity rank
        +
        body-prominence rank

    followed by compact-body rescue.
    """

    required = {
        "roi",
        "component_area",
        "size_robust_z",
    }


    missing = (
        required
        - set(
            table.columns
        )
    )


    if missing:
        raise ValueError(
            f"Missing tubular-QC columns: {sorted(missing)}"
        )


    candidate_z_threshold = float(
        candidate_z_threshold
    )


    if candidate_z_threshold >= 0:
        raise ValueError(
            "candidate_z_threshold must be < 0."
        )


    score_threshold = float(
        score_threshold
    )


    if not (
        0.0
        <= score_threshold
        <= 1.0
    ):
        raise ValueError(
            "score_threshold must be between 0 and 1."
        )


    result = table.copy()


    # ==================================================================
    # ELIGIBILITY
    # ==================================================================

    if eligible_mask is None:

        eligible = np.ones(
            len(result),
            dtype=bool,
        )

    else:

        eligible = np.asarray(
            eligible_mask,
            dtype=bool,
        )


        if eligible.shape != (
            len(result),
        ):
            raise ValueError(
                "eligible_mask must contain one boolean "
                "value per object."
            )


    result[
        "tubular_eligible"
    ] = eligible


    # ==================================================================
    # CANDIDATE POPULATION
    # ==================================================================

    size_z = (
        result[
            "size_robust_z"
        ]
        .to_numpy(
            dtype=float
        )
    )


    candidate = (
        eligible
        &
        np.isfinite(
            size_z
        )
        &
        (
            size_z
            <= candidate_z_threshold
        )
    )


    result[
        "tubular_candidate"
    ] = candidate


    # ==================================================================
    # OUTPUT COLUMNS
    # ==================================================================

    for column in (
        "radius_median",
        "radius_max",
        "thickness_variation",
        "body_prominence",
        "elongation",
        "circularity",
        "solidity",
        "tube_score",
    ):

        result[
            column
        ] = np.nan


    result[
        "compact_body_rescue"
    ] = False


    result[
        "suspicious_tubular"
    ] = False


    # ==================================================================
    # GEOMETRY
    # ==================================================================

    candidate_indices = result.index[
        result[
            "tubular_candidate"
        ]
    ]


    for index in candidate_indices:

        mask = _binary_mask(
            result.at[
                index,
                "roi",
            ]
        )


        (
            radius_median,
            radius_max,
            thickness_variation,
            body_prominence,
        ) = _local_thickness_geometry(
            mask
        )


        elongation = (
            _elongation_from_mask(
                mask
            )
        )


        circularity = (
            _circularity(
                mask
            )
        )


        solidity = (
            _solidity(
                mask
            )
        )


        result.at[
            index,
            "radius_median",
        ] = radius_median


        result.at[
            index,
            "radius_max",
        ] = radius_max


        result.at[
            index,
            "thickness_variation",
        ] = thickness_variation


        result.at[
            index,
            "body_prominence",
        ] = body_prominence


        result.at[
            index,
            "elongation",
        ] = elongation


        result.at[
            index,
            "circularity",
        ] = circularity


        result.at[
            index,
            "solidity",
        ] = solidity


    # ==================================================================
    # RANK-BASED TUBE SCORE
    # ==================================================================

    candidate_table = result.loc[
        candidate_indices
    ].copy()


    if not candidate_table.empty:

        elongation_bad = (
            candidate_table[
                "elongation"
            ]
            .rank(
                pct=True
            )
        )


        uniform_bad = (
            1.0
            - candidate_table[
                "thickness_variation"
            ]
            .rank(
                pct=True
            )
        )


        body_bad = (
            1.0
            - candidate_table[
                "body_prominence"
            ]
            .rank(
                pct=True
            )
        )


        tube_score = (
            elongation_bad
            + uniform_bad
            + body_bad
        ) / 3.0


        result.loc[
            candidate_indices,
            "tube_score",
        ] = tube_score


    # ==================================================================
    # COMPACT-BODY RESCUE
    #
    # This does not classify an object as biologically valid.
    # It only prevents compact objects from being interpreted as tubes.
    # ==================================================================

    compact_body = (
        result[
            "tubular_candidate"
        ]
        &
        (
            result[
                "elongation"
            ]
            < 1.6
        )
        &
        (
            result[
                "circularity"
            ]
            > 0.20
        )
        &
        (
            result[
                "solidity"
            ]
            > 0.45
        )
    )


    result[
        "compact_body_rescue"
    ] = compact_body


    result.loc[
        compact_body,
        "tube_score",
    ] *= 0.25


    # ==================================================================
    # FINAL TUBULAR FLAG
    # ==================================================================

    suspicious = (
        result[
            "tubular_candidate"
        ]
        &
        (
            result[
                "tube_score"
            ]
            >= score_threshold
        )
    )


    result[
        "suspicious_tubular"
    ] = suspicious.fillna(
        False
    )


    # ==================================================================
    # RESULT
    # ==================================================================

    return TubularQCResult(
        table=result,
        candidate_z_threshold=(
            candidate_z_threshold
        ),
        candidate_count=int(
            candidate.sum()
        ),
        score_threshold=(
            score_threshold
        ),
        suspicious_count=int(
            result[
                "suspicious_tubular"
            ].sum()
        ),
    )
