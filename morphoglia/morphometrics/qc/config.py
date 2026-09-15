from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObjectQCConfig:
    """
    Object-level quality control.

    QC scores and visual reports are calculated independently of whether
    a criterion is applied to the final analytical population.

    small_objects
        True:
            exclude statistically suspicious small objects.
        False:
            report them but keep them.

    large_objects
        True:
            exclude statistically suspicious large objects.
        False:
            report them but keep them.

    tubular
        True:
            exclude tubular-suspicious objects.
        False:
            report them but keep them.

    size_z_threshold
        Absolute robust-z threshold used for object size in log-area
        space.

        Objects with:
            z < -size_z_threshold
        are suspiciously small.

        Objects with:
            z > +size_z_threshold
        are suspiciously large.

    tubular_candidate_z
        Upper robust size-z boundary for objects entering tubular QC.

        Example default:
            -1.5

        Together with the small-object eligibility rule, this produces:

            z < -3.5
                suspicious small / fragment

            -3.5 <= z <= -1.5
                low-area but plausible object
                -> evaluate tubular geometry

            z > -1.5
                ordinary-size object
                -> no tubular evaluation

        This avoids dataset-specific absolute pixel thresholds.

    tubular_score_threshold
        Rank-based tube-score threshold.

    grid_n
        QC grid dimension.

        Maximum:
            6 x 6
    """

    small_objects: bool = True
    large_objects: bool = True
    tubular: bool = False

    size_z_threshold: float = 3.5

    tubular_candidate_z: float = -1.5
    tubular_score_threshold: float = 0.80

    grid_n: int = 6

    def __post_init__(self) -> None:

        self.size_z_threshold = float(
            self.size_z_threshold
        )

        if self.size_z_threshold <= 0:
            raise ValueError(
                "size_z_threshold must be > 0."
            )


        self.tubular_candidate_z = float(
            self.tubular_candidate_z
        )

        if self.tubular_candidate_z >= 0:
            raise ValueError(
                "tubular_candidate_z must be < 0."
            )


        self.tubular_score_threshold = float(
            self.tubular_score_threshold
        )

        if not (
            0.0
            <= self.tubular_score_threshold
            <= 1.0
        ):
            raise ValueError(
                "tubular_score_threshold must be between 0 and 1."
            )


        self.grid_n = int(
            self.grid_n
        )

        if not 1 <= self.grid_n <= 6:
            raise ValueError(
                "grid_n must be between 1 and 6."
            )
