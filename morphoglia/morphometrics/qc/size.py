from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MAD_SCALE = 1.4826


@dataclass
class SizeQCResult:
    table: pd.DataFrame

    median_log_area: float
    mad_log_area: float
    robust_sigma: float
    z_threshold: float

    small_count: int
    large_count: int


def score_size_outliers(
    areas,
    z_threshold: float = 3.5,
) -> SizeQCResult:
    """
    Score connected-component areas using robust statistics.

    Areas are transformed as:

        x = log1p(area)

    and standardized using:

        z = (x - median(x)) / (1.4826 * MAD(x))

    where:

        MAD = median(|x - median(x)|)

    Negative extreme scores identify suspiciously small objects.
    Positive extreme scores identify suspiciously large objects.

    This function only SCORES objects. It does not remove anything.
    """

    z_threshold = float(
        z_threshold
    )

    if z_threshold <= 0:

        raise ValueError(
            "z_threshold must be > 0."
        )


    area_array = np.asarray(
        areas,
        dtype=float,
    )


    if area_array.ndim != 1:

        raise ValueError(
            "areas must be one-dimensional."
        )


    if area_array.size == 0:

        raise ValueError(
            "areas cannot be empty."
        )


    if not np.isfinite(
        area_array
    ).all():

        raise ValueError(
            "areas contain non-finite values."
        )


    if (
        area_array <= 0
    ).any():

        raise ValueError(
            "All component areas must be > 0."
        )


    log_area = np.log1p(
        area_array
    )


    median_log_area = float(
        np.median(
            log_area
        )
    )


    absolute_deviation = np.abs(
        log_area
        - median_log_area
    )


    mad_log_area = float(
        np.median(
            absolute_deviation
        )
    )


    robust_sigma = float(
        MAD_SCALE
        * mad_log_area
    )


    if robust_sigma == 0:

        robust_z = np.zeros(
            area_array.shape,
            dtype=float,
        )

    else:

        robust_z = (
            log_area
            - median_log_area
        ) / robust_sigma


    suspicious_small = (
        robust_z
        < -z_threshold
    )


    suspicious_large = (
        robust_z
        > z_threshold
    )


    table = pd.DataFrame(
        {
            "component_area": (
                area_array
            ),
            "log_component_area": (
                log_area
            ),
            "size_robust_z": (
                robust_z
            ),
            "suspicious_small": (
                suspicious_small
            ),
            "suspicious_large": (
                suspicious_large
            ),
        }
    )


    return SizeQCResult(
        table=table,
        median_log_area=median_log_area,
        mad_log_area=mad_log_area,
        robust_sigma=robust_sigma,
        z_threshold=z_threshold,
        small_count=int(
            suspicious_small.sum()
        ),
        large_count=int(
            suspicious_large.sum()
        ),
    )
