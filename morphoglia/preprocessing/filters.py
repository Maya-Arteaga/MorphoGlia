from __future__ import annotations

import cv2
import numpy as np

from skimage import filters as sk_filters
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage import filters as sk_filters
from skimage import morphology
from skimage import measure
from skimage.measure import (
    label,
    regionprops,
    perimeter_crofton,
)


def multiscale_tophat(
    image: np.ndarray,
    radii: tuple[int, ...],
) -> np.ndarray:
    """
    Apply the current MorphoGlia multiscale top-hat operation
    to one 2D image.

    The values in `radii` preserve the behavior of the current
    preprocessing code: each value is used as the OpenCV elliptical
    kernel size after enforcing an odd value >= 3.
    """

    if image.ndim != 2:
        raise ValueError(
            "multiscale_tophat expects one 2D image."
        )

    kernel_sizes = [
        max(3, int(value) | 1)
        for value in radii
    ]

    result = np.zeros_like(
        image,
        dtype=np.float32,
    )

    for kernel_size in kernel_sizes:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )

        filtered = cv2.morphologyEx(
            image,
            cv2.MORPH_TOPHAT,
            kernel,
        )

        result += filtered.astype(
            np.float32
        )

    if image.dtype == np.uint16:
        return np.clip(
            result,
            0,
            65535,
        ).astype(np.uint16)

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def clahe(
    image: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    """
    Apply CLAHE while preserving uint8 or uint16 intensity depth.
    """

    if image.ndim != 2:
        raise ValueError(
            "clahe expects one 2D image."
        )

    if image.dtype not in (
        np.uint8,
        np.uint16,
    ):
        raise ValueError(
            "clahe expects uint8 or uint16 input."
        )

    clahe_object = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size),
    )

    return clahe_object.apply(image)


def unsharp_mask(
    image: np.ndarray,
    radius: float,
    amount: float,
) -> np.ndarray:
    """
    Apply unsharp masking while preserving the input intensity depth.
    """

    if image.ndim != 2:
        raise ValueError(
            "unsharp_mask expects one 2D image."
        )

    if image.dtype == np.uint16:
        image_max = np.iinfo(np.uint16).max

    elif image.dtype == np.uint8:
        image_max = np.iinfo(np.uint8).max

    else:
        raise ValueError(
            "unsharp_mask expects uint8 or uint16 input."
        )

    sharpened = sk_filters.unsharp_mask(
        image,
        radius=float(radius),
        amount=float(amount),
        preserve_range=False,
    )

    sharpened = np.clip(
        sharpened,
        0.0,
        1.0,
    )

    return (
        sharpened * image_max
    ).astype(image.dtype)


def normalize(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize one 2D image to the full range of its current dtype.
    """

    if image.ndim != 2:
        raise ValueError(
            "normalize expects one 2D image."
        )

    if image.dtype == np.uint16:
        image_max = np.iinfo(
            np.uint16
        ).max

    else:
        image_max = np.iinfo(
            np.uint8
        ).max

    normalized = cv2.normalize(
        image,
        None,
        alpha=0,
        beta=image_max,
        norm_type=cv2.NORM_MINMAX,
    )

    return normalized.astype(
        image.dtype
    )


def percentile_threshold(
    image: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """
    Binarize one 2D image using the selected intensity percentile.
    """

    if image.ndim != 2:
        raise ValueError(
            "percentile_threshold expects one 2D image."
        )

    threshold_value = np.percentile(
        image,
        float(percentile),
    )

    binary = (
        image > threshold_value
    )

    return (
        binary.astype(np.uint8)
        * 255
    )


def remove_small_noise(
    image: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """
    Remove connected foreground objects smaller than `min_area`.

    Connectivity=1 is intentionally preserved from the current
    preprocessing implementation.
    """

    if image.ndim != 2:
        raise ValueError(
            "remove_small_noise expects one 2D image."
        )

    labeled = label(
        image > 0,
        connectivity=1,
    )

    cleaned = remove_small_objects(
        labeled,
        min_size=int(min_area),
    )

    return (
        (cleaned > 0).astype(np.uint8)
        * 255
    )
# ============================================================================
# EXTENDED FILTER LIBRARY — MG_PREPROCESSING_REGISTRY_V1
# ============================================================================


def _mg_require_2d(image: np.ndarray, operation: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"{operation} expects one 2D image.")
    if image.size == 0:
        raise ValueError(f"{operation} received an empty image.")
    return image


def _mg_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image

    values = image.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)

    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)

    values = (values - lo) / (hi - lo)
    return np.clip(values * 255.0, 0, 255).astype(np.uint8)


def flip_vertical(image: np.ndarray) -> np.ndarray:
    """Flip a 2D image, or every frame of a (T,H,W) stack, vertically."""
    image = np.asarray(image)
    if image.ndim == 2:
        return np.flip(image, axis=0)
    if image.ndim == 3:
        return np.flip(image, axis=1)
    raise ValueError("flip_vertical expects 2D or (T,H,W) input.")


def convert_gray_scale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA storage to grayscale; preserve grayscale input."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        rgb = image[..., :3]
        if rgb.dtype not in (np.uint8, np.uint16):
            rgb = _mg_to_uint8(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    raise ValueError("convert_gray_scale expects HxW, HxWx3, or HxWx4 input.")


def tophat(image: np.ndarray, disk_size: int) -> np.ndarray:
    image = _mg_require_2d(image, "tophat")
    size = max(3, int(disk_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def gaussian_subtraction(image: np.ndarray, kernel_size: int) -> np.ndarray:
    image = _mg_require_2d(image, "gaussian_subtraction")
    size = max(3, int(kernel_size) | 1)
    background = cv2.GaussianBlur(image, (size, size), 0)
    return cv2.subtract(image, background)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    from skimage import exposure
    image = _mg_require_2d(image, "histogram_equalization")
    result = exposure.equalize_hist(image)
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def adaptive_histogram_equalization(
    image: np.ndarray,
    clip_limit: float,
) -> np.ndarray:
    from skimage import exposure
    image = _mg_require_2d(image, "adaptive_histogram_equalization")
    values = image.astype(np.float64)
    lo = float(values.min())
    hi = float(values.max())
    if hi > lo:
        values = (values - lo) / (hi - lo)
    else:
        values = np.zeros_like(values)
    result = exposure.equalize_adapthist(
        values,
        clip_limit=float(clip_limit),
    )
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    image = _mg_require_2d(image, "gamma_correction")
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    if np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
    else:
        maximum = 1.0

    values = np.clip(image.astype(np.float64) / maximum, 0.0, 1.0)
    result = np.power(values, gamma)

    if np.issubdtype(image.dtype, np.integer):
        return np.clip(result * maximum, 0, maximum).astype(image.dtype)
    return result.astype(image.dtype, copy=False)


def log_transformation(image: np.ndarray) -> np.ndarray:
    image = _mg_require_2d(image, "log_transformation")
    values = np.clip(image.astype(np.float64), 0, None)
    maximum = float(values.max())
    if maximum <= 0:
        return np.zeros(image.shape, dtype=np.uint8)
    result = (255.0 / np.log1p(maximum)) * np.log1p(values)
    return np.clip(result, 0, 255).astype(np.uint8)


def min_max_normalization(image: np.ndarray) -> np.ndarray:
    image = _mg_require_2d(image, "min_max_normalization")
    return _mg_to_uint8(image)


def convert_to_8bit(image: np.ndarray) -> np.ndarray:
    image = _mg_require_2d(image, "convert_to_8bit")
    return _mg_to_uint8(image)


def anisotropic_diffusion(
    image: np.ndarray,
    sigma_s: float,
    sigma_r: float,
) -> np.ndarray:
    """Compatibility implementation of the legacy edge-preserving filter."""
    image = _mg_require_2d(image, "anisotropic_diffusion")
    frame = _mg_to_uint8(image)
    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    filtered = cv2.edgePreservingFilter(
        bgr,
        flags=1,
        sigma_s=float(sigma_s),
        sigma_r=float(sigma_r),
    )
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)


def fast_non_local_means(
    image: np.ndarray,
    h: float,
    template_window_size: int,
    search_window_size: int,
) -> np.ndarray:
    image = _mg_require_2d(image, "fast_non_local_means")
    frame = _mg_to_uint8(image)
    template = max(3, int(template_window_size) | 1)
    search = max(template + 2, int(search_window_size) | 1)
    return cv2.fastNlMeansDenoising(
        frame,
        None,
        h=float(h),
        templateWindowSize=template,
        searchWindowSize=search,
    )


def fixed_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    image = _mg_require_2d(image, "fixed_threshold")
    return ((image > float(threshold)).astype(np.uint8) * 255)


def median_threshold(image: np.ndarray) -> np.ndarray:
    image = _mg_require_2d(image, "median_threshold")
    threshold = float(np.median(image))
    return ((image > threshold).astype(np.uint8) * 255)


def mean_threshold(image: np.ndarray) -> np.ndarray:
    image = _mg_require_2d(image, "mean_threshold")
    threshold = float(np.mean(image))
    return ((image > threshold).astype(np.uint8) * 255)


def adaptive_threshold(
    image: np.ndarray,
    block_size: int,
    constant: float,
) -> np.ndarray:
    image = _mg_require_2d(image, "adaptive_threshold")
    frame = _mg_to_uint8(image)
    block_size = max(3, int(block_size) | 1)
    binary = cv2.adaptiveThreshold(
        frame,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        float(constant),
    )
    binary[frame == 0] = 0
    return binary.astype(np.uint8)


def remove_large_objects(image: np.ndarray, max_area: int) -> np.ndarray:
    from skimage.measure import regionprops
    image = _mg_require_2d(image, "remove_large_objects")
    labeled = label(image > 0, connectivity=1)
    keep = np.zeros(image.shape, dtype=bool)
    for region in regionprops(labeled):
        if int(region.area) <= int(max_area):
            keep[labeled == region.label] = True
    return keep.astype(np.uint8) * 255


def soma(
    image: np.ndarray,
    iterations: int,
    min_area_threshold: int,
) -> np.ndarray:
    image = _mg_require_2d(image, "soma")
    binary = (image > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    iterations = max(0, int(iterations))
    if iterations:
        binary = cv2.erode(binary, kernel, iterations=iterations)
        binary = cv2.dilate(binary, kernel, iterations=iterations)
    labeled = label(binary > 0)
    cleaned = remove_small_objects(
        labeled,
        min_size=max(0, int(min_area_threshold)),
    )
    return (cleaned > 0).astype(np.uint8) * 255


def skeletonize_binary(image: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    image = _mg_require_2d(image, "skeletonize_binary")
    result = skeletonize(image > 0)
    return result.astype(np.uint8) * 255


def eliminate_1_value(image: np.ndarray) -> np.ndarray:
    image = _mg_require_2d(image, "eliminate_1_value").copy()
    image[image == 1] = 0
    return image


def eliminate_below_value(image: np.ndarray, value: float) -> np.ndarray:
    image = _mg_require_2d(image, "eliminate_below_value").copy()
    image[image <= float(value)] = 0
    return image


def canny_edge_detection(
    image: np.ndarray,
    threshold1: float,
    threshold2: float,
) -> np.ndarray:
    image = _mg_require_2d(image, "canny_edge_detection")
    return cv2.Canny(
        _mg_to_uint8(image),
        float(threshold1),
        float(threshold2),
    )


def prep_for_cellpose(
    image: np.ndarray,
    radii: tuple[int, ...],
    sharpen: bool,
    unsharp_radius: float,
    unsharp_amount: float,
) -> np.ndarray:
    """Compatibility implementation of the older PREP_FOR_CELLPOSE helper."""
    from skimage.filters import unsharp_mask as _sk_unsharp
    from skimage.morphology import disk, white_tophat

    image = _mg_require_2d(image, "prep_for_cellpose")
    total = np.zeros_like(image, dtype=np.float32)

    for radius in radii:
        total += white_tophat(
            image,
            footprint=disk(max(1, int(radius))),
        ).astype(np.float32)

    lo = float(total.min())
    hi = float(total.max())
    if hi > lo:
        normalized = (total - lo) / (hi - lo)
    else:
        normalized = np.zeros_like(total)

    if bool(sharpen):
        normalized = _sk_unsharp(
            normalized,
            radius=float(unsharp_radius),
            amount=float(unsharp_amount),
            preserve_range=True,
        )

    normalized = np.clip(normalized, 0.0, 1.0)

    if np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
        return (normalized * maximum).astype(image.dtype)

    return normalized.astype(image.dtype, copy=False)







def median_filter(
    image: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """
    Median filter for suppressing salt-and-pepper noise
    while preserving thin morphological boundaries.
    """

    image = _mg_require_2d(
        image,
        "median_filter",
    )

    kernel_size = max(
        3,
        int(kernel_size) | 1,
    )

    return cv2.medianBlur(
        image,
        kernel_size,
    )

def percentile_hysteresis_threshold(
    image: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    """
    Percentile-based hysteresis threshold.

    Percentiles are calculated only from positive pixels so that
    the zero-valued background created by top-hat filtering does
    not dominate the percentile calculation.

    Pixels above the high threshold are definite foreground.
    Pixels above the low threshold are retained only when connected
    to definite foreground.
    """

    image = _mg_require_2d(
        image,
        "percentile_hysteresis_threshold",
    )

    low_percentile = float(
        low_percentile
    )

    high_percentile = float(
        high_percentile
    )

    if not (
        0.0
        <= low_percentile
        < high_percentile
        <= 100.0
    ):
        raise ValueError(
            "Hysteresis percentiles must satisfy "
            "0 <= low < high <= 100."
        )

    positive = image[
        image > 0
    ]

    if positive.size == 0:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    low_threshold = np.percentile(
        positive,
        low_percentile,
    )

    high_threshold = np.percentile(
        positive,
        high_percentile,
    )

    binary = sk_filters.apply_hysteresis_threshold(
        image,
        low_threshold,
        high_threshold,
    )

    return (
        binary.astype(np.uint8)
        * 255
    )



def fill_small_holes(
    image: np.ndarray,
    area_threshold: int,
) -> np.ndarray:
    """
    Fill small holes inside binary foreground objects without
    performing a general morphological closing.
    """

    image = _mg_require_2d(
        image,
        "fill_small_holes",
    )

    binary = (
        image > 0
    )

    filled = morphology.remove_small_holes(
        binary,
        area_threshold=int(
            area_threshold
        ),
        connectivity=1,
    )

    return (
        filled.astype(np.uint8)
        * 255
    )



def robust_rescale(
    image: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    """
    Robustly rescale image intensities using percentile limits.

    Values below the low percentile are clipped to 0.
    Values above the high percentile are clipped to the maximum
    intensity supported by the input dtype.
    """

    image = _mg_require_2d(
        image,
        "robust_rescale",
    )

    low_percentile = float(
        low_percentile
    )

    high_percentile = float(
        high_percentile
    )

    if not (
        0.0
        <= low_percentile
        < high_percentile
        <= 100.0
    ):
        raise ValueError(
            "Robust rescale percentiles must satisfy "
            "0 <= low < high <= 100."
        )

    values = image[
        np.isfinite(
            image
        )
    ]

    if values.size == 0:
        return np.zeros_like(
            image
        )

    low = float(
        np.percentile(
            values,
            low_percentile,
        )
    )

    high = float(
        np.percentile(
            values,
            high_percentile,
        )
    )

    if high <= low:
        return np.zeros_like(
            image
        )

    normalized = (
        image.astype(
            np.float64
        )
        - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    if np.issubdtype(
        image.dtype,
        np.integer,
    ):
        image_max = np.iinfo(
            image.dtype
        ).max

        return (
            normalized
            * image_max
        ).astype(
            image.dtype
        )

    return normalized.astype(
        image.dtype
    )



def background_percentile_subtraction(
    image: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """
    Estimate a global background floor from an intensity percentile,
    subtract it, and clip negative values to zero.

    Example
    -------
    percentile=60 means that the 60th intensity percentile is treated
    as the background floor.
    """

    image = _mg_require_2d(
        image,
        "background_percentile_subtraction",
    )

    percentile = float(
        percentile
    )

    if not (
        0.0
        <= percentile
        < 100.0
    ):
        raise ValueError(
            "Background percentile must satisfy "
            "0 <= percentile < 100."
        )

    background = float(
        np.percentile(
            image,
            percentile,
        )
    )

    result = (
        image.astype(np.float64)
        - background
    )

    result = np.clip(
        result,
        0.0,
        None,
    )

    if np.issubdtype(
        image.dtype,
        np.integer,
    ):
        image_max = np.iinfo(
            image.dtype
        ).max

        result = np.clip(
            result,
            0,
            image_max,
        )

    return result.astype(
        image.dtype
    )


def huang_threshold_value(
    image: np.ndarray,
    bins: int = 256,
) -> float:
    """
    Huang fuzzy threshold.

    Select the threshold that minimizes fuzzy ambiguity between
    background and foreground classes.

    Returns the threshold intensity value, not a binary image.
    """

    image = _mg_require_2d(
        image,
        "huang_threshold_value",
    )

    values = image[
        np.isfinite(image)
    ]

    if values.size == 0:
        return 0.0

    minimum = float(values.min())
    maximum = float(values.max())

    if maximum <= minimum:
        return minimum

    bins = max(
        16,
        int(bins),
    )

    histogram, edges = np.histogram(
        values,
        bins=bins,
        range=(minimum, maximum),
    )

    histogram = histogram.astype(
        np.float64
    )

    nonzero = np.flatnonzero(
        histogram
    )

    if nonzero.size < 2:
        return minimum

    first = int(nonzero[0])
    last = int(nonzero[-1])

    if last <= first:
        return minimum

    # --------------------------------------------------------------
    # Mean gray-level index of the lower class for every threshold.
    # --------------------------------------------------------------

    mu_lower = np.zeros(
        bins,
        dtype=np.float64,
    )

    count = 0.0
    weighted_sum = 0.0

    for index in range(
        first,
        last + 1,
    ):
        count += histogram[index]
        weighted_sum += (
            index
            * histogram[index]
        )

        if count > 0:
            mu_lower[index] = (
                weighted_sum
                / count
            )

    # --------------------------------------------------------------
    # Mean gray-level index of the upper class for every threshold.
    # mu_upper[t] describes bins t+1 ... last.
    # --------------------------------------------------------------

    mu_upper = np.zeros(
        bins,
        dtype=np.float64,
    )

    count = 0.0
    weighted_sum = 0.0

    for index in range(
        last,
        first,
        -1,
    ):
        count += histogram[index]
        weighted_sum += (
            index
            * histogram[index]
        )

        if count > 0:
            mu_upper[
                index - 1
            ] = (
                weighted_sum
                / count
            )

    scale = 1.0 / float(
        last - first
    )

    minimum_entropy = np.inf
    best_threshold = first

    # --------------------------------------------------------------
    # Huang fuzzy entropy.
    # --------------------------------------------------------------

    for threshold in range(
        first,
        last,
    ):

        entropy = 0.0

        lower_indices = np.arange(
            first,
            threshold + 1,
            dtype=np.float64,
        )

        lower_membership = (
            1.0
            / (
                1.0
                + scale
                * np.abs(
                    lower_indices
                    - mu_lower[threshold]
                )
            )
        )

        lower_membership = np.clip(
            lower_membership,
            1e-12,
            1.0 - 1e-12,
        )

        lower_entropy = (
            -lower_membership
            * np.log(lower_membership)
            - (
                1.0 - lower_membership
            )
            * np.log(
                1.0 - lower_membership
            )
        )

        entropy += float(
            np.sum(
                histogram[
                    first:
                    threshold + 1
                ]
                * lower_entropy
            )
        )

        upper_indices = np.arange(
            threshold + 1,
            last + 1,
            dtype=np.float64,
        )

        upper_membership = (
            1.0
            / (
                1.0
                + scale
                * np.abs(
                    upper_indices
                    - mu_upper[threshold]
                )
            )
        )

        upper_membership = np.clip(
            upper_membership,
            1e-12,
            1.0 - 1e-12,
        )

        upper_entropy = (
            -upper_membership
            * np.log(upper_membership)
            - (
                1.0 - upper_membership
            )
            * np.log(
                1.0 - upper_membership
            )
        )

        entropy += float(
            np.sum(
                histogram[
                    threshold + 1:
                    last + 1
                ]
                * upper_entropy
            )
        )

        if entropy < minimum_entropy:
            minimum_entropy = entropy
            best_threshold = threshold

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    return float(
        centers[
            best_threshold
        ]
    )




def huang_threshold(
    image: np.ndarray,
    bins: int = 256,
) -> np.ndarray:
    """
    Binarize an image using Huang fuzzy thresholding.
    """

    image = _mg_require_2d(
        image,
        "huang_threshold",
    )

    threshold = huang_threshold_value(
        image,
        bins=bins,
    )

    return (
        (
            image > threshold
        )
        .astype(np.uint8)
        * 255
    )






def _oriented_rectangle_kernel(
    width: int,
    length: int,
    angle: float,
) -> np.ndarray:
    """
    Build a thin rectangular structuring element at one orientation.
    """

    width = max(
        1,
        int(width),
    )

    length = max(
        width + 1,
        int(length),
    )

    side = int(
        np.ceil(
            np.hypot(
                width,
                length,
            )
        )
    ) + 4

    if side % 2 == 0:
        side += 1

    center = side // 2

    kernel = np.zeros(
        (
            side,
            side,
        ),
        dtype=np.uint8,
    )

    y0 = (
        center
        - width // 2
    )

    x0 = (
        center
        - length // 2
    )

    kernel[
        y0:
        y0 + width,
        x0:
        x0 + length,
    ] = 1

    matrix = cv2.getRotationMatrix2D(
        (
            center,
            center,
        ),
        float(angle),
        1.0,
    )

    rotated = cv2.warpAffine(
        kernel,
        matrix,
        (
            side,
            side,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        rotated > 0
    ).astype(
        np.uint8
    )
        
        

        
        
        
        
        
        
def branch_aware_consensus_threshold(
    image: np.ndarray,
    support_votes: int,
    strong_votes: int,
    min_seed_area: int,
    line_lengths: tuple[int, ...],
    line_angles: tuple[float, ...],
    line_width: int,
    min_line_occupancy: float,
    bridge_radius: int,
) -> np.ndarray:
    """
    Branch-aware consensus threshold.

    Intensity evidence
    ------------------
    Huang, Li, Triangle, and Yen each generate a foreground mask.

    Pixels supported by >= strong_votes methods form confident cell cores.
    Pixels supported by >= support_votes methods form permissive candidate
    foreground.

    Geometry
    --------
    Ambiguous candidate pixels are retained when they show coherent
    elongated support along at least one tested orientation.

    Connectivity
    ------------
    Weak branch candidates are retained only when they are connected to a
    valid confident cell core.

    This allows faint/pixelated branches to survive while rejecting much of
    the isolated circular and salt-and-pepper background.
    """

    image = _mg_require_2d(
        image,
        "branch_aware_consensus_threshold",
    )

    support_votes = int(
        support_votes
    )

    strong_votes = int(
        strong_votes
    )

    min_seed_area = int(
        min_seed_area
    )

    line_width = max(
        1,
        int(line_width),
    )

    min_line_occupancy = float(
        min_line_occupancy
    )

    bridge_radius = max(
        0,
        int(bridge_radius),
    )

    if not (
        1 <= support_votes <= strong_votes <= 4
    ):
        raise ValueError(
            "Votes must satisfy "
            "1 <= support_votes <= strong_votes <= 4."
        )

    if not (
        0.0
        <= min_line_occupancy
        <= 1.0
    ):
        raise ValueError(
            "min_line_occupancy must be between 0 and 1."
        )

    # ------------------------------------------------------------------
    # AUTOMATIC THRESHOLD MASKS
    # ------------------------------------------------------------------

    finite = np.isfinite(
        image
    )

    if not np.any(
        finite
    ):
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    values = image[
        finite
    ]

    # Li
    li_value = float(
        sk_filters.threshold_li(
            values
        )
    )
    
    triangle_value = float(
        sk_filters.threshold_triangle(
            values
        )
    )
    
    yen_value = float(
        sk_filters.threshold_yen(
            values
        )
    )
    
    huang_value = float(
        huang_threshold_value(
            image
        )
    )
    
    
    li_mask = image > li_value
    triangle_mask = image > triangle_value
    yen_mask = image > yen_value
    huang_mask = image > huang_value

    # ------------------------------------------------------------------
    # CONSENSUS VOTE MAP
    # ------------------------------------------------------------------

    votes = (
        huang_mask.astype(np.uint8)
        + li_mask.astype(np.uint8)
        + triangle_mask.astype(np.uint8)
        + yen_mask.astype(np.uint8)
    )

    support = (
        votes
        >= support_votes
    )

    strong = (
        votes
        >= strong_votes
    )

    # Remove small bright dots before they are allowed to become
    # independent reconstruction seeds.
    strong = remove_small_objects(
        strong,
        min_size=min_seed_area,
        connectivity=2,
    )

    if not np.any(
        strong
    ):
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    # ------------------------------------------------------------------
    # DIRECTIONAL BRANCH SUPPORT
    #
    # Each thin line acts like your rectangular-window idea.
    #
    # For every pixel:
    #     how much of a line passing through this location is occupied by
    #     candidate foreground?
    #
    # We keep the maximum response across orientations/scales.
    # ------------------------------------------------------------------

    support_float = support.astype(
        np.float32
    )

    directional_score = np.zeros(
        image.shape,
        dtype=np.float32,
    )

    valid_lengths = tuple(
        sorted(
            {
                max(
                    3,
                    int(length) | 1,
                )
                for length in line_lengths
            }
        )
    )

    if not valid_lengths:
        raise ValueError(
            "line_lengths must contain at least one value."
        )

    maximum_length = max(
        valid_lengths
    )

    for length in valid_lengths:

        # Larger windows receive more weight.
        # A tiny circular blob can strongly occupy a short line, but
        # should not receive the same score as a structure that remains
        # coherent across a longer window.
        length_weight = (
            float(length)
            / float(maximum_length)
        )

        size = (
            length
            + 2 * line_width
            + 2
        )

        if size % 2 == 0:
            size += 1

        center = (
            size // 2
        )

        half_length = (
            length - 1
        ) / 2.0

        for angle in line_angles:

            radians = np.deg2rad(
                float(angle)
            )

            dx = half_length * np.cos(
                radians
            )

            dy = half_length * np.sin(
                radians
            )

            x1 = int(
                round(
                    center - dx
                )
            )

            y1 = int(
                round(
                    center - dy
                )
            )

            x2 = int(
                round(
                    center + dx
                )
            )

            y2 = int(
                round(
                    center + dy
                )
            )

            kernel = np.zeros(
                (
                    size,
                    size,
                ),
                dtype=np.uint8,
            )

            cv2.line(
                kernel,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                color=1,
                thickness=line_width,
            )

            kernel_sum = float(
                kernel.sum()
            )

            if kernel_sum <= 0:
                continue

            normalized_kernel = (
                kernel.astype(
                    np.float32
                )
                / kernel_sum
            )

            occupancy = cv2.filter2D(
                support_float,
                ddepth=-1,
                kernel=normalized_kernel,
                borderType=cv2.BORDER_CONSTANT,
            )

            occupancy *= (
                length_weight
            )

            directional_score = np.maximum(
                directional_score,
                occupancy,
            )

    directional = (
        directional_score
        >= min_line_occupancy
    )

    # Only ambiguous/permissive foreground can become branch support.
    branch_support = (
        support
        & directional
    )

    # ------------------------------------------------------------------
    # OPTIONAL 1-PIXEL BRANCH REPAIR
    #
    # Allows slightly pixelated branches to bridge tiny interruptions.
    # Keep this small; large closing radii could merge nearby cells.
    # ------------------------------------------------------------------

    if bridge_radius > 0:

        branch_support = (
            morphology.binary_closing(
                branch_support,
                footprint=morphology.disk(
                    bridge_radius
                ),
            )
        )

    # Confident cores are always allowed.
    allowed = (
        strong
        | branch_support
    )

    # ------------------------------------------------------------------
    # GEODESIC RECONSTRUCTION
    #
    # This is the key final gate:
    #
    # branch-like candidate pixels survive only if a confident cell core
    # can reach them through the allowed mask.
    # ------------------------------------------------------------------

    reconstructed = morphology.reconstruction(
        strong.astype(
            np.uint8
        ),
        allowed.astype(
            np.uint8
        ),
        method="dilation",
        footprint=np.ones(
            (
                3,
                3,
            ),
            dtype=np.uint8,
        ),
    )

    result = (
        reconstructed > 0
    )

    return (
        result.astype(
            np.uint8
        )
        * 255
    )




        
        
        
def _directional_opening(
    binary: np.ndarray,
    width: int,
    length: int,
    angles: tuple[float, ...],
) -> np.ndarray:
    """
    Keep binary structures capable of supporting a thin rectangle
    at at least one tested orientation.
    """

    result = np.zeros(
        binary.shape,
        dtype=bool,
    )

    source = (
        binary.astype(
            np.uint8
        )
        * 255
    )

    for angle in angles:

        kernel = (
            _oriented_rectangle_kernel(
                width=width,
                length=length,
                angle=angle,
            )
        )

        opened = cv2.morphologyEx(
            source,
            cv2.MORPH_OPEN,
            kernel,
        )

        result |= (
            opened > 0
        )

    return result




def branch_geometry_cleanup(
    image: np.ndarray,
    angles: tuple[float, ...],
    thin_kernels: tuple[tuple[int, int], ...],
    strong_kernels: tuple[tuple[int, int], ...],
    soma_radius: int,
    min_soma_area: int,
    min_object_area: int,
    bridge_radius: int,
) -> np.ndarray:
    """
    Remove compact/salt-and-pepper threshold noise while retaining
    elongated branch-like structures and thick cellular cores.

    The input must already be binary.

    thin_kernels
        Permissive elongated structures. Example:
            (1, 10), (2, 10)

    strong_kernels
        More selective elongated structures. Example:
            (3, 11), (3, 16)

    Final objects must contain either:
        - a sufficiently large soma/thick-core candidate
        - a strong elongated structure
    """

    image = _mg_require_2d(
        image,
        "branch_geometry_cleanup",
    )

    binary = (
        image > 0
    )

    if not np.any(
        binary
    ):
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    # --------------------------------------------------------------
    # PERMISSIVE THIN-BRANCH SUPPORT
    # --------------------------------------------------------------

    thin_support = np.zeros(
        image.shape,
        dtype=bool,
    )

    for width, length in thin_kernels:

        thin_support |= (
            _directional_opening(
                binary,
                width=int(width),
                length=int(length),
                angles=angles,
            )
        )

    # --------------------------------------------------------------
    # STRONG ELONGATED SUPPORT
    # --------------------------------------------------------------

    strong_support = np.zeros(
        image.shape,
        dtype=bool,
    )

    for width, length in strong_kernels:

        strong_support |= (
            _directional_opening(
                binary,
                width=int(width),
                length=int(length),
                angles=angles,
            )
        )

    # --------------------------------------------------------------
    # SOMA / THICK CELLULAR CORE
    #
    # A disk opening removes thin processes and tiny circular dots,
    # leaving only sufficiently thick foreground.
    # --------------------------------------------------------------

    soma_radius = max(
        1,
        int(soma_radius),
    )

    soma = morphology.opening(
        binary,
        footprint=morphology.disk(
            soma_radius
        ),
    )

    soma = remove_small_objects(
        soma,
        min_size=int(
            min_soma_area
        ),
        connectivity=2,
    )

    # --------------------------------------------------------------
    # ALLOWED TERRITORY
    # --------------------------------------------------------------

    allowed = (
        thin_support
        | strong_support
        | soma
    )

    # Allow only tiny interruptions to reconnect.
    bridge_radius = max(
        0,
        int(bridge_radius),
    )

    if bridge_radius > 0:

        allowed = morphology.closing(
            allowed,
            footprint=morphology.disk(
                bridge_radius
            ),
        )

    # --------------------------------------------------------------
    # SEEDS
    #
    # Something becomes a valid object only if it contains:
    #   - soma-like thick signal
    #   OR
    #   - strong elongated signal
    # --------------------------------------------------------------

    seeds = (
        soma
        | strong_support
    )

    # --------------------------------------------------------------
    # CONNECTED-COMPONENT GATING
    #
    # Keep only allowed components containing at least one seed pixel.
    # This is equivalent to binary reconstruction but much faster here.
    # --------------------------------------------------------------

    labels = measure.label(
        allowed,
        connectivity=2,
    )

    valid_labels = np.unique(
        labels[
            seeds
        ]
    )

    valid_labels = valid_labels[
        valid_labels > 0
    ]

    cleaned = np.isin(
        labels,
        valid_labels,
    )

    cleaned = remove_small_objects(
        cleaned,
        min_size=int(
            min_object_area
        ),
        connectivity=2,
    )

    return (
        cleaned.astype(
            np.uint8
        )
        * 255
    )






def directional_grayscale_opening(
    image: np.ndarray,
    angles: tuple[float, ...],
    branch_kernels: tuple[tuple[int, int], ...],
    soma_radius: int,
) -> np.ndarray:
    """
    Enhance elongated bright structures before binarization.

    Unlike BRANCH_GEOMETRY_CLEANUP, this operates on the grayscale
    intensity image rather than an already-binarized mask.

    Multiple thin rectangular openings preserve branch-like structures
    across different widths, lengths, and orientations. A circular
    opening is included to retain thicker soma/core signal.
    """

    image = _mg_require_2d(
        image,
        "directional_grayscale_opening",
    )

    directional = np.zeros_like(
        image
    )

    # --------------------------------------------------------------
    # ELONGATED BRANCH SUPPORT
    # --------------------------------------------------------------

    for width, length in branch_kernels:

        for angle in angles:

            kernel = _oriented_rectangle_kernel(
                width=int(width),
                length=int(length),
                angle=float(angle),
            )

            opened = cv2.morphologyEx(
                image,
                cv2.MORPH_OPEN,
                kernel,
            )

            directional = np.maximum(
                directional,
                opened,
            )

    # --------------------------------------------------------------
    # SOMA / THICK CELLULAR CORE
    # --------------------------------------------------------------

    soma_radius = max(
        1,
        int(soma_radius),
    )

    soma_size = (
        2 * soma_radius
        + 1
    )

    soma_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            soma_size,
            soma_size,
        ),
    )

    soma = cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        soma_kernel,
    )

    # Keep whichever evidence is stronger:
    # elongated branch response or thick soma response.
    return np.maximum(
        directional,
        soma,
    )




        
def li_threshold(
    image: np.ndarray,
) -> np.ndarray:
    """
    Binarize one 2D image using Li minimum cross-entropy thresholding.
    """

    image = _mg_require_2d(
        image,
        "li_threshold",
    )

    values = image[
        np.isfinite(
            image
        )
    ]

    if values.size == 0:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    threshold_value = float(
        sk_filters.threshold_li(
            values
        )
    )

    binary = (
        image > threshold_value
    )

    return (
        binary.astype(
            np.uint8
        )
        * 255
    )






def remove_circular_objects(
    image: np.ndarray,
    *,
    min_circularity: float = 0.80,
    max_area: int = 45 * 40,
) -> np.ndarray:
    """
    Remove small round connected components while preserving
    elongated/branch-like fragments.

    A component is removed only if:

        circularity >= min_circularity
        AND
        area <= max_area
    """

    from skimage.measure import label, regionprops
    import numpy as np

    binary = np.asarray(image, dtype=bool)

    labeled = label(
        binary,
        connectivity=2,
    )

    output = binary.copy()

    for region in regionprops(labeled):

        # Do not consider large objects background.
        if region.area > max_area:
            continue

        perimeter = region.perimeter_crofton

        if perimeter <= 0:
            continue

        circularity = (
            4.0
            * np.pi
            * region.area
            / (perimeter ** 2)
        )

        # Rasterization can produce values slightly > 1.
        circularity = min(float(circularity), 1.0)

        if circularity >= min_circularity:
            output[labeled == region.label] = False

    return output



def remove_circular_noise(
    image: np.ndarray,
    min_circularity: float,
    max_area: int,
) -> np.ndarray:
    """
    Remove small, approximately circular connected components while
    preserving elongated components that may represent disconnected
    cellular branches.

    An object is removed only when:

        circularity >= min_circularity
        AND
        area <= max_area

    Circularity
    -----------
    4 * pi * area / perimeter^2

    Values near:
        1.0 -> circular
        0.0 -> elongated / irregular

    Crofton perimeter is used because it is less sensitive to pixel-grid
    orientation than a simple boundary-pixel perimeter.
    """

    image = _mg_require_2d(
        image,
        "remove_circular_noise",
    )

    min_circularity = float(
        min_circularity
    )

    max_area = int(
        max_area
    )

    if not (
        0.0
        <= min_circularity
        <= 1.0
    ):
        raise ValueError(
            "min_circularity must be between 0 and 1."
        )

    if max_area < 1:
        raise ValueError(
            "max_area must be >= 1."
        )

    binary = (
        image > 0
    )

    labels = label(
        binary,
        connectivity=2,
    )

    output = binary.copy()

    for region in regionprops(
        labels
    ):

        area = int(
            region.area
        )

        # Large objects are never removed by this filter.
        if area > max_area:
            continue

        component = (
            labels
            == region.label
        )

        perimeter = float(
            perimeter_crofton(
                component,
                directions=4,
            )
        )

        if perimeter <= 0.0:
            continue

        circularity = float(
            (
                4.0
                * np.pi
                * area
            )
            / (
                perimeter ** 2
            )
        )

        # Numerical/perimeter discretization can occasionally produce
        # values slightly above 1.
        circularity = min(
            1.0,
            circularity,
        )

        if circularity >= min_circularity:

            output[
                component
            ] = False

    return (
        output.astype(
            np.uint8
        )
        * 255
    )





def remove_compact_objects(
    image: np.ndarray,
    min_solidity: float,
    min_circularity: float,
    max_aspect_ratio: float,
) -> np.ndarray:
    """
    Remove compact connected components while preserving irregular,
    elongated, or ramified objects.

    A component is removed only when all three conditions are satisfied:

        solidity >= min_solidity
        AND circularity >= min_circularity
        AND major_axis_length / minor_axis_length <= max_aspect_ratio

    Circularity is computed as:

        4 * pi * area / perimeter^2

    using the Crofton perimeter. Aspect ratio is rotation-independent
    because it uses the fitted major and minor region axes rather than
    the image-aligned bounding box.

    Retained components are copied pixel-for-pixel without erosion,
    dilation, opening, closing, or any other morphological modification.
    """

    image = _mg_require_2d(
        image,
        "remove_compact_objects",
    )

    min_solidity = float(
        min_solidity
    )

    min_circularity = float(
        min_circularity
    )

    max_aspect_ratio = float(
        max_aspect_ratio
    )

    if not (
        0.0
        <= min_solidity
        <= 1.0
    ):
        raise ValueError(
            "min_solidity must be between 0 and 1."
        )

    if not (
        0.0
        <= min_circularity
        <= 1.0
    ):
        raise ValueError(
            "min_circularity must be between 0 and 1."
        )

    if max_aspect_ratio < 1.0:
        raise ValueError(
            "max_aspect_ratio must be >= 1."
        )

    binary = (
        image > 0
    )

    labels = label(
        binary,
        connectivity=2,
    )

    output = binary.copy()

    for region in regionprops(
        labels
    ):

        component = (
            labels
            == region.label
        )

        solidity = float(
            region.solidity
        )

        perimeter = float(
            perimeter_crofton(
                component,
                directions=4,
            )
        )

        if perimeter <= 0.0:
            continue

        circularity = float(
            (
                4.0
                * np.pi
                * float(region.area)
            )
            / (
                perimeter ** 2
            )
        )

        # Pixel discretization can occasionally produce values
        # slightly above the theoretical maximum of 1.
        circularity = min(
            1.0,
            circularity,
        )

        minor_axis = float(
            region.axis_minor_length
        )

        if minor_axis <= 0.0:
            aspect_ratio = np.inf

        else:
            aspect_ratio = float(
                region.axis_major_length
                / minor_axis
            )

        if (
            solidity >= min_solidity
            and
            circularity >= min_circularity
            and
            aspect_ratio <= max_aspect_ratio
        ):
            output[
                component
            ] = False

    return (
        output.astype(
            np.uint8
        )
        * 255
    )





