from __future__ import annotations

import numpy as np

from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


# ======================================================================
# LABEL-MAP VALIDATION
# ======================================================================

def validate_instance_map(
    labels: np.ndarray,
) -> np.ndarray:
    """
    Validate an externally supplied instance-label map.

    Semantic contract
    -----------------
    0
        Background.

    Positive integer
        Instance identity.

    IMPORTANT
    ---------
    Label values are identities.

    Therefore this function does NOT:

        - run connected components,
        - split disconnected labels,
        - merge labels,
        - renumber labels.

    A supplied label value must preserve its meaning.

    Supports arbitrary spatial dimensionality, including:

        2D: (Y, X)
        3D: (Z, Y, X)
    """

    labels = np.asarray(
        labels
    )


    if labels.ndim < 2:

        raise ValueError(
            "Instance maps must have at least "
            "two spatial dimensions."
        )


    if not np.issubdtype(
        labels.dtype,
        np.number,
    ):

        raise TypeError(
            "Instance-map labels must be numeric."
        )


    if not np.all(
        np.isfinite(
            labels
        )
    ):

        raise ValueError(
            "Instance maps cannot contain NaN or infinity."
        )


    if np.any(
        labels < 0
    ):

        raise ValueError(
            "Instance-map labels cannot be negative."
        )


    if not np.all(
        labels
        == np.floor(
            labels
        )
    ):

        raise ValueError(
            "Instance-map labels must be integers."
        )


    return labels


# ======================================================================
# INSTANCE IDENTITIES
# ======================================================================

def instance_labels(
    labels: np.ndarray,
) -> np.ndarray:
    """
    Return the positive instance identities present in a label map.

    Background label 0 is excluded.
    """

    labels = validate_instance_map(
        labels
    )


    values = np.unique(
        labels
    )


    return values[
        values > 0
    ]


# ======================================================================
# BINARY → FOREGROUND
# ======================================================================

def binary_to_foreground(
    image: np.ndarray,
) -> np.ndarray:
    """
    Interpret a binary-like image as foreground/background.

    IMPORTANT
    ---------
    Pixel value is NOT treated as object identity.

    Foreground is inferred as the minority side of the intensity
    separation. This supports both:

        dark objects / light background
        light objects / dark background

    Returns
    -------
    bool ndarray
        True = foreground.

    This operation does not identify individual objects.
    """

    image = np.asarray(
        image
    )


    if image.ndim < 2:

        raise ValueError(
            "Binary input must have at least "
            "two spatial dimensions."
        )


    values = image.astype(
        np.float64,
        copy=False,
    )


    finite = np.isfinite(
        values
    )


    if not finite.any():

        raise ValueError(
            "Binary input contains no finite pixels."
        )


    observed = values[
        finite
    ]


    minimum = float(
        observed.min()
    )

    maximum = float(
        observed.max()
    )


    if minimum == maximum:

        return np.zeros(
            image.shape,
            dtype=bool,
        )


    unique_values = np.unique(
        observed
    )


    if unique_values.size <= 8:

        threshold = (
            minimum
            + maximum
        ) / 2.0

    else:

        threshold = float(
            threshold_otsu(
                observed
            )
        )


    dark = (
        values
        <= threshold
    )

    light = (
        values
        > threshold
    )


    dark_count = int(
        np.count_nonzero(
            dark
        )
    )

    light_count = int(
        np.count_nonzero(
            light
        )
    )


    return (
        dark
        if dark_count <= light_count
        else light
    )


# ======================================================================
# CANONICAL GENERATED INSTANCE ORDER
# ======================================================================

def _canonicalize_generated_labels(
    labels: np.ndarray,
) -> np.ndarray:
    """
    Canonicalize labels created internally by MorphoGlia.

    Connected-component libraries may assign numeric labels in
    implementation-specific order.

    MorphoGlia instead orders generated instances by the
    lexicographically first occupied spatial coordinate.

    Because NumPy C-order flattening follows:

        2D: (Y, X)
        3D: (Z, Y, X)

    the first flat occurrence gives exactly that ordering.

    Only internally generated labels are canonicalized.

    Externally supplied instance labels must never be passed through
    this function because their integer values are identities.
    """

    labels = np.asarray(
        labels
    )


    if labels.size == 0:

        return labels.astype(
            np.int32,
            copy=False,
        )


    maximum = int(
        labels.max()
    )


    if maximum == 0:

        return labels.astype(
            np.int32,
            copy=False,
        )


    flat = labels.ravel()


    positions = np.flatnonzero(
        flat > 0
    )


    observed_labels = flat[
        positions
    ].astype(
        np.int64,
        copy=False,
    )


    first_position = np.full(
        maximum + 1,
        flat.size,
        dtype=np.int64,
    )


    np.minimum.at(
        first_position,
        observed_labels,
        positions,
    )


    existing = np.flatnonzero(
        first_position[
            1:
        ] < flat.size
    ) + 1


    ordered_old_labels = existing[
        np.argsort(
            first_position[
                existing
            ],
            kind="stable",
        )
    ]


    remap = np.zeros(
        maximum + 1,
        dtype=np.int32,
    )


    remap[
        ordered_old_labels
    ] = np.arange(
        1,
        len(
            ordered_old_labels
        ) + 1,
        dtype=np.int32,
    )


    return remap[
        labels
    ]


# ======================================================================
# BINARY → INSTANCE MAP
# ======================================================================

def foreground_to_instance_map(
    foreground: np.ndarray,
    *,
    connectivity: int | None = None,
) -> np.ndarray:
    """
    Convert an already interpreted boolean foreground mask into instances.

    True means foreground. No thresholding or polarity inference occurs here.
    """

    foreground = np.asarray(
        foreground,
        dtype=bool,
    )

    if foreground.ndim < 2:
        raise ValueError(
            "Foreground masks must have at least two spatial dimensions."
        )

    ndim = foreground.ndim

    if connectivity is None:
        connectivity = ndim

    connectivity = int(
        connectivity
    )

    if not (
        1
        <= connectivity
        <= ndim
    ):
        raise ValueError(
            "connectivity must satisfy "
            f"1 <= connectivity <= {ndim}."
        )

    structure = (
        ndi.generate_binary_structure(
            ndim,
            connectivity,
        )
    )

    labels, _count = ndi.label(
        foreground,
        structure=structure,
    )

    return _canonicalize_generated_labels(
        labels
    )


def binary_to_instance_map(
    image: np.ndarray,
    *,
    connectivity: int | None = None,
    invert: bool = False,
) -> np.ndarray:
    """
    Convert binary foreground/background data into an instance map.

    Existing MorphoGlia semantics are preserved:
    infer foreground, apply the explicit invert override, then label
    connected components.
    """

    foreground = binary_to_foreground(
        image
    )

    if invert:
        foreground = np.logical_not(
            foreground
        )

    return foreground_to_instance_map(
        foreground,
        connectivity=connectivity,
    )


__all__ = [
    "validate_instance_map",
    "instance_labels",
    "binary_to_foreground",
    "binary_to_instance_map",
]
