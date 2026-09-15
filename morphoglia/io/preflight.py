from __future__ import annotations

"""Fast image-shape preflight and 2D binary storage canonicalization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from PIL import Image

PREFLIGHT_VERSION = "MG_2D3D_PREFLIGHT_V1"


@dataclass(frozen=True)
class ImageStorageInfo:
    path: Path
    shape: tuple[int, ...]
    dtype: str
    axes: str
    storage: str


@dataclass(frozen=True)
class PreflightResult:
    label: str
    spatial_dimension: str
    input_mode: str
    total_images: int
    direct_2d: int
    color_2d: int
    singleton_squeezed: int
    projected_raw: int
    projected_binary: int
    projected_labels: int


_CACHE = {}


def _dimension(value):
    x = str(value).strip().lower()
    aliases = {"2": "2d", "2-d": "2d", "2d": "2d", "3": "3d", "3-d": "3d", "3d": "3d"}
    if x not in aliases:
        raise ValueError(f"Image geometry must be '2d' or '3d'; received {value!r}.")
    return aliases[x]


def _mode(value):
    x = str(value).strip().lower()
    if x not in {"raw", "binary", "labels"}:
        raise ValueError(f"Input type must be raw, binary, or labels; received {value!r}.")
    return x


def inspect_image_storage(path: str | Path) -> ImageStorageInfo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(str(path)) as tif:
            if not tif.series:
                raise ValueError(f"{path.name}: TIFF contains no readable image series.")
            series = tif.series[0]
            shape = tuple(int(v) for v in series.shape)
            dtype = str(series.dtype)
            axes = str(getattr(series, "axes", "") or "")
        return ImageStorageInfo(path, shape, dtype, axes, "TIFF")

    if suffix in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as image:
            w, h = image.size
            mode = str(image.mode)
            frames = int(getattr(image, "n_frames", 1))
            channels = {"RGB": 3, "RGBA": 4, "CMYK": 4}.get(mode)
            if frames > 1 and channels is None:
                shape, axes = (frames, h, w), "QYX"
            elif frames > 1:
                shape, axes = (frames, h, w, channels), "QYXS"
            elif channels is not None:
                shape, axes = (h, w, channels), "YXS"
            else:
                shape, axes = (h, w), "YX"
        return ImageStorageInfo(path, tuple(shape), mode, axes, suffix[1:].upper())

    raise ValueError(f"{path.name}: unsupported image extension {suffix!r}.")


def _squeezed(shape):
    out = tuple(int(v) for v in shape if int(v) != 1)
    return out if out else (1,)


def _classify_2d(info: ImageStorageInfo, mode: str) -> str:
    original = tuple(info.shape)
    shape = _squeezed(original)

    if len(shape) == 2:
        return "singleton" if shape != original else "direct"

    if len(shape) == 3 and shape[-1] in {3, 4} and mode in {"raw", "binary"}:
        return "color"

    if len(shape) >= 3:

        if mode == "raw":
            return "project_raw"

        if mode == "binary":
            return "project_binary"

        if mode == "labels":

            # H x W x RGB/RGBA is not an integer instance-label stack.
            if (
                len(shape) == 3
                and shape[-1] in {
                    3,
                    4,
                }
            ):
                raise ValueError(
                    f"{info.path.name}: color storage cannot be interpreted "
                    "as integer instance labels."
                )

            return "project_labels"

    axes = f" with axes {info.axes!r}" if info.axes else ""

    raise ValueError(
        f"{info.path.name}: cannot reconcile declared 2D geometry "
        f"with storage shape {info.shape}{axes}."
    )


def preflight_paths(
    paths: Iterable[str | Path],
    *,
    input_mode: str,
    spatial_dimension: str,
    label: str = "Input",
) -> PreflightResult:
    mode = _mode(input_mode)
    dimension = _dimension(spatial_dimension)
    paths = [Path(p).expanduser() for p in paths]

    if dimension == "3d":
        raise NotImplementedError(
            "3D image geometry was selected, but 3D analysis is not implemented "
            "in this MorphoGlia release. Choose 2D for XY morphology analysis."
        )

    if not paths:
        raise ValueError(f"{label} preflight received no images.")

    key = (tuple(str(p.resolve()) for p in paths), mode, dimension)
    if key in _CACHE:
        return _CACHE[key]

    counts = {
        "direct": 0,
        "singleton": 0,
        "color": 0,
        "project_raw": 0,
        "project_binary": 0,
        "project_labels": 0,
    }
    errors = []

    for path in paths:
        try:
            info = inspect_image_storage(path)
            counts[_classify_2d(info, mode)] += 1
        except Exception as exc:
            errors.append(str(exc))

    print()
    print("=" * 72)
    print("INPUT")
    print("=" * 72)
    print(f"Dataset:             {label}")
    print(f"Image:      {dimension.upper()}")
    print(f"Input type:          {mode}")
    print(f"Images checked:      {len(paths)}")
    print(f"Direct 2D:           {counts['direct']}")
    if counts["singleton"]:
        print(f"Singleton squeeze:   {counts['singleton']}")
    if counts["color"]:
        print(f"2D color storage:    {counts['color']}")
    if counts["project_raw"]:
        print(
            f"Raw projections:     {counts['project_raw']} "
            "(extra storage planes -> maximum intensity)"
        )

    if counts["project_binary"]:
        print(
            f"Binary projections:  {counts['project_binary']} "
            "(extra storage planes -> foreground union)"
        )

    if counts["project_labels"]:
        print(
            f"Label projections:   {counts['project_labels']} "
            "(extra storage planes -> identity-preserving projection)"
        )

    if errors:
        print(f"Prescan errors:    {len(errors)}")
        preview = "\n".join(f"  - {x}" for x in errors[:12])
        if len(errors) > 12:
            preview += f"\n  - ... and {len(errors) - 12} more"
        raise ValueError(
            "Input prescan failed before image processing began:\n" + preview
        )

    print("Status:              PASS")
    print("=" * 72)
    print()

    result = PreflightResult(
        label=str(label),
        spatial_dimension=dimension,
        input_mode=mode,
        total_images=len(paths),
        direct_2d=counts["direct"],
        color_2d=counts["color"],
        singleton_squeezed=counts["singleton"],
        projected_raw=counts["project_raw"],
        projected_binary=counts["project_binary"],
        projected_labels=counts["project_labels"],
    )
    _CACHE[key] = result
    return result


def preflight_metadata_input(*, input_dir, metadata_scan, preprocessing_config):
    base = Path(input_dir)
    paths = [
        base / interpretation.original_filename
        for interpretation in metadata_scan.recognized
    ]
    return preflight_paths(
        paths,
        input_mode=preprocessing_config.effective_input_mode,
        spatial_dimension=preprocessing_config.effective_spatial_dimension,
        label="Source images",
    )



# ======================================================================
# RAW -> DECLARED 2D
# ======================================================================

def canonicalize_raw_storage_2d(
    image,
    *,
    source_name: str = "<image>",
) -> np.ndarray:
    """
    Return one biologically 2D raw intensity image.

    Declared-2D is the biological prior. Any additional non-spatial
    storage planes are therefore maximum-intensity projected before
    preprocessing begins.

    Examples
    --------
    (Y, X)
        unchanged

    (1, Y, X)
        singleton squeeze

    (N, Y, X)
        maximum intensity over N

    (..., Y, X)
        maximum intensity over every leading storage axis
    """

    array = np.asarray(
        image
    )

    original_shape = (
        array.shape
    )

    original_dtype = (
        array.dtype
    )

    if array.size == 0:
        raise ValueError(
            f"{source_name}: image contains no pixels."
        )

    array = np.squeeze(
        array
    )

    if array.ndim < 2:
        raise ValueError(
            f"{source_name}: cannot interpret storage shape "
            f"{original_shape} as declared 2D raw data."
        )

    # --------------------------------------------------------------
    # Conventional RGB / RGBA 2D storage.
    #
    # Color channels are reduced to grayscale intensity; they are not
    # treated as biological Z planes.
    # --------------------------------------------------------------

    if (
        array.ndim >= 3
        and array.shape[-1] in {
            3,
            4,
        }
    ):

        array = (
            array[
                ...,
                :3,
            ]
            .astype(
                np.float64
            )
            .mean(
                axis=-1
            )
        )

    if array.ndim == 2:

        if np.issubdtype(
            original_dtype,
            np.integer,
        ):

            info = np.iinfo(
                original_dtype
            )

            return np.clip(
                array,
                info.min,
                info.max,
            ).astype(
                original_dtype
            )

        return array.astype(
            original_dtype,
            copy=False,
        )

    values = np.asarray(
        array,
        dtype=np.float64,
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            f"{source_name}: raw image contains NaN or infinite values."
        )

    # The final two axes are the declared biological Y, X plane.
    # Everything before them is storage depth and is projected.
    projection_axes = tuple(
        range(
            values.ndim - 2
        )
    )

    projected = np.max(
        values,
        axis=projection_axes,
    )

    if projected.ndim != 2:
        raise RuntimeError(
            f"{source_name}: raw projection produced "
            f"shape {projected.shape}; expected 2D."
        )

    if np.issubdtype(
        original_dtype,
        np.integer,
    ):

        info = np.iinfo(
            original_dtype
        )

        return np.clip(
            projected,
            info.min,
            info.max,
        ).astype(
            original_dtype
        )

    return projected.astype(
        original_dtype,
        copy=False,
    )


# ======================================================================
# LABELS -> DECLARED 2D
# ======================================================================

def canonicalize_label_storage_2d(
    image,
    *,
    source_name: str = "<image>",
) -> np.ndarray:
    """
    Return one biologically 2D instance-label image.

    Label values are categorical identities, so numerical maximum
    projection is NOT valid.

    Projection rule
    ---------------
    At every XY pixel:

        all planes zero
            -> output 0

        one non-zero identity
            -> preserve that identity

        same non-zero identity repeated across planes
            -> preserve that identity

        different non-zero identities at the same XY position
            -> error

    This preserves label identities without silently resolving conflicts.
    """

    array = np.asarray(
        image
    )

    original_shape = (
        array.shape
    )

    original_dtype = (
        array.dtype
    )

    if array.size == 0:
        raise ValueError(
            f"{source_name}: image contains no pixels."
        )

    array = np.squeeze(
        array
    )

    if array.ndim < 2:
        raise ValueError(
            f"{source_name}: cannot interpret storage shape "
            f"{original_shape} as declared 2D label data."
        )

    if not (
        np.issubdtype(
            array.dtype,
            np.integer,
        )
        or np.issubdtype(
            array.dtype,
            np.bool_,
        )
    ):
        raise ValueError(
            f"{source_name}: label images must contain integer identities."
        )

    if np.any(
        array < 0
    ):
        raise ValueError(
            f"{source_name}: label identities must be >= 0."
        )

    if array.ndim == 2:
        return array.astype(
            original_dtype,
            copy=False,
        )

    if (
        array.ndim == 3
        and array.shape[-1] in {
            3,
            4,
        }
    ):
        raise ValueError(
            f"{source_name}: color storage cannot be interpreted "
            "as integer instance labels."
        )

    height = int(
        array.shape[-2]
    )

    width = int(
        array.shape[-1]
    )

    planes = array.reshape(
        -1,
        height,
        width,
    )

    projected = np.zeros(
        (
            height,
            width,
        ),
        dtype=array.dtype,
    )

    for plane_index, plane in enumerate(
        planes,
        start=1,
    ):

        occupied = (
            plane != 0
        )

        conflict = (
            occupied
            & (
                projected != 0
            )
            & (
                projected != plane
            )
        )

        if np.any(
            conflict
        ):

            conflict_count = int(
                np.count_nonzero(
                    conflict
                )
            )

            raise ValueError(
                f"{source_name}: declared-2D label projection is "
                f"ambiguous at {conflict_count} XY pixel(s). "
                f"Storage plane {plane_index}/{len(planes)} contains "
                "a different non-zero instance identity at positions "
                "already occupied by another label."
            )

        fill = (
            occupied
            & (
                projected == 0
            )
        )

        projected[
            fill
        ] = plane[
            fill
        ]

    return projected.astype(
        original_dtype,
        copy=False,
    )


def canonicalize_binary_storage_2d(
    image,
    *,
    invert: bool,
    source_name: str = "<image>",
) -> np.ndarray:
    """
    Return one biologically 2D binary storage image.

    MorphoGlia polarity contract:
        invert=False -> WHITE objects on BLACK background
        invert=True  -> BLACK objects on WHITE background

    Extra non-spatial planes are combined as a foreground logical union.
    """
    array = np.asarray(image)
    original_dtype = array.dtype
    array = np.squeeze(array)

    if array.ndim == 2:
        return array

    if array.ndim >= 3 and array.shape[-1] in {3, 4}:
        array = array[..., :3].astype(np.float64).mean(axis=-1)
        if array.ndim == 2:
            return array

    if array.ndim < 2:
        raise ValueError(
            f"{source_name}: cannot interpret shape {np.asarray(image).shape} "
            "as one 2D binary mask."
        )

    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).any():
        raise ValueError(f"{source_name}: image contains no finite pixels.")

    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    reduce_axes = tuple(range(values.ndim - 2))

    if lo == hi:
        output = np.full(values.shape[-2:], lo, dtype=np.float64)
    else:
        threshold = lo + 0.5 * (hi - lo)
        if bool(invert):
            foreground = values <= threshold
            fg_value, bg_value = lo, hi
        else:
            foreground = values > threshold
            fg_value, bg_value = hi, lo

        union = np.any(foreground, axis=reduce_axes)
        output = np.where(union, fg_value, bg_value)

    if np.issubdtype(original_dtype, np.bool_):
        return output.astype(bool)
    if np.issubdtype(original_dtype, np.integer):
        return output.astype(original_dtype, copy=False)
    return output


__all__ = [
    "PREFLIGHT_VERSION",
    "ImageStorageInfo",
    "PreflightResult",
    "inspect_image_storage",
    "preflight_paths",
    "preflight_metadata_input",
    "canonicalize_binary_storage_2d",
]
