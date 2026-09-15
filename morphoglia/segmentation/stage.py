from __future__ import annotations

# MG_RESUME_SEGMENTATION_V1

from ..checkpoint import (
    CheckpointJournal,
    file_identity,
    fingerprint,
    source_digest,
)

from ..io.preflight import (
    canonicalize_label_storage_2d,
    preflight_paths,
)
from ..core.instance_map import (
    binary_to_foreground,
    foreground_to_instance_map,
)


from dataclasses import dataclass
from pathlib import Path
import json
from uuid import uuid4

import cv2
import numpy as np
import tifffile

from ..core.instance_map import (
    binary_to_instance_map,
    validate_instance_map,
)

from ..io.overlays import (
    render_instance_map_diagnostic,
)
from ..metadata.files import (
    NomenclatureScan,
)
from ..preprocessing.config import (
    PreprocessingConfig,
)
from ..preprocessing.stage import (
    PreprocessingResult,
)

from .config import (
    SegmentationConfig,
)


@dataclass
class SegmentationResult:
    """
    Result produced by the Segmentation stage.
    """

    instance_paths: list[Path]
    input_mode: str
    skipped_others: int
    processed_count: int
    generation_id: str | None = None
    manifest_path: Path | None = None


def _load_image(
    path: Path,
) -> np.ndarray:
    """
    Load source pixels without interpreting their biological meaning.

    TIFF, PNG, and JPEG therefore enter Segmentation under the same
    dimensionality-validation rules.
    """

    suffix = path.suffix.lower()

    if suffix in {
        ".tif",
        ".tiff",
    }:

        image = tifffile.imread(
            str(path)
        )

    elif suffix in {
        ".png",
        ".jpg",
        ".jpeg",
    }:

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:

            raise ValueError(
                f"Could not load image: {path}"
            )

    else:

        raise ValueError(
            f"Unsupported image format: "
            f"{path.suffix}"
        )

    image = np.asarray(
        image
    )

    if image.size == 0:

        raise ValueError(
            f"Image is empty: {path.name}"
        )

    return image


def _to_2d_intensity(
    image: np.ndarray,
    path: Path,
) -> np.ndarray:
    """
    Convert representations of a 2D grayscale image into H x W form.

    Accepted
    --------
    H x W
        Native grayscale.

    H x W x 1
        Singleton channel.

    1 x H x W
        Singleton image plane.

    H x W x 3 / H x W x 4
        Accepted only when the RGB channels are effectively equivalent,
        meaning that the file is a grayscale/binary mask stored in a
        multichannel container.

    Rejected
    --------
    True multicolor RGB/RGBA images.
    Multi-plane image stacks.
    Other ambiguous dimensionalities.
    """

    image = np.asarray(
        image
    )

    if image.size == 0:

        raise ValueError(
            f"{path.name}: image contains no pixels."
        )

    # ------------------------------------------------------------------
    # Native grayscale
    # ------------------------------------------------------------------

    if image.ndim == 2:

        gray = image

    # ------------------------------------------------------------------
    # Singleton channel / plane
    # ------------------------------------------------------------------

    elif (
        image.ndim == 3
        and image.shape[-1] == 1
    ):

        gray = image[
            ...,
            0,
        ]

    elif (
        image.ndim == 3
        and image.shape[0] == 1
        and image.shape[-1] not in {
            3,
            4,
        }
    ):

        gray = image[
            0
        ]

    # ------------------------------------------------------------------
    # RGB / RGBA grayscale-equivalent image
    # ------------------------------------------------------------------

    elif (
        image.ndim == 3
        and image.shape[-1] in {
            3,
            4,
        }
    ):

        rgb = (
            image[
                ...,
                :3,
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        if not np.isfinite(
            rgb
        ).all():

            raise ValueError(
                f"{path.name}: multichannel image "
                "contains NaN or infinite values."
            )

        channel_spread = (
            rgb.max(
                axis=2
            )
            - rgb.min(
                axis=2
            )
        )

        dynamic_range = float(
            rgb.max()
            - rgb.min()
        )

        if np.issubdtype(
            image.dtype,
            np.integer,
        ):

            tolerance = max(
                1.0,
                0.01
                * max(
                    dynamic_range,
                    1.0,
                ),
            )

        else:

            tolerance = max(
                1e-6,
                0.01
                * max(
                    dynamic_range,
                    1e-12,
                ),
            )

        discordant_fraction = float(
            np.mean(
                channel_spread
                > tolerance
            )
        )

        # Permit tiny encoding/channel noise but never reinterpret an
        # actual color image as a morphology mask.
        max_discordant_fraction = 0.001

        if (
            discordant_fraction
            > max_discordant_fraction
        ):

            raise ValueError(
                f"{path.name}: input_mode='binary' "
                "received a genuine multichannel "
                f"image with shape {image.shape}. "
                f"{discordant_fraction:.2%} of pixels "
                "show meaningful RGB disagreement. "
                "A binary mask may be stored as RGB/RGBA "
                "only when its color channels encode the "
                "same grayscale mask."
            )

        gray = rgb.mean(
            axis=2
        )

    else:

        raise ValueError(
            f"{path.name}: cannot interpret image shape "
            f"{image.shape} as one 2D morphology mask. "
            "Multi-plane image stacks require explicit "
            "processing before MorphoGlia."
        )

    gray = np.asarray(
        gray
    )

    if gray.ndim != 2:

        raise RuntimeError(
            f"{path.name}: internal image normalization "
            f"produced shape {gray.shape} instead of 2D."
        )

    if np.issubdtype(
        gray.dtype,
        np.complexfloating,
    ):

        raise ValueError(
            f"{path.name}: complex-valued images "
            "cannot represent binary morphology masks."
        )

    values = gray.astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            f"{path.name}: image contains "
            "NaN or infinite values."
        )

    return gray


def _normalize_binary_image(
    image: np.ndarray,
    path: Path,
) -> np.ndarray:
    """
    Convert a declared binary source into canonical 2D uint8 values.

    Returns
    -------
    np.ndarray
        H x W uint8 array containing only 0 and 255.

    Notes
    -----
    Exact binary masks are accepted directly.

    Near-binary masks are also accepted. This handles lossy formats such
    as JPEG where originally binary edges can acquire small intensity
    deviations.

    Genuine continuous-intensity images are rejected rather than silently
    thresholded. Such images should use input_mode='raw'.

    Foreground polarity is NOT guessed here. ``invert`` remains the
    explicit user control.
    """

    gray = _to_2d_intensity(
        image=image,
        path=path,
    )

    values = gray.astype(
        np.float64,
        copy=False,
    )

    lo = float(
        values.min()
    )

    hi = float(
        values.max()
    )

    # ------------------------------------------------------------------
    # Constant mask
    # ------------------------------------------------------------------

    if hi == lo:

        if lo == 0.0:

            return np.zeros(
                values.shape,
                dtype=np.uint8,
            )

        return np.full(
            values.shape,
            255,
            dtype=np.uint8,
        )

    # ------------------------------------------------------------------
    # Normalize observed intensity range
    # ------------------------------------------------------------------

    normalized = (
        values - lo
    ) / (
        hi - lo
    )

    # ------------------------------------------------------------------
    # Binary-likeness check
    #
    # Pixels close to either observed endpoint are binary-like.
    # Intermediate pixels are tolerated only as a minority, principally
    # to support compression/anti-aliasing artifacts.
    # ------------------------------------------------------------------

    intermediate = (
        (normalized > 0.10)
        & (normalized < 0.90)
    )

    intermediate_fraction = float(
        np.mean(
            intermediate
        )
    )

    max_intermediate_fraction = 0.05

    if (
        intermediate_fraction
        > max_intermediate_fraction
    ):

        raise ValueError(
            f"{path.name}: input_mode='binary' "
            "does not appear to contain a binary "
            "or near-binary mask. "
            f"{intermediate_fraction:.2%} of pixels "
            "have intermediate intensities "
            "(maximum tolerated: "
            f"{max_intermediate_fraction:.0%}). "
            "If this is an intensity image, use "
            "input_mode='raw'."
        )

    binary = (
        normalized
        > 0.5
    )

    result = (
        binary.astype(
            np.uint8
        )
        * 255
    )

    if result.ndim != 2:

        raise RuntimeError(
            f"{path.name}: canonical binary "
            f"normalization produced shape "
            f"{result.shape}."
        )

    unique = set(
        np.unique(
            result
        ).tolist()
    )

    if not unique.issubset(
        {
            0,
            255,
        }
    ):

        raise RuntimeError(
            f"{path.name}: canonical binary "
            "normalization produced values "
            f"{sorted(unique)}."
        )

    return result


def _save_tiff(
    path: Path,
    image: np.ndarray,
) -> None:
    """Write one canonical map atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if temporary.exists():
        temporary.unlink()
    try:
        tifffile.imwrite(str(temporary), image)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _binary_foreground_2d(
    image: np.ndarray,
    path: Path,
    *,
    invert: bool,
) -> np.ndarray:
    """
    Return one canonical 2D boolean foreground mask.

    Normal 2D/singleton/RGB-grayscale-equivalent inputs retain the existing
    MorphoGlia binary interpretation.

    For a declared 2D binary image stored across several pages, every page is
    independently validated as binary. Foreground is interpreted with the
    existing MorphoGlia rule and maximum-projected in foreground space:

        projected_foreground(y, x) = OR_k foreground_k(y, x)

    For 0/255 white-on-black masks this is exactly an intensity maximum
    projection. Foreground-space projection also remains valid for the
    opposite binary intensity polarity.
    """

    array = np.asarray(
        image
    )

    if array.size == 0:
        raise ValueError(
            f"{path.name}: image contains no pixels."
        )

    direct = (
        array.ndim == 2
        or (
            array.ndim == 3
            and (
                array.shape[0] == 1
                or array.shape[-1] in {
                    1,
                    3,
                    4,
                }
            )
        )
    )

    if direct:
        binary = _normalize_binary_image(
            image=array,
            path=path,
        )

        foreground = binary_to_foreground(
            binary
        )

    else:
        # Multi-page grayscale: (..., Y, X)
        # Multi-page grayscale-equivalent RGB/RGBA: (..., Y, X, C)
        if (
            array.ndim >= 4
            and array.shape[-1] in {
                3,
                4,
            }
        ):
            height = int(
                array.shape[-3]
            )
            width = int(
                array.shape[-2]
            )
            channels = int(
                array.shape[-1]
            )

            planes = array.reshape(
                -1,
                height,
                width,
                channels,
            )

        else:
            if array.ndim < 3:
                raise ValueError(
                    f"{path.name}: cannot interpret storage shape "
                    f"{array.shape} as declared 2D binary data."
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

        foreground = np.zeros(
            (
                height,
                width,
            ),
            dtype=bool,
        )

        for plane_index, plane in enumerate(
            planes,
            start=1,
        ):
            try:
                binary_plane = _normalize_binary_image(
                    image=plane,
                    path=path,
                )

                plane_foreground = binary_to_foreground(
                    binary_plane
                )

            except Exception as exc:
                raise ValueError(
                    f"{path.name}: storage page {plane_index}/"
                    f"{len(planes)} is not a valid binary page: {exc}"
                ) from exc

            foreground |= plane_foreground

    if invert:
        foreground = np.logical_not(
            foreground
        )

    foreground = np.asarray(
        foreground,
        dtype=bool,
    )

    if foreground.ndim != 2:
        raise RuntimeError(
            f"{path.name}: 2D binary projection produced "
            f"shape {foreground.shape}."
        )

    return foreground


def run_segmentation(
    preprocessing_result: PreprocessingResult,
    metadata_scan: NomenclatureScan,
    output_dir: str | Path,
    preprocessing_config: PreprocessingConfig,
    config: SegmentationConfig,
    resume: bool = False,
) -> SegmentationResult:
    """
    Establish MorphoGlia's canonical 2D instance maps.

    raw / binary
        Foreground/background membership is normalized to a validated
        2D binary representation and converted to connected instances.

    labels
        Supplied positive integer labels are validated and preserved.
        No connected-component operation or renumbering is performed.

    Canonical output
    ----------------

    0
        background

    positive integer
        instance identity
    """

    output_dir = Path(
        output_dir
    )

    instance_dir = (
        output_dir
        / "Segmentation"
        / "_Instance_Maps"
    )

    instance_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_paths = list(
        preprocessing_result.prepared_paths
    )

    recognized = (
        metadata_scan.recognized
    )

    if len(
        prepared_paths
    ) != len(
        recognized
    ):

        raise ValueError(
            "Segmentation received a different number of "
            "prepared images and Metadata-recognized images: "
            f"{len(prepared_paths)} versus "
            f"{len(recognized)}."
        )

    mode = (
        preprocessing_result.input_mode
    )

    if mode not in {
        "binary",
        "raw",
        "labels",
    }:

        raise ValueError(
            f"Unsupported Segmentation input mode: {mode}"
        )

    # This also protects Segmentation when Preprocessing is OFF and
    # canonical/prepared inputs are loaded from an earlier run.
    prepared_preflight_mode = (
        "labels"
        if mode == "labels"
        else "binary"
    )

    preflight_paths(
        prepared_paths,
        input_mode=prepared_preflight_mode,
        spatial_dimension=(
            preprocessing_config.effective_spatial_dimension
        ),
        label="Segmentation input",
    )

    print()
    print("=" * 72)
    print("SEGMENTATION")
    print("=" * 72)
    print()

    print(
        f"Input mode:          {mode}"
    )

    if mode == "labels":

        print(
            "Instance operation:  Preserve supplied identities"
        )

    else:

        print(
            "Instance operation:  Connected components"
        )

    print()

    instance_paths: list[
        Path
    ] = []

    total = len(
        prepared_paths
    )

    stage_signature = fingerprint(
        "segmentation_resume_v1",
        config,
        {
            "input_mode": mode,
            "spatial_dimension": preprocessing_config.effective_spatial_dimension,
            "invert": preprocessing_config.effective_invert if mode == "binary" else False,
        },
        source_digest(Path(__file__)),
        source_digest(Path(__file__).resolve().parents[1] / "core" / "instance_map.py"),
        source_digest(Path(__file__).resolve().parents[1] / "io" / "overlays.py"),
    )
    journal = CheckpointJournal(
        output_dir=output_dir,
        stage="segmentation",
        resume=resume,
        stage_signature=stage_signature,
    )
    recomputed_count = 0

    for index, (
        prepared_path,
        interpretation,
    ) in enumerate(
        zip(
            prepared_paths,
            recognized,
        ),
        start=1,
    ):

        prepared_path = Path(
            prepared_path
        )

        canonical_filename = (
            interpretation.canonical_filename
        )

        if canonical_filename is None:

            raise ValueError(
                "Segmentation received a recognized "
                "Metadata entry without a canonical filename."
            )

        canonical_stem = Path(
            canonical_filename
        ).stem

        output_path = (
            instance_dir
            / f"{canonical_stem}.tif"
        )

        diagnostic_path = (
            output_dir
            / "Segmentation"
            / f"{canonical_stem}.png"
        )

        item_signature = fingerprint(
            stage_signature,
            file_identity(prepared_path),
            canonical_filename,
        )
        if journal.reusable_record(
            item_id=canonical_stem,
            item_signature=item_signature,
            outputs=[output_path, diagnostic_path],
        ) is not None:
            print(f"[{index}/{total}] RESUME {prepared_path.name}")
            print(f"    -> {output_path.name}")
            instance_paths.append(output_path)
            continue

        print(
            f"[{index}/{total}] "
            f"{prepared_path.name}"
        )

        print(
            f"    -> {output_path.name}"
        )

        image = _load_image(
            prepared_path
        )

        # ==============================================================
        # EXTERNAL INSTANCE LABELS
        # ==============================================================

        if mode == "labels":

            if preprocessing_config.invert not in (
                None,
                False,
            ):

                raise ValueError(
                    "invert does not apply to "
                    "input_mode='labels'."
                )

            image = canonicalize_label_storage_2d(
                image,
                source_name=prepared_path.name,
            )

            instance_map = (
                validate_instance_map(
                    image
                )
            )

        # ==============================================================
        # GENERATED INSTANCES
        # ==============================================================

        else:

            invert = (
                preprocessing_config.effective_invert
                if mode == "binary"
                else False
            )

            foreground = _binary_foreground_2d(
                image=image,
                path=prepared_path,
                invert=invert,
            )

            instance_map = (
                foreground_to_instance_map(
                    foreground,
                    connectivity=(
                        config.connectivity
                    ),
                )
            )

            # The connected-component result must obey the same canonical
            # instance-map contract as externally supplied labels.
            instance_map = (
                validate_instance_map(
                    instance_map
                )
            )

        # ==============================================================
        # ABSOLUTE CANONICAL OUTPUT INVARIANT
        # ==============================================================

        instance_map = np.asarray(
            instance_map
        )

        if instance_map.ndim != 2:

            raise RuntimeError(
                f"{prepared_path.name}: Segmentation "
                "attempted to produce a non-2D canonical "
                f"instance map with shape "
                f"{instance_map.shape}. "
                "This is an internal MorphoGlia error."
            )

        _save_tiff(
            output_path,
            instance_map,
        )

        # Verify the persisted representation as well.
        persisted = np.asarray(
            tifffile.imread(
                str(
                    output_path
                )
            )
        )

        if persisted.ndim != 2:

            raise RuntimeError(
                f"{output_path.name}: persisted canonical "
                "instance map is not 2D; "
                f"loaded shape {persisted.shape}."
            )

        render_instance_map_diagnostic(
            instance_map=instance_map,
            output_path=diagnostic_path,
        )

        instance_paths.append(
            output_path
        )
        journal.commit(
            item_id=canonical_stem,
            item_signature=item_signature,
            outputs=[output_path, diagnostic_path],
        )
        recomputed_count += 1

    print()
    print("=" * 72)
    print("SEGMENTATION COMPLETE")
    print("=" * 72)
    print()

    print(
        "Instance maps:   ",
        len(
            instance_paths
        ),
    )

    print(
        "Output:          ",
        instance_dir,
    )

    print()

    # Preserve the prior generation only when every image was reused.
    manifest_path = (
        output_dir
        / "Segmentation"
        / "segmentation_manifest.json"
    )
    expected_image_ids = [path.stem for path in instance_paths]
    generation_id = None
    if resume and recomputed_count == 0 and manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                previous_manifest = json.load(file)
            if (
                str(previous_manifest.get("resume_signature", "")) == stage_signature
                and str(previous_manifest.get("input_mode", "")) == str(mode)
                and [str(v) for v in previous_manifest.get("image_ids", [])] == expected_image_ids
            ):
                previous_generation = str(previous_manifest.get("generation_id", "")).strip()
                if previous_generation:
                    generation_id = previous_generation
        except Exception:
            generation_id = None
    if generation_id is None:
        generation_id = uuid4().hex

    manifest_payload = {
        "generation_id": generation_id,
        "resume_signature": stage_signature,
        "input_mode": str(mode),
        "processed_count": int(len(instance_paths)),
        "image_ids": expected_image_ids,
    }

    temporary_manifest = (
        manifest_path
        .with_suffix(
            ".json.tmp"
        )
    )

    with temporary_manifest.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest_payload,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    temporary_manifest.replace(
        manifest_path
    )

    return SegmentationResult(
        instance_paths=instance_paths,
        input_mode=mode,
        skipped_others=(
            metadata_scan.others_count
        ),
        processed_count=len(
            instance_paths
        ),
        generation_id=generation_id,
        manifest_path=manifest_path,
    )
