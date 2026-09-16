from __future__ import annotations

# MG_RESUME_INSTANCE_POSTPROCESSING_V1

import io
import zipfile

from dataclasses import dataclass
from pathlib import Path
import errno
import hashlib
import json
import math
import os
import shutil
import time
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import tifffile

from matplotlib import colormaps

from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

from ..core.instance_map import validate_instance_map
from ..checkpoint import (
    CheckpointJournal,
    atomic_write_bytes,
    file_identity,
    fingerprint,
    source_digest,
)
from ..morphometrics.qc.grid import plot_qc_grid, select_qc_examples
from ..morphometrics.qc.size import score_size_outliers
from ..morphometrics.qc.tubular import score_tubular_objects

from .config import InstanceRefinementConfig


@dataclass
class InstanceRefinementResult:
    effective_instance_paths: list[Path]
    effective_instance_dir: Path | None
    effective_instance_paths_by_image: dict[str, Path]
    modified_instance_paths: list[Path]
    modified_count: int
    reconnect_count: int
    split_parent_count: int
    removed_small_count: int
    removed_large_count: int
    removed_tubular_count: int
    manifest_path: Path
    reused: bool = False


@dataclass(frozen=True)
class _MaskSnapshot:
    """Compact event-local mask geometry using an exclusive-end bbox."""

    label: int | None
    y0: int
    x0: int
    y1: int
    x1: int
    mask: np.ndarray


@dataclass(frozen=True)
class _RefinementEventGeometry:
    """Exact geometry associated with one successful refinement event."""

    event_id: str
    image_id: str
    operation: str
    snapshots: tuple[_MaskSnapshot, ...]

    # Optional semantic context used only by event-driven QC rendering.
    # Existing removal events need only ``snapshots``.
    before_snapshot: _MaskSnapshot | None = None
    result_snapshot: _MaskSnapshot | None = None
    changed_snapshot: _MaskSnapshot | None = None


_REMOVAL_OPERATIONS = frozenset(
    {
        "remove_small",
        "remove_tubular",
        "remove_large",
    }
)

_RETAINED_OPERATIONS = frozenset(
    {
        "reconnect",
        "split",
    }
)

_MODIFYING_OPERATIONS = (
    _REMOVAL_OPERATIONS
    | _RETAINED_OPERATIONS
)

_EVENT_COLUMNS = [
    "event_id",
    "image_id",
    "operation",
    "status",
    "success",
    "changed",
    "source_label",
    "target_label",
    "parent_label",
    "result_labels",
    "gap_px",
    "soma_y",
    "soma_x",
    "source_endpoint_y",
    "source_endpoint_x",
    "target_endpoint_y",
    "target_endpoint_x",
    "bridge_radius_px",
    "pixels_added",
    "pixels_removed",
    "pixels_relabelled",
    "pixels_changed",
    "radial_extent_px",
    "orphan_radius_px",
    "seed_count",
    "bbox_y0",
    "bbox_x0",
    "bbox_y1_exclusive",
    "bbox_x1_exclusive",
    "reason",
]

_QC_BOX_LINE_WIDTH = 5
_QC_BOX_PADDING_PX = 3
_QC_MIN_BOX_SIZE_PX = 15


def _mask_snapshot(
    mask: np.ndarray,
    *,
    label: int | None,
) -> _MaskSnapshot | None:
    """Copy only the occupied crop needed to reproduce event geometry."""

    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)

    if ys.size == 0:
        return None

    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    return _MaskSnapshot(
        label=(
            None
            if label is None
            else int(label)
        ),
        y0=y0,
        x0=x0,
        y1=y1,
        x1=x1,
        mask=mask[y0:y1, x0:x1].copy(),
    )


def _snapshot_bbox_fields(
    snapshot: _MaskSnapshot | None,
) -> dict[str, int | str]:
    """Return an unambiguous exclusive-end bbox for the event table."""

    if snapshot is None:
        return {
            "bbox_y0": "",
            "bbox_x0": "",
            "bbox_y1_exclusive": "",
            "bbox_x1_exclusive": "",
        }

    return {
        "bbox_y0": int(snapshot.y0),
        "bbox_x0": int(snapshot.x0),
        "bbox_y1_exclusive": int(snapshot.y1),
        "bbox_x1_exclusive": int(snapshot.x1),
    }


# ======================================================================
# INSTANCE-MAP I/O
# ======================================================================


def _load_instance_map(path: Path) -> np.ndarray:
    labels = np.asarray(tifffile.imread(str(path)))
    labels = validate_instance_map(labels)
    if labels.ndim != 2:
        raise ValueError(
            f"Instance Refinement requires 2D maps; {path.name!r} has "
            f"shape {labels.shape}."
        )
    return labels.astype(np.int32, copy=False)


def _save_instance_map(path: Path, labels: np.ndarray) -> None:
    """Atomically write and read-back validate one refined instance map."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected = np.asarray(
        labels,
        dtype=np.int32,
    )

    temporary = path.with_name(
        f"{path.stem}.tmp{path.suffix}"
    )

    if temporary.exists():
        temporary.unlink()

    try:

        tifffile.imwrite(
            str(
                temporary
            ),
            expected,
        )

        persisted = _load_instance_map(
            temporary
        )

        if not np.array_equal(
            persisted,
            expected,
        ):
            raise RuntimeError(
                "Persisted refined instance map differs from "
                f"the computed map: {path}"
            )

        temporary.replace(
            path
        )

    finally:

        if temporary.exists():
            temporary.unlink()


def _positive_labels(labels: np.ndarray) -> np.ndarray:
    values = np.unique(labels)
    return values[values > 0].astype(int)


def _count_instances(labels: np.ndarray) -> int:
    return int(len(_positive_labels(labels)))


# ======================================================================
# OBJECT TABLE / EXISTING QC DETECTORS
# ======================================================================


def _object_records(
    image_id: str,
    labels: np.ndarray,
) -> list[dict]:
    """
    Build the minimal object table for one already-loaded instance map.

    Keeping this separate lets refinement update only images whose maps
    actually changed instead of re-reading the full dataset between phases.
    """

    labels = np.asarray(
        labels,
        dtype=np.int32,
    )

    records: list[dict] = []

    for map_label in _positive_labels(
        labels
    ):

        mask = (
            labels
            == int(
                map_label
            )
        )

        ys, xs = np.where(
            mask
        )

        if len(
            ys
        ) == 0:

            continue

        y0 = int(
            ys.min()
        )

        y1 = int(
            ys.max()
        ) + 1

        x0 = int(
            xs.min()
        )

        x1 = int(
            xs.max()
        ) + 1

        roi = np.where(
            mask[
                y0:y1,
                x0:x1,
            ],
            255,
            0,
        ).astype(
            np.uint8
        )

        distance = cv2.distanceTransform(
            (roi > 0).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        records.append(
            {
                "image_id": str(
                    image_id
                ),
                "cell_id": (
                    f"{image_id}_label_"
                    f"{int(map_label)}"
                ),
                "map_label": int(
                    map_label
                ),
                "component_area": int(
                    mask.sum()
                ),
                "object_radius_max": float(
                    distance.max()
                ),
                "roi": roi,
            }
        )

    return records


def _object_table(
    instance_paths: list[Path],
    *,
    print_progress: bool = False,
    progress_label: str = "Inspecting",
) -> pd.DataFrame:
    """
    Build the initial object table with one disk read per source image.

    Later refinement phases update only the rows belonging to modified
    images through ``_refresh_object_table``; the complete source dataset
    is not re-read between tubular, large, and small actions.
    """

    records: list[dict] = []

    paths = [
        Path(
            path
        )
        for path in instance_paths
    ]

    total = len(
        paths
    )

    for index, path in enumerate(
        paths,
        start=1,
    ):

        if print_progress:

            print(
                f"[{index}/{total}] "
                f"{progress_label}: "
                f"{path.name}"
            )

        labels = _load_instance_map(
            path
        )

        records.extend(
            _object_records(
                path.stem,
                labels,
            )
        )

    table = pd.DataFrame(
        records
    )

    if table.empty:

        raise ValueError(
            "Instance Refinement received no objects."
        )

    return table


def _refresh_object_table(
    table: pd.DataFrame,
    *,
    image_ids: set[str],
    working_maps: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Refresh object rows only for images modified in the preceding phase.

    The robust QC statistics are still recomputed globally afterward, but
    unchanged image masks are not loaded from disk again.
    """

    image_ids = {
        str(
            image_id
        )
        for image_id in image_ids
    }

    if not image_ids:

        return table

    base_columns = [
        "image_id",
        "cell_id",
        "map_label",
        "component_area",
        "object_radius_max",
    ]

    remaining = (
        table.loc[
            ~table[
                "image_id"
            ].astype(str).isin(
                image_ids
            ),
            base_columns,
        ]
        .copy()
    )

    replacement_records: list[
        dict
    ] = []

    for image_id in sorted(
        image_ids
    ):

        if image_id not in working_maps:

            raise KeyError(
                "Missing working instance map for "
                f"modified image {image_id!r}."
            )

        replacement_records.extend(
            _object_records(
                image_id,
                working_maps[
                    image_id
                ],
            )
        )

    replacement = pd.DataFrame(
        replacement_records,
        columns=base_columns,
    )

    refreshed = pd.concat(
        [
            remaining,
            replacement,
        ],
        axis=0,
        ignore_index=True,
    )

    if refreshed.empty:

        raise ValueError(
            "Instance Refinement produced no objects "
            "while refreshing QC geometry."
        )

    return refreshed


def _working_map(
    image_id: str,
    *,
    source_by_id: dict[str, Path],
    working_maps: dict[str, np.ndarray],
    original_cache: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Return a mutable current map without writing intermediate TIFFs.

    Only candidate images are loaded after the initial dataset scan.
    """

    image_id = str(
        image_id
    )

    if image_id in working_maps:

        return np.asarray(
            working_maps[
                image_id
            ],
            dtype=np.int32,
        ).copy()

    if image_id not in source_by_id:

        raise KeyError(
            f"No source instance map for {image_id!r}."
        )

    original = _load_instance_map(
        source_by_id[
            image_id
        ]
    )

    original_cache.setdefault(
        image_id,
        original.copy(),
    )

    return original.copy()


def _score_qc(
    table: pd.DataFrame,
    config: InstanceRefinementConfig,
    *,
    include_tubular: bool = True,
) -> pd.DataFrame:
    result = table.copy().reset_index(drop=True)

    size = score_size_outliers(
        result["component_area"].to_numpy(),
        z_threshold=config.size_z_threshold,
    )

    for column in (
        "log_component_area",
        "size_robust_z",
        "suspicious_small",
        "suspicious_large",
    ):
        result[column] = size.table[column].to_numpy()

    if include_tubular:
        eligible = ~result["suspicious_small"].to_numpy(dtype=bool)
        tubular = score_tubular_objects(
            result,
            candidate_z_threshold=config.tubular_candidate_z,
            score_threshold=config.tubular_score_threshold,
            eligible_mask=eligible,
        )

        for column in (
            "tubular_eligible",
            "tubular_candidate",
            "radius_median",
            "radius_max",
            "thickness_variation",
            "body_prominence",
            "elongation",
            "circularity",
            "solidity",
            "tube_score",
            "compact_body_rescue",
            "suspicious_tubular",
        ):
            result[column] = tubular.table[column].to_numpy()

    return result


def _typical_scale(table: pd.DataFrame) -> tuple[float, float]:
    ordinary = table.loc[
        ~table["suspicious_small"].astype(bool)
        & ~table["suspicious_large"].astype(bool)
    ]

    if ordinary.empty:
        ordinary = table

    median_area = float(
        np.median(
            ordinary[
                "component_area"
            ].to_numpy(
                float
            )
        )
    )

    radii: list[float] = []

    if "object_radius_max" in ordinary.columns:

        values = ordinary[
            "object_radius_max"
        ].to_numpy(
            dtype=float
        )

        radii = [
            float(value)
            for value in values
            if np.isfinite(value)
            and value > 0
        ]

    elif "roi" in ordinary.columns:

        # Compatibility path for in-memory tables created before
        # object_radius_max became a persisted QC coordinate.
        for roi in ordinary["roi"]:
            mask = np.asarray(roi) > 0

            if not np.any(
                mask
            ):
                continue

            distance = cv2.distanceTransform(
                mask.astype(np.uint8),
                cv2.DIST_L2,
                5,
            )

            radius = float(
                distance.max()
            )

            if (
                np.isfinite(radius)
                and radius > 0
            ):
                radii.append(
                    radius
                )

    if not radii:
        typical_radius = 1.0
    else:
        typical_radius = float(
            np.median(
                np.asarray(
                    radii,
                    dtype=float,
                )
            )
        )

    return (
        median_area,
        typical_radius,
    )


def _write_qc_overview(
    table: pd.DataFrame,
    root: Path,
    config: InstanceRefinementConfig,
) -> None:
    overview = root / "QC" / "Overview"
    overview.mkdir(parents=True, exist_ok=True)

    specs = (
        (
            "suspicious_small",
            "size_robust_z",
            "low",
            "Suspicious Small Objects",
            "Small_Objects.png",
        ),
        (
            "suspicious_large",
            "size_robust_z",
            "high",
            "Suspicious Large Objects",
            "Large_Objects.png",
        ),
        (
            "suspicious_tubular",
            "tube_score",
            "high",
            "Suspicious Tubular Objects",
            "Tubular_Objects.png",
        ),
    )

    for flag, score, extreme, title, filename in specs:
        if flag not in table.columns:
            continue
        selected = select_qc_examples(
            table,
            flag_column=flag,
            score_column=score,
            grid_n=config.grid_n,
            extreme=extreme,
        )
        plot_qc_grid(
            selected,
            overview / filename,
            grid_n=config.grid_n,
            title=title,
            score_column=score,
        )



# ======================================================================
# PERSISTENT DETECTION CACHE
# ======================================================================

_DETECTION_CACHE_VERSION = "instance_qc_v2_radius_cache"

_DETECTION_CACHE_BOOLEAN_COLUMNS = (
    "suspicious_small",
    "suspicious_large",
    "tubular_eligible",
    "tubular_candidate",
    "compact_body_rescue",
    "suspicious_tubular",
)


def _detection_cache_paths(
    output_dir: Path,
) -> tuple[Path, Path]:
    technical_dir = (
        Path(
            output_dir
        )
        / "Technical_Record"
    )

    return (
        technical_dir
        / "instance_refinement_object_qc.csv",
        technical_dir
        / "instance_refinement_detection.json",
    )


def _detection_signature(
    config: InstanceRefinementConfig,
) -> str:
    """
    Fingerprint only parameters that decide which objects are suspicious.

    Action choices and split/reconnect implementation parameters are
    deliberately excluded so they can be changed without rescanning the
    complete Segmentation dataset.
    """

    payload = {
        "cache_version": _DETECTION_CACHE_VERSION,
        "size_z_threshold": float(
            config.size_z_threshold
        ),
        "tubular_candidate_z": float(
            config.tubular_candidate_z
        ),
        "tubular_score_threshold": float(
            config.tubular_score_threshold
        ),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def _restore_cached_booleans(
    table: pd.DataFrame,
) -> pd.DataFrame:
    table = table.copy()

    for column in _DETECTION_CACHE_BOOLEAN_COLUMNS:

        if column not in table.columns:
            continue

        if pd.api.types.is_bool_dtype(
            table[
                column
            ]
        ):
            table[
                column
            ] = (
                table[
                    column
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
            )

        else:
            table[
                column
            ] = (
                table[
                    column
                ]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(
                    {
                        "true",
                        "1",
                        "yes",
                        "y",
                    }
                )
            )

    return table


def _load_detection_cache(
    *,
    output_dir: Path,
    segmentation_result,
    config: InstanceRefinementConfig,
    source_by_id: dict[str, Path],
) -> pd.DataFrame | None:
    generation_id = getattr(
        segmentation_result,
        "generation_id",
        None,
    )

    if not generation_id:
        return None

    csv_path, metadata_path = (
        _detection_cache_paths(
            output_dir
        )
    )

    if (
        not csv_path.is_file()
        or not metadata_path.is_file()
    ):
        return None

    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )
    except Exception:
        return None

    if str(
        metadata.get(
            "cache_version",
            "",
        )
    ) != _DETECTION_CACHE_VERSION:
        return None

    if str(
        metadata.get(
            "segmentation_generation_id",
            "",
        )
    ) != str(
        generation_id
    ):
        return None

    if str(
        metadata.get(
            "detector_signature",
            "",
        )
    ) != _detection_signature(
        config
    ):
        return None

    try:
        table = pd.read_csv(
            csv_path
        )
    except Exception:
        return None

    required = {
        "image_id",
        "cell_id",
        "map_label",
        "component_area",
        "object_radius_max",
        "size_robust_z",
        "suspicious_small",
        "suspicious_large",
        "suspicious_tubular",
    }

    if not required.issubset(
        table.columns
    ):
        return None

    cached_ids = set(
        table[
            "image_id"
        ]
        .astype(str)
        .unique()
    )

    if not cached_ids.issubset(
        set(
            source_by_id
        )
    ):
        return None

    return _restore_cached_booleans(
        table
    )


def _save_detection_cache(
    *,
    table: pd.DataFrame,
    output_dir: Path,
    segmentation_result,
    config: InstanceRefinementConfig,
) -> tuple[Path, Path]:
    generation_id = getattr(
        segmentation_result,
        "generation_id",
        None,
    )

    csv_path, metadata_path = (
        _detection_cache_paths(
            output_dir
        )
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    persisted = table.drop(
        columns=[
            "roi",
        ],
        errors="ignore",
    )

    persisted.to_csv(
        csv_path,
        index=False,
    )

    metadata = {
        "cache_version": _DETECTION_CACHE_VERSION,
        "segmentation_generation_id": str(
            generation_id
        ),
        "detector_signature": _detection_signature(
            config
        ),
        "object_count": int(
            len(
                persisted
            )
        ),
    }

    temporary_path = (
        metadata_path
        .with_suffix(
            ".json.tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write(
            "\n"
        )

    temporary_path.replace(
        metadata_path
    )

    return (
        csv_path,
        metadata_path,
    )


# ======================================================================
# SOMA / SPLIT SEEDS
# ======================================================================


def _seed_points(
    mask: np.ndarray,
    *,
    typical_radius: float,
    median_area: float,
    config: InstanceRefinementConfig,
    allow_multiple: bool,
) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []

    distance = ndi.distance_transform_edt(mask)

    if not allow_multiple:
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        return [(int(y), int(x))]

    smoothed = ndi.gaussian_filter(
        distance,
        sigma=config.split_smoothing_sigma,
    )

    threshold_abs = max(
        1.0,
        float(typical_radius) * config.split_peak_radius_fraction,
    )

    min_distance = max(
        3,
        int(round(
            float(typical_radius)
            * config.split_peak_min_distance_factor
        )),
    )

    coords = peak_local_max(
        smoothed,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        labels=mask.astype(np.uint8),
        exclude_border=False,
    )

    points = [
        (int(y), int(x))
        for y, x in coords
    ]

    # Guarantee at least one seed for each disconnected component carrying
    # the same label identity, so watershed can never drop foreground pixels.
    components, n_components = ndi.label(mask)
    for component_id in range(1, int(n_components) + 1):
        component = components == component_id
        if any(component[y, x] for y, x in points):
            continue
        local_distance = np.where(component, distance, -1.0)
        y, x = np.unravel_index(
            int(np.argmax(local_distance)),
            local_distance.shape,
        )
        points.append((int(y), int(x)))

    # Large-object area provides a second, independent upper bound on how
    # many cell bodies can plausibly be represented by the object.
    expected = max(
        1.0,
        float(mask.sum()) / max(float(median_area), 1.0),
    )
    max_seeds = max(
        2,
        int(math.ceil(expected * config.split_max_seed_factor)),
    )

    points = sorted(
        set(points),
        key=lambda point: float(smoothed[point]),
        reverse=True,
    )[:max_seeds]

    return points


# ======================================================================
# RECONNECTION
# ======================================================================


def _angle_difference(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _radial_extent(
    parent_mask: np.ndarray,
    soma: tuple[int, int],
    orphan_point: tuple[int, int],
    *,
    half_width_degrees: float,
) -> tuple[float, float]:
    coords = np.column_stack(np.nonzero(parent_mask)).astype(float)
    if len(coords) == 0:
        return 0.0, 0.0

    soma_array = np.asarray(soma, dtype=float)
    vectors = coords - soma_array
    radii = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])

    orphan_vector = np.asarray(orphan_point, dtype=float) - soma_array
    orphan_radius = float(np.linalg.norm(orphan_vector))
    orphan_angle = float(np.arctan2(orphan_vector[0], orphan_vector[1]))

    half_width = math.radians(float(half_width_degrees))
    sector = _angle_difference(angles, orphan_angle) <= half_width

    if np.any(sector):
        extent = float(np.max(radii[sector]))
    else:
        extent = float(np.percentile(radii, 95.0))

    return extent, orphan_radius


def _nearest_mask_points(
    parent_mask: np.ndarray,
    orphan_mask: np.ndarray,
) -> tuple[float, tuple[int, int], tuple[int, int]]:
    parent_coords = np.column_stack(np.nonzero(parent_mask))
    orphan_coords = np.column_stack(np.nonzero(orphan_mask))

    if len(parent_coords) == 0 or len(orphan_coords) == 0:
        return math.inf, (-1, -1), (-1, -1)

    tree = cKDTree(parent_coords.astype(float))
    distances, indices = tree.query(orphan_coords.astype(float), k=1)
    j = int(np.argmin(distances))
    parent = parent_coords[int(indices[j])]
    orphan = orphan_coords[j]

    return (
        float(distances[j]),
        (int(parent[0]), int(parent[1])),
        (int(orphan[0]), int(orphan[1])),
    )


def _bridge_mask(
    shape: tuple[int, int],
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    radius: int,
) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.line(
        canvas,
        (int(point_a[1]), int(point_a[0])),
        (int(point_b[1]), int(point_b[0])),
        color=1,
        thickness=max(1, 2 * int(radius) + 1),
        lineType=cv2.LINE_8,
    )
    return canvas.astype(bool)


def _skeleton_geometry(
    mask: np.ndarray,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
]:
    """
    Return skeleton size, global skeleton coordinates, and global endpoints.

    Skeletonization is restricted to the component bounding box so fragment
    reconnection never performs a whole-image skeletonization for one object.
    Endpoints use 8-neighbour degree == 1.
    """

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    ys, xs = np.where(
        mask
    )

    if ys.size == 0:
        empty = np.empty(
            (
                0,
                2,
            ),
            dtype=int,
        )

        return (
            0,
            empty,
            empty.copy(),
        )

    y0 = int(
        ys.min()
    )
    y1 = int(
        ys.max()
    ) + 1
    x0 = int(
        xs.min()
    )
    x1 = int(
        xs.max()
    ) + 1

    crop = mask[
        y0:y1,
        x0:x1,
    ]

    skeleton = skeletonize(
        crop
    )

    skeleton_local = np.column_stack(
        np.nonzero(
            skeleton
        )
    )

    if len(
        skeleton_local
    ) == 0:
        empty = np.empty(
            (
                0,
                2,
            ),
            dtype=int,
        )

        return (
            0,
            empty,
            empty.copy(),
        )

    neighbor_count = ndi.convolve(
        skeleton.astype(
            np.uint8
        ),
        np.ones(
            (
                3,
                3,
            ),
            dtype=np.uint8,
        ),
        mode="constant",
        cval=0,
    ) - skeleton.astype(
        np.uint8
    )

    endpoints_local = np.column_stack(
        np.nonzero(
            skeleton
            & (
                neighbor_count
                == 1
            )
        )
    )

    offset = np.asarray(
        [
            y0,
            x0,
        ],
        dtype=int,
    )

    skeleton_global = (
        skeleton_local.astype(
            int,
            copy=False,
        )
        + offset
    )

    endpoints_global = (
        endpoints_local.astype(
            int,
            copy=False,
        )
        + offset
    )

    return (
        int(
            len(
                skeleton_local
            )
        ),
        skeleton_global,
        endpoints_global,
    )


def _nearest_coordinate_points(
    target_coords: np.ndarray,
    source_coords: np.ndarray,
) -> tuple[
    float,
    tuple[int, int],
    tuple[int, int],
]:
    """Return the closest target/source coordinate pair."""

    target_coords = np.asarray(
        target_coords,
        dtype=int,
    )
    source_coords = np.asarray(
        source_coords,
        dtype=int,
    )

    if (
        len(
            target_coords
        ) == 0
        or len(
            source_coords
        ) == 0
    ):
        return (
            math.inf,
            (
                -1,
                -1,
            ),
            (
                -1,
                -1,
            ),
        )

    tree = cKDTree(
        target_coords.astype(
            float
        )
    )

    distances, indices = tree.query(
        source_coords.astype(
            float
        ),
        k=1,
    )

    source_index = int(
        np.argmin(
            distances
        )
    )

    target = target_coords[
        int(
            indices[
                source_index
            ]
        )
    ]

    source = source_coords[
        source_index
    ]

    return (
        float(
            distances[
                source_index
            ]
        ),
        (
            int(
                target[0]
            ),
            int(
                target[1]
            ),
        ),
        (
            int(
                source[0]
            ),
            int(
                source[1]
            ),
        ),
    )


def _nearby_parent_labels(
    labels: np.ndarray,
    source_mask: np.ndarray,
    *,
    source_label: int,
    excluded_labels: set[int],
    max_gap_px: float,
) -> list[int]:
    """
    Return only labels whose foreground occurs in the source-local search box.

    The search box is expanded by max_gap_px plus one safety pixel. Exact
    endpoint distances are still checked afterward; this is only a cheap
    candidate-pruning step.
    """

    labels = np.asarray(
        labels,
        dtype=np.int32,
    )
    source_mask = np.asarray(
        source_mask,
        dtype=bool,
    )

    ys, xs = np.where(
        source_mask
    )

    if ys.size == 0:
        return []

    padding = max(
        1,
        int(
            math.ceil(
                float(
                    max_gap_px
                )
            )
        ) + 1,
    )

    y0 = max(
        0,
        int(
            ys.min()
        ) - padding,
    )
    y1 = min(
        labels.shape[0],
        int(
            ys.max()
        ) + 1 + padding,
    )
    x0 = max(
        0,
        int(
            xs.min()
        ) - padding,
    )
    x1 = min(
        labels.shape[1],
        int(
            xs.max()
        ) + 1 + padding,
    )

    values = np.unique(
        labels[
            y0:y1,
            x0:x1,
        ]
    )

    excluded = {
        int(
            value
        )
        for value in excluded_labels
    }
    excluded.add(
        int(
            source_label
        )
    )
    excluded.add(
        0
    )

    return sorted(
        int(
            value
        )
        for value in values
        if int(
            value
        ) not in excluded
    )


def _distance_radius_at(
    mask: np.ndarray,
    point: tuple[int, int],
) -> float:
    """Measure local half-thickness at one component point using a bbox DT."""

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    ys, xs = np.where(
        mask
    )

    if ys.size == 0:
        return 0.0

    y0 = int(
        ys.min()
    )
    y1 = int(
        ys.max()
    ) + 1
    x0 = int(
        xs.min()
    )
    x1 = int(
        xs.max()
    ) + 1

    crop = mask[
        y0:y1,
        x0:x1,
    ]

    distance = cv2.distanceTransform(
        crop.astype(
            np.uint8
        ),
        cv2.DIST_L2,
        5,
    )

    local_y = int(
        point[0]
    ) - y0
    local_x = int(
        point[1]
    ) - x0

    if not (
        0
        <= local_y
        < distance.shape[0]
        and 0
        <= local_x
        < distance.shape[1]
    ):
        return 0.0

    return float(
        distance[
            local_y,
            local_x,
        ]
    )


def _reconnect_image(
    labels: np.ndarray,
    *,
    orphan_labels: set[int],
    large_labels: set[int],
    median_area: float,
    typical_radius: float,
    config: InstanceRefinementConfig,
    image_id: str,
    event_counter: list[int],
) -> tuple[
    np.ndarray,
    list[dict],
    dict[str, _RefinementEventGeometry],
]:
    """
    Conservatively reconnect detached branch fragments to one local parent.

    Parent identity is decided by local skeleton-endpoint geometry, not by
    global soma Voronoi ownership. Soma location is used only afterward as a
    radial-envelope validation constraint.

    A source fragment must have enough skeleton support to prevent tiny
    isolated specks from being attached merely because they lie within the
    maximum gap. The threshold is data-independent but tied to the already
    explicit reconnection scale:

        minimum skeleton length = 2 * max_gap_px

    A reconnection is committed only when exactly one non-fragment parent has
    a skeleton endpoint within max_gap_px. Ambiguous sources remain unchanged.
    """

    current = np.asarray(
        labels,
        dtype=np.int32,
    ).copy()

    events: list[dict] = []

    event_geometries: dict[
        str,
        _RefinementEventGeometry,
    ] = {}

    remaining = {
        int(
            label
        )
        for label in orphan_labels
        if np.any(
            current
            == int(
                label
            )
        )
    }

    minimum_skeleton_length = max(
        2,
        int(
            math.ceil(
                2.0
                * float(
                    config.max_gap_px
                )
            )
        ),
    )

    failure_details: dict[
        int,
        dict,
    ] = {}

    # Geometry and soma seeds are cached per current label. Only the selected
    # parent changes after a successful merge, so only that cache entry needs
    # invalidation. This keeps repeated candidate passes local and cheap.
    geometry_cache: dict[
        int,
        tuple[
            int,
            np.ndarray,
            np.ndarray,
        ],
    ] = {}

    soma_seed_cache: dict[
        int,
        list[tuple[int, int]],
    ] = {}

    def geometry_for(
        map_label: int,
    ) -> tuple[
        int,
        np.ndarray,
        np.ndarray,
    ]:

        map_label = int(
            map_label
        )

        if map_label not in geometry_cache:
            geometry_cache[
                map_label
            ] = _skeleton_geometry(
                current
                == map_label
            )

        return geometry_cache[
            map_label
        ]

    def soma_seeds_for(
        map_label: int,
    ) -> list[tuple[int, int]]:

        map_label = int(
            map_label
        )

        if map_label not in soma_seed_cache:
            soma_seed_cache[
                map_label
            ] = _seed_points(
                current
                == map_label,
                typical_radius=typical_radius,
                median_area=median_area,
                config=config,
                allow_multiple=(
                    map_label
                    in large_labels
                ),
            )

        return soma_seed_cache[
            map_label
        ]

    # One reconnection is committed per iteration. This deliberately lets a
    # newly enlarged parent absorb another fragment during the next pass.
    while remaining:

        proposals = []

        for orphan_label in sorted(
            remaining
        ):

            orphan_mask = (
                current
                == orphan_label
            )

            if not np.any(
                orphan_mask
            ):
                failure_details[
                    orphan_label
                ] = {
                    "reason": "source_label_absent",
                }
                continue

            (
                skeleton_length,
                _,
                orphan_endpoints,
            ) = geometry_for(
                orphan_label
            )

            if (
                skeleton_length
                < minimum_skeleton_length
            ):
                failure_details[
                    orphan_label
                ] = {
                    "reason": "fragment_skeleton_too_short",
                }
                continue

            if len(
                orphan_endpoints
            ) == 0:
                failure_details[
                    orphan_label
                ] = {
                    "reason": "fragment_has_no_skeleton_endpoint",
                }
                continue

            # Other unresolved fragment candidates cannot become parents yet.
            # If one fragment is first absorbed into a valid parent, its pixels
            # become part of that parent and can support a later reconnection.
            parent_candidates = _nearby_parent_labels(
                current,
                orphan_mask,
                source_label=orphan_label,
                excluded_labels=remaining,
                max_gap_px=(
                    config.max_gap_px
                ),
            )

            valid_parents = []
            nearest_rejected = None

            for parent_label in parent_candidates:

                parent_mask = (
                    current
                    == int(
                        parent_label
                    )
                )

                (
                    _,
                    parent_skeleton,
                    parent_endpoints,
                ) = geometry_for(
                    parent_label
                )

                # A rare closed skeleton has no topological endpoint. Falling
                # back to the skeleton itself preserves a conservative local
                # distance test without manufacturing crop-edge endpoints.
                target_points = (
                    parent_endpoints
                    if len(
                        parent_endpoints
                    ) > 0
                    else parent_skeleton
                )

                (
                    endpoint_gap,
                    parent_point,
                    orphan_point,
                ) = _nearest_coordinate_points(
                    target_points,
                    orphan_endpoints,
                )

                rejected_record = (
                    float(
                        endpoint_gap
                    ),
                    int(
                        parent_label
                    ),
                    parent_point,
                    orphan_point,
                )

                if (
                    nearest_rejected is None
                    or endpoint_gap
                    < nearest_rejected[0]
                ):
                    nearest_rejected = (
                        rejected_record
                    )

                if (
                    np.isfinite(
                        endpoint_gap
                    )
                    and endpoint_gap
                    <= config.max_gap_px
                ):
                    valid_parents.append(
                        rejected_record
                    )

            if not valid_parents:

                details = {
                    "reason": "no_parent_endpoint_within_max_gap_px",
                }

                if nearest_rejected is not None:
                    (
                        nearest_gap,
                        nearest_parent,
                        nearest_parent_point,
                        nearest_orphan_point,
                    ) = nearest_rejected

                    details.update(
                        {
                            "target_label": int(
                                nearest_parent
                            ),
                            "parent_label": int(
                                nearest_parent
                            ),
                            "gap_px": float(
                                nearest_gap
                            ),
                            "source_endpoint": (
                                nearest_orphan_point
                            ),
                            "target_endpoint": (
                                nearest_parent_point
                            ),
                        }
                    )

                failure_details[
                    orphan_label
                ] = details
                continue

            if len(
                valid_parents
            ) > 1:

                valid_parents.sort(
                    key=lambda item: (
                        item[0],
                        item[1],
                    )
                )

                (
                    nearest_gap,
                    nearest_parent,
                    nearest_parent_point,
                    nearest_orphan_point,
                ) = valid_parents[0]

                failure_details[
                    orphan_label
                ] = {
                    "target_label": int(
                        nearest_parent
                    ),
                    "parent_label": int(
                        nearest_parent
                    ),
                    "gap_px": float(
                        nearest_gap
                    ),
                    "source_endpoint": (
                        nearest_orphan_point
                    ),
                    "target_endpoint": (
                        nearest_parent_point
                    ),
                    "reason": "multiple_parent_endpoints_within_max_gap_px",
                }
                continue

            (
                endpoint_gap,
                parent_label,
                parent_point,
                orphan_point,
            ) = valid_parents[0]

            parent_label = int(
                parent_label
            )

            parent_mask = (
                current
                == parent_label
            )

            soma_seeds = soma_seeds_for(
                parent_label
            )

            if not soma_seeds:
                failure_details[
                    orphan_label
                ] = {
                    "target_label": parent_label,
                    "parent_label": parent_label,
                    "gap_px": float(
                        endpoint_gap
                    ),
                    "source_endpoint": orphan_point,
                    "target_endpoint": parent_point,
                    "reason": "parent_has_no_soma_seed",
                }
                continue

            soma = min(
                soma_seeds,
                key=lambda seed: math.hypot(
                    float(
                        seed[0]
                        - parent_point[0]
                    ),
                    float(
                        seed[1]
                        - parent_point[1]
                    ),
                ),
            )

            soma = (
                int(
                    soma[0]
                ),
                int(
                    soma[1]
                ),
            )

            (
                radial_extent,
                orphan_radius,
            ) = _radial_extent(
                parent_mask,
                soma,
                orphan_point,
                half_width_degrees=(
                    config.radial_sector_half_width_degrees
                ),
            )

            if orphan_radius > (
                radial_extent
                + config.effective_radial_tolerance_px
            ):
                failure_details[
                    orphan_label
                ] = {
                    "target_label": parent_label,
                    "parent_label": parent_label,
                    "gap_px": float(
                        endpoint_gap
                    ),
                    "soma": soma,
                    "source_endpoint": orphan_point,
                    "target_endpoint": parent_point,
                    "radial_extent_px": float(
                        radial_extent
                    ),
                    "orphan_radius_px": float(
                        orphan_radius
                    ),
                    "reason": "outside_parent_radial_envelope",
                }
                continue

            radius_a = _distance_radius_at(
                parent_mask,
                parent_point,
            )
            radius_b = _distance_radius_at(
                orphan_mask,
                orphan_point,
            )

            bridge_radius = max(
                1,
                int(
                    round(
                        (
                            radius_a
                            + radius_b
                        )
                        / 2.0
                    )
                ),
            )

            bridge = _bridge_mask(
                current.shape,
                parent_point,
                orphan_point,
                bridge_radius,
            )

            collision = (
                bridge
                & (
                    current
                    != 0
                )
                & (
                    current
                    != parent_label
                )
                & (
                    current
                    != orphan_label
                )
            )

            if np.any(
                collision
            ):
                failure_details[
                    orphan_label
                ] = {
                    "target_label": parent_label,
                    "parent_label": parent_label,
                    "gap_px": float(
                        endpoint_gap
                    ),
                    "soma": soma,
                    "source_endpoint": orphan_point,
                    "target_endpoint": parent_point,
                    "bridge_radius_px": bridge_radius,
                    "radial_extent_px": float(
                        radial_extent
                    ),
                    "orphan_radius_px": float(
                        orphan_radius
                    ),
                    "reason": "bridge_crosses_other_label",
                }
                continue

            proposals.append(
                (
                    float(
                        endpoint_gap
                    ),
                    orphan_label,
                    parent_label,
                    soma,
                    parent_point,
                    orphan_point,
                    bridge_radius,
                    bridge,
                    float(
                        radial_extent
                    ),
                    float(
                        orphan_radius
                    ),
                )
            )

        if not proposals:
            break

        proposal = min(
            proposals,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            ),
        )

        (
            gap,
            orphan_label,
            parent_label,
            soma,
            parent_point,
            orphan_point,
            bridge_radius,
            bridge,
            radial_extent,
            orphan_radius,
        ) = proposal

        orphan_mask = (
            current
            == orphan_label
        )

        before_foreground = (
            current
            > 0
        )

        affected_snapshot = _mask_snapshot(
            orphan_mask
            | bridge,
            label=parent_label,
        )

        if affected_snapshot is None:
            raise RuntimeError(
                "Successful reconnection has no event-local geometry."
            )

        pixels_relabelled = int(
            np.count_nonzero(
                orphan_mask
            )
        )

        current[
            orphan_mask
        ] = parent_label
        current[
            bridge
        ] = parent_label

        # The selected parent geometry changed. The source label disappeared.
        # Invalidate only those cache entries before the next local pass.
        geometry_cache.pop(
            int(
                parent_label
            ),
            None,
        )
        geometry_cache.pop(
            int(
                orphan_label
            ),
            None,
        )
        soma_seed_cache.pop(
            int(
                parent_label
            ),
            None,
        )
        soma_seed_cache.pop(
            int(
                orphan_label
            ),
            None,
        )

        pixels_added = int(
            np.count_nonzero(
                (
                    current
                    > 0
                )
                & ~before_foreground
            )
        )

        result_snapshot = _mask_snapshot(
            current == int(parent_label),
            label=int(parent_label),
        )

        if result_snapshot is None:
            raise RuntimeError(
                "Successful reconnection has no whole-result geometry."
            )

        event = _event_row(
            event_counter,
            image_id=image_id,
            operation="reconnect",
            status="success",
            changed=True,
            source_label=orphan_label,
            target_label=parent_label,
            parent_label=parent_label,
            result_labels=str(
                parent_label
            ),
            gap_px=float(
                gap
            ),
            soma=soma,
            source_endpoint=orphan_point,
            target_endpoint=parent_point,
            bridge_radius_px=bridge_radius,
            pixels_added=pixels_added,
            pixels_relabelled=pixels_relabelled,
            radial_extent_px=float(
                radial_extent
            ),
            orphan_radius_px=float(
                orphan_radius
            ),
            bbox_snapshot=affected_snapshot,
            reason="fragment_reconnected_to_unique_local_parent",
        )

        events.append(
            event
        )

        event_geometries[
            event[
                "event_id"
            ]
        ] = _RefinementEventGeometry(
            event_id=event[
                "event_id"
            ],
            image_id=str(
                image_id
            ),
            operation="reconnect",
            snapshots=(
                affected_snapshot,
            ),
            result_snapshot=result_snapshot,
            changed_snapshot=affected_snapshot,
        )

        remaining.remove(
            orphan_label
        )
        failure_details.pop(
            orphan_label,
            None,
        )

    for orphan_label in sorted(
        remaining
    ):

        source_snapshot = _mask_snapshot(
            current
            == orphan_label,
            label=orphan_label,
        )

        details = dict(
            failure_details.get(
                orphan_label,
                {
                    "reason": "no_valid_reconnection_proposal",
                },
            )
        )

        reason = str(
            details.pop(
                "reason"
            )
        )

        events.append(
            _event_row(
                event_counter,
                image_id=image_id,
                operation="reconnect",
                status="unresolved",
                changed=False,
                source_label=orphan_label,
                bbox_snapshot=source_snapshot,
                reason=reason,
                **details,
            )
        )

    return (
        current,
        events,
        event_geometries,
    )


# ======================================================================
# SPLITTING
# ======================================================================


def _split_parent(
    labels: np.ndarray,
    *,
    parent_label: int,
    median_area: float,
    typical_radius: float,
    config: InstanceRefinementConfig,
    next_label: int,
    image_id: str,
    event_counter: list[int],
) -> tuple[
    np.ndarray,
    int,
    dict,
    _RefinementEventGeometry | None,
]:
    current = np.asarray(labels, dtype=np.int32).copy()
    mask = current == int(parent_label)
    parent_snapshot = _mask_snapshot(
        mask,
        label=parent_label,
    )

    seeds = _seed_points(
        mask,
        typical_radius=typical_radius,
        median_area=median_area,
        config=config,
        allow_multiple=True,
    )

    if len(seeds) < 2:
        event = _event_row(
            event_counter,
            image_id=image_id,
            operation="split",
            status="unresolved",
            changed=False,
            source_label=int(parent_label),
            parent_label=int(parent_label),
            seed_count=len(seeds),
            bbox_snapshot=parent_snapshot,
            reason="fewer_than_two_credible_seeds",
        )
        return current, next_label, event, None

    distance = ndi.distance_transform_edt(mask)
    smoothed = ndi.gaussian_filter(
        distance,
        sigma=config.split_smoothing_sigma,
    )

    active_seeds = list(seeds)
    minimum_child_area = max(
        20,
        int(round(
            median_area
            * config.split_min_child_area_fraction
        )),
    )

    split_labels = None

    while len(active_seeds) >= 2:
        markers = np.zeros(mask.shape, dtype=np.int32)
        for marker_id, (y, x) in enumerate(active_seeds, start=1):
            markers[y, x] = marker_id

        proposal = watershed(
            -smoothed,
            markers=markers,
            mask=mask,
        )

        areas = np.bincount(proposal.ravel())
        bad_marker_ids = {
            marker_id
            for marker_id in range(1, len(active_seeds) + 1)
            if marker_id >= len(areas)
            or int(areas[marker_id]) < minimum_child_area
        }

        if not bad_marker_ids:
            split_labels = proposal
            break

        active_seeds = [
            seed
            for marker_id, seed in enumerate(active_seeds, start=1)
            if marker_id not in bad_marker_ids
        ]

    if split_labels is None or len(active_seeds) < 2:
        event = _event_row(
            event_counter,
            image_id=image_id,
            operation="split",
            status="unresolved",
            changed=False,
            source_label=int(parent_label),
            parent_label=int(parent_label),
            seed_count=len(active_seeds),
            bbox_snapshot=parent_snapshot,
            reason="fewer_than_two_valid_children",
        )
        return current, next_label, event, None

    child_ids = [
        int(value)
        for value in np.unique(split_labels)
        if int(value) > 0
    ]

    child_areas = {
        child_id: int(np.count_nonzero(split_labels == child_id))
        for child_id in child_ids
    }

    child_ids.sort(
        key=lambda child_id: child_areas[child_id],
        reverse=True,
    )

    if len(child_ids) < 2:
        event = _event_row(
            event_counter,
            image_id=image_id,
            operation="split",
            status="unresolved",
            changed=False,
            source_label=int(parent_label),
            parent_label=int(parent_label),
            seed_count=len(active_seeds),
            bbox_snapshot=parent_snapshot,
            reason="watershed_produced_fewer_than_two_children",
        )
        return current, next_label, event, None

    result_labels = [int(parent_label)]

    current[mask] = 0
    current[split_labels == child_ids[0]] = int(parent_label)

    for child_id in child_ids[1:]:
        while np.any(current == int(next_label)):
            next_label += 1
        current[split_labels == child_id] = int(next_label)
        result_labels.append(int(next_label))
        next_label += 1

    child_snapshots = tuple(
        snapshot
        for result_label in result_labels
        for snapshot in [
            _mask_snapshot(
                current == int(result_label),
                label=int(result_label),
            )
        ]
        if snapshot is not None
    )

    if len(child_snapshots) < 2:
        raise RuntimeError(
            "Successful split did not preserve at least two child geometries."
        )

    pixels_relabelled = int(
        sum(
            np.count_nonzero(
                current
                == int(result_label)
            )
            for result_label in result_labels[1:]
        )
    )

    event = _event_row(
        event_counter,
        image_id=image_id,
        operation="split",
        status="success",
        changed=True,
        source_label=int(parent_label),
        parent_label=int(parent_label),
        result_labels="|".join(
            str(value)
            for value in result_labels
        ),
        pixels_relabelled=pixels_relabelled,
        seed_count=len(active_seeds),
        bbox_snapshot=parent_snapshot,
        reason="large_multi_soma_split",
    )

    geometry = _RefinementEventGeometry(
        event_id=event[
            "event_id"
        ],
        image_id=str(
            image_id
        ),
        operation="split",
        snapshots=child_snapshots,
        before_snapshot=parent_snapshot,
    )

    return (
        current,
        next_label,
        event,
        geometry,
    )


# ======================================================================
# OUTPUT / MANIFEST
# ======================================================================


def _event_row(
    event_counter: list[int],
    *,
    image_id: str,
    operation: str,
    status: str,
    changed: bool,
    reason: str,
    source_label: int | str = "",
    target_label: int | str = "",
    parent_label: int | str = "",
    result_labels: str = "",
    gap_px: float | str = "",
    soma: tuple[int, int] | None = None,
    source_endpoint: tuple[int, int] | None = None,
    target_endpoint: tuple[int, int] | None = None,
    bridge_radius_px: int | str = "",
    pixels_added: int = 0,
    pixels_removed: int = 0,
    pixels_relabelled: int = 0,
    radial_extent_px: float | str = "",
    orphan_radius_px: float | str = "",
    seed_count: int | str = "",
    bbox_snapshot: _MaskSnapshot | None = None,
) -> dict:
    status = str(status).strip().lower()

    if status not in {
        "success",
        "unresolved",
    }:
        raise ValueError(
            "Refinement event status must be 'success' or 'unresolved'."
        )

    success = (
        status
        == "success"
    )

    operation = str(
        operation
    )

    if operation not in _MODIFYING_OPERATIONS:
        raise ValueError(
            f"Unknown Instance Refinement event operation: {operation!r}."
        )

    if bool(changed) != success:
        raise ValueError(
            "A successful refinement event must change the map, and an "
            "unresolved event must not change it."
        )

    pixels_added = int(
        pixels_added
    )
    pixels_removed = int(
        pixels_removed
    )
    pixels_relabelled = int(
        pixels_relabelled
    )

    if min(
        pixels_added,
        pixels_removed,
        pixels_relabelled,
    ) < 0:
        raise ValueError(
            "Refinement event pixel counts cannot be negative."
        )

    pixels_changed = int(
        pixels_added
        + pixels_removed
        + pixels_relabelled
    )

    if success and pixels_changed <= 0:
        raise ValueError(
            "A successful refinement event must record changed pixels."
        )

    if not success and pixels_changed != 0:
        raise ValueError(
            "An unresolved refinement event cannot record changed pixels."
        )

    event_counter[0] += 1

    row = {
        "event_id": f"E{event_counter[0]:07d}",
        "image_id": str(image_id),
        "operation": operation,
        "status": status,
        "success": bool(success),
        "changed": bool(changed),
        "source_label": source_label,
        "target_label": target_label,
        "parent_label": parent_label,
        "result_labels": str(result_labels),
        "gap_px": gap_px,
        "soma_y": (
            ""
            if soma is None
            else int(soma[0])
        ),
        "soma_x": (
            ""
            if soma is None
            else int(soma[1])
        ),
        "source_endpoint_y": (
            ""
            if source_endpoint is None
            else int(source_endpoint[0])
        ),
        "source_endpoint_x": (
            ""
            if source_endpoint is None
            else int(source_endpoint[1])
        ),
        "target_endpoint_y": (
            ""
            if target_endpoint is None
            else int(target_endpoint[0])
        ),
        "target_endpoint_x": (
            ""
            if target_endpoint is None
            else int(target_endpoint[1])
        ),
        "bridge_radius_px": bridge_radius_px,
        "pixels_added": pixels_added,
        "pixels_removed": pixels_removed,
        "pixels_relabelled": pixels_relabelled,
        "pixels_changed": pixels_changed,
        "radial_extent_px": radial_extent_px,
        "orphan_radius_px": orphan_radius_px,
        "seed_count": seed_count,
        "reason": str(reason),
    }

    row.update(
        _snapshot_bbox_fields(
            bbox_snapshot
        )
    )

    return row


def _successful_events(
    events_table: pd.DataFrame,
) -> pd.DataFrame:
    """Return only successful modifying events with explicit map changes."""

    if events_table.empty:
        return events_table.copy()

    required = {
        "operation",
        "status",
        "success",
        "changed",
    }

    missing = required - set(
        events_table.columns
    )

    if missing:
        raise RuntimeError(
            "Instance Refinement event table lacks required semantic fields: "
            f"{sorted(missing)}"
        )

    mask = (
        events_table[
            "operation"
        ].astype(str).isin(
            _MODIFYING_OPERATIONS
        )
        & events_table[
            "status"
        ].astype(str).eq(
            "success"
        )
        & events_table[
            "success"
        ].astype(bool)
        & events_table[
            "changed"
        ].astype(bool)
    )

    return (
        events_table.loc[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


_BUSY_REMOVE_RETRY_DELAYS_SECONDS = (
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
    4.00,
    8.00,
)


def _retry_busy_remove(
    function,
    path: str | Path,
    error: BaseException,
) -> None:
    """Retry one transient EBUSY removal without hiding other failures."""

    if (
        not isinstance(
            error,
            OSError,
        )
        or error.errno
        != errno.EBUSY
    ):
        raise error

    path = Path(
        path
    )
    last_error = error
    retry_total = len(
        _BUSY_REMOVE_RETRY_DELAYS_SECONDS
    )

    for retry_index, delay in enumerate(
        _BUSY_REMOVE_RETRY_DELAYS_SECONDS,
        start=1,
    ):
        print(
            "Filesystem resource busy while removing stage output: "
            f"{path} (retry {retry_index}/{retry_total} in {delay:g}s)"
        )

        time.sleep(
            delay
        )

        try:
            function(
                path
            )
            return

        except FileNotFoundError:
            return

        except OSError as retry_error:
            if (
                retry_error.errno
                != errno.EBUSY
            ):
                raise

            last_error = retry_error

    raise OSError(
        errno.EBUSY,
        "Could not remove the stage-owned output after explicit resource-"
        f"busy retries ({sum(_BUSY_REMOVE_RETRY_DELAYS_SECONDS):g}s total)",
        os.fspath(
            path
        ),
    ) from last_error


def _remove_stage_tree(
    path: str | Path,
) -> None:
    """Remove a stage tree, retrying only transient busy filesystem entries."""

    def _on_remove_error(
        function,
        failed_path,
        exc_info,
    ) -> None:
        _retry_busy_remove(
            function,
            failed_path,
            exc_info[1],
        )

    shutil.rmtree(
        path,
        onerror=_on_remove_error,
    )


def _link_effective(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    relative = os.path.relpath(source, destination.parent)

    try:
        destination.symlink_to(relative)
        return "symlink"
    except Exception:
        pass

    try:
        os.link(source, destination)
        return "hardlink"
    except Exception:
        pass

    shutil.copy2(source, destination)
    return "copy_fallback"


def _write_configuration_record(
    output_dir: Path,
    config: InstanceRefinementConfig,
) -> Path:
    technical = output_dir / "Technical_Record"
    technical.mkdir(parents=True, exist_ok=True)
    path = technical / "instance_refinement_configuration.csv"
    pd.DataFrame(config.configuration_rows()).to_csv(path, index=False)
    return path


def _passthrough_result(segmentation_result) -> InstanceRefinementResult:
    paths = [
        Path(
            path
        )
        for path in (
            segmentation_result
            .instance_paths
        )
    ]

    paths_by_image = {
        path.stem: path
        for path in paths
    }

    parents = {
        path.parent.resolve()
        for path in paths
    }

    effective_dir = (
        next(
            iter(
                parents
            )
        )
        if len(
            parents
        ) == 1
        else None
    )

    return InstanceRefinementResult(
        effective_instance_paths=paths,
        effective_instance_dir=effective_dir,
        effective_instance_paths_by_image=(
            paths_by_image
        ),
        modified_instance_paths=[],
        modified_count=0,
        reconnect_count=0,
        split_parent_count=0,
        removed_small_count=0,
        removed_large_count=0,
        removed_tubular_count=0,
        manifest_path=Path(""),
        reused=True,
    )


def passthrough_instance_refinement(segmentation_result) -> InstanceRefinementResult:
    return _passthrough_result(segmentation_result)


# ======================================================================
# SAVED REUSE
# ======================================================================


def load_saved_instance_refinement(
    *,
    segmentation_result,
    output_dir: str | Path,
    allow_missing: bool = True,
) -> InstanceRefinementResult | None:
    """
    Reconstruct the authoritative per-image instance-map resolver.

    New manifests point unchanged images directly to immutable Segmentation
    maps and modified images to Instance_Postprocessing/Postprocessed_Maps. No
    symlink/hardlink/copy mirror directory is required.

    Older manifests remain readable through the historical file-stat fallback.
    """

    output_dir = Path(
        output_dir
    )

    root = (
        output_dir
        / "Instance_Postprocessing"
    )

    legacy_root = (
        output_dir
        / "Instance_Refinement"
    )

    manifest_path = (
        root
        / "refinement_manifest.csv"
    )

    # Backward-compatible read only. New runs always write the canonical
    # Instance_Postprocessing tree.
    if (
        not manifest_path.is_file()
        and (
            legacy_root
            / "refinement_manifest.csv"
        ).is_file()
    ):
        root = legacy_root
        manifest_path = (
            root
            / "refinement_manifest.csv"
        )

    if not manifest_path.is_file():

        if allow_missing:
            return None

        raise FileNotFoundError(
            "Saved Instance Postprocessing manifest "
            f"was not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    basic_required = {
        "image_id",
        "modified",
    }

    missing = (
        basic_required
        - set(
            manifest.columns
        )
    )

    if missing:
        raise ValueError(
            "Saved refinement_manifest.csv is incompatible. "
            f"Missing columns: {sorted(missing)}"
        )

    rows = {
        str(
            row.image_id
        ): row
        for row in manifest.itertuples(
            index=False
        )
    }

    current_generation = getattr(
        segmentation_result,
        "generation_id",
        None,
    )

    has_generation = (
        "segmentation_generation_id"
        in manifest.columns
    )

    if has_generation:

        generations = {
            str(
                value
            ).strip()
            for value in manifest[
                "segmentation_generation_id"
            ].dropna().tolist()
            if str(
                value
            ).strip()
        }

        if len(
            generations
        ) != 1:
            raise ValueError(
                "Saved refinement_manifest.csv contains an ambiguous "
                "Segmentation generation identity."
            )

        saved_generation = next(
            iter(
                generations
            )
        )

        if (
            not current_generation
            or saved_generation
            != str(
                current_generation
            )
        ):
            raise ValueError(
                "Saved Instance Refinement is stale because it belongs "
                "to a different Segmentation generation. Re-run "
                "config.run.instance_refinement = True."
            )

    else:

        legacy_required = {
            "source_size",
            "source_mtime_ns",
            "effective_path",
        }

        legacy_missing = (
            legacy_required
            - set(
                manifest.columns
            )
        )

        if legacy_missing:
            raise ValueError(
                "Saved legacy refinement_manifest.csv is incompatible. "
                f"Missing columns: {sorted(legacy_missing)}"
            )

    effective_paths: list[
        Path
    ] = []

    effective_paths_by_image: dict[
        str,
        Path,
    ] = {}

    modified_paths: list[
        Path
    ] = []

    for source in [
        Path(
            path
        )
        for path in (
            segmentation_result
            .instance_paths
        )
    ]:

        image_id = (
            source.stem
        )

        if image_id not in rows:
            raise ValueError(
                "Saved Instance Refinement does not contain every "
                "canonical Segmentation map. "
                f"Missing: {image_id}"
            )

        row = rows[
            image_id
        ]

        modified = (
            str(
                row.modified
            )
            .strip()
            .lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

        if has_generation:

            if modified:

                refined_value = getattr(
                    row,
                    "refined_path",
                    "",
                )

                refined_text = (
                    ""
                    if pd.isna(
                        refined_value
                    )
                    else str(
                        refined_value
                    ).strip()
                )

                if not refined_text:
                    effective_value = getattr(
                        row,
                        "effective_path",
                        "",
                    )
                    refined_text = (
                        ""
                        if pd.isna(
                            effective_value
                        )
                        else str(
                            effective_value
                        ).strip()
                    )

                if not refined_text:
                    raise ValueError(
                        "Saved modified refinement row has no refined path: "
                        f"{image_id}"
                    )

                effective = Path(
                    refined_text
                )

                if not effective.is_file():
                    raise FileNotFoundError(
                        "Saved refined instance map is missing: "
                        f"{effective}"
                    )

                modified_paths.append(
                    effective
                )

            else:

                # Use the current Segmentation path, not a persisted absolute
                # path, so moving/mounting the dataset does not invalidate
                # unchanged images.
                effective = source

        else:

            # Historical manifest validation.
            stat = source.stat()

            if (
                int(
                    row.source_size
                )
                != int(
                    stat.st_size
                )
                or int(
                    row.source_mtime_ns
                )
                != int(
                    stat.st_mtime_ns
                )
            ):
                raise ValueError(
                    "Saved Instance Refinement is stale because its source "
                    f"Segmentation map changed: {source}. Re-run "
                    "config.run.instance_refinement = True."
                )

            effective = Path(
                str(
                    row.effective_path
                )
            )

            if not effective.is_file():
                raise FileNotFoundError(
                    "Saved effective instance map is missing: "
                    f"{effective}"
                )

            if modified:
                modified_paths.append(
                    effective
                )

        effective_paths.append(
            effective
        )

        effective_paths_by_image[
            image_id
        ] = effective

    summary_path = (
        output_dir
        / "Technical_Record"
        / "instance_refinement_summary.csv"
    )

    if "summary_path" in manifest.columns:

        summary_paths = {
            str(value).strip()
            for value in manifest[
                "summary_path"
            ].dropna().tolist()
            if str(value).strip()
        }

        if len(summary_paths) != 1:
            raise ValueError(
                "Saved refinement_manifest.csv contains an ambiguous "
                "generation summary path."
            )

        summary_path = Path(
            next(
                iter(
                    summary_paths
                )
            )
        )

        if not summary_path.is_file():
            raise FileNotFoundError(
                "Saved Instance Postprocessing generation summary "
                f"is missing: {summary_path}"
            )

    if summary_path.is_file():

        summary = pd.read_csv(
            summary_path
        )

        reconnect_count = int(
            summary.get(
                "reconnections",
                pd.Series(dtype=int),
            ).sum()
        )
        split_parent_count = int(
            summary.get(
                "split_parents",
                pd.Series(dtype=int),
            ).sum()
        )
        removed_small_count = int(
            summary.get(
                "small_removed",
                pd.Series(dtype=int),
            ).sum()
        )
        removed_large_count = int(
            summary.get(
                "large_removed",
                pd.Series(dtype=int),
            ).sum()
        )
        removed_tubular_count = int(
            summary.get(
                "tubular_removed",
                pd.Series(dtype=int),
            ).sum()
        )

    else:

        reconnect_count = 0
        split_parent_count = 0
        removed_small_count = 0
        removed_large_count = 0
        removed_tubular_count = 0

    parents = {
        path.parent.resolve()
        for path in effective_paths
    }

    effective_dir = (
        next(
            iter(
                parents
            )
        )
        if len(
            parents
        ) == 1
        else None
    )

    return InstanceRefinementResult(
        effective_instance_paths=(
            effective_paths
        ),
        effective_instance_dir=(
            effective_dir
        ),
        effective_instance_paths_by_image=(
            effective_paths_by_image
        ),
        modified_instance_paths=(
            modified_paths
        ),
        modified_count=len(
            modified_paths
        ),
        reconnect_count=(
            reconnect_count
        ),
        split_parent_count=(
            split_parent_count
        ),
        removed_small_count=(
            removed_small_count
        ),
        removed_large_count=(
            removed_large_count
        ),
        removed_tubular_count=(
            removed_tubular_count
        ),
        manifest_path=(
            manifest_path
        ),
        reused=True,
    )


# ======================================================================
# PUBLIC STAGE
# ======================================================================


def _refinement_base_canvas(
    final: np.ndarray,
) -> np.ndarray:
    """Render retained instance fills without competing boxes or label text."""

    final = np.asarray(
        final
    )

    height, width = final.shape
    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    tab10 = colormaps[
        "tab10"
    ]

    for label in _positive_labels(
        final
    ):
        rgb_float = tab10.colors[
            (
                int(label)
                - 1
            )
            % len(
                tab10.colors
            )
        ]
        rgb = tuple(
            int(
                round(
                    255.0
                    * channel
                )
            )
            for channel in rgb_float[:3]
        )
        canvas[
            final
            == int(label)
        ] = (
            rgb[2],
            rgb[1],
            rgb[0],
        )

    return canvas


def _expanded_qc_bbox(
    snapshot: _MaskSnapshot,
    *,
    shape: tuple[int, int],
) -> tuple[
    tuple[int, int],
    tuple[int, int],
]:
    """Return a padded, minimum-size OpenCV bbox clipped to the image."""

    height, width = (
        int(shape[0]),
        int(shape[1]),
    )

    raw_height = int(
        snapshot.y1
        - snapshot.y0
    )
    raw_width = int(
        snapshot.x1
        - snapshot.x0
    )

    box_height = min(
        height,
        max(
            _QC_MIN_BOX_SIZE_PX,
            raw_height
            + 2 * _QC_BOX_PADDING_PX,
        ),
    )
    box_width = min(
        width,
        max(
            _QC_MIN_BOX_SIZE_PX,
            raw_width
            + 2 * _QC_BOX_PADDING_PX,
        ),
    )

    center_y = (
        snapshot.y0
        + snapshot.y1
        - 1
    ) / 2.0
    center_x = (
        snapshot.x0
        + snapshot.x1
        - 1
    ) / 2.0

    y0 = int(
        math.floor(
            center_y
            - (
                box_height
                - 1
            ) / 2.0
        )
    )
    x0 = int(
        math.floor(
            center_x
            - (
                box_width
                - 1
            ) / 2.0
        )
    )

    y0 = max(
        0,
        min(
            y0,
            height
            - box_height,
        ),
    )
    x0 = max(
        0,
        min(
            x0,
            width
            - box_width,
        ),
    )

    return (
        (
            int(x0),
            int(y0),
        ),
        (
            int(
                x0
                + box_width
                - 1
            ),
            int(
                y0
                + box_height
                - 1
            ),
        ),
    )


def _snapshot_is_retained(
    snapshot: _MaskSnapshot,
    final: np.ndarray,
) -> bool:
    """Test whether any event-local object pixels remain foreground."""

    final_crop = final[
        snapshot.y0:snapshot.y1,
        snapshot.x0:snapshot.x1,
    ]

    return bool(
        np.any(
            snapshot.mask
            & (
                final_crop
                > 0
            )
        )
    )


def _draw_dashed_rectangle(
    canvas: np.ndarray,
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    *,
    dash_length: int = 12,
    gap_length: int = 8,
) -> None:
    """Draw one clipped dashed OpenCV rectangle in-place."""

    x0, y0 = (int(point_a[0]), int(point_a[1]))
    x1, y1 = (int(point_b[0]), int(point_b[1]))

    dash_length = max(1, int(dash_length))
    gap_length = max(1, int(gap_length))
    period = dash_length + gap_length

    for start in range(x0, x1 + 1, period):
        end = min(x1, start + dash_length - 1)
        cv2.line(canvas, (start, y0), (end, y0), color, int(thickness))
        cv2.line(canvas, (start, y1), (end, y1), color, int(thickness))

    for start in range(y0, y1 + 1, period):
        end = min(y1, start + dash_length - 1)
        cv2.line(canvas, (x0, start), (x0, end), color, int(thickness))
        cv2.line(canvas, (x1, start), (x1, end), color, int(thickness))


def _render_modified_refinement_qc(
    *,
    original: np.ndarray,
    final: np.ndarray,
    image_events: pd.DataFrame,
    event_geometries: dict[
        str,
        _RefinementEventGeometry,
    ],
    output_path: str | Path,
) -> tuple[
    Path,
    int,
    int,
]:
    """
    Render one event-driven, modified-image refinement QC panel.

    Red
        A successful removal. Its exact event-local pre-removal mask is
        restored as a visible ghost and surrounded by a linewidth-5 bbox.

    Reconnection
        Dark cyan box: whole final reconnected cell.
        Bright cyan box: fragment + bridge that produced the reconnection.

    Split
        Dashed yellow box: original fused parent before splitting.
        Bright cyan boxes: retained post-split children.

    Generic array differences are used only to verify that the image really
    changed. They never determine operation meaning or annotation color.
    """

    original = np.asarray(
        original
    )
    final = np.asarray(
        final
    )

    if original.ndim != 2 or final.ndim != 2:
        raise ValueError(
            "Modified-image refinement QC requires two 2D instance maps."
        )

    if original.shape != final.shape:
        raise ValueError(
            "Original and refined instance maps have different shapes: "
            f"{original.shape} versus {final.shape}."
        )

    if np.array_equal(
        original,
        final,
    ):
        raise ValueError(
            "Modified-image QC was requested for a map with no changed pixels."
        )

    successful = _successful_events(
        image_events
    )

    if successful.empty:
        raise RuntimeError(
            "Modified-image QC requires at least one successful modifying event."
        )

    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas = _refinement_base_canvas(
        final
    )

    # OpenCV uses BGR. The ghost is deliberately lighter than the pure-red
    # event box so the removed morphology remains visible inside it.
    red = (
        0,
        0,
        255,
    )
    removed_ghost = (
        64,
        64,
        255,
    )
    # Bright cyan marks the directly changed/retained result.
    cyan = (
        255,
        255,
        0,
    )

    # Darker blue-green context marks the whole reconnected parent.
    reconnect_whole = (
        180,
        140,
        0,
    )

    # Yellow dashed context marks the pre-split fused parent.
    split_parent = (
        0,
        255,
        255,
    )

    red_annotations = 0
    cyan_annotations = 0

    for row in successful.itertuples(
        index=False
    ):
        event_id = str(
            row.event_id
        )
        operation = str(
            row.operation
        )

        geometry = event_geometries.get(
            event_id
        )

        if geometry is None:
            raise RuntimeError(
                "Missing event-local QC geometry for successful event "
                f"{event_id}."
            )

        if (
            geometry.image_id
            != str(
                row.image_id
            )
            or geometry.operation
            != operation
        ):
            raise RuntimeError(
                "Event table and event-local QC geometry disagree for "
                f"{event_id}."
            )

        if not geometry.snapshots:
            raise RuntimeError(
                f"Successful event {event_id} contains no QC snapshots."
            )

        if operation in _REMOVAL_OPERATIONS:

            for snapshot in geometry.snapshots:
                expected_shape = (
                    snapshot.y1 - snapshot.y0,
                    snapshot.x1 - snapshot.x0,
                )

                if (
                    snapshot.mask.shape != expected_shape
                    or snapshot.y0 < 0
                    or snapshot.x0 < 0
                    or snapshot.y1 > final.shape[0]
                    or snapshot.x1 > final.shape[1]
                ):
                    raise RuntimeError(
                        f"Invalid QC snapshot geometry for event {event_id}."
                    )

                crop = canvas[
                    snapshot.y0:snapshot.y1,
                    snapshot.x0:snapshot.x1,
                ]
                crop[snapshot.mask] = removed_ghost

                point_a, point_b = _expanded_qc_bbox(
                    snapshot,
                    shape=final.shape,
                )
                cv2.rectangle(
                    canvas,
                    point_a,
                    point_b,
                    red,
                    _QC_BOX_LINE_WIDTH,
                )
                red_annotations += 1

        elif operation == "reconnect":

            whole_snapshot = geometry.result_snapshot
            changed_snapshot = (
                geometry.changed_snapshot
                if geometry.changed_snapshot is not None
                else geometry.snapshots[0]
            )

            if whole_snapshot is None:
                raise RuntimeError(
                    f"Reconnect event {event_id} lacks whole-result QC geometry."
                )

            # Context first: bbox around the complete final cell.
            point_a, point_b = _expanded_qc_bbox(
                whole_snapshot,
                shape=final.shape,
            )
            cv2.rectangle(
                canvas,
                point_a,
                point_b,
                reconnect_whole,
                _QC_BOX_LINE_WIDTH,
            )

            # Foreground change second so bright cyan stays visually dominant.
            if _snapshot_is_retained(
                changed_snapshot,
                final,
            ):
                point_a, point_b = _expanded_qc_bbox(
                    changed_snapshot,
                    shape=final.shape,
                )
                cv2.rectangle(
                    canvas,
                    point_a,
                    point_b,
                    cyan,
                    _QC_BOX_LINE_WIDTH,
                )
                cyan_annotations += 1

        elif operation == "split":

            parent_snapshot = geometry.before_snapshot

            if parent_snapshot is None:
                raise RuntimeError(
                    f"Split event {event_id} lacks pre-split parent QC geometry."
                )

            # Draw the previous fused parent first as dashed yellow context.
            point_a, point_b = _expanded_qc_bbox(
                parent_snapshot,
                shape=final.shape,
            )
            _draw_dashed_rectangle(
                canvas,
                point_a,
                point_b,
                split_parent,
                _QC_BOX_LINE_WIDTH,
            )

            # Then show every retained child with the existing bright-cyan box.
            for snapshot in geometry.snapshots:
                if not _snapshot_is_retained(
                    snapshot,
                    final,
                ):
                    continue

                point_a, point_b = _expanded_qc_bbox(
                    snapshot,
                    shape=final.shape,
                )
                cv2.rectangle(
                    canvas,
                    point_a,
                    point_b,
                    cyan,
                    _QC_BOX_LINE_WIDTH,
                )
                cyan_annotations += 1

        else:
            raise RuntimeError(
                "Unknown successful refinement operation in QC: "
                f"{operation!r}."
            )

    if (
        red_annotations
        + cyan_annotations
        == 0
    ):
        raise RuntimeError(
            "Modified-image QC produced no explicit refinement annotation."
        )

    if not cv2.imwrite(
        str(output_path),
        canvas,
    ):
        raise RuntimeError(
            f"Could not save modified-image refinement QC: {output_path}"
        )

    return (
        output_path,
        red_annotations,
        cyan_annotations,
    )


# ======================================================================
# INSTANCE POSTPROCESSING CRASH-RECOVERY
# ======================================================================

_INSTANCE_PHASE_CHECKPOINT_VERSION = 1

_INSTANCE_PHASES = (
    ("tubular", 1),
    ("large", 2),
    ("small", 3),
)


def _resume_dataframe_atomic(
    table: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
) -> Path:
    """Atomically replace one CSV after pandas finishes writing it."""

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f"{path.name}.tmp"
    )

    if temporary.exists():
        temporary.unlink()

    try:

        table.to_csv(
            temporary,
            index=index,
        )

        if (
            not temporary.is_file()
            or temporary.stat().st_size <= 0
        ):
            raise RuntimeError(
                "Atomic CSV staging produced no data: "
                f"{temporary}"
            )

        temporary.replace(
            path
        )

    finally:

        if temporary.exists():
            temporary.unlink()

    return path


def _resume_stage_signature(
    *,
    config: InstanceRefinementConfig,
    segmentation_result,
    source_paths: list[Path],
) -> str:
    """
    Fingerprint every dependency that can change postprocessing semantics.

    Microscopy TIFF contents are not hashed here. Segmentation generation
    identity plus path/size/mtime identity provides the cheap upstream
    invalidation contract established by the pipeline.
    """

    source_file = Path(
        __file__
    ).resolve()

    package_root = source_file.parent

    qc_root = (
        package_root
        .parent
        / "morphometrics"
        / "qc"
    )

    code_files = [
        source_file,
        package_root / "config.py",
        qc_root / "size.py",
        qc_root / "tubular.py",
    ]

    code_identity = [
        (
            str(
                path
            ),
            source_digest(
                path
            ),
        )
        for path in code_files
        if path.is_file()
    ]

    return fingerprint(
        "instance_postprocessing_resume_v1",
        config,
        {
            "segmentation_generation_id":
                getattr(
                    segmentation_result,
                    "generation_id",
                    None,
                ),

            "sources":
                [
                    file_identity(
                        path
                    )
                    for path in source_paths
                ],
        },
        code_identity,
    )


def _resume_phase_path(
    output_dir: Path,
    phase: str,
    rank: int,
) -> Path:

    return (
        Path(
            output_dir
        )
        / "Technical_Record"
        / "Checkpoints"
        / "Instance_Postprocessing_Phases"
        / f"{int(rank):02d}_{phase}.zip"
    )


def _resume_encode_value(
    value,
    arrays: dict[
        str,
        np.ndarray,
    ],
):
    """Encode checkpoint state without pickle."""

    if value is pd.NA:
        return None

    if value is None or isinstance(
        value,
        (
            bool,
            int,
            float,
            str,
        ),
    ):
        return value

    if isinstance(
        value,
        np.generic,
    ):
        return _resume_encode_value(
            value.item(),
            arrays,
        )

    if isinstance(
        value,
        Path,
    ):
        return {
            "__mg_type__":
                "path",

            "value":
                str(
                    value
                ),
        }

    if isinstance(
        value,
        np.ndarray,
    ):

        array_name = (
            f"arrays/"
            f"{len(arrays):08d}.npy"
        )

        arrays[
            array_name
        ] = np.asarray(
            value
        )

        return {
            "__mg_type__":
                "ndarray",

            "path":
                array_name,
        }

    if isinstance(
        value,
        _MaskSnapshot,
    ):

        from dataclasses import fields

        return {
            "__mg_type__":
                "mask_snapshot",

            "fields":
                {
                    field.name:
                        _resume_encode_value(
                            getattr(
                                value,
                                field.name,
                            ),
                            arrays,
                        )
                    for field in fields(
                        value
                    )
                },
        }

    if isinstance(
        value,
        _RefinementEventGeometry,
    ):

        from dataclasses import fields

        return {
            "__mg_type__":
                "event_geometry",

            "fields":
                {
                    field.name:
                        _resume_encode_value(
                            getattr(
                                value,
                                field.name,
                            ),
                            arrays,
                        )
                    for field in fields(
                        value
                    )
                },
        }

    if isinstance(
        value,
        tuple,
    ):
        return {
            "__mg_type__":
                "tuple",

            "items":
                [
                    _resume_encode_value(
                        item,
                        arrays,
                    )
                    for item in value
                ],
        }

    if isinstance(
        value,
        (
            set,
            frozenset,
        ),
    ):
        return {
            "__mg_type__":
                "set",

            "items":
                [
                    _resume_encode_value(
                        item,
                        arrays,
                    )
                    for item in sorted(
                        value,
                        key=repr,
                    )
                ],
        }

    if isinstance(
        value,
        list,
    ):
        return {
            "__mg_type__":
                "list",

            "items":
                [
                    _resume_encode_value(
                        item,
                        arrays,
                    )
                    for item in value
                ],
        }

    if isinstance(
        value,
        dict,
    ):
        return {
            "__mg_type__":
                "dict",

            "items":
                [
                    [
                        _resume_encode_value(
                            key,
                            arrays,
                        ),
                        _resume_encode_value(
                            item,
                            arrays,
                        ),
                    ]
                    for key, item in value.items()
                ],
        }

    raise TypeError(
        "Instance Postprocessing checkpoint cannot serialize "
        f"{type(value).__name__}: {value!r}"
    )


def _resume_decode_value(
    value,
    arrays: dict[
        str,
        np.ndarray,
    ],
):
    """Decode state written by _resume_encode_value()."""

    if not isinstance(
        value,
        dict,
    ) or "__mg_type__" not in value:
        return value

    kind = str(
        value[
            "__mg_type__"
        ]
    )

    if kind == "path":
        return Path(
            str(
                value[
                    "value"
                ]
            )
        )

    if kind == "ndarray":

        name = str(
            value[
                "path"
            ]
        )

        if name not in arrays:
            raise ValueError(
                "Checkpoint references a missing array: "
                f"{name}"
            )

        return np.asarray(
            arrays[
                name
            ]
        ).copy()

    if kind == "mask_snapshot":

        fields = {
            str(
                name
            ):
                _resume_decode_value(
                    item,
                    arrays,
                )
            for name, item in value[
                "fields"
            ].items()
        }

        return _MaskSnapshot(
            **fields
        )

    if kind == "event_geometry":

        fields = {
            str(
                name
            ):
                _resume_decode_value(
                    item,
                    arrays,
                )
            for name, item in value[
                "fields"
            ].items()
        }

        return _RefinementEventGeometry(
            **fields
        )

    if kind == "tuple":
        return tuple(
            _resume_decode_value(
                item,
                arrays,
            )
            for item in value[
                "items"
            ]
        )

    if kind == "set":
        return set(
            _resume_decode_value(
                item,
                arrays,
            )
            for item in value[
                "items"
            ]
        )

    if kind == "list":
        return [
            _resume_decode_value(
                item,
                arrays,
            )
            for item in value[
                "items"
            ]
        ]

    if kind == "dict":
        return {
            _resume_decode_value(
                key,
                arrays,
            ):
                _resume_decode_value(
                    item,
                    arrays,
                )
            for key, item in value[
                "items"
            ]
        }

    raise ValueError(
        "Unknown Instance Postprocessing checkpoint type: "
        f"{kind!r}"
    )


def _resume_write_phase_state(
    path: Path,
    *,
    phase: str,
    rank: int,
    working_maps: dict[
        str,
        np.ndarray,
    ],
    events: list[
        dict
    ],
    event_geometries: dict[
        str,
        _RefinementEventGeometry,
    ],
    event_counter: int,
) -> None:
    """
    Persist complete phase-end state as one atomic ZIP archive.

    Only images changed so far are stored. Immutable Segmentation maps are
    never copied into the checkpoint.
    """

    arrays: dict[
        str,
        np.ndarray,
    ] = {}

    state = {
        "working_maps":
            working_maps,

        "events":
            events,

        "event_geometries":
            event_geometries,

        "event_counter":
            int(
                event_counter
            ),
    }

    payload = {
        "format_version":
            _INSTANCE_PHASE_CHECKPOINT_VERSION,

        "phase":
            str(
                phase
            ),

        "rank":
            int(
                rank
            ),

        "state":
            _resume_encode_value(
                state,
                arrays,
            ),
    }

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=(
            zipfile.ZIP_DEFLATED
        ),
    ) as archive:

        archive.writestr(
            "state.json",
            json.dumps(
                payload,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
                allow_nan=True,
            ),
        )

        for (
            name,
            array,
        ) in arrays.items():

            array_buffer = (
                io.BytesIO()
            )

            np.save(
                array_buffer,
                np.asarray(
                    array
                ),
                allow_pickle=False,
            )

            archive.writestr(
                name,
                array_buffer.getvalue(),
            )

    atomic_write_bytes(
        path,
        buffer.getvalue(),
    )


def _resume_read_phase_state(
    path: Path,
    *,
    expected_phase: str,
    expected_rank: int,
) -> dict:
    """
    Read and validate one complete phase checkpoint.
    """

    path = Path(
        path
    )

    arrays: dict[
        str,
        np.ndarray,
    ] = {}

    try:

        with zipfile.ZipFile(
            path,
            mode="r",
        ) as archive:

            names = set(
                archive.namelist()
            )

            if "state.json" not in names:
                raise ValueError(
                    "Checkpoint archive has no state.json."
                )

            payload = json.loads(
                archive.read(
                    "state.json"
                ).decode(
                    "utf-8"
                )
            )

            for name in sorted(
                names
            ):

                if not (
                    name.startswith(
                        "arrays/"
                    )
                    and name.endswith(
                        ".npy"
                    )
                ):
                    continue

                arrays[
                    name
                ] = np.load(
                    io.BytesIO(
                        archive.read(
                            name
                        )
                    ),
                    allow_pickle=False,
                )

    except Exception as exc:

        raise ValueError(
            "Could not read Instance Postprocessing phase checkpoint: "
            f"{path}"
        ) from exc

    if int(
        payload.get(
            "format_version",
            -1,
        )
    ) != int(
        _INSTANCE_PHASE_CHECKPOINT_VERSION
    ):
        raise ValueError(
            "Incompatible Instance Postprocessing checkpoint version."
        )

    if (
        str(
            payload.get(
                "phase",
                "",
            )
        )
        != str(
            expected_phase
        )
        or int(
            payload.get(
                "rank",
                -1,
            )
        )
        != int(
            expected_rank
        )
    ):
        raise ValueError(
            "Instance Postprocessing checkpoint phase identity mismatch."
        )

    state = _resume_decode_value(
        payload[
            "state"
        ],
        arrays,
    )

    if not isinstance(
        state,
        dict,
    ):
        raise ValueError(
            "Decoded Instance Postprocessing checkpoint is not a mapping."
        )

    working_maps = state.get(
        "working_maps"
    )

    events = state.get(
        "events"
    )

    geometries = state.get(
        "event_geometries"
    )

    if not isinstance(
        working_maps,
        dict,
    ):
        raise ValueError(
            "Checkpoint working_maps is not a mapping."
        )

    if not isinstance(
        events,
        list,
    ):
        raise ValueError(
            "Checkpoint events is not a list."
        )

    if not isinstance(
        geometries,
        dict,
    ):
        raise ValueError(
            "Checkpoint event_geometries is not a mapping."
        )

    validated_maps: dict[
        str,
        np.ndarray,
    ] = {}

    for image_id, labels in working_maps.items():

        labels = validate_instance_map(
            np.asarray(
                labels
            )
        )

        if labels.ndim != 2:
            raise ValueError(
                "Checkpoint contains a non-2D working instance map: "
                f"{image_id!r}"
            )

        validated_maps[
            str(
                image_id
            )
        ] = labels.astype(
            np.int32,
            copy=False,
        )

    validated_geometries: dict[
        str,
        _RefinementEventGeometry,
    ] = {}

    for event_id, geometry in geometries.items():

        if not isinstance(
            geometry,
            _RefinementEventGeometry,
        ):
            raise ValueError(
                "Checkpoint event geometry has the wrong type: "
                f"{event_id!r}"
            )

        if str(
            geometry.event_id
        ) != str(
            event_id
        ):
            raise ValueError(
                "Checkpoint event geometry key/id mismatch."
            )

        validated_geometries[
            str(
                event_id
            )
        ] = geometry

    return {
        "working_maps":
            validated_maps,

        "events":
            [
                dict(
                    event
                )
                for event in events
            ],

        "event_geometries":
            validated_geometries,

        "event_counter":
            int(
                state.get(
                    "event_counter",
                    0,
                )
            ),
    }


def _resume_commit_phase(
    *,
    journal: CheckpointJournal,
    output_dir: Path,
    stage_signature: str,
    phase: str,
    rank: int,
    working_maps: dict[
        str,
        np.ndarray,
    ],
    events: list[
        dict
    ],
    event_geometries: dict[
        str,
        _RefinementEventGeometry,
    ],
    event_counter: int,
) -> Path:
    """
    Write, read back, validate, then commit one phase checkpoint.
    """

    path = _resume_phase_path(
        output_dir,
        phase,
        rank,
    )

    _resume_write_phase_state(
        path,
        phase=phase,
        rank=rank,
        working_maps=(
            working_maps
        ),
        events=events,
        event_geometries=(
            event_geometries
        ),
        event_counter=(
            event_counter
        ),
    )

    # A journal record can only be created for a readable phase archive.
    restored = _resume_read_phase_state(
        path,
        expected_phase=(
            phase
        ),
        expected_rank=(
            rank
        ),
    )

    if (
        set(
            restored[
                "working_maps"
            ]
        )
        != set(
            working_maps
        )
        or int(
            restored[
                "event_counter"
            ]
        )
        != int(
            event_counter
        )
    ):
        raise RuntimeError(
            "Instance Postprocessing phase checkpoint failed "
            "its read-back validation."
        )

    item_signature = fingerprint(
        stage_signature,
        str(
            phase
        ),
        int(
            rank
        ),
    )

    journal.commit(
        item_id=(
            str(
                phase
            )
        ),
        item_signature=(
            item_signature
        ),
        outputs=[
            path
        ],
        metadata={
            "phase_rank":
                int(
                    rank
                ),

            "modified_images":
                int(
                    len(
                        working_maps
                    )
                ),

            "events":
                int(
                    len(
                        events
                    )
                ),
        },
    )

    return path


def _resume_load_latest_phase(
    *,
    journal: CheckpointJournal,
    output_dir: Path,
    stage_signature: str,
) -> tuple[
    int,
    str | None,
    dict | None,
]:
    """
    Restore the latest readable committed phase, falling back if necessary.
    """

    for phase, rank in reversed(
        _INSTANCE_PHASES
    ):

        path = _resume_phase_path(
            output_dir,
            phase,
            rank,
        )

        item_signature = fingerprint(
            stage_signature,
            str(
                phase
            ),
            int(
                rank
            ),
        )

        record = journal.reusable_record(
            item_id=(
                phase
            ),
            item_signature=(
                item_signature
            ),
            outputs=[
                path
            ],
        )

        if record is None:
            continue

        try:

            state = _resume_read_phase_state(
                path,
                expected_phase=(
                    phase
                ),
                expected_rank=(
                    rank
                ),
            )

        except Exception as exc:

            print(
                "Ignoring invalid Instance Postprocessing "
                f"{phase!r} checkpoint: {exc}"
            )

            continue

        return (
            int(
                rank
            ),
            str(
                phase
            ),
            state,
        )

    return (
        0,
        None,
        None,
    )


def _resume_load_complete_result(
    *,
    segmentation_result,
    output_dir: Path,
    stage_signature: str,
):
    """
    Reuse a fully committed generation only when its signature matches.
    """

    manifest_path = (
        Path(
            output_dir
        )
        / "Instance_Postprocessing"
        / "refinement_manifest.csv"
    )

    if not manifest_path.is_file():
        return None

    try:

        manifest = pd.read_csv(
            manifest_path
        )

        if (
            "resume_signature"
            not in manifest.columns
        ):
            return None

        signatures = {
            str(
                value
            ).strip()
            for value in manifest[
                "resume_signature"
            ].dropna().tolist()
            if str(
                value
            ).strip()
        }

        if signatures != {
            str(
                stage_signature
            )
        }:
            return None

        return load_saved_instance_refinement(
            segmentation_result=(
                segmentation_result
            ),
            output_dir=(
                output_dir
            ),
            allow_missing=False,
        )

    except Exception as exc:

        print(
            "Saved complete Instance Postprocessing generation "
            f"is not reusable: {exc}"
        )

        return None


def run_instance_refinement(
    *,
    segmentation_result,
    output_dir: str | Path,
    config: InstanceRefinementConfig,
    resume: bool = False,
) -> InstanceRefinementResult:
    """
    Refine canonical instance maps without rewriting unchanged image data.

    Disk policy
    -----------
    Segmentation/_Instance_Maps
        Immutable source maps.

    Instance_Postprocessing/Postprocessed_Maps
        Contains real TIFFs only for images actually changed by
        postprocessing. Unchanged images remain in immutable Segmentation
        storage and are resolved per image through the effective-path map.

    No intermediate postprocessed TIFF is written between tubular, large,
    and small phases.
    """

    config.__post_init__()

    output_dir = Path(
        output_dir
    )

    root = (
        output_dir
        / "Instance_Postprocessing"
    )

    legacy_root = (
        output_dir
        / "Instance_Refinement"
    )

    technical_dir = (
        output_dir
        / "Technical_Record"
    )

    # A fresh invocation writes into an unpublished generation. The
    # canonical manifest is switched only after the generation is complete.
    refinement_generation_id = (
        uuid4().hex
    )

    modified_dir = (
        root
        / "Postprocessed_Maps"
        / refinement_generation_id
    )

    modified_qc_dir = (
        root
        / "QC"
        / "Postprocessed_Images"
        / refinement_generation_id
    )

    generation_technical_dir = (
        technical_dir
        / "Instance_Postprocessing"
        / refinement_generation_id
    )

    generation_summary_path = (
        generation_technical_dir
        / "instance_refinement_summary.csv"
    )

    generation_events_path = (
        generation_technical_dir
        / "instance_refinement_events.csv"
    )

    source_paths = [
        Path(
            path
        )
        for path in (
            segmentation_result
            .instance_paths
        )
    ]

    if not source_paths:
        raise ValueError(
            "Instance Refinement received no instance maps."
        )

    source_by_id = {
        path.stem: path
        for path in source_paths
    }

    if len(
        source_by_id
    ) != len(
        source_paths
    ):
        raise ValueError(
            "Instance Refinement received duplicate canonical image stems."
        )


    stage_signature = _resume_stage_signature(
        config=config,
        segmentation_result=(
            segmentation_result
        ),
        source_paths=(
            source_paths
        ),
    )

    if resume:

        complete_result = (
            _resume_load_complete_result(
                segmentation_result=(
                    segmentation_result
                ),
                output_dir=(
                    output_dir
                ),
                stage_signature=(
                    stage_signature
                ),
            )
        )

        if complete_result is not None:

            print()
            print("=" * 72)
            print("INSTANCE POSTPROCESSING")
            print("=" * 72)
            print()
            print(
                "Resume: compatible completed generation reused."
            )
            print()

            return complete_result

    cached_detection = None

    if resume:

        cached_detection = _load_detection_cache(
            output_dir=output_dir,
            segmentation_result=(
                segmentation_result
            ),
            config=config,
            source_by_id=(
                source_by_id
            ),
        )

    reuse_detection = (
        cached_detection
        is not None
    )

    # First run after the naming migration: preserve the previously generated
    # detection overview without rescanning the dataset. Later runs preserve
    # the canonical Instance_Postprocessing/QC/Overview in-place.
    legacy_overview = (
        legacy_root
        / "QC"
        / "Overview"
    )
    canonical_overview = (
        root
        / "QC"
        / "Overview"
    )

    if (
        reuse_detection
        and legacy_overview.is_dir()
        and not canonical_overview.exists()
    ):
        canonical_overview.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copytree(
            legacy_overview,
            canonical_overview,
        )

    # --------------------------------------------------------------
    # NON-DESTRUCTIVE GENERATION PREPARATION
    #
    # Existing committed outputs remain untouched while this invocation
    # computes. A new generation becomes authoritative only when the
    # canonical refinement_manifest.csv is atomically replaced at the end.
    # --------------------------------------------------------------

    modified_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    modified_qc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    technical_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generation_technical_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("INSTANCE POSTPROCESSING")
    print("=" * 72)
    print()
    print(
        f"Remove small:       "
        f"{config.remove_small}"
    )
    print(
        f"Tubular action:     "
        f"{config.tubular_action}"
    )
    print(
        f"Large action:       "
        f"{config.large_action}"
    )
    print(
        f"Detection cache:    "
        f"{'REUSED' if reuse_detection else 'REBUILT'}"
    )
    print()

    _write_configuration_record(
        output_dir,
        config,
    )

    # ==================================================================
    # INITIAL DETECTION
    #
    # Full disk scan only when Segmentation generation or detector
    # parameters changed. Action-only reruns load the persisted object QC.
    # ==================================================================

    if reuse_detection:

        print(
            "Initial object detection"
        )
        print(
            "-" * 72
        )
        print(
            "Using cached Instance-QC detections; "
            "canonical Segmentation maps are not rescanned."
        )
        print()

        initial_table = (
            cached_detection
            .copy()
            .reset_index(
                drop=True
            )
        )

    else:

        print(
            "Initial object scan"
        )
        print(
            "-" * 72
        )

        scanned_table = _object_table(
            source_paths,
            print_progress=True,
            progress_label="Inspecting",
        )

        print()

        initial_table = _score_qc(
            scanned_table,
            config,
            include_tubular=True,
        )

        _write_qc_overview(
            initial_table,
            root,
            config,
        )

        _save_detection_cache(
            table=initial_table,
            output_dir=output_dir,
            segmentation_result=(
                segmentation_result
            ),
            config=config,
        )

    # From this point forward, phase-to-phase rescoring needs only scalar
    # object geometry. ROIs are intentionally not retained for unchanged
    # images on cached reruns.
    base_columns = [
        "image_id",
        "cell_id",
        "map_label",
        "component_area",
        "object_radius_max",
    ]

    base_table = (
        initial_table[
            base_columns
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # Empty instance maps intentionally have no object-table rows. Seed every
    # canonical image with zero before adding counts for non-empty maps so the
    # final per-image summary remains total over the complete source set.
    initial_counts = {
        str(
            image_id
        ): 0
        for image_id in source_by_id
    }

    initial_counts.update(
        base_table
        .groupby(
            "image_id",
            sort=False,
        )[
            "map_label"
        ]
        .nunique()
        .astype(int)
        .to_dict()
    )

    # Only images that actually change are retained in memory.
    working_maps: dict[
        str,
        np.ndarray,
    ] = {}

    # Original arrays are cached only for candidate images that are opened
    # after the initial scan. This makes final equality validation cheap.
    original_cache: dict[
        str,
        np.ndarray,
    ] = {}

    events: list[
        dict
    ] = []

    event_geometries: dict[
        str,
        _RefinementEventGeometry,
    ] = {}

    event_counter = [
        0
    ]

    phase_journal = CheckpointJournal(
        output_dir=output_dir,
        stage="instance_postprocessing_phases",
        resume=resume,
        stage_signature=(
            stage_signature
        ),
    )

    resume_phase_rank = 0
    resume_phase_name = None

    if resume:

        (
            resume_phase_rank,
            resume_phase_name,
            resume_state,
        ) = _resume_load_latest_phase(
            journal=(
                phase_journal
            ),
            output_dir=(
                output_dir
            ),
            stage_signature=(
                stage_signature
            ),
        )

        if resume_state is not None:

            working_maps = (
                resume_state[
                    "working_maps"
                ]
            )

            events = (
                resume_state[
                    "events"
                ]
            )

            event_geometries = (
                resume_state[
                    "event_geometries"
                ]
            )

            event_counter = [
                int(
                    resume_state[
                        "event_counter"
                    ]
                )
            ]

            unknown_images = (
                set(
                    working_maps
                )
                - set(
                    source_by_id
                )
            )

            if unknown_images:
                raise ValueError(
                    "Instance Postprocessing checkpoint contains "
                    "unknown image IDs: "
                    f"{sorted(unknown_images)}"
                )

            if working_maps:

                base_table = _refresh_object_table(
                    base_table,
                    image_ids=set(
                        working_maps
                    ),
                    working_maps=(
                        working_maps
                    ),
                )

            print(
                "Resume checkpoint: "
                f"{resume_phase_name} phase complete."
            )
            print()

    # ==================================================================
    # 1. TUBULAR / ORPHAN ACTION
    # ==================================================================

    if resume_phase_rank < 1:

        if config.tubular_action in {
            "remove",
            "reconnect",
        }:

            median_area, typical_radius = (
                _typical_scale(
                    initial_table
                )
            )

            tubular_work = []

            for image_id, group in initial_table.groupby(
                "image_id",
                sort=False,
            ):

                if (
                    config.tubular_action
                    == "reconnect"
                ):

                    # Reconnection is a repair attempt, so statistically small
                    # fragments must be offered a chance to reconnect BEFORE the
                    # final small-object removal phase. Suspicious tubular objects
                    # remain candidates as before.
                    reconnect_candidate = (
                        group[
                            "suspicious_small"
                        ].astype(
                            bool
                        )
                        | group[
                            "suspicious_tubular"
                        ].astype(
                            bool
                        )
                    )

                    orphan_labels = {
                        int(
                            value
                        )
                        for value in group.loc[
                            reconnect_candidate,
                            "map_label",
                        ].tolist()
                    }

                else:

                    orphan_labels = {
                        int(
                            value
                        )
                        for value in group.loc[
                            group[
                                "suspicious_tubular"
                            ].astype(
                                bool
                            ),
                            "map_label",
                        ].tolist()
                    }

                if orphan_labels:

                    tubular_work.append(
                        (
                            str(
                                image_id
                            ),
                            group,
                            orphan_labels,
                        )
                    )

            if tubular_work:

                print(
                    (
                        "Fragment/orphan action"
                        if config.tubular_action == "reconnect"
                        else "Tubular/orphan action"
                    )
                )
                print(
                    "-" * 72
                )

            tubular_modified: set[
                str
            ] = set()

            tubular_total = len(
                tubular_work
            )

            for index, (
                image_id,
                group,
                orphan_labels,
            ) in enumerate(
                tubular_work,
                start=1,
            ):

                print(
                    f"[{index}/{tubular_total}] "
                    f"{image_id}.tif "
                    f"({len(orphan_labels)} candidate"
                    f"{'' if len(orphan_labels) == 1 else 's'})"
                )

                labels = _working_map(
                    image_id,
                    source_by_id=(
                        source_by_id
                    ),
                    working_maps=(
                        working_maps
                    ),
                    original_cache=(
                        original_cache
                    ),
                )

                before = labels.copy()

                if (
                    config.tubular_action
                    == "remove"
                ):

                    for orphan_label in sorted(
                        orphan_labels
                    ):

                        removed_mask = (
                            labels
                            == orphan_label
                        )

                        if np.any(
                            removed_mask
                        ):

                            snapshot = _mask_snapshot(
                                removed_mask,
                                label=orphan_label,
                            )

                            if snapshot is None:
                                raise RuntimeError(
                                    "Tubular removal has no event-local geometry."
                                )

                            pixels_removed = int(
                                np.count_nonzero(
                                    removed_mask
                                )
                            )

                            labels[
                                removed_mask
                            ] = 0

                            event = _event_row(
                                event_counter,
                                image_id=image_id,
                                operation="remove_tubular",
                                status="success",
                                changed=True,
                                source_label=orphan_label,
                                pixels_removed=pixels_removed,
                                bbox_snapshot=snapshot,
                                reason="suspicious_tubular_removed",
                            )

                            events.append(
                                event
                            )

                            event_geometries[
                                event[
                                    "event_id"
                                ]
                            ] = _RefinementEventGeometry(
                                event_id=event[
                                    "event_id"
                                ],
                                image_id=image_id,
                                operation="remove_tubular",
                                snapshots=(
                                    snapshot,
                                ),
                            )

                else:

                    large_labels = {
                        int(
                            value
                        )

                        for value in group.loc[
                            group[
                                "suspicious_large"
                            ].astype(
                                bool
                            ),
                            "map_label",
                        ].tolist()
                    }

                    (
                        labels,
                        reconnect_events,
                        reconnect_geometries,
                    ) = _reconnect_image(
                        labels,
                        orphan_labels=(
                            orphan_labels
                        ),
                        large_labels=(
                            large_labels
                        ),
                        median_area=(
                            median_area
                        ),
                        typical_radius=(
                            typical_radius
                        ),
                        config=config,
                        image_id=(
                            image_id
                        ),
                        event_counter=(
                            event_counter
                        ),
                    )

                    events.extend(
                        reconnect_events
                    )

                    event_geometries.update(
                        reconnect_geometries
                    )

                if not np.array_equal(
                    labels,
                    before,
                ):

                    working_maps[
                        image_id
                    ] = labels

                    tubular_modified.add(
                        image_id
                    )

            if tubular_work:

                print()

            if tubular_modified:

                base_table = (
                    _refresh_object_table(
                        base_table,
                        image_ids=(
                            tubular_modified
                        ),
                        working_maps=(
                            working_maps
                        ),
                    )
                )


        _resume_commit_phase(
            journal=phase_journal,
            output_dir=output_dir,
            stage_signature=stage_signature,
            phase='tubular',
            rank=1,
            working_maps=working_maps,
            events=events,
            event_geometries=event_geometries,
            event_counter=event_counter[0],
        )

        resume_phase_rank = 1

    else:

        print(
            "Resume: tubular/orphan phase already complete."
        )

    # ==================================================================
    # 2. RECOMPUTE GLOBAL SIZE COORDINATE; LARGE ACTION
    #
    # Global robust statistics are recalculated, but unchanged masks are
    # not read from disk again.
    # ==================================================================

    if resume_phase_rank < 2:

        large_table = _score_qc(
            base_table,
            config,
            include_tubular=False,
        )

        if config.large_action in {
            "remove",
            "split",
        }:

            median_area, typical_radius = (
                _typical_scale(
                    large_table
                )
            )

            large_work = []

            for image_id, group in large_table.groupby(
                "image_id",
                sort=False,
            ):

                large_labels = [
                    int(
                        value
                    )

                    for value in group.loc[
                        group[
                            "suspicious_large"
                        ].astype(
                            bool
                        ),
                        "map_label",
                    ].tolist()
                ]

                if large_labels:

                    large_work.append(
                        (
                            str(
                                image_id
                            ),
                            large_labels,
                        )
                    )

            if large_work:

                print(
                    "Large-object action"
                )
                print(
                    "-" * 72
                )

            large_modified: set[
                str
            ] = set()

            large_total = len(
                large_work
            )

            for index, (
                image_id,
                large_labels,
            ) in enumerate(
                large_work,
                start=1,
            ):

                print(
                    f"[{index}/{large_total}] "
                    f"{image_id}.tif "
                    f"({len(large_labels)} candidate"
                    f"{'' if len(large_labels) == 1 else 's'})"
                )

                labels = _working_map(
                    image_id,
                    source_by_id=(
                        source_by_id
                    ),
                    working_maps=(
                        working_maps
                    ),
                    original_cache=(
                        original_cache
                    ),
                )

                before = labels.copy()

                next_label = (
                    int(
                        labels.max()
                    )
                    + 1
                )

                if (
                    config.large_action
                    == "remove"
                ):

                    for parent_label in (
                        large_labels
                    ):

                        removed_mask = (
                            labels
                            == parent_label
                        )

                        if np.any(
                            removed_mask
                        ):

                            snapshot = _mask_snapshot(
                                removed_mask,
                                label=parent_label,
                            )

                            if snapshot is None:
                                raise RuntimeError(
                                    "Large-object removal has no event-local geometry."
                                )

                            pixels_removed = int(
                                np.count_nonzero(
                                    removed_mask
                                )
                            )

                            labels[
                                removed_mask
                            ] = 0

                            event = _event_row(
                                event_counter,
                                image_id=image_id,
                                operation="remove_large",
                                status="success",
                                changed=True,
                                source_label=parent_label,
                                pixels_removed=pixels_removed,
                                bbox_snapshot=snapshot,
                                reason="suspicious_large_removed",
                            )

                            events.append(
                                event
                            )

                            event_geometries[
                                event[
                                    "event_id"
                                ]
                            ] = _RefinementEventGeometry(
                                event_id=event[
                                    "event_id"
                                ],
                                image_id=image_id,
                                operation="remove_large",
                                snapshots=(
                                    snapshot,
                                ),
                            )

                else:

                    for parent_label in (
                        large_labels
                    ):

                        if not np.any(
                            labels
                            == parent_label
                        ):

                            continue

                        (
                            labels,
                            next_label,
                            event,
                            geometry,
                        ) = _split_parent(
                            labels,
                            parent_label=(
                                parent_label
                            ),
                            median_area=(
                                median_area
                            ),
                            typical_radius=(
                                typical_radius
                            ),
                            config=config,
                            next_label=(
                                next_label
                            ),
                            image_id=(
                                image_id
                            ),
                            event_counter=(
                                event_counter
                            ),
                        )

                        events.append(
                            event
                        )

                        if geometry is not None:

                            event_geometries[
                                geometry.event_id
                            ] = geometry

                if not np.array_equal(
                    labels,
                    before,
                ):

                    working_maps[
                        image_id
                    ] = labels

                    large_modified.add(
                        image_id
                    )

            if large_work:

                print()

            if large_modified:

                base_table = (
                    _refresh_object_table(
                        base_table,
                        image_ids=(
                            large_modified
                        ),
                        working_maps=(
                            working_maps
                        ),
                    )
                )


        _resume_commit_phase(
            journal=phase_journal,
            output_dir=output_dir,
            stage_signature=stage_signature,
            phase='large',
            rank=2,
            working_maps=working_maps,
            events=events,
            event_geometries=event_geometries,
            event_counter=event_counter[0],
        )

        resume_phase_rank = 2

    else:

        print(
            "Resume: large-object phase already complete."
        )

    # ==================================================================
    # 3. RECOMPUTE GLOBAL SIZE COORDINATE; SMALL ACTION LAST
    # ==================================================================

    if resume_phase_rank < 3:

        small_table = _score_qc(
            base_table,
            config,
            include_tubular=False,
        )

        if config.remove_small:

            small_work = []

            for image_id, group in small_table.groupby(
                "image_id",
                sort=False,
            ):

                small_labels = [
                    int(
                        value
                    )

                    for value in group.loc[
                        group[
                            "suspicious_small"
                        ].astype(
                            bool
                        ),
                        "map_label",
                    ].tolist()
                ]

                if small_labels:

                    small_work.append(
                        (
                            str(
                                image_id
                            ),
                            small_labels,
                        )
                    )

            if small_work:

                print(
                    "Small-object action"
                )
                print(
                    "-" * 72
                )

            small_total = len(
                small_work
            )

            for index, (
                image_id,
                small_labels,
            ) in enumerate(
                small_work,
                start=1,
            ):

                print(
                    f"[{index}/{small_total}] "
                    f"{image_id}.tif "
                    f"({len(small_labels)} candidate"
                    f"{'' if len(small_labels) == 1 else 's'})"
                )

                labels = _working_map(
                    image_id,
                    source_by_id=(
                        source_by_id
                    ),
                    working_maps=(
                        working_maps
                    ),
                    original_cache=(
                        original_cache
                    ),
                )

                before = labels.copy()

                for small_label in (
                    small_labels
                ):

                    removed_mask = (
                        labels
                        == small_label
                    )

                    if np.any(
                        removed_mask
                    ):

                        snapshot = _mask_snapshot(
                            removed_mask,
                            label=small_label,
                        )

                        if snapshot is None:
                            raise RuntimeError(
                                "Small-object removal has no event-local geometry."
                            )

                        pixels_removed = int(
                            np.count_nonzero(
                                removed_mask
                            )
                        )

                        labels[
                            removed_mask
                        ] = 0

                        event = _event_row(
                            event_counter,
                            image_id=image_id,
                            operation="remove_small",
                            status="success",
                            changed=True,
                            source_label=small_label,
                            pixels_removed=pixels_removed,
                            bbox_snapshot=snapshot,
                            reason="suspicious_small_removed",
                        )

                        events.append(
                            event
                        )

                        event_geometries[
                            event[
                                "event_id"
                            ]
                        ] = _RefinementEventGeometry(
                            event_id=event[
                                "event_id"
                            ],
                            image_id=image_id,
                            operation="remove_small",
                            snapshots=(
                                snapshot,
                            ),
                        )

                if not np.array_equal(
                    labels,
                    before,
                ):

                    working_maps[
                        image_id
                    ] = labels

            if small_work:

                print()


        _resume_commit_phase(
            journal=phase_journal,
            output_dir=output_dir,
            stage_signature=stage_signature,
            phase='small',
            rank=3,
            working_maps=working_maps,
            events=events,
            event_geometries=event_geometries,
            event_counter=event_counter[0],
        )

        resume_phase_rank = 3

    else:

        print(
            "Resume: small-object phase already complete."
        )

    # ==================================================================
    # FINAL WRITE
    #
    # Each modified image is written exactly once. Unmodified images become
    # lightweight links to the immutable Segmentation source map.
    # ==================================================================

    manifest_rows: list[
        dict
    ] = []

    effective_paths: list[
        Path
    ] = []

    effective_paths_by_image: dict[
        str,
        Path,
    ] = {}

    modified_paths: list[
        Path
    ] = []

    summary_rows: list[
        dict
    ] = []

    events_table = pd.DataFrame(
        events,
        columns=_EVENT_COLUMNS,
    )

    if not events_table[
        "event_id"
    ].is_unique:
        raise RuntimeError(
            "Instance Refinement produced duplicate event IDs."
        )

    successful_events_table = _successful_events(
        events_table
    )

    modified_image_ids = []

    for image_id, labels in (
        working_maps.items()
    ):

        original = (
            original_cache.get(
                image_id
            )
        )

        if original is None:

            original = _load_instance_map(
                source_by_id[
                    image_id
                ]
            )

            original_cache[
                image_id
            ] = original

        if not np.array_equal(
            original,
            labels,
        ):

            modified_image_ids.append(
                image_id
            )

    modified_image_ids = sorted(
        modified_image_ids
    )

    if modified_image_ids:

        print(
            "Writing modified maps and QC"
        )
        print(
            "-" * 72
        )

    modified_total = len(
        modified_image_ids
    )

    modified_set = set(
        modified_image_ids
    )

    successful_image_ids = set(
        successful_events_table[
            "image_id"
        ].astype(str)
    )

    if successful_image_ids != modified_set:
        raise RuntimeError(
            "Successful refinement events and modified maps disagree. "
            f"Events only: {sorted(successful_image_ids - modified_set)}; "
            f"maps only: {sorted(modified_set - successful_image_ids)}."
        )

    successful_event_ids = set(
        successful_events_table[
            "event_id"
        ].astype(str)
    )

    if successful_event_ids != set(
        event_geometries
    ):
        raise RuntimeError(
            "Successful refinement events and event-local QC geometry disagree."
        )

    total_red_annotations = 0
    total_cyan_annotations = 0

    for index, image_id in enumerate(
        modified_image_ids,
        start=1,
    ):

        print(
            f"[{index}/{modified_total}] "
            f"{image_id}.tif"
        )

        output_path = (
            modified_dir
            / f"{image_id}.tif"
        )

        final = working_maps[
            image_id
        ]

        _save_instance_map(
            output_path,
            final,
        )

        modified_paths.append(
            output_path
        )

        original_for_qc = original_cache.get(
            image_id
        )

        if original_for_qc is None:
            original_for_qc = _load_instance_map(
                source_by_id[
                    image_id
                ]
            )

            original_cache[
                image_id
            ] = original_for_qc

        image_events = successful_events_table.loc[
            successful_events_table[
                "image_id"
            ].astype(str).eq(
                image_id
            )
        ]

        (
            _,
            red_annotations,
            cyan_annotations,
        ) = _render_modified_refinement_qc(
            original=original_for_qc,
            final=final,
            image_events=image_events,
            event_geometries=event_geometries,
            output_path=(
                modified_qc_dir
                / f"{image_id}.png"
            ),
        )

        total_red_annotations += int(
            red_annotations
        )
        total_cyan_annotations += int(
            cyan_annotations
        )

    if modified_image_ids:

        print()

    written_qc_ids = {
        path.stem
        for path in modified_qc_dir.glob(
            "*.png"
        )
        if not path.name.startswith(
            "._"
        )
    }

    if written_qc_ids != modified_set:
        raise RuntimeError(
            "Modified-map and modified-QC image sets disagree. "
            f"QC only: {sorted(written_qc_ids - modified_set)}; "
            f"maps only: {sorted(modified_set - written_qc_ids)}."
        )

    # Build one authoritative per-image resolver without mirroring unchanged
    # TIFFs into Instance_Postprocessing. Modified images point to Postprocessed_Maps;
    # unchanged images point directly to immutable Segmentation outputs.
    segmentation_generation_id = getattr(
        segmentation_result,
        "generation_id",
        None,
    )

    for source in source_paths:

        image_id = (
            source.stem
        )

        modified = (
            image_id
            in modified_set
        )

        if modified:

            effective = (
                modified_dir
                / f"{image_id}.tif"
            )

            final = working_maps[
                image_id
            ]

            effective_source = (
                "instance_refinement"
            )

            refined_path = str(
                effective.resolve()
            )

        else:

            effective = source
            final = None

            effective_source = (
                "segmentation"
            )

            refined_path = ""

        effective_paths.append(
            effective
        )

        effective_paths_by_image[
            image_id
        ] = effective

        manifest_rows.append(
            {
                "image_id": (
                    image_id
                ),
                "segmentation_generation_id": (
                    ""
                    if segmentation_generation_id is None
                    else str(
                        segmentation_generation_id
                    )
                ),
                "refinement_generation_id": (
                    refinement_generation_id
                ),
                "resume_signature": (
                    stage_signature
                ),
                "summary_path": str(
                    generation_summary_path.resolve()
                ),
                "events_path": str(
                    generation_events_path.resolve()
                ),
                "modified": bool(
                    modified
                ),
                "source_path": str(
                    source.resolve()
                ),
                "refined_path": (
                    refined_path
                ),
                "effective_path": str(
                    effective.resolve()
                ),
                "effective_source": (
                    effective_source
                ),
            }
        )

        image_events = (
            successful_events_table.loc[
                successful_events_table[
                    "image_id"
                ]
                == image_id
            ]

            if not successful_events_table.empty

            else pd.DataFrame()
        )

        operation_counts = (
            image_events[
                "operation"
            ]
            .value_counts()
            .to_dict()

            if not image_events.empty

            else {}
        )

        final_instances = (
            _count_instances(
                final
            )
            if final is not None
            else int(
                initial_counts[
                    image_id
                ]
            )
        )

        summary_rows.append(
            {
                "image_id": (
                    image_id
                ),
                "modified": bool(
                    modified
                ),
                "initial_instances": int(
                    initial_counts[
                        image_id
                    ]
                ),
                "final_instances": int(
                    final_instances
                ),
                "reconnections": int(
                    operation_counts.get(
                        "reconnect",
                        0,
                    )
                ),
                "split_parents": int(
                    operation_counts.get(
                        "split",
                        0,
                    )
                ),
                "small_removed": int(
                    operation_counts.get(
                        "remove_small",
                        0,
                    )
                ),
                "large_removed": int(
                    operation_counts.get(
                        "remove_large",
                        0,
                    )
                ),
                "tubular_removed": int(
                    operation_counts.get(
                        "remove_tubular",
                        0,
                    )
                ),
            }
        )

    manifest_path = (
        root
        / "refinement_manifest.csv"
    )

    manifest_table = pd.DataFrame(
        manifest_rows
    )

    summary_table = pd.DataFrame(
        summary_rows
    )

    # Generation-owned technical tables are completed first.
    _resume_dataframe_atomic(
        summary_table,
        generation_summary_path,
        index=False,
    )

    _resume_dataframe_atomic(
        events_table,
        generation_events_path,
        index=False,
    )

    # The manifest is the publication/commit pointer. Prior committed
    # generations remain authoritative until this atomic replacement.
    _resume_dataframe_atomic(
        manifest_table,
        manifest_path,
        index=False,
    )

    # Maintain the traditional fixed Technical_Record paths as convenient
    # mirrors. Saved loading uses the generation path recorded in manifest.
    summary_path = (
        technical_dir
        / "instance_refinement_summary.csv"
    )

    events_path = (
        technical_dir
        / "instance_refinement_events.csv"
    )

    _resume_dataframe_atomic(
        summary_table,
        summary_path,
        index=False,
    )

    _resume_dataframe_atomic(
        events_table,
        events_path,
        index=False,
    )

    reconnect_count = int(
        (
            successful_events_table[
                "operation"
            ]
            == "reconnect"
        ).sum()
    )

    split_parent_count = int(
        (
            successful_events_table[
                "operation"
            ]
            == "split"
        ).sum()
    )

    removed_small_count = int(
        (
            successful_events_table[
                "operation"
            ]
            == "remove_small"
        ).sum()
    )

    removed_large_count = int(
        (
            successful_events_table[
                "operation"
            ]
            == "remove_large"
        ).sum()
    )

    removed_tubular_count = int(
        (
            successful_events_table[
                "operation"
            ]
            == "remove_tubular"
        ).sum()
    )

    if (
        reconnect_count == 0
        and split_parent_count == 0
        and total_cyan_annotations != 0
    ):
        raise RuntimeError(
            "Cyan QC annotations exist without a successful reconnect or split."
        )

    print(
        f"Detected objects:    "
        f"{len(initial_table)}"
    )
    print(
        f"Suspicious small:    "
        f"{int(initial_table['suspicious_small'].sum())}"
    )
    print(
        f"Suspicious large:    "
        f"{int(initial_table['suspicious_large'].sum())}"
    )
    print(
        f"Tubular candidates:  "
        f"{int(initial_table['tubular_candidate'].sum())}"
    )
    print(
        f"Suspicious tubular:  "
        f"{int(initial_table['suspicious_tubular'].sum())}"
    )
    print()
    print(
        f"Reconnected:         "
        f"{reconnect_count}"
    )
    print(
        f"Split parents:       "
        f"{split_parent_count}"
    )
    print(
        f"Small removed:       "
        f"{removed_small_count}"
    )
    print(
        f"Large removed:       "
        f"{removed_large_count}"
    )
    print(
        f"Tubular removed:     "
        f"{removed_tubular_count}"
    )
    print(
        f"Modified images:     "
        f"{len(modified_paths)}"
    )
    print(
        f"Unchanged sources:   "
        f"{len(source_paths) - len(modified_paths)}"
    )
    print()
    print(
        f"Postprocessed maps:  "
        f"{modified_dir}"
    )
    print(
        f"Postprocessed QC:    "
        f"{modified_qc_dir}"
    )
    print()

    if (
        legacy_root.exists()
        and legacy_root.resolve() != root.resolve()
    ):
        _remove_stage_tree(
            legacy_root
        )
    print("=" * 72)
    print("INSTANCE POSTPROCESSING COMPLETE")
    print("=" * 72)
    print()

    effective_parents = {
        path.parent.resolve()
        for path in effective_paths
    }

    effective_dir = (
        next(
            iter(
                effective_parents
            )
        )
        if len(
            effective_parents
        ) == 1
        else None
    )

    return InstanceRefinementResult(
        effective_instance_paths=(
            effective_paths
        ),
        effective_instance_dir=(
            effective_dir
        ),
        effective_instance_paths_by_image=(
            effective_paths_by_image
        ),
        modified_instance_paths=(
            modified_paths
        ),
        modified_count=len(
            modified_paths
        ),
        reconnect_count=(
            reconnect_count
        ),
        split_parent_count=(
            split_parent_count
        ),
        removed_small_count=(
            removed_small_count
        ),
        removed_large_count=(
            removed_large_count
        ),
        removed_tubular_count=(
            removed_tubular_count
        ),
        manifest_path=(
            manifest_path
        ),
        reused=False,
    )
