from __future__ import annotations

# MG_RESUME_MORPHOMETRICS_V1
import inspect
import json

from ..checkpoint import (
    CheckpointJournal,
    atomic_write_bytes,
    file_identity,
    fingerprint,
    source_digest,
)

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tifffile
from tqdm.auto import tqdm

from ..core.models import (
    CellRecord,
    DatasetIndex,
    ImageRecord,
)
from ..core.instance_map import (
    validate_instance_map,
)

from ..metadata.files import (
    NomenclatureScan,
)

from ..io.table import (
    MasterTableBuilder,
)

from .builtin import (
    METRIC_DESCRIPTIONS,
)

from .calculators import (
    build_calculators_from_config,
)

from .engine import (
    MorphometricEngine,
)

from ..io.overlays import (
    render_instance_map_diagnostic,
)



if TYPE_CHECKING:
    from ..config import PipelineConfig


# ======================================================================
# RESULT
# ======================================================================

@dataclass
class MorphometricsResult:
    """
    Result of the dataset-level Morphometrics stage.

    master
        Canonical per-cell table containing metadata, object information,
        and requested morphometric outputs.

    dataset
        Final in-memory dataset containing only objects accepted by QC.

    morphometric_columns
        Morphometric outputs actually present in the master table.

    image_count
        Number of canonical instance maps analyzed.

    cell_count
        Number of objects accepted by QC and analyzed morphometrically.

    detected_count
        Number of connected objects detected before Object QC.
    """

    master: pd.DataFrame

    dataset: DatasetIndex

    morphometric_columns: list[str]

    image_count: int

    cell_count: int

    detected_count: int | None = None


# ======================================================================
# METADATA ↔ BINARY IMAGE LINK
# ======================================================================

def _metadata_by_canonical_stem(
    metadata_scan: NomenclatureScan,
) -> dict[str, dict[str, str]]:
    """
    Build the canonical identity map created by Metadata.

    Preprocessing preserves the canonical stem when writing binary TIFFs,
    so Morphometrics can recover metadata without parsing filenames again.
    """

    mapping: dict[
        str,
        dict[str, str],
    ] = {}


    for interpretation in metadata_scan.recognized:

        canonical_filename = (
            interpretation
            .canonical_filename
        )


        if canonical_filename is None:
            continue


        stem = Path(
            canonical_filename
        ).stem


        if stem in mapping:

            raise ValueError(
                "Metadata produced duplicate canonical image identity: "
                f"{stem!r}."
            )


        mapping[
            stem
        ] = dict(
            interpretation.metadata
        )


    return mapping


# ======================================================================
# DATASET CONSTRUCTION FROM CANONICAL INSTANCE MAPS
# ======================================================================

def _load_instance_map(
    path: Path,
) -> np.ndarray:
    """
    Load and validate one canonical instance-map TIFF.

    Morphometrics is currently 2D. The instance-map representation itself
    supports N dimensions, but 3D morphometric calculators are not yet
    implemented.
    """

    instance_map = np.asarray(
        tifffile.imread(
            str(path)
        )
    )

    instance_map = validate_instance_map(
        instance_map
    )

    if instance_map.ndim != 2:

        raise ValueError(
            "Current Morphometrics implementation requires 2D "
            f"instance maps, but {path.name!r} has shape "
            f"{instance_map.shape}."
        )

    return instance_map


def _instance_objects_2d(
    instance_map: np.ndarray,
) -> list[dict]:
    """
    Describe every positive instance identity in a 2D instance map.

    Object identity comes exclusively from map_label.

    Objects are ordered by their lexicographically first occupied
    coordinate (Y, X), independently of label value or connected-
    component library behavior.
    """

    foreground = (
        instance_map > 0
    )

    coordinates = np.argwhere(
        foreground
    )

    if coordinates.size == 0:

        return []


    values = instance_map[
        foreground
    ]


    # Group coordinates by map label while preserving the original
    # row-major coordinate order inside each group.
    order = np.argsort(
        values,
        kind="stable",
    )

    sorted_values = values[
        order
    ]

    sorted_coordinates = coordinates[
        order
    ]


    (
        unique_labels,
        starts,
        counts,
    ) = np.unique(
        sorted_values,
        return_index=True,
        return_counts=True,
    )


    objects: list[dict] = []


    for (
        label_value,
        start,
        count,
    ) in zip(
        unique_labels,
        starts,
        counts,
    ):

        group = sorted_coordinates[
            int(start):
            int(start + count)
        ]


        y_values = group[
            :,
            0,
        ]

        x_values = group[
            :,
            1,
        ]


        y_min = int(
            y_values.min()
        )

        y_max = int(
            y_values.max()
        )

        x_min = int(
            x_values.min()
        )

        x_max = int(
            x_values.max()
        )


        first_y = int(
            group[
                0,
                0,
            ]
        )

        first_x = int(
            group[
                0,
                1,
            ]
        )


        objects.append(
            {
                "map_label": int(
                    label_value
                ),
                "area": int(
                    count
                ),
                "bbox": (
                    x_min,
                    y_min,
                    x_max - x_min + 1,
                    y_max - y_min + 1,
                ),
                "first_coordinate": (
                    first_y,
                    first_x,
                ),
            }
        )


    objects.sort(
        key=lambda item: item[
            "first_coordinate"
        ]
    )


    return objects


def _build_dataset(
    instance_paths: list[Path],
    metadata_scan: NomenclatureScan,
    config: "PipelineConfig",
) -> DatasetIndex:
    """
    Construct ImageRecord / CellRecord objects directly from canonical
    instance maps.

    Morphometrics does not detect connected components.

    The positive integer already stored in the instance map is the
    object's map_label.

    For externally supplied label maps:

        source_label = map_label

    For binary/raw-derived maps:

        source_label = None
    """

    metadata_lookup = (
        _metadata_by_canonical_stem(
            metadata_scan
        )
    )


    source_has_labels = (
        config
        .preprocessing
        .effective_input_mode
        == "labels"
    )


    dataset = DatasetIndex()


    for instance_path in tqdm(
        instance_paths,
        desc="Instance extraction",
        leave=True,
        dynamic_ncols=True,
    ):

        instance_path = Path(
            instance_path
        )


        image_id = (
            instance_path.stem
        )


        if image_id not in metadata_lookup:

            raise ValueError(
                "Preprocessed instance map has no matching Metadata "
                f"identity: {instance_path.name!r}."
            )


        instance_map = (
            _load_instance_map(
                instance_path
            )
        )


        image_record = ImageRecord(
            image_id=image_id,
            path=instance_path,
            metadata=(
                metadata_lookup[
                    image_id
                ]
                .copy()
            ),
            size=(
                int(
                    instance_map.shape[
                        0
                    ]
                ),
                int(
                    instance_map.shape[
                        1
                    ]
                ),
            ),
        )


        dataset.add_image(
            image_record
        )


        objects = (
            _instance_objects_2d(
                instance_map
            )
        )


        cell_records: list[
            CellRecord
        ] = []


        for index, object_info in enumerate(
            objects,
            start=1,
        ):

            (
                x,
                y,
                w,
                h,
            ) = object_info[
                "bbox"
            ]


            map_label = int(
                object_info[
                    "map_label"
                ]
            )


            cell = CellRecord(
                cell_id=(
                    f"{image_id}_cell_{index}"
                ),
                image_id=image_id,
                bbox=(
                    x,
                    y,
                    w,
                    h,
                ),
                center=(
                    x + w // 2,
                    y + h // 2,
                ),
                metadata=(
                    image_record
                    .metadata
                    .copy()
                ),
                map_label=map_label,
                source_label=(
                    map_label
                    if source_has_labels
                    else None
                ),
                component_area=int(
                    object_info[
                        "area"
                    ]
                ),
            )


            cell_records.append(
                cell
            )


        dataset.add_cells(
            cell_records
        )


    return dataset


# ======================================================================
# EXACT ROI TABLE FOR OBJECT QC
# ======================================================================

def _build_object_qc_table(
    dataset: DatasetIndex,
) -> tuple[
    pd.DataFrame,
    dict[
        tuple[str, str],
        np.ndarray,
    ],
]:
    """
    Build the Object-QC table directly from canonical instance maps.

    Exact ROI identity is:

        instance_map == cell.map_label

    No thresholding, polarity inference, anchor recovery, largest-
    component selection, or connected-component reconstruction occurs.
    """

    by_image: dict[
        str,
        list,
    ] = {}


    for cell in dataset.iter_cells():

        by_image.setdefault(
            cell.image_id,
            [],
        ).append(
            cell
        )


    records: list[
        dict
    ] = []


    roi_lookup: dict[
        tuple[str, str],
        np.ndarray,
    ] = {}


    for (
        image_id,
        cells,
    ) in tqdm(
        by_image.items(),
        total=len(
            by_image
        ),
        desc="Object QC ROIs",
        leave=True,
        dynamic_ncols=True,
    ):

        image_record = (
            dataset.get_image(
                image_id
            )
        )


        instance_map = (
            _load_instance_map(
                Path(
                    image_record.path
                )
            )
        )


        for cell in cells:

            if cell.map_label is None:

                raise ValueError(
                    "Object QC requires map_label, but "
                    f"{cell.cell_id!r} has no map label."
                )


            (
                x,
                y,
                w,
                h,
            ) = cell.bbox


            cropped_labels = instance_map[
                y:
                y + h,
                x:
                x + w,
            ]


            roi = np.where(
                cropped_labels
                == int(
                    cell.map_label
                ),
                255,
                0,
            ).astype(
                np.uint8
            )


            measured_area = int(
                np.count_nonzero(
                    roi
                )
            )


            if measured_area == 0:

                raise RuntimeError(
                    "Instance-map ROI contains no pixels for "
                    f"{cell.cell_id!r} "
                    f"(map_label={cell.map_label})."
                )


            component_area = (
                cell.component_area
            )


            if component_area is None:

                raise ValueError(
                    "Object QC requires component_area, but "
                    f"{cell.cell_id!r} has no component area."
                )


            if measured_area != int(
                component_area
            ):

                raise RuntimeError(
                    "Instance-map ROI area does not match "
                    f"CellRecord for {cell.cell_id!r}: "
                    f"{measured_area} versus "
                    f"{component_area}."
                )


            key = (
                str(
                    cell.image_id
                ),
                str(
                    cell.cell_id
                ),
            )


            if key in roi_lookup:

                raise ValueError(
                    "Duplicate object identity while preparing QC: "
                    f"{key!r}."
                )


            roi_lookup[
                key
            ] = roi


            records.append(
                {
                    "image_id": str(
                        cell.image_id
                    ),
                    "cell_id": str(
                        cell.cell_id
                    ),
                    "map_label": int(
                        cell.map_label
                    ),
                    "component_area": int(
                        component_area
                    ),
                    "roi": roi,
                }
            )


    object_table = pd.DataFrame(
        records
    )


    if object_table.empty:

        raise ValueError(
            "Object QC received no instances."
        )


    return (
        object_table,
        roi_lookup,
    )


# ======================================================================
# DATASET FILTERING
# ======================================================================

def _filter_dataset(
    dataset: DatasetIndex,
    included_keys: set[
        tuple[str, str]
    ],
) -> DatasetIndex:
    """
    Construct a DatasetIndex containing only QC-accepted cells.

    ImageRecords are all retained, including images for which no object
    survives QC.
    """

    filtered = DatasetIndex()


    for image_record in (
        dataset.images.values()
    ):

        filtered.add_image(
            image_record
        )


    accepted_cells = []


    for cell in dataset.iter_cells():

        key = (
            str(
                cell.image_id
            ),
            str(
                cell.cell_id
            ),
        )


        if key in included_keys:

            accepted_cells.append(
                cell
            )


    filtered.add_cells(
        accepted_cells
    )


    if len(
        accepted_cells
    ) != len(
        included_keys
    ):

        raise RuntimeError(
            "Object QC accepted-key count does not match "
            "the CellRecords recovered from DatasetIndex: "
            f"{len(included_keys)} keys versus "
            f"{len(accepted_cells)} cells."
        )


    return filtered


# ======================================================================
# MORPHOMETRIC COMPUTATION
# ======================================================================

# ======================================================================
# MORPHOMETRICS RESUME SHARDS
# ======================================================================

_MORPHOMETRICS_SHARD_VERSION = 1


def _feature_json_safe(
    value,
):
    """Convert one raw feature value to a JSON-round-trippable value."""

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
        return _feature_json_safe(
            value.item()
        )


    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            _feature_json_safe(
                item
            )
            for item in value.tolist()
        ]


    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ):
            _feature_json_safe(
                item
            )
            for key, item in value.items()
        }


    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _feature_json_safe(
                item
            )
            for item in value
        ]


    raise TypeError(
        "Morphometrics checkpoint cannot serialize "
        f"feature value of type {type(value).__name__}: {value!r}"
    )


def _morphometrics_code_signature(
    calculators,
) -> str:
    """
    Fingerprint MorphoGlia morphometric code plus any external calculator
    implementation files.

    Only small Python source files are hashed here; microscopy TIFFs are not.
    """

    package_root = Path(
        __file__
    ).resolve().parent


    code_files = sorted(
        path
        for path in package_root.rglob(
            "*.py"
        )
        if path.is_file()
    )


    code_payload = [
        (
            str(
                path.relative_to(
                    package_root
                )
            ),
            source_digest(
                path
            ),
        )
        for path in code_files
    ]


    external_payload = []


    for calculator in calculators:

        calculator_class = (
            calculator.__class__
        )


        entry = {
            "module":
                str(
                    calculator_class.__module__
                ),

            "qualname":
                str(
                    calculator_class.__qualname__
                ),
        }


        try:
            calculator_file = inspect.getsourcefile(
                calculator_class
            )

            if calculator_file:

                calculator_path = Path(
                    calculator_file
                ).resolve()

                if (
                    calculator_path.is_file()
                    and package_root
                    not in calculator_path.parents
                ):

                    entry[
                        "source_digest"
                    ] = source_digest(
                        calculator_path
                    )

        except Exception:
            pass


        external_payload.append(
            entry
        )


    return fingerprint(
        "morphometrics_code_v1",
        code_payload,
        external_payload,
    )


def _feature_shard_path(
    output_dir: Path,
    image_id: str,
) -> Path:

    return (
        Path(
            output_dir
        )
        / "Technical_Record"
        / "Checkpoints"
        / "Morphometrics_Features"
        / f"{image_id}.json"
    )


def _write_feature_shard(
    path: Path,
    *,
    image_id: str,
    cells,
) -> None:
    """
    Atomically persist raw features for exactly one fully completed image.
    """

    payload = {
        "format_version":
            _MORPHOMETRICS_SHARD_VERSION,

        "image_id":
            str(
                image_id
            ),

        "cell_count":
            int(
                len(
                    cells
                )
            ),

        "cells":
            [],
    }


    for cell in cells:

        if cell.raw_features is None:

            raise RuntimeError(
                "Cannot checkpoint Morphometrics before every cell in "
                f"image {image_id!r} has raw_features."
            )


        payload[
            "cells"
        ].append(
            {
                "cell_id":
                    str(
                        cell.cell_id
                    ),

                "features":
                    _feature_json_safe(
                        dict(
                            cell.raw_features
                        )
                    ),
            }
        )


    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=True,
        )
        + "\n"
    ).encode(
        "utf-8"
    )


    atomic_write_bytes(
        path,
        encoded,
    )


def _restore_feature_shard(
    path: Path,
    *,
    image_id: str,
    cells,
) -> None:
    """
    Validate one shard against the current image population and restore
    CellRecord.raw_features in the existing deterministic cell order.
    """

    path = Path(
        path
    )


    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(
                file
            )

    except Exception as exc:

        raise ValueError(
            f"Could not read Morphometrics feature shard: {path}"
        ) from exc


    if not isinstance(
        payload,
        dict,
    ):

        raise ValueError(
            f"Morphometrics feature shard is not a mapping: {path}"
        )


    if int(
        payload.get(
            "format_version",
            -1,
        )
    ) != int(
        _MORPHOMETRICS_SHARD_VERSION
    ):

        raise ValueError(
            f"Incompatible Morphometrics feature shard version: {path}"
        )


    if str(
        payload.get(
            "image_id",
            "",
        )
    ) != str(
        image_id
    ):

        raise ValueError(
            "Morphometrics feature shard image identity mismatch: "
            f"{path}"
        )


    rows = payload.get(
        "cells"
    )


    if not isinstance(
        rows,
        list,
    ):

        raise ValueError(
            f"Morphometrics feature shard has no cell list: {path}"
        )


    expected_ids = [
        str(
            cell.cell_id
        )
        for cell in cells
    ]


    observed_ids = [
        str(
            row.get(
                "cell_id",
                "",
            )
        )
        if isinstance(
            row,
            dict,
        )
        else ""
        for row in rows
    ]


    if (
        int(
            payload.get(
                "cell_count",
                -1,
            )
        )
        != len(
            cells
        )
        or observed_ids
        != expected_ids
    ):

        raise ValueError(
            "Morphometrics feature shard cell population/order does not "
            f"match the current instance map for image {image_id!r}."
        )


    restored = []


    for row in rows:

        features = row.get(
            "features"
        )


        if not isinstance(
            features,
            dict,
        ):

            raise ValueError(
                "Morphometrics feature shard contains a non-mapping "
                f"feature payload for image {image_id!r}."
            )


        restored.append(
            {
                str(
                    key
                ):
                    value
                for key, value in features.items()
            }
        )


    for cell, features in zip(
        cells,
        restored,
    ):

        cell.raw_features = (
            features
        )


def _compute_morphometrics(
    dataset: DatasetIndex,
    config: "PipelineConfig",
    output_dir: Path,
    roi_lookup: dict[
        tuple[str, str],
        np.ndarray,
    ] | None = None,
    instance_paths: list[Path] | None = None,
) -> None:
    """
    Compute configured morphometric families for accepted cells.

    Crash-recovery boundary
    -----------------------
    One checkpoint is committed only after every cell belonging to one image
    has completed all configured morphometric calculators and the persisted
    per-image feature shard has been read back and validated.

    The canonical MasterTableBuilder contract is unchanged: whether features
    are newly computed or restored, they are placed on CellRecord.raw_features.
    """

    calculators = (
        build_calculators_from_config(
            config
            .morphometrics
            .effective_calculators
        )
    )


    engine = MorphometricEngine(
        calculators
    )


    if roi_lookup is None:

        raise ValueError(
            "Morphometrics requires the exact instance ROIs "
            "prepared from the canonical instance maps."
        )


    if instance_paths is None:

        raise ValueError(
            "Morphometrics resume requires the authoritative instance-map "
            "paths for per-image cache validation."
        )


    source_by_id = {
        Path(
            path
        ).stem:
            Path(
                path
            )
        for path in instance_paths
    }


    by_image: dict[
        str,
        list,
    ] = {}


    for cell in dataset.iter_cells():

        by_image.setdefault(
            str(
                cell.image_id
            ),
            [],
        ).append(
            cell
        )


    missing_sources = (
        set(
            by_image
        )
        - set(
            source_by_id
        )
    )


    if missing_sources:

        raise KeyError(
            "Morphometrics cannot resolve authoritative instance maps for "
            f"images: {sorted(missing_sources)}"
        )


    total_cells = sum(
        len(
            cells
        )
        for cells in by_image.values()
    )


    calculator_names = [
        calculator
        .__class__
        .__name__

        for calculator in calculators
    ]


    stage_signature = fingerprint(
        "morphometrics_resume_v1",

        config.morphometrics,

        {
            "microns_per_pixel":
                config.metadata
                .effective_microns_per_pixel,

            "input_mode":
                config.preprocessing
                .effective_input_mode,
        },

        _morphometrics_code_signature(
            calculators
        ),
    )


    journal = CheckpointJournal(
        output_dir=output_dir,
        stage="morphometrics",
        resume=bool(
            config.resume
        ),
        stage_signature=(
            stage_signature
        ),
    )


    bars = {
        name:
            tqdm(
                total=total_cells,
                desc=name,
                leave=True,
                dynamic_ncols=True,
                miniters=1,
                position=index,
            )

        for (
            index,
            name,
        ) in enumerate(
            calculator_names
        )
    }


    overall_bar = tqdm(
        total=total_cells,
        desc="Morphometrics (all)",
        leave=True,
        dynamic_ncols=True,
        miniters=1,
        position=len(
            calculator_names
        ),
    )


    image_total = len(
        by_image
    )


    try:

        for image_index, (
            image_id,
            cells,
        ) in enumerate(
            by_image.items(),
            start=1,
        ):

            source_path = source_by_id[
                image_id
            ]


            shard_path = _feature_shard_path(
                output_dir,
                image_id,
            )


            expected_cell_ids = [
                str(
                    cell.cell_id
                )
                for cell in cells
            ]


            item_signature = fingerprint(
                stage_signature,
                file_identity(
                    source_path
                ),
                expected_cell_ids,
            )


            reusable = journal.reusable_record(
                item_id=(
                    image_id
                ),
                item_signature=(
                    item_signature
                ),
                outputs=[
                    shard_path
                ],
            )


            if reusable is not None:

                try:

                    _restore_feature_shard(
                        shard_path,
                        image_id=(
                            image_id
                        ),
                        cells=(
                            cells
                        ),
                    )

                except Exception:

                    reusable = None


            if reusable is not None:

                print(
                    f"[{image_index}/{image_total}] "
                    f"RESUME {image_id}"
                )


                reused_cell_count = len(
                    cells
                )


                for bar in bars.values():

                    bar.update(
                        reused_cell_count
                    )


                overall_bar.update(
                    reused_cell_count
                )


                continue


            print(
                f"[{image_index}/{image_total}] "
                f"{image_id}"
            )


            for cell in cells:

                key = (
                    str(
                        cell.image_id
                    ),
                    str(
                        cell.cell_id
                    ),
                )


                roi = roi_lookup.get(
                    key
                )


                if roi is None:

                    raise KeyError(
                        "Missing exact instance ROI for "
                        f"{key!r}."
                    )


                features_by_calculator = (
                    engine
                    .compute_per_calc(
                        roi
                    )
                )


                merged: dict = {}


                for (
                    calculator_name,
                    features,
                ) in (
                    features_by_calculator.items()
                ):

                    merged.update(
                        features
                    )


                    if calculator_name in bars:

                        bars[
                            calculator_name
                        ].update(
                            1
                        )


                cell.raw_features = (
                    merged
                )


                overall_bar.update(
                    1
                )


            _write_feature_shard(
                shard_path,
                image_id=(
                    image_id
                ),
                cells=(
                    cells
                ),
            )


            # Re-open the persisted shard before committing it. This means a
            # journal record can only exist for a shard that is syntactically
            # valid and matches the current image/cell population.
            _restore_feature_shard(
                shard_path,
                image_id=(
                    image_id
                ),
                cells=(
                    cells
                ),
            )


            journal.commit(
                item_id=(
                    image_id
                ),
                item_signature=(
                    item_signature
                ),
                outputs=[
                    shard_path
                ],
                metadata={
                    "cell_count":
                        len(
                            cells
                        )
                },
            )


    finally:

        for bar in bars.values():

            bar.close()


        overall_bar.close()


# ======================================================================
# PUBLIC STAGE
# ======================================================================

def run_morphometrics(
    instance_paths: list[Path],
    metadata_scan: NomenclatureScan,
    output_dir: str | Path,
    config: "PipelineConfig",
) -> MorphometricsResult:
    """
    Run canonical dataset-level Morphometrics from the effective instance maps.

    Object detection/refinement is deliberately upstream. Morphometrics never
    decides whether to use original or refined maps and never performs Object
    QC. It receives exactly one authoritative instance-map path per image.
    """

    output_dir = Path(output_dir)
    instance_paths = [Path(path) for path in instance_paths]

    if not instance_paths:
        raise ValueError("Morphometrics received no instance maps.")

    print()
    print("=" * 72)
    print("MORPHOMETRICS")
    print("=" * 72)
    print()

    dataset = _build_dataset(
        instance_paths=instance_paths,
        metadata_scan=metadata_scan,
        config=config,
    )

    image_count = len(dataset.images)
    cell_count = len(dataset.cells)

    print(f"Images:              {image_count}")
    print(f"Objects:             {cell_count}")
    print()

    if cell_count == 0:
        raise ValueError(
            "Morphometrics received instance maps containing no objects."
        )

    # Reuse the exact instance-map ROI extractor. Despite its historical name,
    # it performs no filtering; it only extracts one exact ROI per map label.
    object_table, roi_lookup = _build_object_qc_table(
        dataset=dataset,
    )

    if len(object_table) != cell_count:
        raise RuntimeError(
            "Exact ROI count does not match instance population: "
            f"{len(object_table)} versus {cell_count}."
        )

    print()
    print("=" * 72)
    print("MORPHOMETRIC COMPUTATION")
    print("=" * 72)
    print()

    _compute_morphometrics(
        dataset=dataset,
        config=config,
        output_dir=output_dir,
        roi_lookup=roi_lookup,
        instance_paths=(
            instance_paths
        ),
    )

    del roi_lookup

    master = MasterTableBuilder().build(dataset)

    if len(master) != cell_count:
        raise RuntimeError(
            "Master-table row count does not match instance population: "
            f"{len(master)} versus {cell_count}."
        )

    morphometric_columns = [
        feature
        for feature in METRIC_DESCRIPTIONS
        if feature in master.columns
    ]

    print()
    print("=" * 72)
    print("MORPHOMETRICS COMPLETE")
    print("=" * 72)
    print()
    print(f"Images analyzed:     {image_count}")
    print(f"Objects analyzed:    {cell_count}")
    print(f"Morphometric outputs: {len(morphometric_columns)}")
    print()

    return MorphometricsResult(
        master=master,
        dataset=dataset,
        morphometric_columns=morphometric_columns,
        image_count=image_count,
        cell_count=cell_count,
        detected_count=cell_count,
    )
