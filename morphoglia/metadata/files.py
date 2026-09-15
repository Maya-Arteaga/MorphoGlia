from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import NomenclatureConfig


@dataclass
class FilenameInterpretation:
    """
    Result of interpreting one raw filename.
    """

    original_filename: str
    status: str
    metadata: dict[str, str]
    canonical_filename: str | None
    issue: str | None = None


def _validate_config(
    config: NomenclatureConfig,
) -> None:
    """
    Validate the declared nomenclature structure itself.

    Only explicitly mapped metadata positions are required.

    Raw filename positions that are not mapped to metadata are ignored
    automatically. ``ignored_positions`` remains supported for explicit
    documentation and backwards compatibility, but is not required.

    Configuration problems are global problems and stop Metadata.
    """

    if config.source_separator is None:

        raise ValueError(
            "No source separator has been configured."
        )


    if config.source_separator == "":

        raise ValueError(
            "The source separator cannot be empty."
        )


    active_fields = config.active_fields()


    if not active_fields:

        raise ValueError(
            "No nomenclature fields have been configured."
        )


    # ------------------------------------------------------------------
    # Validate mapped positions.
    # ------------------------------------------------------------------

    position_to_field: dict[
        int,
        str,
    ] = {}


    for (
        field_name,
        position,
    ) in active_fields.items():

        if not isinstance(
            position,
            int,
        ):

            raise ValueError(
                f"Position for '{field_name}' "
                "must be an integer."
            )


        if position < 1:

            raise ValueError(
                f"Invalid position for "
                f"'{field_name}': {position}. "
                "Positions must start at 1."
            )


        if (
            position
            in position_to_field
        ):

            other_field = (
                position_to_field[
                    position
                ]
            )

            raise ValueError(
                f"Raw position {position} "
                "is assigned to both "
                f"'{other_field}' and "
                f"'{field_name}'."
            )


        position_to_field[
            position
        ] = field_name


    # ------------------------------------------------------------------
    # Explicit ignored positions remain valid but optional.
    #
    # They do NOT define filename length and are NOT required to fill
    # gaps between mapped positions.
    # ------------------------------------------------------------------

    for position in config.ignored_positions:

        if not isinstance(
            position,
            int,
        ):

            raise ValueError(
                "Ignored positions must be integers."
            )


        if position < 1:

            raise ValueError(
                f"Invalid ignored position: "
                f"{position}. "
                "Positions must start at 1."
            )


        if (
            position
            in position_to_field
        ):

            field_name = (
                position_to_field[
                    position
                ]
            )

            raise ValueError(
                f"Raw position {position} "
                "is assigned to "
                f"'{field_name}' and also "
                "marked as ignored."
            )


    # ------------------------------------------------------------------
    # Canonical filename keys must remain unique.
    # ------------------------------------------------------------------

    canonical_keys: dict[
        str,
        str,
    ] = {}


    for field_name in active_fields:

        canonical_key = (
            config.canonical_key(
                field_name
            )
        )


        if (
            canonical_key
            in canonical_keys
        ):

            other_field = (
                canonical_keys[
                    canonical_key
                ]
            )

            raise ValueError(
                "Metadata fields "
                f"'{other_field}' and "
                f"'{field_name}' both produce "
                "canonical key "
                f"'{canonical_key}'."
            )


        canonical_keys[
            canonical_key
        ] = field_name


def interpret_filename(
    filename: str | Path,
    config: NomenclatureConfig,
) -> tuple[
    dict[str, str],
    str,
]:
    """
    Interpret one raw filename using the declared MorphoGlia nomenclature.

    Returns
    -------
    metadata
        Dictionary containing only metadata fields used in this
        experiment.

    canonical_filename
        MorphoGlia-standard filename using:

            key-value_key-value_key-value.ext

    Notes
    -----
    - The raw file is never modified.
    - Raw positions are 1-based.
    - Only mapped positions are extracted.
    - Unmapped raw positions are ignored automatically.
    - Fields configured as None are unused.
    - Extra raw positions are allowed and ignored when unmapped.
    - A mapped position missing from a particular filename is an error.
    - Standard fields use MorphoGlia's fixed macro-to-micro order.
    - Custom fields are appended after standard fields.
    """

    _validate_config(
        config
    )


    path = Path(
        filename
    )

    stem = path.stem

    extension = path.suffix


    parts = stem.split(
        config.source_separator
    )


    active_fields = (
        config.active_fields()
    )


    # ------------------------------------------------------------------
    # Every explicitly requested position must exist in THIS filename.
    #
    # Unmapped positions do not matter.
    # ------------------------------------------------------------------

    for (
        field_name,
        position,
    ) in active_fields.items():

        if position > len(
            parts
        ):

            raise ValueError(
                f"Missing '{field_name}': "
                "filename contains only "
                f"{len(parts)} raw positions "
                f"but position {position} "
                "is required."
            )


    # ------------------------------------------------------------------
    # Extract mapped metadata only.
    # ------------------------------------------------------------------

    metadata: dict[
        str,
        str,
    ] = {}


    for (
        field_name,
        position,
    ) in active_fields.items():

        value = (
            parts[
                position - 1
            ]
            .strip()
        )


        if not value:

            raise ValueError(
                "Missing value for "
                f"'{field_name}' at raw "
                f"position {position}."
            )


        metadata[
            field_name
        ] = value


    # ------------------------------------------------------------------
    # Build canonical MorphoGlia filename.
    #
    # Unmapped raw tokens disappear from the canonical identity.
    # ------------------------------------------------------------------

    canonical_stem = "_".join(
        (
            f"{config.canonical_key(field_name)}"
            f"-{value}"
        )

        for (
            field_name,
            value,
        ) in metadata.items()
    )


    canonical_filename = (
        canonical_stem
        + extension
    )


    return (
        metadata,
        canonical_filename,
    )


def validate_filename(
    filename: str | Path,
    config: NomenclatureConfig,
) -> FilenameInterpretation:
    """
    Validate one raw filename.

    Individual filename problems become:
        status = "other"

    Global configuration problems are allowed to raise and should be
    corrected by the researcher before Metadata continues.
    """

    # Validate outside the try block deliberately.
    #
    # If the researcher configured the nomenclature incorrectly,
    # Metadata should stop rather than classify every file as "other".
    _validate_config(config)

    path = Path(filename)

    try:
        metadata, canonical_filename = interpret_filename(
            path,
            config,
        )

        return FilenameInterpretation(
            original_filename=path.name,
            status="recognized",
            metadata=metadata,
            canonical_filename=canonical_filename,
            issue=None,
        )

    except ValueError as exc:
        return FilenameInterpretation(
            original_filename=path.name,
            status="other",
            metadata={},
            canonical_filename=None,
            issue=str(exc),
        )


# ======================================================================
# Dataset-level nomenclature scanning
# ======================================================================

from collections import Counter


IMAGE_EXTENSIONS = (
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
)


@dataclass
class NomenclatureScan:
    """
    Result of validating all image filenames in one input directory.

    No image pixel data are loaded.
    """

    input_dir: Path
    interpretations: list[FilenameInterpretation]
    field_order: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.interpretations)

    @property
    def recognized(self) -> list[FilenameInterpretation]:
        return [
            result
            for result in self.interpretations
            if result.status == "recognized"
        ]

    @property
    def others(self) -> list[FilenameInterpretation]:
        return [
            result
            for result in self.interpretations
            if result.status == "other"
        ]

    @property
    def recognized_count(self) -> int:
        return len(self.recognized)

    @property
    def others_count(self) -> int:
        return len(self.others)

    def value_counts(
        self,
    ) -> dict[str, dict[str, int]]:
        """
        Count observed values for every active metadata field.

        Only recognized files contribute to these counts.
        """
        counts: dict[str, Counter] = {
            field_name: Counter()
            for field_name in self.field_order
        }

        for result in self.recognized:
            for field_name in self.field_order:
                value = result.metadata.get(field_name)

                if value is not None:
                    counts[field_name][value] += 1

        return {
            field_name: dict(counter)
            for field_name, counter in counts.items()
        }


def discover_image_paths(
    input_dir: str | Path,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
) -> list[Path]:
    """
    Discover supported image files directly inside an input directory.

    This reads directory entries only. Image contents are never loaded.
    """

    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir}"
        )

    normalized_extensions = {
        extension.lower()
        for extension in extensions
    }

    paths = [
        path
        for path in input_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in normalized_extensions
        )
    ]

    return sorted(
        paths,
        key=lambda path: path.name.lower(),
    )


def scan_nomenclature(
    input_dir: str | Path,
    config: NomenclatureConfig,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
) -> NomenclatureScan:
    """
    Validate all supported image filenames in an input directory.

    The scan:

        - does not load image pixels
        - does not rename raw files
        - does not modify the input directory
        - automatically ignores unmapped raw filename positions
        - rejects canonical identity collisions before preprocessing
    """

    # Global configuration problems should stop immediately.

    _validate_config(
        config
    )


    input_dir = Path(
        input_dir
    )


    image_paths = (
        discover_image_paths(
            input_dir,
            extensions=extensions,
        )
    )


    interpretations = [
        validate_filename(
            path,
            config,
        )

        for path in image_paths
    ]


    # ------------------------------------------------------------------
    # Canonical identity must remain one-to-one.
    #
    # Extensions are deliberately ignored in this comparison because
    # preprocessing can normalize different source formats to TIFF.
    # ------------------------------------------------------------------

    canonical_to_raw: dict[
        str,
        str,
    ] = {}


    collisions: list[
        tuple[
            str,
            str,
            str,
        ]
    ] = []


    for result in interpretations:

        if (
            result.status
            != "recognized"
        ):

            continue


        if (
            result.canonical_filename
            is None
        ):

            continue


        canonical_stem = (
            Path(
                result.canonical_filename
            )
            .stem
        )


        previous = (
            canonical_to_raw.get(
                canonical_stem
            )
        )


        if previous is not None:

            collisions.append(
                (
                    canonical_stem,
                    previous,
                    result.original_filename,
                )
            )

        else:

            canonical_to_raw[
                canonical_stem
            ] = (
                result.original_filename
            )


    if collisions:

        lines = [
            (
                "Multiple raw files collapse to the same "
                "canonical MorphoGlia image identity."
            ),
            (
                "This usually means that a filename position "
                "that distinguishes images was left unmapped."
            ),
            "",
        ]


        for (
            canonical_stem,
            first,
            second,
        ) in collisions[:10]:

            lines.extend(
                [
                    f"Canonical identity: {canonical_stem}",
                    f"  - {first}",
                    f"  - {second}",
                    "",
                ]
            )


        if len(
            collisions
        ) > 10:

            lines.append(
                "Additional collisions: "
                f"{len(collisions) - 10}"
            )


        raise ValueError(
            "\n".join(
                lines
            )
        )


    return NomenclatureScan(
        input_dir=input_dir,
        interpretations=interpretations,
        field_order=tuple(
            config.active_fields().keys()
        ),
    )


def print_nomenclature_summary(
    scan: NomenclatureScan,
    preview_limit: int = 10,
) -> None:
    """
    Print a clear Metadata nomenclature summary for the researcher.
    """

    separator = "=" * 72

    print()
    print(separator)
    print("METADATA — NOMENCLATURE")
    print(separator)
    print()

    print(f"Input directory:   {scan.input_dir}")
    print(f"Images found:      {scan.total}")
    print(f"Recognized:        {scan.recognized_count}")
    print(f"Others:            {scan.others_count}")

    # ------------------------------------------------------------------
    # Observed metadata values
    # ------------------------------------------------------------------

    print()
    print("Recognized metadata")
    print("-" * 72)
    print()

    counts = scan.value_counts()

    if scan.recognized_count == 0:
        print("No filenames were recognized.")

    else:
        for field_name in scan.field_order:

            print(f"{field_name}:")

            field_counts = counts.get(
                field_name,
                {},
            )

            for value, count in sorted(
                field_counts.items(),
                key=lambda item: item[0].lower(),
            ):
                print(
                    f"  {value}: {count}"
                )

            print()

    # ------------------------------------------------------------------
    # Canonical preview
    # ------------------------------------------------------------------

    print("Canonical preview")
    print("-" * 72)
    print()

    if not scan.recognized:
        print("No canonical filenames available.")

    else:
        for result in scan.recognized[:preview_limit]:
            print(
                f"{result.original_filename}"
            )
            print(
                f"  -> {result.canonical_filename}"
            )

        remaining = (
            scan.recognized_count
            - min(
                preview_limit,
                scan.recognized_count,
            )
        )

        if remaining > 0:
            print()
            print(
                f"... {remaining} additional recognized image(s)"
            )

    # ------------------------------------------------------------------
    # Others
    # ------------------------------------------------------------------

    print()
    print(separator)
    print("OTHERS")
    print(separator)
    print()

    if not scan.others:
        print("No nomenclature problems detected.")

    else:
        for result in scan.others[:preview_limit]:
            print(result.original_filename)
            print(
                f"  Issue: {result.issue}"
            )
            print()

        remaining = (
            scan.others_count
            - min(
                preview_limit,
                scan.others_count,
            )
        )

        if remaining > 0:
            print(
                f"... {remaining} additional file(s) in Others"
            )

    print()
    print("Raw files have not been modified.")
    print(separator)
    print()
