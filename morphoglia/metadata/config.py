from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class NomenclatureConfig:
    """
    Configuration describing how raw filenames are interpreted.

    MorphoGlia canonical filenames follow:

        key-value_key-value_key-value.ext

    The researcher defines:
        1. the separator used in the raw filenames
        2. which raw position corresponds to each metadata field

    Raw positions are 1-based.

    Fields left as None are not used in the experiment and are omitted
    from canonical filenames and downstream metadata tables.

    Raw filename positions not assigned to a metadata field are ignored
    automatically.

    ``ignored_positions`` remains available for explicit documentation
    and backwards compatibility, but it is not required.
    """

    # ------------------------------------------------------------------
    # Raw filename structure
    # ------------------------------------------------------------------

    source_separator: str | None = None

    ignored_positions: set[int] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Canonical MorphoGlia metadata vocabulary
    #
    # Fixed macro -> micro order.
    # ------------------------------------------------------------------

    subject: int | None = None

    genotype: int | None = None
    sex: int | None = None
    cell_type: int | None = None
    condition: int | None = None
    treatment: int | None = None

    region: int | None = None
    hemisphere: int | None = None
    eye: int | None = None
    tissue: int | None = None
    layer: int | None = None
    quadrant: int | None = None
    sample: int | None = None
    section: int | None = None
    spatial_bin: int | None = None

    replicate: int | None = None
    session: int | None = None
    timepoint: int | None = None

    date: int | None = None
    stain: int | None = None
    objective: int | None = None
    channel: int | None = None
    acquisition: int | None = None
    run: int | None = None

    # ------------------------------------------------------------------
    # Experiment-specific extensions
    # ------------------------------------------------------------------

    custom_fields: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Fixed MorphoGlia ordering
    # ------------------------------------------------------------------

    CANONICAL_FIELDS: ClassVar[tuple[str, ...]] = (
        "subject",
        "genotype",
        "sex",
        "cell_type",
        "condition",
        "treatment",
        "region",
        "hemisphere",
        "eye",
        "tissue",
        "layer",
        "quadrant",
        "sample",
        "section",
        "spatial_bin",
        "replicate",
        "session",
        "timepoint",
        "date",
        "stain",
        "objective",
        "channel",
        "acquisition",
        "run",
    )

    # ------------------------------------------------------------------
    # Canonical filename keys
    # ------------------------------------------------------------------

    CANONICAL_KEYS: ClassVar[dict[str, str]] = {
        "subject": "subject",
        "genotype": "genotype",
        "sex": "sex",
        "cell_type": "celltype",
        "condition": "condition",
        "treatment": "treatment",
        "region": "region",
        "hemisphere": "hemisphere",
        "eye": "eye",
        "tissue": "tissue",
        "layer": "layer",
        "quadrant": "quadrant",
        "sample": "sample",
        "section": "section",
        "spatial_bin": "spatialbin",
        "replicate": "replicate",
        "session": "session",
        "timepoint": "timepoint",
        "date": "date",
        "stain": "stain",
        "objective": "objective",
        "channel": "channel",
        "acquisition": "acquisition",
        "run": "run",
    }

    def active_fields(self) -> dict[str, int]:
        """
        Return only metadata fields used in this experiment.

        Standard fields follow MorphoGlia canonical order.
        Custom fields are appended afterwards.
        """

        active: dict[str, int] = {}

        for key in self.CANONICAL_FIELDS:
            position = getattr(self, key)

            if position is not None:
                active[key] = position

        for key, position in self.custom_fields.items():
            active[key] = position

        return active

    def canonical_key(self, field_name: str) -> str:
        """
        Return the filename-safe canonical key for a metadata field.
        """

        if field_name in self.CANONICAL_KEYS:
            return self.CANONICAL_KEYS[field_name]

        canonical = "".join(
            character.lower()
            for character in field_name
            if character.isalnum()
        )

        if not canonical:
            raise ValueError(
                f"Custom metadata field '{field_name}' does not produce "
                "a valid canonical filename key."
            )

        return canonical


@dataclass
class MetadataConfig:
    """
    Configuration for the Metadata stage.
    """

    microns_per_pixel: float | None = None

    nomenclature: NomenclatureConfig = field(
        default_factory=NomenclatureConfig
    )

    @property
    def effective_microns_per_pixel(self) -> float:
        if self.microns_per_pixel is None:
            return 1.0

        return float(self.microns_per_pixel)

    @property
    def calibration_source(self) -> str:
        if self.microns_per_pixel is None:
            return "default"

        return "user"

    def configuration_rows(self) -> list[dict[str, object]]:
        """
        Return Metadata settings in the format expected by
        Technical_Record/configuration.csv.

        All standard nomenclature fields are recorded here, including
        fields not used in the experiment.

        Unused fields are recorded for traceability but are not created
        as columns in downstream biological tables.
        """

        rows: list[dict[str, object]] = []

        # Pixel calibration
        rows.append(
            {
                "section": "Metadata",
                "parameter": "microns_per_pixel",
                "value": self.effective_microns_per_pixel,
                "source": self.calibration_source,
            }
        )

        # Input separator
        rows.append(
            {
                "section": "Metadata.Nomenclature",
                "parameter": "source_separator",
                "value": self.nomenclature.source_separator,
                "source": (
                    "user"
                    if self.nomenclature.source_separator is not None
                    else "default"
                ),
            }
        )

        # Standard MorphoGlia vocabulary
        for field_name in self.nomenclature.CANONICAL_FIELDS:

            position = getattr(
                self.nomenclature,
                field_name,
            )

            rows.append(
                {
                    "section": "Metadata.Nomenclature",
                    "parameter": f"{field_name}_position",
                    "value": position,
                    "source": (
                        "user"
                        if position is not None
                        else "default"
                    ),
                }
            )

        # Explicitly ignored raw positions
        ignored = ",".join(
            str(position)
            for position in sorted(
                self.nomenclature.ignored_positions
            )
        )

        rows.append(
            {
                "section": "Metadata.Nomenclature",
                "parameter": "ignored_positions",
                "value": ignored,
                "source": (
                    "user"
                    if self.nomenclature.ignored_positions
                    else "default"
                ),
            }
        )

        # Custom metadata fields
        custom = ";".join(
            f"{field_name}={position}"
            for field_name, position
            in self.nomenclature.custom_fields.items()
        )

        rows.append(
            {
                "section": "Metadata.Nomenclature",
                "parameter": "custom_fields",
                "value": custom,
                "source": (
                    "user"
                    if self.nomenclature.custom_fields
                    else "default"
                ),
            }
        )

        return rows
