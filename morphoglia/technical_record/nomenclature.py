from __future__ import annotations

import csv
from pathlib import Path

from ..metadata.config import NomenclatureConfig


from ..metadata.files import NomenclatureScan


def save_nomenclature_csv(
    output_dir: str | Path,
    scan: NomenclatureScan,
    config: NomenclatureConfig,
) -> Path:
    """
    Save the complete filename interpretation record to:

        <output_dir>/Technical_Record/nomenclature.csv

    The file records:
        - original raw filename
        - canonical MorphoGlia filename
        - recognized / other status
        - parsing issue, when present
        - source separator
        - raw-position mapping
        - explicitly ignored raw positions
        - parsed metadata fields

    Raw image files are never modified.
    """

    output_dir = Path(output_dir)

    technical_record_dir = output_dir / "Technical_Record"
    technical_record_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = technical_record_dir / "nomenclature.csv"

    active_fields = config.active_fields()

    field_positions = ";".join(
        f"{field_name}={position}"
        for field_name, position in active_fields.items()
    )

    ignored_positions = ";".join(
        str(position)
        for position in sorted(config.ignored_positions)
    )

    columns = [
        "original_filename",
        "canonical_filename",
        "status",
        "issue",
        "source_separator",
        "field_positions",
        "ignored_positions",
        *active_fields.keys(),
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for result in scan.interpretations:

            row = {
                "original_filename": result.original_filename,
                "canonical_filename": (
                    result.canonical_filename or ""
                ),
                "status": result.status,
                "issue": result.issue or "",
                "source_separator": (
                    config.source_separator or ""
                ),
                "field_positions": field_positions,
                "ignored_positions": ignored_positions,
            }

            for field_name in active_fields:
                row[field_name] = result.metadata.get(
                    field_name,
                    "",
                )

            writer.writerow(row)

    return path
