from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Any


def save_configuration_csv(
    output_dir: str | Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    """
    Save the effective configuration used in the analysis to:

        <output_dir>/Technical_Record/configuration.csv

    Expected row fields:
        section
        parameter
        value
        source

    source should currently be:
        - "default"
        - "automatic"
        - "user"
    """
    output_dir = Path(output_dir)

    technical_record_dir = output_dir / "Technical_Record"
    technical_record_dir.mkdir(parents=True, exist_ok=True)

    path = technical_record_dir / "configuration.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "section",
            "parameter",
            "value",
            "source",
        ])

        for row in rows:
            writer.writerow([
                row.get("section", ""),
                row.get("parameter", ""),
                row.get("value", ""),
                row.get("source", ""),
            ])

    return path
