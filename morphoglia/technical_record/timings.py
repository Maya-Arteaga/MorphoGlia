from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def format_seconds(seconds: float | int | None) -> str:
    """
    Convert elapsed seconds to HH:MM:SS using whole seconds.

    Examples
    --------
    0       -> 00:00:00
    126     -> 00:02:06
    4472    -> 01:14:32
    None    -> ""
    """
    if seconds is None:
        return ""

    total_seconds = max(0, int(seconds))

    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_timings_csv(
    output_dir: str | Path,
    rows: Iterable[dict],
) -> Path:
    """
    Save timing information to:

        <output_dir>/Technical_Record/timings.csv

    Expected row fields:
        stage
        operation
        ran
        seconds

    The stored CSV converts raw seconds to HH:MM:SS.
    """
    output_dir = Path(output_dir)

    technical_record_dir = output_dir / "Technical_Record"
    technical_record_dir.mkdir(parents=True, exist_ok=True)

    path = technical_record_dir / "timings.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "stage",
            "operation",
            "ran",
            "time",
        ])

        for row in rows:
            writer.writerow([
                row.get("stage", ""),
                row.get("operation", ""),
                row.get("ran", ""),
                format_seconds(row.get("seconds")),
            ])

    return path
