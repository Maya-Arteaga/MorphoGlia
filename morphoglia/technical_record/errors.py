from __future__ import annotations

from pathlib import Path
from typing import Any


_SEPARATOR = "=" * 72


def _technical_record_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)

    technical_record_dir = output_dir / "Technical_Record"
    technical_record_dir.mkdir(parents=True, exist_ok=True)

    return technical_record_dir


def _append_entry(
    output_dir: str | Path,
    *,
    level: str,
    stage: str,
    message: str,
    operation: str | None = None,
    exception: Any | None = None,
) -> Path:
    """
    Append one warning or error entry to:

        <output_dir>/Technical_Record/errors.log
    """
    path = _technical_record_dir(output_dir) / "errors.log"

    with path.open("a", encoding="utf-8") as f:
        f.write(f"{_SEPARATOR}\n")
        f.write(f"{level.upper()}\n")
        f.write(f"{_SEPARATOR}\n\n")

        f.write(f"Stage: {stage}\n")

        if operation:
            f.write(f"Operation: {operation}\n")

        f.write(f"Message: {message}\n")

        if exception is not None:
            f.write(
                f"Exception: "
                f"{exception.__class__.__name__}: {exception}\n"
            )

        f.write("\n")

    return path


def log_warning(
    output_dir: str | Path,
    *,
    stage: str,
    message: str,
    operation: str | None = None,
) -> Path:
    """
    Append a warning to errors.log.
    """
    return _append_entry(
        output_dir,
        level="WARNING",
        stage=stage,
        operation=operation,
        message=message,
    )


def log_error(
    output_dir: str | Path,
    *,
    stage: str,
    message: str,
    operation: str | None = None,
    exception: Any | None = None,
) -> Path:
    """
    Append an error to errors.log.
    """
    return _append_entry(
        output_dir,
        level="ERROR",
        stage=stage,
        operation=operation,
        message=message,
        exception=exception,
    )
