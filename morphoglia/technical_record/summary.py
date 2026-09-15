from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:
    np = None


def _jsonable(value: Any) -> Any:
    """
    Recursively convert common MorphoGlia values into JSON-safe objects.
    """
    if is_dataclass(value):
        return _jsonable(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            return value.tolist()

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def save_analysis_summary(
    output_dir: str | Path,
    summary: dict[str, Any],
) -> Path:
    """
    Save the complete structured analysis record to:

        <output_dir>/Technical_Record/analysis_summary.json
    """
    output_dir = Path(output_dir)

    technical_record_dir = output_dir / "Technical_Record"
    technical_record_dir.mkdir(parents=True, exist_ok=True)

    path = technical_record_dir / "analysis_summary.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            _jsonable(summary),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return path
