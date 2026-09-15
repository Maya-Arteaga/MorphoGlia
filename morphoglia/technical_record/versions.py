from __future__ import annotations

import csv
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


_PACKAGES = (
    "numpy",
    "pandas",
    "cv2",
    "skimage",
    "scipy",
    "sklearn",
    "umap",
    "hdbscan",
    "tqdm",
)


def _safe_version(module_name: str) -> str | None:
    """
    Return the installed version of a Python module.

    First try the module's own version attribute.
    If that is unavailable, fall back to Python package metadata.

    If the module cannot be imported or its version cannot be determined,
    return None instead of failing the analysis.
    """
    try:
        module = __import__(module_name)
    except Exception:
        return None

    module_version = (
        getattr(module, "__version__", None)
        or getattr(module, "version", None)
    )

    if module_version is not None:
        return str(module_version)

    try:
        return version(module_name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def collect_versions() -> dict[str, Any]:
    """
    Collect Python, operating-system, and scientific-package versions.
    """
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    for package in _PACKAGES:
        versions[package] = _safe_version(package)

    return versions


def save_versions_csv(
    output_dir: str | Path,
    versions: dict[str, Any] | None = None,
) -> Path:
    """
    Save software versions to:

        <output_dir>/Technical_Record/versions.csv
    """
    output_dir = Path(output_dir)

    technical_record_dir = output_dir / "Technical_Record"
    technical_record_dir.mkdir(parents=True, exist_ok=True)

    path = technical_record_dir / "versions.csv"

    if versions is None:
        versions = collect_versions()

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["software", "version"])

        for software, software_version in versions.items():
            writer.writerow([
                software,
                "not available" if software_version is None else software_version,
            ])

    return path
