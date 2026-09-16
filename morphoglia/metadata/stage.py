from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import MetadataConfig
from .files import (
    NomenclatureScan,
    print_nomenclature_summary,
    scan_nomenclature,
)


from ..technical_record.configuration import save_configuration_csv

from ..technical_record.nomenclature import save_nomenclature_csv



@dataclass
class MetadataResult:
    """
    Result produced by the Metadata stage.

    Raw images are not modified.
    """

    scan: NomenclatureScan
    microns_per_pixel: float
    calibration_source: str
    configuration_csv: Path
    nomenclature_csv: Path


def print_calibration_summary(
    config: MetadataConfig,
) -> None:
    """
    Print the effective spatial calibration used by MorphoGlia.
    """

    separator = "=" * 72

    print()
    print(separator)
    print("METADATA — PIXEL CALIBRATION")
    print(separator)
    print()

    print(
        "Conversion: "
        f"1 pixel = {config.effective_microns_per_pixel} microns"
    )

    #print(
    #    f"Source:     {config.calibration_source}"
    #)

    if config.calibration_source == "default":
        print()
        print(
            "No pixel calibration was provided."
        )
        print(
            "Default value: "
            "1 pixel = 1 micron."
        )

    print()
    #print(separator)
    print()


def run_metadata(
    input_dir: str | Path,
    output_dir: str | Path,
    config: MetadataConfig,
    preview_limit: int = 10,
) -> MetadataResult:
    """
    Run the MorphoGlia Metadata stage.

    The stage:
        1. reports the effective pixel calibration
        2. scans supported image filenames
        3. validates the declared nomenclature
        4. prints recognized files and Others
        5. saves Technical_Record/configuration.csv
        6. saves Technical_Record/nomenclature.csv

    Raw image pixels are never loaded.
    Raw files are never renamed or modified.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Pixel calibration
    # ------------------------------------------------------------------

    print_calibration_summary(config)

    # ------------------------------------------------------------------
    # Nomenclature
    # ------------------------------------------------------------------

    scan = scan_nomenclature(
        input_dir=input_dir,
        config=config.nomenclature,
    )

    print_nomenclature_summary(
        scan,
        preview_limit=preview_limit,
    )

    # ------------------------------------------------------------------
    # Technical Record
    # ------------------------------------------------------------------

    configuration_csv = save_configuration_csv(
        output_dir=output_dir,
        rows=config.configuration_rows(),
    )

    nomenclature_csv = save_nomenclature_csv(
        output_dir=output_dir,
        scan=scan,
        config=config.nomenclature,
    )

    return MetadataResult(
        scan=scan,
        microns_per_pixel=config.effective_microns_per_pixel,
        calibration_source=config.calibration_source,
        configuration_csv=configuration_csv,
        nomenclature_csv=nomenclature_csv,
    )
