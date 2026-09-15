from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class FluorescenceV1Config:
    """
    Parameters for MorphoGlia's first raw-image preprocessing preset.

    Sequence
    --------
    MULTISCALE_TOPHAT
    -> CLAHE
    -> UNSHARP_MASK
    -> NORMALIZE
    -> PERCENTILE_THRESHOLD
    -> REMOVE_SMALL_NOISE

    Parameters left as None use the preset defaults.

    This allows MorphoGlia to distinguish:
        default value
    from:
        value explicitly supplied by the user
    """

    top_hat_radii: tuple[int, ...] | None = None

    clahe_clip_limit: float | None = None
    clahe_tile_grid_size: tuple[int, int] | None = None

    unsharp_radius: float | None = None
    unsharp_amount: float | None = None

    percentile: float | None = None

    min_area: int | None = None

    # Preset sequences are defined in preprocessing/presets.py.

    DEFAULT_TOP_HAT_RADII: ClassVar[tuple[int, ...]] = (
        6,
        15,
        35,
    )

    DEFAULT_CLAHE_CLIP_LIMIT: ClassVar[float] = 0.01

    DEFAULT_CLAHE_TILE_GRID_SIZE: ClassVar[tuple[int, int]] = (
        8,
        8,
    )

    DEFAULT_UNSHARP_RADIUS: ClassVar[float] = 1.0
    DEFAULT_UNSHARP_AMOUNT: ClassVar[float] = 1.0

    DEFAULT_PERCENTILE: ClassVar[float] = 96.6

    DEFAULT_MIN_AREA: ClassVar[int] = 250

    @property
    def effective_top_hat_radii(self) -> tuple[int, ...]:
        if self.top_hat_radii is None:
            return self.DEFAULT_TOP_HAT_RADII

        return tuple(self.top_hat_radii)

    @property
    def effective_clahe_clip_limit(self) -> float:
        if self.clahe_clip_limit is None:
            return self.DEFAULT_CLAHE_CLIP_LIMIT

        return float(self.clahe_clip_limit)

    @property
    def effective_clahe_tile_grid_size(self) -> tuple[int, int]:
        if self.clahe_tile_grid_size is None:
            return self.DEFAULT_CLAHE_TILE_GRID_SIZE

        return tuple(self.clahe_tile_grid_size)

    @property
    def effective_unsharp_radius(self) -> float:
        if self.unsharp_radius is None:
            return self.DEFAULT_UNSHARP_RADIUS

        return float(self.unsharp_radius)

    @property
    def effective_unsharp_amount(self) -> float:
        if self.unsharp_amount is None:
            return self.DEFAULT_UNSHARP_AMOUNT

        return float(self.unsharp_amount)

    @property
    def effective_percentile(self) -> float:
        if self.percentile is None:
            return self.DEFAULT_PERCENTILE

        return float(self.percentile)

    @property
    def effective_min_area(self) -> int:
        if self.min_area is None:
            return self.DEFAULT_MIN_AREA

        return int(self.min_area)

    @staticmethod
    def _source(value: object) -> str:
        if value is None:
            return "default"

        return "user"

    def configuration_rows(self) -> list[dict[str, object]]:
        """
        Return the effective parameters actually used by this preset.
        """

        return [
            {
                "section": "Preprocessing.FluorescenceV1",
                "parameter": "top_hat_radii",
                "value": ",".join(
                    str(value)
                    for value in self.effective_top_hat_radii
                ),
                "source": self._source(self.top_hat_radii),
            },
            {
                "section": "Preprocessing.FluorescenceV1",
                "parameter": "clahe_clip_limit",
                "value": self.effective_clahe_clip_limit,
                "source": self._source(self.clahe_clip_limit),
            },
            {
                "section": "Preprocessing.FluorescenceV1",
                "parameter": "clahe_tile_grid_size",
                "value": ",".join(
                    str(value)
                    for value in self.effective_clahe_tile_grid_size
                ),
                "source": self._source(
                    self.clahe_tile_grid_size
                ),
            },
            {
                "section": "Preprocessing.FluorescenceV1",
                "parameter": "unsharp_radius",
                "value": self.effective_unsharp_radius,
                "source": self._source(self.unsharp_radius),
            },
            {
                "section": "Preprocessing.FluorescenceV1",
                "parameter": "unsharp_amount",
                "value": self.effective_unsharp_amount,
                "source": self._source(self.unsharp_amount),
            },
            {
                "section": "Preprocessing.FluorescenceV1",
                "parameter": "percentile",
                "value": self.effective_percentile,
                "source": self._source(self.percentile),
            },
            {
                "section": "Preprocessing.FluorescenceV1",
                "parameter": "min_area",
                "value": self.effective_min_area,
                "source": self._source(self.min_area),
            },
        ]


@dataclass
class PreprocessingConfig:
    """
    Configuration for the MorphoGlia Preprocessing stage.

    input_mode
    ----------
    None:
        Uses the conservative default: binary.

    "binary":
        The researcher already provides binary images.
        MorphoGlia performs no filtering.
        Optional inversion can be applied.

    "raw":
        MorphoGlia applies a preprocessing preset and produces the
        canonical binary images.

    invert
    ------
    Used for binary-input mode.

    save_intermediate
    -----------------
    Used for raw-input mode.

    False by default. The final binary image is always produced.
    """

    # Biological image geometry; independent of ndarray/storage rank.
    spatial_dimension: str | None = None

    input_mode: str | None = None

    invert: bool | None = None

    preset: str | None = None

    save_intermediate: bool | None = None

    fluorescence_v1: FluorescenceV1Config = field(
        default_factory=FluorescenceV1Config
    )

    DEFAULT_SPATIAL_DIMENSION: ClassVar[str] = "2d"
    DEFAULT_INPUT_MODE: ClassVar[str] = "binary"
    DEFAULT_PRESET: ClassVar[str] = "mg1"
    DEFAULT_INVERT: ClassVar[bool] = False
    DEFAULT_SAVE_INTERMEDIATE: ClassVar[bool] = False

    @property
    def effective_spatial_dimension(self) -> str:
        dimension = (
            self.DEFAULT_SPATIAL_DIMENSION
            if self.spatial_dimension is None
            else str(self.spatial_dimension).strip().lower()
        )

        aliases = {
            "2": "2d",
            "2-d": "2d",
            "2d": "2d",
            "3": "3d",
            "3-d": "3d",
            "3d": "3d",
        }

        if dimension not in aliases:
            raise ValueError(
                "Preprocessing spatial_dimension must be '2d' or '3d'."
            )

        return aliases[dimension]


    @property
    def effective_input_mode(self) -> str:
        mode = (
            self.DEFAULT_INPUT_MODE
            if self.input_mode is None
            else self.input_mode.lower()
        )

        if mode not in {
            "binary",
            "labels",
            "raw",
        }:
            raise ValueError(
                "Preprocessing input_mode must be "
                "'binary', 'labels', or 'raw'."
            )

        return mode

    @property
    def effective_invert(self) -> bool:
        if self.invert is None:
            return self.DEFAULT_INVERT

        return bool(self.invert)

    @property
    def effective_save_intermediate(self) -> bool:
        if self.save_intermediate is None:
            return self.DEFAULT_SAVE_INTERMEDIATE

        return bool(self.save_intermediate)

    @property
    def effective_preset(self) -> str | None:
        if self.effective_input_mode in {
            "binary",
            "labels",
        }:
            return None

        from .presets import (
            available_presets,
            normalize_preset,
        )

        requested = (
            self.DEFAULT_PRESET
            if self.preset is None
            else str(self.preset).strip()
        )

        preset = normalize_preset(
            requested
        )

        available = available_presets()

        if preset not in available:
            raise ValueError(
                "Unknown raw preprocessing preset "
                f"{preset!r}. Available presets: "
                + ", ".join(available)
                + "."
            )

        return preset

    @property
    def sequence(self) -> tuple[str, ...]:
        """Return the active sequence declared in preprocessing/presets.py."""
        if self.effective_input_mode in {
            "binary",
            "labels",
        }:
            return ()

        from .presets import sequence_for_preset

        return sequence_for_preset(
            self.effective_preset
        )

    def configuration_rows(self) -> list[dict[str, object]]:
        rows = list(
            self._configuration_rows_without_spatial_dimension()
        )

        geometry_row = {
            "section": "Preprocessing",
            "parameter": "spatial_dimension",
            "value": self.effective_spatial_dimension,
            "source": (
                "default"
                if self.spatial_dimension is None
                else "user"
            ),
        }

        rows = [
            row
            for row in rows
            if not (
                str(row.get("section", "")) == "Preprocessing"
                and str(row.get("parameter", "")) == "spatial_dimension"
            )
        ]

        return [geometry_row] + rows


    def _configuration_rows_without_spatial_dimension(self) -> list[dict[str, object]]:
        """
        Return only configuration that is relevant to the active mode.
        """

        mode = self.effective_input_mode

        rows: list[dict[str, object]] = [
            {
                "section": "Preprocessing",
                "parameter": "input_mode",
                "value": mode,
                "source": (
                    "default"
                    if self.input_mode is None
                    else "user"
                ),
            }
        ]

        if mode == "labels":
            return rows

        if mode == "binary":

            rows.append(
                {
                    "section": "Preprocessing",
                    "parameter": "invert",
                    "value": self.effective_invert,
                    "source": (
                        "default"
                        if self.invert is None
                        else "user"
                    ),
                }
            )

            return rows

        rows.extend(
            [
                {
                    "section": "Preprocessing",
                    "parameter": "preset",
                    "value": self.effective_preset,
                    "source": (
                        "default"
                        if self.preset is None
                        else "user"
                    ),
                },
                {
                    "section": "Preprocessing",
                    "parameter": "save_intermediate",
                    "value": self.effective_save_intermediate,
                    "source": (
                        "default"
                        if self.save_intermediate is None
                        else "user"
                    ),
                },
            ]
        )

        from .processor import active_parameter_rows

        rows.extend(
            active_parameter_rows(self)
        )

        return rows
