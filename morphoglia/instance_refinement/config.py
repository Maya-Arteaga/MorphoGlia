from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InstanceRefinementConfig:
    """
    Optional refinement of canonical 2D instance maps.

    Public controls
    ---------------
    remove_small
        Remove statistically suspicious small objects after repair/splitting.

    tubular_action
        "keep"      -> detect/report tubular candidates but do not modify them.
        "remove"    -> remove suspicious tubular/orphan objects.
        "reconnect" -> attempt soma-territory-constrained reconnection.

    large_action
        "keep"   -> detect/report suspicious large objects but do not modify them.
        "remove" -> remove suspicious large objects.
        "split"  -> attempt evidence-based multi-soma splitting.

    The remaining values are advanced developer defaults. Normal users should
    not need to tune them.
    """

    remove_small: bool = True

    tubular_action: str = "reconnect"
    large_action: str = "split"

    # Existing robust Object-QC detector coordinates.
    size_z_threshold: float = 3.5
    tubular_candidate_z: float = -1.5
    tubular_score_threshold: float = 0.80
    grid_n: int = 6

    # Reconnection.
    max_gap_px: float = 8.0
    radial_sector_half_width_degrees: float = 30.0
    radial_tolerance_px: float | None = None

    # Soma/split discovery. These are intentionally data-scaled rather than
    # absolute soma-size thresholds.
    split_smoothing_sigma: float = 3.0
    split_peak_radius_fraction: float = 0.65
    split_peak_min_distance_factor: float = 2.5
    split_max_seed_factor: float = 1.5
    split_min_child_area_fraction: float = 0.15

    def __post_init__(self) -> None:
        self.remove_small = bool(self.remove_small)

        self.tubular_action = str(self.tubular_action).strip().lower()
        self.large_action = str(self.large_action).strip().lower()

        if self.tubular_action not in {"keep", "remove", "reconnect"}:
            raise ValueError(
                "instance_refinement.tubular_action must be one of: "
                "'keep', 'remove', 'reconnect'."
            )

        if self.large_action not in {"keep", "remove", "split"}:
            raise ValueError(
                "instance_refinement.large_action must be one of: "
                "'keep', 'remove', 'split'."
            )

        self.size_z_threshold = float(self.size_z_threshold)
        self.tubular_candidate_z = float(self.tubular_candidate_z)
        self.tubular_score_threshold = float(self.tubular_score_threshold)
        self.grid_n = int(self.grid_n)

        if self.size_z_threshold <= 0:
            raise ValueError("size_z_threshold must be > 0.")
        if self.tubular_candidate_z >= 0:
            raise ValueError("tubular_candidate_z must be < 0.")
        if not 0.0 <= self.tubular_score_threshold <= 1.0:
            raise ValueError("tubular_score_threshold must be between 0 and 1.")
        if not 1 <= self.grid_n <= 6:
            raise ValueError("grid_n must be between 1 and 6.")

        self.max_gap_px = float(self.max_gap_px)
        self.radial_sector_half_width_degrees = float(
            self.radial_sector_half_width_degrees
        )
        self.radial_tolerance_px = (
            None
            if self.radial_tolerance_px is None
            else float(self.radial_tolerance_px)
        )

        if self.max_gap_px <= 0:
            raise ValueError("max_gap_px must be > 0.")
        if not 0 < self.radial_sector_half_width_degrees <= 180:
            raise ValueError(
                "radial_sector_half_width_degrees must be in (0, 180]."
            )
        if self.radial_tolerance_px is not None and self.radial_tolerance_px < 0:
            raise ValueError("radial_tolerance_px must be >= 0 or None.")

        self.split_smoothing_sigma = float(self.split_smoothing_sigma)
        self.split_peak_radius_fraction = float(self.split_peak_radius_fraction)
        self.split_peak_min_distance_factor = float(
            self.split_peak_min_distance_factor
        )
        self.split_max_seed_factor = float(self.split_max_seed_factor)
        self.split_min_child_area_fraction = float(
            self.split_min_child_area_fraction
        )

        if self.split_smoothing_sigma < 0:
            raise ValueError("split_smoothing_sigma must be >= 0.")
        if not 0 < self.split_peak_radius_fraction <= 1:
            raise ValueError("split_peak_radius_fraction must be in (0, 1].")
        if self.split_peak_min_distance_factor <= 0:
            raise ValueError("split_peak_min_distance_factor must be > 0.")
        if self.split_max_seed_factor < 1:
            raise ValueError("split_max_seed_factor must be >= 1.")
        if not 0 < self.split_min_child_area_fraction < 1:
            raise ValueError(
                "split_min_child_area_fraction must be between 0 and 1."
            )

    @property
    def effective_radial_tolerance_px(self) -> float:
        return (
            self.max_gap_px
            if self.radial_tolerance_px is None
            else float(self.radial_tolerance_px)
        )

    # ------------------------------------------------------------------
    # Legacy Object-QC compatibility.
    # Existing scripts using config.qc.* continue to work. The old
    # booleans map to the historical REMOVE semantics.
    # ------------------------------------------------------------------

    @property
    def small_objects(self) -> bool:
        return bool(self.remove_small)

    @small_objects.setter
    def small_objects(self, value) -> None:
        self.remove_small = bool(value)

    @property
    def large_objects(self) -> bool:
        return self.large_action == "remove"

    @large_objects.setter
    def large_objects(self, value) -> None:
        self.large_action = "remove" if bool(value) else "keep"

    @property
    def tubular(self) -> bool:
        return self.tubular_action == "remove"

    @tubular.setter
    def tubular(self, value) -> None:
        self.tubular_action = "remove" if bool(value) else "keep"

    def configuration_rows(self) -> list[dict]:
        rows = []
        values = {
            "remove_small": self.remove_small,
            "tubular_action": self.tubular_action,
            "large_action": self.large_action,
            "size_z_threshold": self.size_z_threshold,
            "tubular_candidate_z": self.tubular_candidate_z,
            "tubular_score_threshold": self.tubular_score_threshold,
            "grid_n": self.grid_n,
            "max_gap_px": self.max_gap_px,
            "radial_sector_half_width_degrees": (
                self.radial_sector_half_width_degrees
            ),
            "radial_tolerance_px": self.effective_radial_tolerance_px,
            "split_smoothing_sigma": self.split_smoothing_sigma,
            "split_peak_radius_fraction": self.split_peak_radius_fraction,
            "split_peak_min_distance_factor": (
                self.split_peak_min_distance_factor
            ),
            "split_max_seed_factor": self.split_max_seed_factor,
            "split_min_child_area_fraction": (
                self.split_min_child_area_fraction
            ),
        }
        for parameter, value in values.items():
            rows.append(
                {
                    "section": "Instance_Refinement",
                    "parameter": parameter,
                    "value": value,
                    "source": "user" if parameter in {
                        "remove_small",
                        "tubular_action",
                        "large_action",
                    } else "default",
                }
            )
        return rows
