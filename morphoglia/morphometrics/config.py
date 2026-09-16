from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ======================================================================
# Built-in MorphoGlia morphometric calculators
# ======================================================================

DEFAULT_CALCULATORS = (
    "cell",
    "soma",
    "convex_hull",
    "fractal",
    "sholl",
    "branches",
    "branch_order_dist",
    "centroids",
)


@dataclass
class MorphometricsConfig:
    """
    Configuration for the Morphometrics stage.

    By default, MorphoGlia runs all built-in morphometric calculators.

    If `calculators` is explicitly supplied, only those calculators are
    used.

    Examples
    --------
    Default:
        calculators = None

        -> all built-in morphometrics

    Subset:
        calculators = [
            "cell",
            "soma",
            "branches",
            "sholl",
        ]

        -> only those calculator groups

    Custom calculator specifications supported by the existing registry
    builder can also be supplied later.
    """

    calculators: list[Any] | None = None

    @property
    def source(self) -> str:
        """
        Whether calculator selection came from MorphoGlia defaults
        or an explicit user selection.
        """

        if self.calculators is None:
            return "default"

        return "user"

    @property
    def effective_calculators(
        self,
    ) -> list[Any]:
        """
        Return the calculator specifications that should actually run.
        """

        if self.calculators is None:
            return list(
                DEFAULT_CALCULATORS
            )

        return list(
            self.calculators
        )

    def builtin_status(
        self,
    ) -> dict[str, bool]:
        """
        Return enabled/disabled status for every built-in calculator.

        This representation is useful for GUI checkboxes and the
        Technical Record.
        """

        selected_names = {
            spec
            for spec in self.effective_calculators
            if isinstance(spec, str)
        }

        return {
            name: name in selected_names
            for name in DEFAULT_CALCULATORS
        }

    def configuration_rows(
        self,
    ) -> list[dict]:
        """
        Return Technical Record rows for built-in calculator selection.
        """

        status = self.builtin_status()

        return [
            {
                "section": "Morphometrics",
                "parameter": name,
                "value": enabled,
                "source": self.source,
            }
            for name, enabled in status.items()
        ]
