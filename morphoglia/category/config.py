from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CategoryConfig:
    """
    Configuration for the Experimental Category stage.

    category_fields:
        Metadata columns that MorphoGlia combines to create the
        reserved 'category' column.

        Example:

            category_fields = [
                "condition",
                "region",
                "spatial_bin",
            ]

        could produce:

            CTRL_CA1_center
            LPS_CA1_periphery

        By default no fields are selected.

    Notes
    -----
    - Category is derived from existing metadata.
    - Category is not part of the canonical image filename.
    - Changing Category does not require rerunning Metadata,
      Preprocessing, or Morphometrics.
    """

    category_fields: list[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        """
        Report whether the category definition was supplied by the user.
        """
        if self.category_fields:
            return "user"

        return "default"

    def configuration_rows(self) -> list[dict[str, object]]:
        """
        Return Category settings in the format expected by
        Technical_Record/configuration.csv.
        """

        fields = ",".join(self.category_fields)

        return [
            {
                "section": "Category",
                "parameter": "category_fields",
                "value": fields,
                "source": self.source,
            }
        ]
