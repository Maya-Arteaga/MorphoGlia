from __future__ import annotations

import pandas as pd

from .config import CategoryConfig


# ======================================================================
# CANONICAL CATEGORY CONTRACT
# ======================================================================

CATEGORY_COLUMN = "category"


def apply_category(
    table: pd.DataFrame,
    config: CategoryConfig,
) -> pd.DataFrame:
    """
    Create the reserved MorphoGlia 'category' column.

    Category is derived only from existing metadata columns.

    Examples
    --------
    category_fields = ["condition"]

        LPS
        CTRL

    category_fields = ["condition", "region"]

        LPS_CA1
        CTRL_CA1

    category_fields = ["condition", "region", "spatial_bin"]

        LPS_CA1_center
        CTRL_CA1_periphery

    Notes
    -----
    - The input table is not modified in place.
    - If no category fields are selected, the table is returned unchanged.
    - Every selected field must exist in the table.
    - Missing values in selected category fields are not accepted.
    - An existing 'category' column is replaced.
    """

    result = table.copy()

    # ------------------------------------------------------------------
    # No category requested
    # ------------------------------------------------------------------

    if not config.category_fields:
        return result

    # ------------------------------------------------------------------
    # Validate requested metadata columns
    # ------------------------------------------------------------------

    missing_columns = [
        field_name
        for field_name in config.category_fields
        if field_name not in result.columns
    ]

    if missing_columns:
        fields = ", ".join(missing_columns)

        raise ValueError(
            "Category cannot be created because the following "
            f"metadata column(s) are missing: {fields}"
        )

    # ------------------------------------------------------------------
    # Validate missing values
    # ------------------------------------------------------------------

    selected = result[config.category_fields]

    missing_mask = selected.isna()

    if missing_mask.any().any():

        problems: list[str] = []

        for field_name in config.category_fields:

            count = int(
                missing_mask[field_name].sum()
            )

            if count > 0:
                problems.append(
                    f"{field_name} ({count} missing)"
                )

        details = ", ".join(problems)

        raise ValueError(
            "Category cannot be created because selected metadata "
            f"contain missing values: {details}"
        )

    # ------------------------------------------------------------------
    # Create category
    # ------------------------------------------------------------------

    category = (
        selected
        .astype(str)
        .agg("_".join, axis=1)
    )

    result[CATEGORY_COLUMN] = category

    return result
