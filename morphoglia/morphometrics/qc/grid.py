from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================================================================
# SELECTION
# ======================================================================

def select_qc_examples(
    table: pd.DataFrame,
    *,
    flag_column: str,
    score_column: str,
    grid_n: int = 6,
    extreme: str,
) -> pd.DataFrame:
    """
    Select suspicious objects for an n x n QC grid.

    When there are more suspicious objects than available grid cells,
    half of the grid shows the most extreme objects and half shows
    suspicious objects closest to the statistical cutoff.

    The returned DataFrame stores the total number of suspicious objects
    in:

        result.attrs["detected_count"]

    so the plotting function can report the total number detected rather
    than merely the number displayed in the grid.

    Parameters
    ----------
    table
        Object-level QC table.

    flag_column
        Boolean column identifying suspicious objects.

    score_column
        Continuous QC score used to rank suspicious objects.

    grid_n
        Grid dimension. Maximum is 6 x 6.

    extreme
        "low"
            Lower scores are more extreme.

        "high"
            Higher scores are more extreme.
    """

    grid_n = int(
        grid_n
    )

    if not 1 <= grid_n <= 6:
        raise ValueError(
            "grid_n must be between 1 and 6."
        )


    if extreme not in {
        "low",
        "high",
    }:
        raise ValueError(
            "extreme must be 'low' or 'high'."
        )


    required = {
        flag_column,
        score_column,
    }


    missing = (
        required
        - set(
            table.columns
        )
    )


    if missing:
        raise ValueError(
            f"Missing QC columns: {sorted(missing)}"
        )


    suspicious = (
        table.loc[
            table[
                flag_column
            ].astype(bool)
        ]
        .copy()
    )


    detected_count = len(
        suspicious
    )


    # ------------------------------------------------------------------
    # NO SUSPICIOUS OBJECTS
    # ------------------------------------------------------------------

    if suspicious.empty:

        suspicious.attrs[
            "detected_count"
        ] = 0

        return suspicious


    # ------------------------------------------------------------------
    # GRID CAPACITY
    # ------------------------------------------------------------------

    capacity = (
        grid_n
        * grid_n
    )


    ascending_extreme = (
        extreme
        == "low"
    )


    suspicious = (
        suspicious
        .sort_values(
            score_column,
            ascending=ascending_extreme,
            kind="mergesort",
        )
    )


    # ------------------------------------------------------------------
    # EVERYTHING FITS IN THE GRID
    # ------------------------------------------------------------------

    if len(
        suspicious
    ) <= capacity:

        selected = (
            suspicious
            .copy()
            .reset_index(
                drop=True
            )
        )


        selected[
            "_qc_position"
        ] = "suspicious"


        selected.attrs[
            "detected_count"
        ] = detected_count


        return selected


    # ------------------------------------------------------------------
    # SPLIT GRID:
    #
    # first half  = strongest outliers
    # second half = suspicious objects closest to cutoff
    # ------------------------------------------------------------------

    n_extreme = (
        capacity
        // 2
    )


    n_borderline = (
        capacity
        - n_extreme
    )


    extreme_rows = (
        suspicious
        .head(
            n_extreme
        )
        .copy()
    )


    extreme_rows[
        "_qc_position"
    ] = "extreme"


    borderline_rows = (
        suspicious
        .sort_values(
            score_column,
            ascending=not ascending_extreme,
            kind="mergesort",
        )
        .head(
            n_borderline
        )
        .copy()
    )


    borderline_rows[
        "_qc_position"
    ] = "borderline"


    selected = pd.concat(
        [
            extreme_rows,
            borderline_rows,
        ],
        axis=0,
    )


    selected = (
        selected
        .loc[
            ~selected.index.duplicated(
                keep="first"
            )
        ]
        .reset_index(
            drop=True
        )
    )


    selected.attrs[
        "detected_count"
    ] = detected_count


    return selected


# ======================================================================
# PLOTTING
# ======================================================================

def plot_qc_grid(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    grid_n: int = 6,
    title: str,
    score_column: str,
) -> Path | None:
    """
    Plot isolated objects in an n x n QC grid.

    No PNG is produced when no suspicious objects were detected.

    Expected columns
    ----------------
    roi
    cell_id
    component_area
    score_column
    _qc_position

    Returns
    -------
    Path
        Path to the generated PNG.

    None
        When no suspicious objects were detected.
    """

    grid_n = int(
        grid_n
    )


    if not 1 <= grid_n <= 6:
        raise ValueError(
            "grid_n must be between 1 and 6."
        )


    # ------------------------------------------------------------------
    # NUMBER DETECTED
    #
    # This is the TOTAL suspicious population, not merely the objects
    # selected for display.
    # ------------------------------------------------------------------

    detected_count = int(
        table.attrs.get(
            "detected_count",
            len(table),
        )
    )


    # ------------------------------------------------------------------
    # NOTHING DETECTED -> NO EMPTY PNG
    # ------------------------------------------------------------------

    if detected_count == 0 or table.empty:
        return None


    # ------------------------------------------------------------------
    # VALIDATE INPUT
    # ------------------------------------------------------------------

    required = {
        "roi",
        "cell_id",
        "component_area",
        score_column,
    }


    missing = (
        required
        - set(
            table.columns
        )
    )


    if missing:
        raise ValueError(
            f"Missing grid columns: {sorted(missing)}"
        )


    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------------------------------------------
    # FIGURE
    # ------------------------------------------------------------------

    figure, axes = plt.subplots(
        grid_n,
        grid_n,
        figsize=(
            2.3 * grid_n,
            2.3 * grid_n,
        ),
        squeeze=False,
    )


    flat_axes = axes.ravel()


    for axis in flat_axes:
        axis.axis(
            "off"
        )


    # ------------------------------------------------------------------
    # OBJECTS
    # ------------------------------------------------------------------

    for axis, (_, row) in zip(
        flat_axes,
        table.iterrows(),
    ):

        roi = np.asarray(
            row[
                "roi"
            ]
        )


        axis.imshow(
            roi,
            cmap="gray",
            interpolation="nearest",
        )


        position = str(
            row.get(
                "_qc_position",
                "suspicious",
            )
        )


        if position == "extreme":

            marker = "EXTREME"

        elif position == "borderline":

            marker = "BORDER"

        else:

            marker = ""


        cell_id = str(
            row[
                "cell_id"
            ]
        )


        # --------------------------------------------------------------
        # PANEL TITLE
        #
        # Only EXTREME / BORDER is bold.
        # --------------------------------------------------------------

        if marker:

            first_line = (
                rf"$\bf{{{marker}}}$"
                f" | {cell_id}"
            )

        else:

            first_line = (
                cell_id
            )


        axis.set_title(
            (
                f"{first_line}\n"
                f"A={int(row['component_area'])}  "
                f"z={float(row[score_column]):.2f}"
            ),
            fontsize=7,
        )


        axis.axis(
            "off"
        )


    # ------------------------------------------------------------------
    # MAIN TITLE
    # ------------------------------------------------------------------

    figure.suptitle(
        (
            f"{title} "
            f"| Detected: {detected_count}"
        ),
        fontsize=24,
        fontweight="bold",
        y=0.965,
    )


    # ------------------------------------------------------------------
    # LEGEND / EXPLANATION
    # ------------------------------------------------------------------

    figure.text(
        0.5,
        0.015,
        (
            r"$\bf{EXTREME}$"
            " = strongest outliers   |   "
            r"$\bf{BORDER}$"
            " = suspicious objects closest to cutoff"
        ),
        ha="center",
        fontsize=15,
    )


    figure.tight_layout(
        rect=[
            0,
            0.035,
            1,
            0.955,
        ]
    )


    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


    return output_path
