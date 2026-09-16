from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd

from .config import PlotConfig


# ======================================================================
# MORPHOGLIA PALETTE
# ======================================================================

MG_PALETTE_COLORS = (

    "#D95F83",  # arcade raspberry
    "#6268C4",  # electric periwinkle
    "#8A72B5",  # luminous violet
    "#16A0A0",  # vivid arcade teal

    
    
    "#C7A83D",  # ochre gold
    "#D77A3D",  # burnt orange
    #"#C94F45",  # brick red


)




MG2_PALETTE_COLORS = (
    "#E4574F",  # retro red
    "#3F83C5",  # arcade blue
    "#62A85B",  # vivid vintage green
    "#F29A38",  # warm game orange
    "#9A63B5",  # Vintage purple
    "#E3C33D",  # electric mustard yellow
)



MG3_PALETTE_COLORS = (
    "#D95D4F",  # coral red
    "#E58A47",  # burnt orange
    "#D8B84A",  # muted arcade gold

    "#4F9B83",  # jade green
    "#477FA3",  # dusty arcade blue
    "#765F9E",  # muted violet
)

MG4_PALETTE_COLORS = (
    "#C94F45",  # brick red
    "#D77A3D",  # burnt orange
    "#C7A83D",  # ochre gold

    "#438C78",  # teal green
    "#426E91",  # steel blue
    "#70588F",  # plum violet
)


# ======================================================================
# QUANTITATIVE / STATISTICAL PALETTES
# ======================================================================

# Continuous diverging palette used for morphometric profiles.
#
# These colors are deliberately distinct from the categorical MG palette:
#
#     MG palette         -> morphology-state identity
#     diverging palette  -> signed quantitative values
#     viridis            -> stability / robustness
#
MG_DIVERGING_COLORS = (
    "#C64F4A",  # restrained arcade red
    "#D97871",  # faded warm red
    "#E9AAA0",  # soft salmon
    "#F3F0E8",  # warm vintage ivory
    "#BDD6E2",  # pale arcade blue
    "#76A9C2",  # softened retro blue
    "#477FA3",  # restrained arcade blue
)


# Discrete version used for standardized residuals.
#
# Bins:
#     residual < -3
#     -3 to -2
#     -2 to 0
#      0 to +2
#     +2 to +3
#     > +3
#
MG_RESIDUAL_COLORS = (
    "#C64F4A",  # < -3
    "#D97871",  # -3 to -2
    "#E9AAA0",  # -2 to 0
    "#BDD6E2",  #  0 to +2
    "#76A9C2",  # +2 to +3
    "#477FA3",  # > +3
)

# Neutral reference population used by category plots.
LIGHT_GRAY = "#C7C7C7"
LIGHT_GRAY_COLOR = LIGHT_GRAY
LIGHT_GRAY_ALPHA = 0.5

# ---------------------------------------------------------------------
# Shared plot styling helpers
# ---------------------------------------------------------------------

CATEGORY_BACKGROUND_SIZE = 34
CATEGORY_FOREGROUND_SIZE = 42
CATEGORY_FOREGROUND_ALPHA = 0.82

LEGEND_MARKER_SIZE = 12
LEGEND_ANCHOR = (0.985, 0.93)

PLOT_FRAME_LINEWIDTH = 2.4

BAR_OUTLINE_COLOR = "black"
BAR_OUTLINE_WIDTH = 2.0

# Generic heatmap frame/grid defaults.
#
# These do NOT define heatmap semantics:
#   - Stability uses viridis.
#   - Residuals use the discrete statistical palette.
#   - Morphometric profiles use the continuous diverging palette.
HEATMAP_FRAME_LINEWIDTH = 2.0
HEATMAP_GRID_LINEWIDTH = 0.5

def apply_clean_axis_style(ax):
    """
    Minimal clean plotting style:
    - thicker black frame
    - no tick numbers
    """
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_FRAME_LINEWIDTH)
        spine.set_color("black")


def category_display_name(value: object) -> str:
    """
    Human-facing category labels:
    SCOP_CA1 -> SCOP-CA1
    """
    return str(value).replace("_", "-")


def build_cluster_legend_handles(cluster_colors):
    from matplotlib.lines import Line2D

    handles = []

    for cluster in sorted(cluster_colors):
        handles.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker="o",
                markersize=LEGEND_MARKER_SIZE,
                markerfacecolor=cluster_colors[cluster],
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=f"C{cluster}",
            )
        )

    return handles


# MG_MORPHOLOGY_STATE_PALETTE_V1
# ======================================================================
# MORPHOLOGY-STATE PALETTE REGISTRY
# ======================================================================
#
# These palettes affect ONLY morphology-state identity:
#
#     C1, C2, C3, ...
#
# They do NOT alter:
#
#     stability / robustness colors
#     morphometric profile heatmaps
#     standardized residual heatmaps
#     category reference gray
#
# To add another custom palette later:
#
#     1. define its tuple of colors above;
#     2. add one entry to _CUSTOM_MORPHOLOGY_STATE_PALETTES.
#
# The GUI discovers the available names from this registry.

_CUSTOM_MORPHOLOGY_STATE_PALETTES = {
    "MG1": MG_PALETTE_COLORS,
    "MG2": MG2_PALETTE_COLORS,
    "MG3": MG3_PALETTE_COLORS,
    "MG4": MG4_PALETTE_COLORS,
}

_MATPLOTLIB_MORPHOLOGY_STATE_PALETTES = (
    "plasma",
    "tab10",
    "tab20",
)

_SEABORN_MORPHOLOGY_STATE_PALETTES = (
    "hls",
    "rocket",
    "flare",
    "magma",
    "Spectral",
)

MORPHOLOGY_STATE_PALETTE_NAMES = (
    tuple(
        _CUSTOM_MORPHOLOGY_STATE_PALETTES
    )
    + _MATPLOTLIB_MORPHOLOGY_STATE_PALETTES
    + _SEABORN_MORPHOLOGY_STATE_PALETTES
)

_ACTIVE_MORPHOLOGY_STATE_PALETTE = "MG1"



def morphology_state_palette(
    palette: str | None = None,
    n_colors: int = 6,
) -> tuple:
    """
    Return exactly K morphology-state identity colors.

    MG1-MG4
    -------
    Their first six authored colors are preserved exactly.

    If K > 6, supplementary categorical colors are appended.
    The MG colors themselves are never interpolated.

    Continuous palettes
    -------------------
    plasma, rocket, flare, magma, Spectral

    Exactly K colors are sampled across the useful range of the
    complete colormap.

    hls
    ---
    Exactly K evenly distributed qualitative hues are generated.

    tab10 / tab20
    -------------
    Native categorical colors are preserved and extended only when
    the requested K exceeds their native capacity.

    This affects only morphology-state identity colors.
    """

    name = (
        _ACTIVE_MORPHOLOGY_STATE_PALETTE
        if palette is None
        else str(
            palette
        ).strip()
    )

    n_colors = int(
        n_colors
    )

    if n_colors < 1:
        raise ValueError(
            "n_colors must be >= 1."
        )


    def continuous_positions(
        count: int,
    ) -> np.ndarray:

        if count == 1:
            return np.asarray(
                [0.50],
                dtype=float,
            )

        return np.linspace(
            0.08,
            0.92,
            count,
        )


    def supplementary_colors(
        count: int,
    ) -> tuple:

        if count <= 0:
            return tuple()

        pool = []

        for cmap_name in (
            "tab20",
            "tab20b",
            "tab20c",
        ):

            cmap = plt.get_cmap(
                cmap_name
            )

            pool.extend(
                tuple(
                    cmap.colors
                )
            )

        if count > len(
            pool
        ):

            remaining = (
                count
                - len(
                    pool
                )
            )

            cmap = plt.get_cmap(
                "hsv"
            )

            pool.extend(
                cmap(
                    float(position)
                )
                for position in np.linspace(
                    0.0,
                    1.0,
                    remaining,
                    endpoint=False,
                )
            )

        return tuple(
            pool[
                :count
            ]
        )


    # ------------------------------------------------------------------
    # MG1-MG4
    # ------------------------------------------------------------------

    if name in _CUSTOM_MORPHOLOGY_STATE_PALETTES:

        authored = tuple(
            _CUSTOM_MORPHOLOGY_STATE_PALETTES[
                name
            ]
        )

        if n_colors <= len(
            authored
        ):

            return authored[
                :n_colors
            ]

        extra = (
            n_colors
            - len(
                authored
            )
        )

        return (
            authored
            + supplementary_colors(
                extra
            )
        )


    # ------------------------------------------------------------------
    # tab10
    # ------------------------------------------------------------------

    if name == "tab10":

        base = tuple(
            plt.get_cmap(
                "tab10"
            ).colors
        )

        if n_colors <= len(
            base
        ):
            return base[
                :n_colors
            ]

        # Lighter partners from tab20 extend tab10 to twenty
        # related categorical colors.
        extension = tuple(
            plt.get_cmap(
                "tab20"
            ).colors[
                1::2
            ]
        )

        combined = (
            base
            + extension
        )

        if n_colors <= len(
            combined
        ):
            return combined[
                :n_colors
            ]

        return (
            combined
            + supplementary_colors(
                n_colors
                - len(
                    combined
                )
            )
        )[
            :n_colors
        ]


    # ------------------------------------------------------------------
    # tab20
    # ------------------------------------------------------------------

    if name == "tab20":

        base = tuple(
            plt.get_cmap(
                "tab20"
            ).colors
        )

        if n_colors <= len(
            base
        ):
            return base[
                :n_colors
            ]

        return (
            base
            + supplementary_colors(
                n_colors
                - len(
                    base
                )
            )
        )[
            :n_colors
        ]


    # ------------------------------------------------------------------
    # Matplotlib continuous palettes
    # ------------------------------------------------------------------

    if name == "plasma":

        cmap = plt.get_cmap(
            name
        )

        return tuple(
            cmap(
                float(position)
            )
            for position in continuous_positions(
                n_colors
            )
        )


    # ------------------------------------------------------------------
    # Seaborn palettes
    # ------------------------------------------------------------------

    if name in _SEABORN_MORPHOLOGY_STATE_PALETTES:

        try:
            import seaborn as sns

        except ImportError as exc:
            raise ImportError(
                f"Palette {name!r} requires seaborn."
            ) from exc


        if name == "hls":

            return tuple(
                sns.color_palette(
                    "hls",
                    n_colors=n_colors,
                )
            )


        cmap = sns.color_palette(
            name,
            as_cmap=True,
        )

        return tuple(
            cmap(
                float(position)
            )
            for position in continuous_positions(
                n_colors
            )
        )


    raise ValueError(
        "Unknown morphology-state palette "
        f"{name!r}. Available palettes: "
        f"{', '.join(MORPHOLOGY_STATE_PALETTE_NAMES)}."
    )



def set_morphology_state_palette(
    palette: str,
) -> str:
    """
    Set the process-wide morphology-state identity palette.

    MorphoGlia runs one pipeline configuration at a time, so a shared
    presentation palette keeps DRC, Mapping, and Plots state colors
    consistent without changing analytical cluster identities.
    """

    global _ACTIVE_MORPHOLOGY_STATE_PALETTE

    name = str(
        palette
    ).strip()

    # Validate before changing global state.
    morphology_state_palette(
        name
    )

    _ACTIVE_MORPHOLOGY_STATE_PALETTE = (
        name
    )

    return name


def active_morphology_state_palette() -> str:
    """Return the currently active morphology-state palette name."""

    return (
        _ACTIVE_MORPHOLOGY_STATE_PALETTE
    )



def mg_color(
    index: int,
    palette: str | None = None,
    n_colors: int | None = None,
):
    """
    Return one morphology-state identity color.

    Production rendering code should pass the total number of states K
    through n_colors so continuous palettes can span their full range.
    """

    index = int(
        index
    )

    if index < 0:
        raise ValueError(
            "Morphology-state color indices must be >= 0."
        )


    if n_colors is not None:

        n_colors = max(
            int(
                n_colors
            ),
            index + 1,
        )

        colors = morphology_state_palette(
            palette=palette,
            n_colors=n_colors,
        )

        return colors[
            index
        ]


    # ------------------------------------------------------------------
    # Backward-compatible fallback for external/legacy direct calls.
    #
    # The production MorphoGlia renderers pass K explicitly.
    # ------------------------------------------------------------------

    if index < 6:

        colors = morphology_state_palette(
            palette=palette,
            n_colors=6,
        )

        return colors[
            index
        ]


    cmap = plt.get_cmap(
        "tab20"
    )

    return cmap.colors[
        (
            index
            - 6
        )
        % len(
            cmap.colors
        )
    ]




def morphology_state_rgb(
    state_number: int,
    palette: str | None = None,
    n_colors: int | None = None,
) -> tuple[int, int, int]:
    """
    Return one one-based morphology-state color as uint8 RGB.
    """

    state_number = int(
        state_number
    )

    if state_number < 1:
        raise ValueError(
            "Morphology-state numbers are one-based "
            "and must be >= 1."
        )

    color = to_rgb(
        mg_color(
            state_number - 1,
            palette=palette,
            n_colors=n_colors,
        )
    )

    return tuple(
        int(
            round(
                255 * channel
            )
        )
        for channel in color
    )



# ======================================================================
# CLUSTER DISPLAY ORDER
# ======================================================================

def cluster_display_order(
    master: pd.DataFrame,
    labels,
    complexity_column: str = "Cell_area",
):
    """
    Order cluster DISPLAY labels by median cell area.

    Important
    ---------
    The analytical GMM cluster identity is NOT modified.

    Only the human-facing labels C1 ... Cn are reordered.

    Therefore:

        smaller median Cell_area
            ->
        lower display state number

        larger median Cell_area
            ->
        higher display state number

    This creates a reproducible morphology-magnitude ordering while
    preserving the original analytical labels in all saved tables.
    """

    if complexity_column not in master.columns:

        raise ValueError(
            "Cannot order morphology states because "
            f"{complexity_column!r} is missing."
        )


    labels = np.asarray(
        labels,
        dtype=int,
    )


    if len(
        labels
    ) != len(
        master
    ):

        raise ValueError(
            "Cluster-label count does not match master table."
        )


    table = pd.DataFrame(
        {
            "cluster":
                labels,

            "complexity":
                master[
                    complexity_column
                ].to_numpy(
                    dtype=float
                ),
        }
    )


    medians = (
        table
        .groupby(
            "cluster"
        )[
            "complexity"
        ]
        .median()
        .sort_values(
            kind="mergesort"
        )
    )


    ordered_clusters = [
        int(
            cluster
        )
        for cluster in medians.index
    ]


    display_rank = {
        raw_cluster:
            rank + 1

        for rank, raw_cluster in enumerate(
            ordered_clusters
        )
    }


    return (
        ordered_clusters,
        display_rank,
    )


# ======================================================================
# AXIS STYLE
# ======================================================================

def style_axis(
    axis,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    config: PlotConfig,
    hide_ticks: bool = True,
) -> None:
    """
    Apply the canonical MorphoGlia visual style to one axis.
    """

    axis.set_title(
        title,
        fontsize=(
            config
            .effective_panel_title_size
        ),
        fontweight="bold",
        pad=12,
    )


    axis.set_xlabel(
        xlabel,
        fontsize=(
            config
            .effective_axis_label_size
        ),
        fontweight="bold",
        labelpad=10,
    )


    axis.set_ylabel(
        ylabel,
        fontsize=(
            config
            .effective_axis_label_size
        ),
        fontweight="bold",
        labelpad=10,
    )


    # ------------------------------------------------------------------
    # TICKS
    # ------------------------------------------------------------------

    if hide_ticks:

        axis.set_xticks(
            []
        )

        axis.set_yticks(
            []
        )


        axis.tick_params(
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    else:

        axis.tick_params(
            labelsize=10,
            width=1.2,
        )


        for label in (
            axis.get_xticklabels()
            + axis.get_yticklabels()
        ):

            label.set_fontweight(
                "bold"
            )


    # ------------------------------------------------------------------
    # FRAME
    # ------------------------------------------------------------------

    for spine in axis.spines.values():

        spine.set_linewidth(
            config
            .effective_frame_line_width
        )


# ======================================================================
# CLUSTER LEGEND
# ======================================================================

def add_cluster_legend(
    axis,
    config: PlotConfig,
) -> None:
    """
    Place bold cluster labels outside the upper-right corner.
    """

    legend = axis.legend(
        loc="upper left",
        bbox_to_anchor=(
            1.015,
            1.0,
        ),
        frameon=False,
        borderaxespad=0.0,
        markerscale=1.30,
        prop={
            "size":
                config
                .effective_legend_font_size,

            "weight":
                "bold",
        },
    )


    if legend is not None:

        for text in (
            legend.get_texts()
        ):

            text.set_fontweight(
                "bold"
            )


# ======================================================================
# DOUBLE-OUTLINE MARKERS
# ======================================================================

def scatter_double_outline(
    axis,
    x,
    y,
    *,
    marker: str,
    size: float,
    color,
    alpha: float = 1.0,
    zorder: int = 20,
):
    """
    Draw a marker with:

        black outer border
        white inner border
        colored center

    This is used for prototypes and selected points that need to remain
    visible over dense scatter plots.
    """

    # ------------------------------------------------------------------
    # BLACK OUTER BORDER
    # ------------------------------------------------------------------

    axis.scatter(
        x,
        y,
        marker=marker,
        s=(
            size
            * 2.1
        ),
        color="black",
        edgecolors="none",
        alpha=alpha,
        zorder=zorder,
    )


    # ------------------------------------------------------------------
    # WHITE INNER BORDER
    # ------------------------------------------------------------------

    axis.scatter(
        x,
        y,
        marker=marker,
        s=(
            size
            * 1.5
        ),
        color="white",
        edgecolors="none",
        alpha=alpha,
        zorder=(
            zorder
            + 1
        ),
    )


    # ------------------------------------------------------------------
    # COLORED CENTER
    # ------------------------------------------------------------------

    axis.scatter(
        x,
        y,
        marker=marker,
        s=size,
        color=color,
        edgecolors="none",
        alpha=alpha,
        zorder=(
            zorder
            + 2
        ),
    )


