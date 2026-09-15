from __future__ import annotations

from dataclasses import dataclass, field


# ======================================================================
# CANONICAL MORPHOGLIA VISUAL STYLE
# ======================================================================

DEFAULT_DPI = 300
DEFAULT_MORPHOLOGY_STATE_PALETTE = "MG1"


# MG_CATEGORY_PLOT_DIRECTORY_V1
def plot_output_directory_name(
    category_fields,
) -> str:
    """
    Return the canonical Plots output directory for one category definition.

    Examples
    --------
    [] ->
        Plots

    ["condition"] ->
        Plots_condition

    ["condition", "layer"] ->
        Plots_condition_layer

    The directory encodes the category *definition*, not individual category
    values, so multiple analytical category definitions can coexist for the
    same dataset without overwriting one another.
    """

    tokens = []

    for field in (
        category_fields
        or ()
    ):
        token = "".join(
            character
            if (
                character.isalnum()
                or character
                in {
                    "-",
                    "_",
                    ".",
                }
            )
            else "_"
            for character
            in str(field).strip()
        )

        while "__" in token:
            token = token.replace(
                "__",
                "_",
            )

        token = token.strip(
            "._-"
        )

        if token:
            tokens.append(
                token
            )

    if not tokens:
        return "Plots"

    return (
        "Plots_"
        + "_".join(tokens)
    )

DEFAULT_UMAP_N_NEIGHBORS = 10
DEFAULT_UMAP_MIN_DIST = 0.10

# Main scatter
DEFAULT_POINT_SIZE = 70.0
DEFAULT_POINT_ALPHA = 0.55
DEFAULT_POINT_EDGE_WIDTH = 0.40

# Prototype triangles
DEFAULT_PROTOTYPE_MARKER_SIZE = 420.0

# Frames / typography
DEFAULT_FRAME_LINE_WIDTH = 1.8
DEFAULT_AXIS_LABEL_SIZE = 14
DEFAULT_PANEL_TITLE_SIZE = 15
DEFAULT_FIGURE_TITLE_SIZE = 19
DEFAULT_LEGEND_FONT_SIZE = 11

# Stability plots
DEFAULT_LINE_WIDTH = 2.8
DEFAULT_LINE_ALPHA = 0.80
DEFAULT_HEATMAP_GRID_WIDTH = 0.75

# Category highlighting
DEFAULT_BACKGROUND_ALPHA = 0.55
DEFAULT_HIGHLIGHT_ALPHA = 0.72

DEFAULT_RANDOM_STATE = 24


@dataclass
class PlotConfig:
    """
    Canonical MorphoGlia plot configuration.

    The public plotting style is deliberately:
        bold
        clean
        minimal
        consistent

    UMAP remains visualization only.
    """

    dpi: int | None = None
    morphology_state_palette: str | None = None

    umap_n_neighbors: int | None = None
    umap_min_dist: float | None = None

    point_size: float | None = None
    point_alpha: float | None = None
    point_edge_width: float | None = None

    prototype_marker_size: float | None = None

    frame_line_width: float | None = None

    axis_label_size: int | None = None
    panel_title_size: int | None = None
    figure_title_size: int | None = None
    legend_font_size: int | None = None

    line_width: float | None = None
    line_alpha: float | None = None

    heatmap_grid_width: float | None = None

    background_alpha: float | None = None
    highlight_alpha: float | None = None

    # Optional user-defined display order for metadata values.
    #
    # Example:
    #     {
    #         "sex": ["M", "F"],
    #         "condition": ["SS", "SCOP"],
    #     }
    #
    # Timepoint fields are numerically ordered automatically when absent.
    metadata_order: dict[str, list[str]] = field(
        default_factory=dict
    )

    random_state: int | None = None

    show_prototypes: bool = True


    @property
    def effective_dpi(self) -> int:
        value = (
            DEFAULT_DPI
            if self.dpi is None
            else int(self.dpi)
        )

        if value <= 0:
            raise ValueError(
                "Plots DPI must be a positive integer."
            )

        return value


    @property
    def effective_morphology_state_palette(self) -> str:
        value = (
            DEFAULT_MORPHOLOGY_STATE_PALETTE
            if self.morphology_state_palette is None
            else str(
                self.morphology_state_palette
            ).strip()
        )

        if not value:
            raise ValueError(
                "Morphology-state palette cannot be empty."
            )

        return value


    @property
    def effective_umap_n_neighbors(self) -> int:
        return (
            DEFAULT_UMAP_N_NEIGHBORS
            if self.umap_n_neighbors is None
            else int(self.umap_n_neighbors)
        )


    @property
    def effective_umap_min_dist(self) -> float:
        return (
            DEFAULT_UMAP_MIN_DIST
            if self.umap_min_dist is None
            else float(self.umap_min_dist)
        )


    @property
    def effective_point_size(self) -> float:
        return (
            DEFAULT_POINT_SIZE
            if self.point_size is None
            else float(self.point_size)
        )


    @property
    def effective_point_alpha(self) -> float:
        return (
            DEFAULT_POINT_ALPHA
            if self.point_alpha is None
            else float(self.point_alpha)
        )


    @property
    def effective_point_edge_width(self) -> float:
        return (
            DEFAULT_POINT_EDGE_WIDTH
            if self.point_edge_width is None
            else float(self.point_edge_width)
        )


    @property
    def effective_prototype_marker_size(self) -> float:
        return (
            DEFAULT_PROTOTYPE_MARKER_SIZE
            if self.prototype_marker_size is None
            else float(self.prototype_marker_size)
        )


    @property
    def effective_frame_line_width(self) -> float:
        return (
            DEFAULT_FRAME_LINE_WIDTH
            if self.frame_line_width is None
            else float(self.frame_line_width)
        )


    @property
    def effective_axis_label_size(self) -> int:
        return (
            DEFAULT_AXIS_LABEL_SIZE
            if self.axis_label_size is None
            else int(self.axis_label_size)
        )


    @property
    def effective_panel_title_size(self) -> int:
        return (
            DEFAULT_PANEL_TITLE_SIZE
            if self.panel_title_size is None
            else int(self.panel_title_size)
        )


    @property
    def effective_figure_title_size(self) -> int:
        return (
            DEFAULT_FIGURE_TITLE_SIZE
            if self.figure_title_size is None
            else int(self.figure_title_size)
        )


    @property
    def effective_legend_font_size(self) -> int:
        return (
            DEFAULT_LEGEND_FONT_SIZE
            if self.legend_font_size is None
            else int(self.legend_font_size)
        )


    @property
    def effective_line_width(self) -> float:
        return (
            DEFAULT_LINE_WIDTH
            if self.line_width is None
            else float(self.line_width)
        )


    @property
    def effective_line_alpha(self) -> float:
        return (
            DEFAULT_LINE_ALPHA
            if self.line_alpha is None
            else float(self.line_alpha)
        )


    @property
    def effective_heatmap_grid_width(self) -> float:
        return (
            DEFAULT_HEATMAP_GRID_WIDTH
            if self.heatmap_grid_width is None
            else float(self.heatmap_grid_width)
        )


    @property
    def effective_background_alpha(self) -> float:
        return (
            DEFAULT_BACKGROUND_ALPHA
            if self.background_alpha is None
            else float(self.background_alpha)
        )


    @property
    def effective_highlight_alpha(self) -> float:
        return (
            DEFAULT_HIGHLIGHT_ALPHA
            if self.highlight_alpha is None
            else float(self.highlight_alpha)
        )


    @property
    def effective_metadata_order(
        self,
    ) -> dict[str, list[str]]:
        """
        Return a defensive string-normalized copy of metadata display order.
        """

        if self.metadata_order is None:
            return {}

        if not isinstance(
            self.metadata_order,
            dict,
        ):
            raise TypeError(
                "plots.metadata_order must be a dictionary "
                "mapping metadata fields to ordered value lists."
            )

        result: dict[
            str,
            list[str],
        ] = {}

        for (
            field_name,
            values,
        ) in self.metadata_order.items():

            if isinstance(
                values,
                (
                    str,
                    bytes,
                ),
            ):
                raise TypeError(
                    "Each plots.metadata_order entry must "
                    "be a sequence of labels, not one string."
                )

            result[
                str(
                    field_name
                )
            ] = [
                str(
                    value
                )
                for value
                in values
            ]

        return result


    @property
    def effective_random_state(self) -> int:
        return (
            DEFAULT_RANDOM_STATE
            if self.random_state is None
            else int(self.random_state)
        )


    def configuration_rows(self) -> list[dict]:

        return [
            {
                "section": "Plots",
                "parameter": "dpi",
                "value": self.effective_dpi,
                "source": "default" if self.dpi is None else "user",
            },
            {
                "section": "Plots",
                "parameter": "morphology_state_palette",
                "value": self.effective_morphology_state_palette,
                "source": (
                    "default"
                    if self.morphology_state_palette is None
                    else "user"
                ),
            },
            {
                "section": "Plots",
                "parameter": "point_size",
                "value": self.effective_point_size,
                "source": "default" if self.point_size is None else "user",
            },
            {
                "section": "Plots",
                "parameter": "point_alpha",
                "value": self.effective_point_alpha,
                "source": "default" if self.point_alpha is None else "user",
            },
            {
                "section": "Plots",
                "parameter": "frame_line_width",
                "value": self.effective_frame_line_width,
                "source": (
                    "default"
                    if self.frame_line_width is None
                    else "user"
                ),
            },
            {
                "section": "Plots",
                "parameter": "metadata_order",
                "value": repr(
                    self.effective_metadata_order
                ),
                "source": (
                    "default"
                    if not self.metadata_order
                    else "user"
                ),
            },
        ]
