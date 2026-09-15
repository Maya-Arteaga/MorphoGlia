from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import to_rgb

from ..plots.style import (
    mg_color,
)

# ======================================================================
# HELPERS
# ======================================================================

def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
) -> None:

    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )




def cluster_rgb(
    cluster: int,
    n_colors: int | None = None,
    palette: str | None = None,
) -> tuple[int, int, int]:
    """
    Return one human-facing morphology-state display color as uint8 RGB.

    Palette is passed explicitly so presentation-only rerenders do not
    depend on process-global palette state.
    """

    cluster = int(
        cluster
    )

    if cluster < 1:
        raise ValueError(
            "Morphology-state display clusters are one-based."
        )

    rgb = to_rgb(
        mg_color(
            cluster - 1,
            palette=palette,
            n_colors=n_colors,
        )
    )

    return tuple(
        int(
            round(
                255 * value
            )
        )
        for value in rgb
    )






def _short_cell_id(
    cell_id: str,
) -> str:
    """
    Convert MorphoGlia's canonical keyed cell identity into a compact
    values-only display identity.

    Example
    -------
    subject-R7_condition-SCOP_region-CA1_tissue-T2_cell_33

    becomes

    R7_SCOP_CA1_T2_33

    The canonical cell_id itself is never modified.
    """

    text = str(
        cell_id
    ).strip()


    if not text:

        return ""


    parts = text.split(
        "_"
    )


    values = []

    index = 0


    while index < len(
        parts
    ):

        part = parts[
            index
        ]


        # --------------------------------------------------------------
        # Final "_cell_33" structure.
        # --------------------------------------------------------------

        if (
            part.lower()
            == "cell"
            and index + 1 < len(parts)
        ):

            values.append(
                parts[
                    index + 1
                ]
            )

            index += 2

            continue


        # --------------------------------------------------------------
        # Generic keyed value:
        #
        #     subject-R7 -> R7
        #     region-CA1 -> CA1
        # --------------------------------------------------------------

        if "-" in part:

            (
                _,
                value,
            ) = part.split(
                "-",
                1,
            )


            if value:

                values.append(
                    value
                )


        elif (
            part
            and part.lower()
            != "cell"
        ):

            values.append(
                part
            )


        index += 1


    return "_".join(
        values
    )



def _wrap_text(
    text: str,
    width: int = 28,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines



def _load_font(
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass

    return ImageFont.load_default()




def _mask_to_rgba(
    mask: np.ndarray,
    rgb: tuple[int, int, int],
) -> np.ndarray:
    """
    Convert a binary mask {0,255} into a transparent RGBA image.
    Foreground keeps the cluster color, background is fully transparent.
    """
    rgba = np.zeros(
        (mask.shape[0], mask.shape[1], 4),
        dtype=np.uint8,
    )

    foreground = mask > 0
    rgba[foreground, 0] = rgb[0]
    rgba[foreground, 1] = rgb[1]
    rgba[foreground, 2] = rgb[2]
    rgba[foreground, 3] = 255

    return rgba


def _cell_caption(
    cell: pd.Series,
    show_full_id: bool = False,
) -> list[str]:
    """
    Return the compact values-only cell identity.

    show_full_id is retained for API compatibility, but prototype
    montages intentionally use the compact identity so neighboring
    labels cannot collide.
    """

    compact = _short_cell_id(
        str(
            cell[
                "cell_id"
            ]
        )
    )


    return [
        compact
    ]



def render_selection_montage(
    selection: pd.DataFrame,
    image_cache: Mapping,
    output_path: str | Path,
    title: str | None = None,
    label_columns: Sequence[str] = (),
    padding: int = 20,
    cluster_gap: int = 45,
    cell_gap_x: int = 20,
    cell_gap_y: int = 26,
    ncols: int = 3,
    max_cells_per_cluster: int | None = None,
    show_full_id: bool = False,

    precomputed_crops: Mapping[tuple[int, int], object] | None = None,
    palette: str | None = None,
    dpi: int | None = None,) -> Path:
    """
    Render selected observed cells as transparent native-scale grids.

    Important
    ---------
    - Objects are NEVER resized.
    - Every object retains its original pixel dimensions.
    - A common native-scale canvas is used.
    - Background is transparent.
    - Colors come exclusively from plots/style.py through mg_color().
    - display_cluster is used when available so colors match PCA/UMAP.
    - Captions use compact values-only cell identities.
    """

    if selection.empty:

        raise ValueError(
            "Selection table is empty."
        )


    _require_columns(
        selection,
        [
            "image_id",
            "cell_id",
            "cluster",
            "selection_type",
            "selection_rank",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
        ],
    )


    ncols = int(
        ncols
    )


    if ncols < 1:

        raise ValueError(
            "ncols must be >= 1."
        )


    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ==================================================================
    # SORT
    # ==================================================================

    selection = (
        selection
        .copy()
    )


    selection[
        "cluster"
    ] = (
        selection[
            "cluster"
        ]
        .astype(int)
    )


    selection[
        "selection_rank"
    ] = (
        selection[
            "selection_rank"
        ]
        .astype(int)
    )


    selection = (
        selection
        .sort_values(
            [
                "cluster",
                "selection_rank",
                "cell_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ==================================================================
    # EXTRACT EXACT NATIVE-SCALE COMPONENTS
    # ==================================================================

    prepared_rows = []

    global_max_h = 0
    global_max_w = 0


    for (
        _,
        cell,
    ) in selection.iterrows():

        crop_key = (
            int(cell["cluster"]),
            int(cell["selection_rank"]),
        )

        if (
            precomputed_crops is not None
            and crop_key in precomputed_crops
        ):
            cached_crop = precomputed_crops[
                crop_key
            ]

            if isinstance(
                cached_crop,
                (str, Path),
            ):
                crop = np.asarray(
                    Image.open(
                        cached_crop
                    ).convert(
                        "L"
                    )
                )
            else:
                crop = np.asarray(
                    cached_crop
                )
        else:
            crop = extract_component_crop(
                cell=cell,
                image_cache=image_cache,
            )


        if crop is None:

            continue


        height, width = (
            crop.shape
        )


        global_max_h = max(
            global_max_h,
            height,
        )


        global_max_w = max(
            global_max_w,
            width,
        )


        prepared_rows.append(
            {
                "cell":
                    cell,
                "crop":
                    crop,
            }
        )


    if not prepared_rows:

        raise ValueError(
            "No valid crops could be extracted "
            "from the selection table."
        )


    # ==================================================================
    # SHARED NATIVE-SCALE IMAGE CANVAS
    # ==================================================================

    shared_h = (
        global_max_h
        + 2 * padding
    )


    shared_w = (
        global_max_w
        + 2 * padding
    )


    # ==================================================================
    # GROUP BY HUMAN-FACING DISPLAY CLUSTER
    # ==================================================================

    grouped: dict[
        int,
        list[dict],
    ] = {}


    for row in prepared_rows:

        cell = row[
            "cell"
        ]


        display_cluster = int(
            cell.get(
                "display_cluster",
                cell[
                    "cluster"
                ],
            )
        )


        grouped.setdefault(
            display_cluster,
            [],
        ).append(
            row
        )


    for cluster in grouped:

        grouped[
            cluster
        ] = sorted(
            grouped[
                cluster
            ],
            key=lambda row: (
                int(
                    row[
                        "cell"
                    ][
                        "selection_rank"
                    ]
                ),
                str(
                    row[
                        "cell"
                    ][
                        "cell_id"
                    ]
                ),
            ),
        )


        if (
            max_cells_per_cluster
            is not None
        ):

            grouped[
                cluster
            ] = grouped[
                cluster
            ][
                :int(
                    max_cells_per_cluster
                )
            ]


    # ==================================================================
    # TYPOGRAPHY
    # ==================================================================

    title_font = _load_font(
        30
    )


    cluster_font = _load_font(
        22
    )


    cell_font = _load_font(
        15
    )


    # ==================================================================
    # CALCULATE CAPTION WIDTH
    #
    # Tile width expands when necessary.
    #
    # This prevents neighboring cell IDs from overlapping while object
    # geometry itself remains at exact native pixel scale.
    # ==================================================================

    measurement_canvas = Image.new(
        "RGBA",
        (
            10,
            10,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )


    measurement_draw = ImageDraw.Draw(
        measurement_canvas
    )


    max_caption_width = 0


    for rows in grouped.values():

        for row in rows:

            caption = _short_cell_id(
                str(
                    row[
                        "cell"
                    ][
                        "cell_id"
                    ]
                )
            )


            bbox = measurement_draw.textbbox(
                (
                    0,
                    0,
                ),
                caption,
                font=cell_font,
            )


            caption_width = (
                bbox[
                    2
                ]
                - bbox[
                    0
                ]
            )


            max_caption_width = max(
                max_caption_width,
                caption_width,
            )


    tile_w = max(
        shared_w,
        max_caption_width + 24,
    )


    text_block_h = 30


    tile_h = (
        shared_h
        + text_block_h
    )


    overall_title_h = (
        48
        if title
        else 0
    )


    cluster_title_h = 34


    # ==================================================================
    # LAYOUT
    # ==================================================================

    cluster_order = sorted(
        grouped
    )


    block_widths = []

    block_heights = []


    for cluster in cluster_order:

        rows = grouped[
            cluster
        ]


        nrows = math.ceil(
            len(
                rows
            )
            / ncols
        )


        block_width = (
            ncols
            * tile_w
            + (
                ncols
                - 1
            )
            * cell_gap_x
        )


        block_height = (
            cluster_title_h
            + 12
            + nrows
            * tile_h
            + max(
                0,
                nrows - 1,
            )
            * cell_gap_y
        )


        block_widths.append(
            block_width
        )


        block_heights.append(
            block_height
        )


    total_w = (
        max(
            block_widths
        )
        + 2 * padding
    )


    total_h = (
        overall_title_h
        + sum(
            block_heights
        )
        + max(
            0,
            len(
                block_heights
            ) - 1,
        )
        * cluster_gap
        + 2 * padding
    )


    # ==================================================================
    # TRANSPARENT OUTPUT
    # ==================================================================

    canvas = Image.new(
        "RGBA",
        (
            total_w,
            total_h,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )


    draw = ImageDraw.Draw(
        canvas
    )


    y_cursor = padding


    if title:

        draw.text(
            (
                padding,
                y_cursor,
            ),
            title,
            fill=(
                80,
                80,
                80,
                255,
            ),
            font=title_font,
        )


        y_cursor += (
            overall_title_h
        )


    # ==================================================================
    # RENDER CLUSTERS
    # ==================================================================

    for cluster in cluster_order:

        rows = grouped[
            cluster
        ]


        color = cluster_rgb(
            cluster,
            n_colors=len(cluster_order),
            palette=palette,
        )


        draw.text(
            (
                padding,
                y_cursor,
            ),
            (
                f"C{cluster}  "
            ),
            fill=(
                *color,
                255,
            ),
            font=cluster_font,
        )


        y_block_top = (
            y_cursor
            + cluster_title_h
            + 8
        )


        for (
            index,
            row,
        ) in enumerate(
            rows
        ):

            grid_row = (
                index
                // ncols
            )


            grid_column = (
                index
                % ncols
            )


            tile_x = (
                padding
                + grid_column
                * (
                    tile_w
                    + cell_gap_x
                )
            )


            tile_y = (
                y_block_top
                + grid_row
                * (
                    tile_h
                    + cell_gap_y
                )
            )


            centered = (
                center_component_on_canvas(
                    row[
                        "crop"
                    ],
                    canvas_height=(
                        shared_h
                    ),
                    canvas_width=(
                        shared_w
                    ),
                )
            )


            rgba = _mask_to_rgba(
                centered,
                color,
            )


            rgba_image = (
                Image.fromarray(
                    rgba,
                    mode="RGBA",
                )
            )


            image_x = (
                tile_x
                + (
                    tile_w
                    - shared_w
                )
                // 2
            )


            canvas.alpha_composite(
                rgba_image,
                (
                    image_x,
                    tile_y,
                ),
            )


            # ----------------------------------------------------------
            # COMPACT CELL ID
            # ----------------------------------------------------------

            caption = _short_cell_id(
                str(
                    row[
                        "cell"
                    ][
                        "cell_id"
                    ]
                )
            )


            bbox = draw.textbbox(
                (
                    0,
                    0,
                ),
                caption,
                font=cell_font,
            )


            caption_width = (
                bbox[
                    2
                ]
                - bbox[
                    0
                ]
            )


            caption_x = (
                tile_x
                + (
                    tile_w
                    - caption_width
                )
                // 2
            )


            caption_y = (
                tile_y
                + shared_h
                + 5
            )


            draw.text(
                (
                    caption_x,
                    caption_y,
                ),
                caption,
                fill=(
                    *color,
                    255,
                ),
                font=cell_font,
            )


        nrows = math.ceil(
            len(
                rows
            )
            / ncols
        )


        block_height = (
            cluster_title_h
            + 12
            + nrows
            * tile_h
            + max(
                0,
                nrows - 1,
            )
            * cell_gap_y
        )


        y_cursor += (
            block_height
            + cluster_gap
        )


    save_kwargs = {}

    if dpi is not None:
        dpi = int(
            dpi
        )

        if dpi <= 0:
            raise ValueError(
                "Prototype montage DPI must be positive."
            )

        save_kwargs[
            "dpi"
        ] = (
            dpi,
            dpi,
        )

    canvas.save(
        output_path,
        **save_kwargs,
    )


    return output_path



def get_map_label(
    cell: pd.Series,
) -> int | None:
    """
    Return the canonical instance-map identity for one cell.

    map_label is the identity stored directly in MorphoGlia's canonical
    instance map. No bbox matching or connected-component reconstruction
    is performed.
    """

    if (
        "map_label" not in cell.index
        or pd.isna(
            cell[
                "map_label"
            ]
        )
    ):

        return None


    map_label = int(
        cell[
            "map_label"
        ]
    )


    if map_label <= 0:

        return None


    return map_label


def extract_component_crop(
    cell: pd.Series,
    image_cache: Mapping,
) -> np.ndarray | None:
    """
    Extract one exact instance at native pixel scale.

    Identity is determined exclusively by map_label.

    No padding.
    No resizing.

    Returns a uint8 binary crop with values {0, 255}.
    """

    image_id = str(
        cell[
            "image_id"
        ]
    )


    cached = image_cache[
        image_id
    ]


    map_label = get_map_label(
        cell
    )


    if map_label is None:

        return None


    x = int(
        cell[
            "bbox_x"
        ]
    )

    y = int(
        cell[
            "bbox_y"
        ]
    )

    w = int(
        cell[
            "bbox_w"
        ]
    )

    h = int(
        cell[
            "bbox_h"
        ]
    )


    crop_labels = (
        cached[
            "instance_map"
        ][
            y:y + h,
            x:x + w,
        ]
    )


    instance_crop = (
        crop_labels
        == map_label
    ).astype(
        np.uint8
    ) * 255


    if not np.any(
        instance_crop
    ):

        return None


    return instance_crop


def center_component_on_canvas(
    component_crop: np.ndarray,
    canvas_height: int,
    canvas_width: int,
) -> np.ndarray:
    """
    Center a native-scale component on a shared blank canvas.

    The component is NEVER resized.

    Therefore every cell in the same montage remains directly
    comparable in pixel magnitude.
    """

    height, width = (
        component_crop.shape
    )


    if (
        height > canvas_height
        or width > canvas_width
    ):

        raise ValueError(
            "Component is larger than "
            "the shared montage canvas."
        )


    canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
        ),
        dtype=np.uint8,
    )


    y0 = (
        canvas_height
        - height
    ) // 2

    x0 = (
        canvas_width
        - width
    ) // 2


    canvas[
        y0:y0 + height,
        x0:x0 + width,
    ] = component_crop


    return canvas


def _format_cell_title(
    cell: pd.Series,
    label_columns: Sequence[str],
) -> str:
    """
    Build a compact title without hard-coding a biological schema.

    This keeps Mapping portable between astrocytes, microglia, etc.
    """

    cell_id = str(
        cell[
            "cell_id"
        ]
    )


    short_cell_id = (
        cell_id.split(
            "_cell_"
        )[-1]
    )


    lines = [
        f"Cell {short_cell_id}"
    ]


    metadata = [
        str(
            cell[
                column
            ]
        )
        for column in label_columns
        if (
            column in cell.index
            and pd.notna(
                cell[
                    column
                ]
            )
        )
    ]


    if metadata:

        lines.append(
            " ".join(
                metadata
            )
        )


    selection_type = str(
        cell.get(
            "selection_type",
            "",
        )
    )


    if (
        selection_type
        == "prototype"
        and "prototype_distance"
        in cell.index
    ):

        lines.append(
            "D="
            f"{cell['prototype_distance']:.3f}"
        )


    elif (
        "consensus_reliability"
        in cell.index
        and "cluster_probability"
        in cell.index
    ):

        lines.append(
            "R="
            f"{cell['consensus_reliability']:.3f}  "
            "P="
            f"{cell['cluster_probability']:.3f}"
        )


    return "\n".join(
        lines
    )

