# morphoglia/io/overlays.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Optional
import cv2
import numpy as np

from matplotlib import colormaps

# BGR colors
YELLOW = (0, 255, 255)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)


def render_instance_map_diagnostic(
    instance_map: np.ndarray,
    output_path: str | Path,
    rejected_labels: set[int] | None = None,
) -> Path:
    """
    Render the canonical segmentation diagnostic.

    Yellow bbox
        Object survives the effective QC policy.

    Red bbox
        Object is actually excluded after applying the user's
        small / large / tubular True-False settings.

    Suspicion alone does not make an object red.
    """

    instance_map = np.asarray(
        instance_map
    )


    if instance_map.ndim != 2:

        raise ValueError(
            "Segmentation diagnostic rendering "
            "requires a 2D instance map."
        )


    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    rejected_labels = {
        int(label)
        for label in (
            rejected_labels
            or set()
        )
    }


    height, width = (
        instance_map.shape
    )


    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )


    labels = np.unique(
        instance_map
    )

    labels = labels[
        labels > 0
    ]


    tab10 = colormaps[
        "tab10"
    ]


    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    font_scale = 0.8

    text_thickness = 2

    normal_box_thickness = 3

    rejected_box_thickness = 6


    for label in labels:

        map_label = int(
            label
        )


        mask = (
            instance_map
            == map_label
        )


        if not np.any(
            mask
        ):

            continue


        # ==============================================================
        # TAB10 OBJECT FILL
        # ==============================================================

        rgb_float = (
            tab10.colors[
                (
                    map_label
                    - 1
                )
                % len(
                    tab10.colors
                )
            ]
        )


        rgb = tuple(
            int(
                round(
                    255.0
                    * channel
                )
            )
            for channel
            in rgb_float[:3]
        )


        bgr = (
            rgb[2],
            rgb[1],
            rgb[0],
        )


        canvas[
            mask
        ] = bgr


        # ==============================================================
        # EXACT BOUNDING BOX
        # ==============================================================

        ys, xs = np.where(
            mask
        )


        x1 = int(
            xs.min()
        )

        x2 = int(
            xs.max()
        )

        y1 = int(
            ys.min()
        )

        y2 = int(
            ys.max()
        )


        # ==============================================================
        # EFFECTIVE QC STATUS
        # ==============================================================

        if (
            map_label
            in rejected_labels
        ):

            bbox_color = RED

            bbox_thickness = (
                rejected_box_thickness
            )

        else:

            bbox_color = GREEN

            bbox_thickness = (
                normal_box_thickness
            )


        cv2.rectangle(
            canvas,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            bbox_color,
            bbox_thickness,
        )


        # ==============================================================
        # MAP LABEL — UPPER-RIGHT INSIDE BBOX
        # ==============================================================

        text = str(
            map_label
        )


        (
            text_width,
            text_height,
        ), _ = cv2.getTextSize(
            text,
            font,
            font_scale,
            text_thickness,
        )


        text_x = max(
            x1 + 1,
            x2
            - text_width
            - 2,
        )


        text_y = min(
            y2 - 1,
            y1
            + text_height
            + 2,
        )


        text_y = max(
            y1 + 1,
            text_y,
        )


        cv2.putText(
            canvas,
            text,
            (
                int(text_x),
                int(text_y),
            ),
            font,
            font_scale,
            RED,
            text_thickness,
            cv2.LINE_AA,
        )


    success = cv2.imwrite(
        str(
            output_path
        ),
        canvas,
    )


    if not success:

        raise RuntimeError(
            "Could not save segmentation "
            f"diagnostic: {output_path}"
        )


    return output_path


class SpatialMapper:
    """
    Overlay generator for QC:
      - overlay_mode: 'bbox' | 'contour' | 'both'  (case-insensitive)
      - reject_visual: 'none' | 'dim' | 'mask'
          'none' → draw overlays, keep full image
          'dim'  → dim rejected regions (fast alpha blend)
          'mask' → zero/recolor rejected regions (keep only accepted)

    Contours:
      - If a CellRecord carries `contour` (global coords, np.ndarray Nx1x2), we draw/use it.
      - Otherwise, we try to read the persisted ROI PNG and re-extract contours,
        shifting them by the cell's bbox back to full-image coordinates.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────
    # helpers
    # ────────────────────────────────────────────────────────

    def _ensure_bgr(self, img: np.ndarray) -> np.ndarray:
        """Make sure the canvas is 3-channel BGR."""
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            return img.copy()
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img.copy()

    def _rect_from_bbox(self, bbox) -> Optional[Tuple[int, int, int, int]]:
        """
        Interprets bbox as (x, y, w, h) — the format produced by cv2.connectedComponentsWithStats.
        Returns (x1, y1, x2, y2) or None if invalid.
        """
        if bbox is None:
            return None
        if isinstance(bbox, dict) and "bbox" in bbox:
            bbox = bbox["bbox"]
        if not hasattr(bbox, "__len__") or len(bbox) != 4:
            return None
    
        x, y, w, h = map(int, bbox)
        x1, y1 = x, y
        x2, y2 = x + w, y + h
    
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2


    def _roi_contours_from_disk(self, cell, x1: int, y1: int):
        """
        Load the persisted ROI PNG for this cell (if it exists), find external contours,
        and shift them to global coordinates by (x1, y1).
        """
        crop_dir = self.out_dir.parent / "QC_segmented_cells" / cell.image_id
        cell_path = crop_dir / f"{cell.cell_id}.png"
        if not cell_path.exists():
            return []

        roi = cv2.imread(str(cell_path), cv2.IMREAD_GRAYSCALE)
        if roi is None:
            return []

        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in cnts:
            cnt = cnt.astype(np.int32)
            cnt[:, 0, 0] += int(x1)
            cnt[:, 0, 1] += int(y1)
            out.append(cnt)
        return out

    # ────────────────────────────────────────────────────────
    # main
    # ────────────────────────────────────────────────────────

    def boundingbox_qc_ids(
        self,
        dataset,
        image_loader,
        overlay_mode: str = "bbox",
        *,
        reject_visual: str = "none",    # 'none' | 'dim' | 'mask'
        dim_factor: float = 0.25,       # for 'dim' (0..1)
        mask_bg_color: Tuple[int,int,int] = (0, 0, 0),  # for 'mask'
    ):
        """
        Create QC overlays: <out_dir>/<image_id>_ID.png (or _ID_dim/_ID_keep for dim/mask)
        """
        # Normalize modes
        mode = (overlay_mode or "bbox").strip().lower()
        if mode not in ("bbox", "contour", "both"):
            print(f"[SpatialMapper] unknown overlay_mode '{overlay_mode}', defaulting to 'bbox'.")
            mode = "bbox"

        rej = (reject_visual or "none").strip().lower()
        if rej not in ("none", "dim", "mask"):
            print(f"[SpatialMapper] unknown reject_visual '{reject_visual}', defaulting to 'none'.")
            rej = "none"

        draw_bbox = mode in ("bbox", "both")
        draw_cnt  = mode in ("contour", "both")

        # Visual style
        BBOX_COLOR, BBOX_THICK = GREEN, 10
        CONTOUR_COLOR, CONTOUR_THICK = YELLOW, 3
        TEXT_COLOR, TEXT_THICK = RED, 2
        FONT, FONT_SCALE = cv2.FONT_HERSHEY_SIMPLEX, 1.0

        # Collect cells by image
        img_to_cells = {}
        for c in dataset.iter_cells():
            img_to_cells.setdefault(c.image_id, []).append(c)

        for img_id, cells in img_to_cells.items():
            img_rec = dataset.get_image(img_id)
            base = image_loader.load(img_rec)
            canvas = self._ensure_bgr(base)

            # Build keep mask (uint8 0/255) of accepted objects to cheaply dim/mask rejects
            H, W = canvas.shape[:2]
            keep_mask = np.zeros((H, W), np.uint8)

            for idx, cell in enumerate(cells, start=1):
                # 0) bbox (robust parsing)
                rect = self._rect_from_bbox(getattr(cell, "bbox", None))
                if rect is None:
                    continue
                x1, y1, x2, y2 = rect
                w, h = x2 - x1, y2 - y1
                
                # Fallback: if bbox is too thin, recompute from contour
                if w <= 5 or h <= 5:
                    cnt = getattr(cell, "contour", None)
                    if isinstance(cnt, np.ndarray) and cnt.ndim == 3 and cnt.shape[-1] == 2 and len(cnt) > 0:
                        cx, cy, cw, ch = cv2.boundingRect(cnt.astype(np.int32))
                        x1, y1, x2, y2 = cx, cy, cx + cw, cy + ch
                        w, h = cw, ch



                # A) keep mask: prefer contour, else fill the bbox
                cnt = getattr(cell, "contour", None)
                if isinstance(cnt, np.ndarray) and cnt.ndim == 3 and cnt.shape[-1] == 2:
                    cv2.drawContours(keep_mask, [cnt.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
                else:
                    cv2.rectangle(keep_mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

            # Apply reject_visual BEFORE drawing overlays (so overlays stay vivid)
            if rej == "dim":
                dimmed = cv2.convertScaleAbs(canvas, alpha=float(dim_factor), beta=0.0)
                # 3-channel mask
                m3 = keep_mask[..., None]
                canvas = np.where(m3 == 255, canvas, dimmed)
                out_name = f"{img_id}_ID_dim.png"
            elif rej == "mask":
                bg = np.zeros_like(canvas)
                bg[:] = mask_bg_color
                m3 = keep_mask[..., None]
                canvas = np.where(m3 == 255, canvas, bg)
                out_name = f"{img_id}_ID_keep.png"
            else:
                out_name = f"{img_id}_ID.png"

            # Now draw overlays (bbox/contour/IDs)
            for idx, cell in enumerate(cells, start=1):
                rect = self._rect_from_bbox(getattr(cell, "bbox", None))
                if rect is None:
                    # If no bbox, try to place ID at center
                    cx, cy = getattr(cell, "center", (None, None))
                    if cx is not None and cy is not None:
                        cv2.putText(canvas, str(idx), (int(cx), int(cy)), FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICK, cv2.LINE_AA)
                    continue

                x1, y1, x2, y2 = rect
                w, h = x2 - x1, y2 - y1

                if draw_bbox:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICK)

                if draw_cnt:
                    cnt = getattr(cell, "contour", None)
                    if isinstance(cnt, np.ndarray) and cnt.ndim == 3 and cnt.shape[-1] == 2:
                        cv2.drawContours(canvas, [cnt.astype(np.int32)], -1, CONTOUR_COLOR, CONTOUR_THICK)
                    else:
                        # fallback to ROI file if available
                        for cnt_disk in self._roi_contours_from_disk(cell, x1, y1):
                            cv2.drawContours(canvas, [cnt_disk], -1, CONTOUR_COLOR, CONTOUR_THICK)

                # ID (top-right of the box)
                label = str(idx)
                text_size, _ = cv2.getTextSize(label, FONT, FONT_SCALE, TEXT_THICK)
                text_x = x1 + w - text_size[0] - 2
                text_y = y1 + text_size[1] + 2
                cv2.putText(canvas, label, (text_x, text_y), FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICK, cv2.LINE_AA)

            cv2.imwrite(str(self.out_dir / out_name), canvas)
