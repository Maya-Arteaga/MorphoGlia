from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

import pandas as pd

from ..core.models import (
    DatasetIndex,
)

from ..morphometrics.builtin import (
    add_global_centroids,
)


# ======================================================================
# CANONICAL MORPHOMETRIC FEATURE ORDER
# ======================================================================

FEATURE_ORDER: List[str] = [
    # Cell (10)
    "Cell_area","Cell_perimeter","Cell_circularity","Cell_compactness","Cell_orientation",
    "Cell_feret_diameter","Cell_eccentricity","Cell_aspect_ratio","Cell_solidity","Cell_convexity",

    # Soma (8)
    "Soma_area","Soma_perimeter","Soma_circularity","Soma_compactness",
    "Soma_orientation","Soma_feret_diameter","Soma_eccentricity","Soma_aspect_ratio",

    # Branching — core areas & shapes
    "Branches_area",
    "Primary_Branches_area","Intermediate_Branches_area","Terminal_Branches_area",
    
    "Primary_Area_Mean","Intermediate_Area_Mean","Terminal_Area_Mean",
    "Primary_Area_Median","Intermediate_Area_Median","Terminal_Area_Median",
    
    "Primary_Area_Fraction","Intermediate_Area_Fraction","Terminal_Area_Fraction",
    
    "Primary_Area_NormCell","Intermediate_Area_NormCell","Terminal_Area_NormCell",
    
    "Primary_Thickness","Intermediate_Thickness","Terminal_Thickness",
    
    "Primary_Branches_aspect_ratio","Intermediate_Branches_aspect_ratio","Terminal_Branches_aspect_ratio",

    # Counts + compact distribution descriptors
    "Primary_Branch_Count","Intermediate_Branch_Count","Terminal_Branch_Count",
    "Percentiles_Primary_Branch_Area","Percentiles_Intermediate_Branch_Area","Percentiles_Terminal_Branch_Area",
    "Percentiles_Primary_Branch_AR","Percentiles_Intermediate_Branch_AR","Percentiles_Terminal_Branch_AR",

    # Tortuosity & legacy skeleton counts
    "Tortuosity_median","Tortuosity_mad",
    "End_Points","Junctions","Branches","Initial_Points","Total_Branches_Length","ratio_branches",



    # Convex hull + fractal
    "Convex_Hull_area","Convex_Hull_perimeter","Convex_Hull_compactness","Fractal_dimension",
    "Convex_Hull_eccentricity","Convex_Hull_feret_diameter","Convex_Hull_orientation",

    # Sholl
    "Sholl_max_distance","Sholl_crossing_processes","Sholl_circles","Sholl_ring_counts","Sholl_ring_radii",
    
    # Centroids (GLOBAL; derived from *_local + bbox)
    "Soma_centroid","Soma_centroid_x","Soma_centroid_y",
    "CH_centroid","CH_centroid_x","CH_centroid_y",
]


# ======================================================================
# MASTER TABLE CONSTRUCTION
# ======================================================================

class MasterTableBuilder:
    """
    Converts DatasetIndex into a master DataFrame containing:
      1) Nomenclature metadata parsed from image filename (group, region, replicate, ...)
      2) Core cell metadata: ID, bbox, center
      3) Raw morphological features
      4) Derived fields (e.g., category = group_region)
    """

    CORE_COLS = ["cell_id","image_id","map_label","source_label","bbox_x","bbox_y","bbox_w","bbox_h","center_x","center_y"]

    def build(
        self,
        dataset: DatasetIndex,
        feature_order: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        # Get metadata keys from the first image (if present), accept metadata or meta
        first_image = next(iter(getattr(dataset, "images", {}).values()), None)
        _meta_head = {}
        if first_image is not None:
            if hasattr(first_image, "metadata") and isinstance(first_image.metadata, dict):
                _meta_head.update(first_image.metadata)
            if hasattr(first_image, "meta") and isinstance(first_image.meta, dict):
                _meta_head.update(first_image.meta)
        meta_keys = list(_meta_head.keys())

        for c in dataset.iter_cells():
            row: Dict[str, Any] = {}

            # 1) Metadata: prefer cell.metadata, else fall back to image record
            meta = {}
            if hasattr(c, "metadata") and isinstance(c.metadata, dict):
                meta.update(c.metadata)
            else:
                try:
                    img_rec = dataset.get_image(getattr(c, "image_id", None))
                    if hasattr(img_rec, "metadata") and isinstance(img_rec.metadata, dict):
                        meta.update(img_rec.metadata)
                    if hasattr(img_rec, "meta") and isinstance(img_rec.meta, dict):
                        meta.update(img_rec.meta)
                except Exception:
                    pass
            row.update(meta)

            # 2) Core cell-level information
            bbox = getattr(c, "bbox", (None, None, None, None)) or (None, None, None, None)
            center = getattr(c, "center", (None, None)) or (None, None)
            row.update({
                "cell_id": getattr(c, "cell_id", None),
                "image_id": getattr(c, "image_id", None),

                "map_label": getattr(
                    c,
                    "map_label",
                    None,
                ),

                "source_label": getattr(
                    c,
                    "source_label",
                    None,
                ),
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
                "center_x": center[0],
                "center_y": center[1],
            })

            # 3) Raw morphological features
            raw = getattr(c, "raw_features", None)
            if isinstance(raw, dict) and raw:
                row.update(raw)
                # Convert ROI-local centroids to GLOBAL coords using bbox origin
                add_global_centroids(c, row)

            # >>> THIS WAS MISSING <<<
            rows.append(row)

        df = pd.DataFrame(rows)

        # 4) Derived columns
        if "group" in df.columns and "region" in df.columns:
            df["category"] = df["group"].astype(str) + "_" + df["region"].astype(str)
        else:
            df["category"] = "undefined"

        # 5) Reorder columns
        feature_order = list(feature_order) if feature_order is not None else FEATURE_ORDER
        head = [k for k in meta_keys if k in df.columns] + ["category"] + self.CORE_COLS

        feature_cols_present = [c for c in feature_order if c in df.columns]
        other_cols = [c for c in df.columns if c not in set(head + feature_cols_present)]

        head_existing = [c for c in head if c in df.columns]
        df = df[head_existing + feature_cols_present + other_cols]

        return df


__all__ = [
    "FEATURE_ORDER",
    "MasterTableBuilder",
]
