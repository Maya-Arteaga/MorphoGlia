#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 13:55:27 2025

@author: juanpablomayaarteaga
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

BBox = Tuple[int, int, int, int]  # (x, y, w, h)

@dataclass
class ImageRecord:
    image_id: str
    path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    size: Tuple[int, int] | None = None  # (h, w)

@dataclass
class CellRecord:
    cell_id: str
    image_id: str
    bbox: BBox
    center: Tuple[int, int]

    # Metadata inherited from the parent image.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Canonical object identity.
    #
    # map_label
    #     Integer stored in MorphoGlia's canonical instance map.
    #
    # source_label
    #     Original external instance label when input_mode="labels".
    #     None when MorphoGlia created the instance identity from
    #     binary/raw input.
    #
    # These identities are deliberately separate from cell_id.
    map_label: int | None = None
    source_label: int | None = None

    # Pixel/voxel count of this instance.
    # Used by Object QC.
    component_area: int | None = None


    crop_path: Path | None = None
    crop_cached: bool = False
    raw_features: Dict[str, float] | None = None
    embedding: np.ndarray | None = None
    cluster: int | None = None


class DatasetIndex:
    """
    Simple in-memory containers for records; can be replaced by a DB or parquet later.
    """
    def __init__(self):
        self.images: Dict[str, ImageRecord] = {}
        self.cells: Dict[str, CellRecord] = {}

    def add_image(self, record: ImageRecord):
        self.images[record.image_id] = record

    def add_cells(self, cell_records: List[CellRecord]):
        for c in cell_records:
            self.cells[c.cell_id] = c

    def iter_cells(self):
        return self.cells.values()

    def get_image(self, image_id: str) -> ImageRecord:
        return self.images[image_id]
