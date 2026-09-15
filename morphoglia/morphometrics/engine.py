# morphoglia/morphometrics/engine.py
from __future__ import annotations
from typing import Dict, List
import numpy as np

class MorphometricEngine:
    """
    Applies a list of calculator instances to ROIs.
    Each calculator is a callable(roi) -> dict[str, float].
    """
    def __init__(self, calculators: List):
        self.calculators = calculators

    def compute(self, roi: np.ndarray) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        for calc in self.calculators:
            feats.update(calc(roi))
        return feats

    def compute_per_calc(self, roi: np.ndarray) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for calc in self.calculators:
            out[calc.__class__.__name__] = calc(roi)
        return out
