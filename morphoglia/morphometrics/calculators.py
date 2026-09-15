# morphoglia/morphometrics/calculators.py
# -*- coding: utf-8 -*-
"""
Calculator registry for morphometric calculators.

What this provides
------------------
1) A global REGISTRY mapping short names → classes.
2) A @register decorator to register classes under short names.
3) Helpers to instantiate calculators from config:
   - short names (e.g., ["cell", "soma"])
   - dict specs with {"name": "cell", "kwargs": {...}}
   - dict specs with {"class": "module.ClassName", "kwargs": {...}}

This enables YAML like:

  components:
    calculators:
      - cell
      - soma
      - {name: fractal, kwargs: {sizes: [16, 32, 64]}}
      - {class: morphoglia.morphometrics.ConvexHullMorphometrics, kwargs: {}}

to work without touching code.

Notes
-----
- We also eagerly pre-register your built-in calculators from morphoglia/morphometrics.py
  so they’re available by short name right away.
"""

# morphoglia/morphometrics/calculators.py
from __future__ import annotations
from typing import Any, Dict, List, Type


import importlib

# Public registry: short-name -> class
REGISTRY: Dict[str, Type] = {}

# ======================================================================
# DYNAMIC CALCULATOR LOADING
# ======================================================================

def _load_obj(
    qualname: str,
):
    """
    Resolve a fully-qualified Python object path.

    This is intentionally local to Morphometrics because dynamic loading
    is currently used only for configurable morphometric calculators.
    """

    module, separator, name = (
        qualname.rpartition(
            "."
        )
    )

    if (
        not module
        or not separator
        or not name
    ):

        raise ValueError(
            f"Invalid qualified name '{qualname}'. "
            "Expected format like "
            "'package.module.ClassName'."
        )

    imported_module = (
        importlib.import_module(
            module
        )
    )

    try:

        return getattr(
            imported_module,
            name,
        )

    except AttributeError as exc:

        raise ImportError(
            f"'{name}' not found in module "
            f"'{module}'."
        ) from exc


def register(name: str):
    """
    Decorator to register a calculator under a short name.
    Usage:
        @register("cell")
        class CellMorphometrics(...): ...
    """
    def _wrap(cls: Type):
        REGISTRY[name] = cls
        return cls
    return _wrap


# ---- built-in registrations (mapping to legacy implementations) ----
def _register_legacy_defaults():
    from .builtin import (
        CellMorphometrics,
        SomaMorphometrics,
        ConvexHullMorphometrics,
        FractalMorphometrics,
        ShollMorphometrics,
        BranchingMorphometrics,
        BranchOrderDistributionMorphometrics,  
        CentroidMorphometrics,   
    )

    REGISTRY.setdefault("cell", CellMorphometrics)
    REGISTRY.setdefault("soma", SomaMorphometrics)
    REGISTRY.setdefault("convex_hull", ConvexHullMorphometrics)
    REGISTRY.setdefault("fractal", FractalMorphometrics)
    REGISTRY.setdefault("sholl", ShollMorphometrics)
    REGISTRY.setdefault("branches", BranchingMorphometrics)
    REGISTRY.setdefault("branch_order_dist", BranchOrderDistributionMorphometrics)  
    REGISTRY.setdefault("centroids", CentroidMorphometrics)
 

# call on import so the names exist immediately
_register_legacy_defaults()


# ---- builder ----
def build_calculators_from_config(specs: List[Any]):
    """
    Accepts a list like:
      - "cell"
      - {"name": "fractal", "kwargs": {"sizes": [16, 32, 64]}}
      - {"class": "package.module.ClassName", "kwargs": {...}}
      - fully qualified class string "package.module.ClassName"

    Returns: list of instantiated calculator objects.
    """
    out = []
    for spec in specs:
        # short-name string
        if isinstance(spec, str):
            try:
                cls = REGISTRY[spec]
            except KeyError:
                raise ValueError(f"Unknown calculator short-name: {spec}. Known: {list(REGISTRY.keys())}")
            out.append(cls())
            continue

        # dict with explicit class path
        if isinstance(spec, dict) and "class" in spec:
            cls = _load_obj(spec["class"])
            kwargs = spec.get("kwargs", {}) or {}
            out.append(cls(**kwargs))
            continue

        # dict with short-name + kwargs
        if isinstance(spec, dict) and "name" in spec:
            name = spec["name"]
            kwargs = spec.get("kwargs", {}) or {}
            try:
                cls = REGISTRY[name]
            except KeyError:
                raise ValueError(f"Unknown calculator short-name: {name}. Known: {list(REGISTRY.keys())}")
            out.append(cls(**kwargs))
            continue

        raise ValueError(f"Unsupported calculator spec: {spec!r}")

    return out
