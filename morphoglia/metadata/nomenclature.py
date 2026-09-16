# morphoglia/metadata/nomenclature.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nomenclature handling for MorphoGlia filenames.

Two strategies:
1) Key–value mode (preferred): e.g., "group=SS__region=CA1__replicate=R2.tif"
2) Positional fallback:       e.g., "CTRL_CA1_T1_R1.tif"
   Order is configurable via `fallback_keys` (default: ["group","replicate","tissue","region"]).
   In MorphoGlia we standardize this to ["group","region","tissue","replicate"].

Validation:
- A NomenclatureSchema can enforce required keys, allowed values, and regex patterns.
- In strict mode it raises; otherwise, warnings are attached to the returned meta.

Improvements vs. the original:
- Added schema and extractor summaries for logging/printing.
- Added default schema factory for the order group_region_tissue_replicate.
- More robust casting (bool/int/float), duplicate key handling, and parse mode tag.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Set, Pattern, Iterable, Optional
from pathlib import Path
import re


# ─────────────────────────────────────────────────────────────
# SCHEMA CLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class NomenclatureSchema:
    """
    Defines rules for validating metadata parsed from filenames.
    """
    required: Set[str] = field(default_factory=set)
    optional: Set[str] = field(default_factory=set)
    allowed_values: Dict[str, Set[str]] = field(default_factory=dict)
    patterns: Dict[str, Pattern[str]] = field(default_factory=dict)
    strict: bool = False

    def validate(self, meta: Dict[str, Any]) -> list[str]:
        warnings: list[str] = []

        missing = self.required - meta.keys()
        if missing:
            msg = f"Missing keys: {sorted(missing)}"
            if self.strict:
                raise ValueError(msg)
            warnings.append(msg)

        for k, allowed in self.allowed_values.items():
            if k in meta and allowed and str(meta[k]) not in allowed:
                warnings.append(f"Value '{meta[k]}' not allowed for key '{k}'")

        for k, pat in self.patterns.items():
            if k in meta and meta[k] is not None and not pat.match(str(meta[k])):
                warnings.append(f"Value '{meta[k]}' fails pattern for '{k}' (/{pat.pattern}/)")

        return warnings

    # NEW: human-readable description for logs/printing
    def summary(self) -> str:
        lines = [
            "NomenclatureSchema(",
            f"  strict={self.strict},",
            f"  required={sorted(self.required)}",
        ]
        if self.optional:
            lines.append(f"  optional={sorted(self.optional)}")
        if self.allowed_values:
            allowed_preview = {k: (f"{len(v)} allowed" if len(v) > 6 else sorted(v))
                               for k, v in self.allowed_values.items()}
            lines.append(f"  allowed_values={allowed_preview}")
        if self.patterns:
            pat_str = {k: p.pattern for k, p in self.patterns.items()}
            lines.append(f"  patterns={pat_str}")
        lines.append(")")
        return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# MAIN EXTRACTOR CLASS
# ─────────────────────────────────────────────────────────────

class NomenclatureExtractor:
    """
    Extracts metadata from filenames using key-value pairs or positional fallback.

    Examples:
        group=SS__region=CA1__replicate=R2.tif
        CTRL_CA1_T1_R1.tif  →  {'group': 'CTRL', 'region': 'CA1', 'tissue': 'T1', 'replicate': 'R1'}
    """
    def __init__(
        self,
        schema: Optional[NomenclatureSchema] = None,
        *,
        pair_sep: str = "__",
        kv_sep: str = "=",
        fallback_keys: Optional[Iterable[str]] = None,
    ):
        self.schema = schema
        self.pair_sep = pair_sep
        self.kv_sep = kv_sep
        # Default remains backward compatible but you will override below
        self.fallback_keys = list(fallback_keys or ["group", "replicate", "tissue", "region"])

    # NEW: quick config + schema overview
    def summary(self) -> str:
        lines = [
            "NomenclatureExtractor(",
            f"  pair_sep='{self.pair_sep}', kv_sep='{self.kv_sep}',",
            f"  fallback_keys={self.fallback_keys},",
            f"  schema={'set' if self.schema else 'None'}",
        ]
        if self.schema:
            lines.append("  └─ " + self.schema.summary().replace("\n", "\n     "))
        lines.append(")")
        return "\n".join(lines)

    def parse(self, path: Path | str) -> dict:
        p = Path(path)
        stem = p.stem

        # First try key-value parsing
        segments = [s for s in stem.split(self.pair_sep) if self.kv_sep in s]
        if segments:
            meta = self._parse_key_value(segments)
            meta["_nomenclature_mode"] = "kv"
        else:
            meta = self._parse_positional(stem)
            meta["_nomenclature_mode"] = "positional"

        # Validate if schema is provided
        warnings = self.schema.validate(meta) if self.schema else []
        if warnings:
            meta["_nomenclature_warnings"] = warnings
        return meta

    def _parse_key_value(self, segments: list[str]) -> dict:
        meta: Dict[str, Any] = {}
        for seg in segments:
            k, v = seg.split(self.kv_sep, 1)
            k = k.strip()
            v = v.strip()
            # last-one-wins if duplicate keys appear
            meta[k] = self._cast(v)
        return meta

    def _parse_positional(self, stem: str) -> dict:
        parts = [p for p in stem.split("_") if p != ""]
        meta: Dict[str, Any] = {}
        for key, val in zip(self.fallback_keys, parts):
            meta[key] = self._cast(val)
        return meta

    @staticmethod
    def _cast(v: str):
        s = v.strip()
        # bools first
        lower = s.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        # int → float → str
        for caster in (int, float):
            try:
                return caster(s)
            except Exception:
                pass
        return s

# ─────────────────────────────────────────────────────────────
# SCHEMA FACTORIES (your standardized order)
# ─────────────────────────────────────────────────────────────

# Patterns are permissive but useful: tweak as needed.
_REPLICATE_RE = re.compile(r"^R\d+$")      # R1, R2, ...
_TISSUE_RE   = re.compile(r"^T\d+$")      # T1, T2, ...
_REGION_RE   = re.compile(r"^[A-Za-z0-9]+$")  # CA1, CA3, SUB, DG, etc.



def make_custom_order_extractor(order: list[str], *, strict: bool = False,
                                pair_sep: str = "__", kv_sep: str = "="):
    """
    Build an extractor with a user-provided nomenclature order.

    Example:
        order = ["group", "replicate", "tissue", "region"]
        filename = "3xTg-SS_CA1_T1_R1.tif"
        → {"group": "3xTg-SS", "replicate": "CA1", "tissue": "T1", "region": "R1"}
    """

    # Normalize order and filter out ignored slots
    clean_order = [k for k in order if k.upper() != "X"]

    class CustomExtractor:
        def __init__(self, order: list[str]):
            self._user_order = order              # full user order (with X’s if any)
            self.fallback_keys = clean_order      # used for printing
            self.strict = strict
            self.pair_sep = pair_sep
            self.kv_sep = kv_sep

        def parse(self, path: str | Path) -> dict[str, str]:
            """
            Parse filename using the custom positional order.
            """
            stem = Path(path).stem

            # First: key-value mode
            segments = [s for s in stem.split(self.pair_sep) if self.kv_sep in s]
            if segments:
                meta = {}
                for seg in segments:
                    k, v = seg.split(self.kv_sep, 1)
                    meta[k.strip()] = v.strip()
                meta["_nomenclature_mode"] = "kv"
                return meta

            # Otherwise: positional mode with custom order
            parts = [p for p in stem.split("_") if p != ""]
            meta = {}
            for idx, key in enumerate(self._user_order):
                if key.upper() == "X":
                    continue
                if idx < len(parts):
                    meta[key] = parts[idx]
            meta["_nomenclature_mode"] = "custom"
            return meta

    return CustomExtractor(order)







def default_schema_group_region_tissue_replicate(
    *,
    strict: bool = False,
    allowed_groups: Optional[Set[str]] = None,
    allowed_regions: Optional[Set[str]] = None,
) -> NomenclatureSchema:
    """
    Enforce positional order group_region_tissue_replicate and basic validity.
    - Requires all four keys.
    - Optionally restricts 'group' and/or 'region' to allowed sets.
    """
    patterns = {
        "replicate": _REPLICATE_RE,
        "tissue": _TISSUE_RE,
        "region": _REGION_RE,
    }
    allowed_values: Dict[str, Set[str]] = {}
    if allowed_groups:
        allowed_values["group"] = allowed_groups
    if allowed_regions:
        allowed_values["region"] = allowed_regions

    return NomenclatureSchema(
        strict=strict,
        required={"group", "region", "tissue", "replicate"},
        optional=set(),
        allowed_values=allowed_values,
        patterns=patterns,
    )

def make_group_region_tissue_replicate_extractor(
    *,
    strict: bool = False,
    allowed_groups: Optional[Set[str]] = None,
    allowed_regions: Optional[Set[str]] = None,
    pair_sep: str = "__",
    kv_sep: str = "=",
) -> NomenclatureExtractor:
    """
    Convenience factory that returns an extractor configured for the
    `group_region_tissue_replicate` order with a matching schema.
    """
    schema = default_schema_group_region_tissue_replicate(
        strict=strict,
        allowed_groups=allowed_groups,
        allowed_regions=allowed_regions,
    )
    return NomenclatureExtractor(
        schema=schema,
        pair_sep=pair_sep,
        kv_sep=kv_sep,
        fallback_keys=["group", "region", "tissue", "replicate"],
    )
