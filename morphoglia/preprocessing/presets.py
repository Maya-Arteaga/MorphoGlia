from __future__ import annotations

"""
MorphoGlia raw-preprocessing presets.

Public presets
--------------
MG1
    Full MorphoGlia fluorescence workflow.

MG2
    Background subtraction + directional grayscale enhancement.

MG3
    Compact Gaussian-background workflow.

MG4
    Compact percentile-background workflow.

Historical ``fluorescence_v1`` remains accepted as an alias of MG1.
"""


MG1_SEQUENCE = (
    "ROBUST_RESCALE",
    "BACKGROUND_PERCENTILE_SUBTRACTION",
    "ROBUST_RESCALE",
    "GAUSSIAN_SUBTRACTION",
    "DIRECTIONAL_GRAYSCALE_OPENING",
    "ROBUST_RESCALE",
    "LI_THRESHOLD",
    "REMOVE_SMALL_NOISE",
    "REMOVE_COMPACT_OBJECTS",
)


MG2_SEQUENCE = (
    "ROBUST_RESCALE",
    "BACKGROUND_PERCENTILE_SUBTRACTION",
    "GAUSSIAN_SUBTRACTION",
    "DIRECTIONAL_GRAYSCALE_OPENING",
    "ROBUST_RESCALE",
    "LI_THRESHOLD",
    "REMOVE_SMALL_NOISE",
)


MG3_SEQUENCE = (
    "ROBUST_RESCALE",
    "GAUSSIAN_SUBTRACTION",
    "LI_THRESHOLD",
    "REMOVE_SMALL_NOISE",
)


MG4_SEQUENCE = (
    "BACKGROUND_PERCENTILE_SUBTRACTION",
    "ROBUST_RESCALE",
    "LI_THRESHOLD",
    "REMOVE_SMALL_NOISE",
)


PRESET_SEQUENCES: dict[str, tuple[str, ...]] = {
    "mg1": MG1_SEQUENCE,
    "mg2": MG2_SEQUENCE,
    "mg3": MG3_SEQUENCE,
    "mg4": MG4_SEQUENCE,
}


PRESET_ALIASES: dict[str, str] = {
    "fluorescence_v1": "mg1",
}


# Backward-compatible import name.
FLUORESCENCE_V1_SEQUENCE = MG1_SEQUENCE


def normalize_preset(
    preset: str,
) -> str:
    key = str(
        preset
    ).strip().lower()

    return PRESET_ALIASES.get(
        key,
        key,
    )


def available_presets() -> tuple[str, ...]:
    return tuple(
        PRESET_SEQUENCES
    )


def display_name_for_preset(
    preset: str,
) -> str:
    return normalize_preset(
        preset
    ).upper()


def sequence_for_preset(
    preset: str,
) -> tuple[str, ...]:

    key = normalize_preset(
        preset
    )

    if key not in PRESET_SEQUENCES:
        raise ValueError(
            f"Unknown preprocessing preset: {preset!r}. "
            "Available presets: "
            + ", ".join(
                available_presets()
            )
            + "."
        )

    return tuple(
        str(operation)
        .strip()
        .upper()

        for operation
        in PRESET_SEQUENCES[
            key
        ]
    )


__all__ = [
    "MG1_SEQUENCE",
    "MG2_SEQUENCE",
    "MG3_SEQUENCE",
    "MG4_SEQUENCE",
    "FLUORESCENCE_V1_SEQUENCE",
    "PRESET_SEQUENCES",
    "PRESET_ALIASES",
    "normalize_preset",
    "available_presets",
    "display_name_for_preset",
    "sequence_for_preset",
]
