from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import filters
from .filter_parameters import parameters_for


@dataclass(frozen=True)
class OperationSpec:
    function: Callable
    framewise: bool = True


FILTER_REGISTRY: dict[str, OperationSpec] = {
    "FLIP_VERTICAL": OperationSpec(filters.flip_vertical, framewise=False),

    "TOPHAT": OperationSpec(filters.tophat),
    "MULTISCALE_TOPHAT": OperationSpec(filters.multiscale_tophat),
    "MULTISCALE2_TOPHAT": OperationSpec(filters.multiscale_tophat),
    "GAUSSIAN_SUBTRACTION": OperationSpec(filters.gaussian_subtraction),

    "CLAHE": OperationSpec(filters.clahe),
    "HISTOGRAM_EQ": OperationSpec(filters.histogram_equalization),
    "ADAPT_HIST_EQ": OperationSpec(filters.adaptive_histogram_equalization),
    "GAMMA_CORRECTION": OperationSpec(filters.gamma_correction),
    "LOG_TRANSFORMATION": OperationSpec(filters.log_transformation),
    "MIN_MAX_NORMALIZATION": OperationSpec(filters.min_max_normalization),
    "NORMALIZE": OperationSpec(filters.normalize),
    "CONVERT_TO_8BIT": OperationSpec(filters.convert_to_8bit),
    "CONVERT_GRAY_SCALE": OperationSpec(filters.convert_gray_scale, framewise=False),

    "ANISOTROPIC_DIFFUSION": OperationSpec(filters.anisotropic_diffusion),
    "FAST_NON_LOCAL_MEANS": OperationSpec(filters.fast_non_local_means),
    "UNSHARP_MASK": OperationSpec(filters.unsharp_mask),

    "FIXED_THRESHOLD": OperationSpec(filters.fixed_threshold),
    "PERCENTILE_THRESHOLD": OperationSpec(filters.percentile_threshold),
    "MEDIAN_THRESHOLD": OperationSpec(filters.median_threshold),
    "MEAN_THRESHOLD": OperationSpec(filters.mean_threshold),
    "ADAPTIVE_THRESHOLD": OperationSpec(filters.adaptive_threshold),
    "ADAPTATIVE_THRESHOLD": OperationSpec(filters.adaptive_threshold),

    "REMOVE_SMALL_NOISE": OperationSpec(filters.remove_small_noise),
    "REMOVE_LARGE_OBJECTS": OperationSpec(filters.remove_large_objects),
    "SOMA": OperationSpec(filters.soma),
    "SKELETONIZE": OperationSpec(filters.skeletonize_binary),
    "CANNY_EDGE_DETECTION": OperationSpec(filters.canny_edge_detection),

    "ELIMINATE_1_VALUE": OperationSpec(filters.eliminate_1_value),
    "ELIMINATE_N_VALUE": OperationSpec(filters.eliminate_below_value),
    "ELIMINATE_N_VALUE2": OperationSpec(filters.eliminate_below_value),
    "ELIMINATE_N_VALUE3": OperationSpec(filters.eliminate_below_value),
    
    "MEDIAN_FILTER":OperationSpec(filters.median_filter,),

    "PERCENTILE_HYSTERESIS_THRESHOLD": OperationSpec(filters.percentile_hysteresis_threshold,),
    
    "FILL_SMALL_HOLES":OperationSpec(filters.fill_small_holes,),
    
    "ROBUST_RESCALE": OperationSpec(filters.robust_rescale),
    
    "HUANG_THRESHOLD": OperationSpec(filters.huang_threshold),
    "BRANCH_GEOMETRY_CLEANUP": OperationSpec(
        filters.branch_geometry_cleanup
    ),
    "LI_THRESHOLD": OperationSpec(filters.li_threshold),
    
    "DIRECTIONAL_GRAYSCALE_OPENING": OperationSpec(
        filters.directional_grayscale_opening
    ),
    
    "REMOVE_CIRCULAR_NOISE": OperationSpec(
        filters.remove_circular_noise
    ),
    
    
    
    "BRANCH_AWARE_CONSENSUS_THRESHOLD": OperationSpec(
        filters.branch_aware_consensus_threshold
    ),
        
    
    "BACKGROUND_PERCENTILE_SUBTRACTION": OperationSpec(filters.background_percentile_subtraction),
    
    
    "REMOVE_COMPACT_OBJECTS": OperationSpec(
        filters.remove_compact_objects
    ),
    

    "PREP_FOR_CELLPOSE": OperationSpec(filters.prep_for_cellpose),
}


# The supplied legacy method COLORIZE_SKELETON_SOMA had only ``pass`` and
# therefore is not represented as a working preprocessing operation.
UNSUPPORTED_LEGACY_OPERATIONS = {
    "COLORIZE_SKELETON_SOMA",
}


# Preserve the current public FluorescenceV1Config controls exactly for the
# parameters they already expose. Everything else uses filter_parameters.py.
_FLUORESCENCE_V1_CONFIG_OVERRIDES = {
    "MULTISCALE_TOPHAT": {
        "radii": ("effective_top_hat_radii", "top_hat_radii"),
    },
    "CLAHE": {
        "clip_limit": ("effective_clahe_clip_limit", "clahe_clip_limit"),
        "tile_grid_size": (
            "effective_clahe_tile_grid_size",
            "clahe_tile_grid_size",
        ),
    },
    "UNSHARP_MASK": {
        "radius": ("effective_unsharp_radius", "unsharp_radius"),
        "amount": ("effective_unsharp_amount", "unsharp_amount"),
    },
    "PERCENTILE_THRESHOLD": {
        "percentile": ("effective_percentile", "percentile"),
    },
    "REMOVE_SMALL_NOISE": {
        "min_area": ("effective_min_area", "min_area"),
    },
}


def _apply_framewise(
    image: np.ndarray,
    function,
    **kwargs,
) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim == 2:
        return function(image, **kwargs)

    if image.ndim == 3:
        return np.stack(
            [
                function(frame, **kwargs)
                for frame in image
            ],
            axis=0,
        )

    raise ValueError(
        "Framewise preprocessing expects a 2D image or "
        "a 3D stack with shape (T,H,W)."
    )


def available_operations() -> tuple[str, ...]:
    return tuple(FILTER_REGISTRY)


def _operation_parameters(
    operation: str,
    config,
    preset: str,
) -> tuple[dict[str, object], dict[str, str]]:
    key = str(operation).strip().upper()
    parameters = parameters_for(key, preset=preset)
    sources = {name: "default" for name in parameters}

    if preset in {"mg1", "fluorescence_v1"} and hasattr(config, "fluorescence_v1"):
        preset_config = config.fluorescence_v1

        for parameter, (effective_attr, raw_attr) in (
            _FLUORESCENCE_V1_CONFIG_OVERRIDES.get(key, {}).items()
        ):
            # filter_parameters.py is the authoritative developer-level
            # default. The public config overrides it only when the
            # researcher explicitly supplied a value.
            raw_value = getattr(
                preset_config,
                raw_attr,
                None,
            )

            if raw_value is not None:
                parameters[parameter] = getattr(
                    preset_config,
                    effective_attr,
                )
                sources[parameter] = "user"

    return parameters, sources


def _canonical_binary_output(
    image: np.ndarray,
    *,
    preset: str,
    last_operation: str,
) -> np.ndarray:
    """Enforce the raw-preprocessing contract: final output is binary."""
    image = np.asarray(image)

    if image.size == 0:
        raise ValueError(f"Raw preset {preset!r} produced an empty image.")

    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255

    values = image.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(
            f"Raw preset {preset!r} produced NaN or infinite values."
        )

    unique = np.unique(values)
    if len(unique) > 2:
        raise ValueError(
            f"Raw preprocessing preset {preset!r} does not end in a binary "
            f"image. Last operation: {last_operation}. Add a threshold or "
            "other binarizing operation near the end of the sequence."
        )

    if len(unique) == 1:
        return np.full(
            image.shape,
            255 if float(unique[0]) > 0 else 0,
            dtype=np.uint8,
        )

    threshold = (float(unique[0]) + float(unique[-1])) / 2.0
    return (values > threshold).astype(np.uint8) * 255


def run_sequence(
    image: np.ndarray,
    *,
    sequence,
    config,
    preset: str,
    collect_intermediate: bool = False,
    require_binary_output: bool = True,
):
    sequence = tuple(
        str(operation).strip().upper()
        for operation in sequence
    )

    if not sequence:
        raise ValueError(f"Raw preprocessing preset {preset!r} has an empty sequence.")

    current = np.asarray(image)
    intermediate = []

    for operation in sequence:
        if operation in UNSUPPORTED_LEGACY_OPERATIONS:
            raise ValueError(
                f"{operation} cannot be used because the supplied legacy "
                "implementation had no operation body."
            )

        if operation not in FILTER_REGISTRY:
            raise ValueError(
                f"Unknown preprocessing operation: {operation!r}. "
                "Available operations: " + ", ".join(available_operations()) + "."
            )

        spec = FILTER_REGISTRY[operation]
        parameters, _ = _operation_parameters(operation, config, preset)

        if spec.framewise:
            current = _apply_framewise(
                current,
                spec.function,
                **parameters,
            )
        else:
            current = spec.function(current, **parameters)

        current = np.asarray(current)

        
        if collect_intermediate:
            intermediate.append(
                (
                    operation,
                    current.copy(),
                )
            )

    if require_binary_output:
        current = _canonical_binary_output(
            current,
            preset=preset,
            last_operation=sequence[-1],
        )

    if collect_intermediate:
        return current, intermediate

    return current


def process_image(
    image: np.ndarray,
    config,
    collect_intermediate: bool = False,
):
    """Execute the active declarative raw-preprocessing sequence."""
    preset = config.effective_preset

    if preset is None:
        raise ValueError("process_image() applies only to raw preprocessing.")

    return run_sequence(
        image,
        sequence=config.sequence,
        config=config,
        preset=preset,
        collect_intermediate=collect_intermediate,
        require_binary_output=True,
    )


def active_parameter_rows(config) -> list[dict[str, object]]:
    """Technical_Record rows for the actual active sequence and its values."""
    preset = config.effective_preset
    if preset is None:
        return []

    sequence = tuple(config.sequence)
    rows: list[dict[str, object]] = [
        {
            "section": "Preprocessing.Preset",
            "parameter": "sequence",
            "value": " -> ".join(sequence),
            "source": "default",
        }
    ]

    for operation in sequence:
        parameters, sources = _operation_parameters(operation, config, preset)
        for parameter, value in parameters.items():
            rows.append(
                {
                    "section": "Preprocessing.Filter",
                    "parameter": f"{operation}.{parameter}",
                    "value": value,
                    "source": sources.get(parameter, "default"),
                }
            )

    return rows


__all__ = [
    "FILTER_REGISTRY",
    "UNSUPPORTED_LEGACY_OPERATIONS",
    "available_operations",
    "run_sequence",
    "process_image",
    "active_parameter_rows",
]
