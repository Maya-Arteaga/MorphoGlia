from __future__ import annotations

"""
Developer-level defaults for atomic preprocessing operations.

The existing FluorescenceV1Config values still override the matching entries
for the canonical fluorescence_v1 preset. Additional operations can be tuned
here without expanding the normal PipelineConfig surface.
"""


FILTER_PARAMETERS: dict[str, dict[str, object]] = {
    "FLIP_VERTICAL": {},

    "TOPHAT": {"disk_size": 15},
    "MULTISCALE_TOPHAT": {"radii": (6, 15, 35)},
    "MULTISCALE2_TOPHAT": {"radii": (5, 40)},

    "CLAHE": {
        "clip_limit": 0.01,
        "tile_grid_size": (8, 8),
    },
    "HISTOGRAM_EQ": {},
    "ADAPT_HIST_EQ": {"clip_limit": 0.01},
    "GAMMA_CORRECTION": {"gamma": 1.4},
    "LOG_TRANSFORMATION": {},
    "MIN_MAX_NORMALIZATION": {},
    "NORMALIZE": {},
    "CONVERT_TO_8BIT": {},
    "CONVERT_GRAY_SCALE": {},

    "ANISOTROPIC_DIFFUSION": {
        "sigma_s": 5.0,
        "sigma_r": 0.4,
    },
    "FAST_NON_LOCAL_MEANS": {
        "h": 5.0,
        "template_window_size": 5,
        "search_window_size": 30,
    },
    "UNSHARP_MASK": {
        "radius": 9.0,
        "amount": 0.7,
    },

    "FIXED_THRESHOLD": {"threshold": 128.0},
    "PERCENTILE_THRESHOLD": {"percentile": 80.0},
    "MEDIAN_THRESHOLD": {},
    "MEAN_THRESHOLD": {},
    "ADAPTIVE_THRESHOLD": {
        "block_size": 11,
        "constant": 2.0,
    },
    # Legacy spelling retained as an alias.
    "ADAPTATIVE_THRESHOLD": {
        "block_size": 11,
        "constant": 2.0,
    },

    "REMOVE_SMALL_NOISE": {"min_area": 2500},
    "REMOVE_LARGE_OBJECTS": {"max_area": 2000},
    "SOMA": {
        "iterations": 3,
        "min_area_threshold": 0,
    },
    "SKELETONIZE": {},
    "CANNY_EDGE_DETECTION": {
        "threshold1": 100.0,
        "threshold2": 200.0,
    },

    "ELIMINATE_1_VALUE": {},
    "ELIMINATE_N_VALUE": {"value": 6.0},
    "ELIMINATE_N_VALUE2": {"value": 6.0},
    "ELIMINATE_N_VALUE3": {"value": 6.0},
    
    "MEDIAN_FILTER": {"kernel_size": 3},
    
    "PERCENTILE_HYSTERESIS_THRESHOLD": {
        "low_percentile": 70.0,
        "high_percentile": 92.0,
    },
    
    "FILL_SMALL_HOLES": {"area_threshold": 32},
    
    "ROBUST_RESCALE": {
        "low_percentile": 0.5,
        "high_percentile": 99.5,
    },
    
    "BACKGROUND_PERCENTILE_SUBTRACTION": {"percentile": 60.0},
    
    "HUANG_THRESHOLD": {"bins": 256},
    
    "BRANCH_GEOMETRY_CLEANUP": {
        "angles": (
            0, 30, 60,
            90, 120, 150,
        ),
        "thin_kernels": (
            (1, 10),
            (2, 10),
        ),
        "strong_kernels": (
            (3, 11),
            (3, 16),
        ),
        "soma_radius": 4,
        "min_soma_area": 80,
        "min_object_area": 30,
        "bridge_radius": 1,
    },
        
        
    
    
    
    "BRANCH_AWARE_CONSENSUS_THRESHOLD": {
        "support_votes": 1,
        "strong_votes": 3,
        "min_seed_area": 40,
        "line_lengths": (9, 13, 17),
        "line_angles": (0, 30, 60, 90, 120, 150),
        "line_width": 1,
        "min_line_occupancy": 0.65,
        "bridge_radius": 1,
    },
    
    
    "GAUSSIAN_SUBTRACTION": {"kernel_size": 181},
    
    "DIRECTIONAL_GRAYSCALE_OPENING": {
        "angles": (
            0, 15, 30, 45, 60, 75,
            90, 105, 120, 135, 150, 165,
        ),
        "branch_kernels": (
            (1, 10),
            (2, 10),
            (3, 11),
            (3, 16),
        ),
        "soma_radius": 4,
    },
    
    "LI_THRESHOLD": {},
    
    "REMOVE_CIRCULAR_NOISE": {
        "min_circularity": 0.50,
        "max_area": 2300,
    },
    
    
    "REMOVE_COMPACT_OBJECTS": {
        "min_solidity": 0.73,
        "min_circularity": 0.15,
        "max_aspect_ratio": 2.5,
    },
    

        
    
    "PREP_FOR_CELLPOSE": {
        "radii": (10, 40, 120),
        "sharpen": True,
        "unsharp_radius": 1.0,
        "unsharp_amount": 1.0,
    },
    
    
    }


# Optional developer overrides for a particular named preset.
#
# Example:
#
# PRESET_PARAMETER_OVERRIDES = {
#     "fluorescence_v2": {
#         "GAUSSIAN_SUBTRACTION": {"kernel_size": 31},
#         "PERCENTILE_THRESHOLD": {"percentile": 95.0},
#     },
# }
#
PRESET_PARAMETER_OVERRIDES: dict[
    str,
    dict[str, dict[str, object]],
] = {
    "mg1": {},
    "mg2": {},
    "mg3": {},
    "mg4": {},

    # Historical compatibility.
    "fluorescence_v1": {},
}


def parameters_for(
    operation: str,
    *,
    preset: str | None = None,
) -> dict[str, object]:
    key = str(operation).strip().upper()
    parameters = dict(FILTER_PARAMETERS.get(key, {}))

    if preset is not None:
        preset_key = str(preset).strip().lower()
        parameters.update(
            PRESET_PARAMETER_OVERRIDES
            .get(preset_key, {})
            .get(key, {})
        )

    return parameters


__all__ = [
    "FILTER_PARAMETERS",
    "PRESET_PARAMETER_OVERRIDES",
    "parameters_for",
]
