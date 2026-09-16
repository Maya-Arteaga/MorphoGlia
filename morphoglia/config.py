from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
    is_dataclass,
)
from pathlib import Path
from typing import Any

import yaml

from .metadata.config import (
    MetadataConfig,
)

from .preprocessing.config import (
    PreprocessingConfig,
)
from .segmentation.config import (
    SegmentationConfig,
)

from .morphometrics.config import (
    MorphometricsConfig,
)

from .instance_refinement.config import (
    InstanceRefinementConfig,
)

from .category.config import (
    CategoryConfig,
)

from .dim_reduction_clustering.config import (
    FeaturePreparationConfig,
    DimensionalityReductionConfig,
    ClusteringConfig,
)

from .mapping.config import (
    MappingConfig,
)

from .plots.config import (
    PlotConfig,
)

from .compute import (
    ComputeConfig,
)



# ======================================================================
# PIPELINE EXECUTION CONFIGURATION
# ======================================================================

PIPELINE_STAGES = (
    "metadata",
    "preprocessing",
    "segmentation",
    "instance_refinement",
    "morphometrics",
    "category",
    "dim_reduction_clustering",
    "mapping",
    "spatial_analysis",
    "plots",
)


DEFAULT_STAGE_ENABLED = {
    "metadata": True,
    "preprocessing": True,
    "segmentation": True,
    "instance_refinement": True,
    "morphometrics": True,
    "category": True,
    "dim_reduction_clustering": True,
    "mapping": True,
    "spatial_analysis": False,
    "plots": True,
}


class StageRunConfig:
    """
    Public MorphoGlia stage execution switches.

    None
        Use the package default.

    True / False
        Explicit user choice.

    Feature Preparation, Dimensionality Reduction, and Clustering remain
    separate internal engines/configuration namespaces, but execute as
    one coupled public stage.
    """

    metadata: bool | None = None
    preprocessing: bool | None = None
    segmentation: bool | None = None
    instance_refinement: bool | None = None
    morphometrics: bool | None = None
    category: bool | None = None

    dim_reduction_clustering: bool | None = None

    mapping: bool | None = None
    spatial_analysis: bool | None = None
    plots: bool | None = None

    def is_enabled(
        self,
        stage: str,
    ) -> bool:

        if stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Unknown pipeline stage: {stage}"
            )

        value = getattr(
            self,
            stage,
        )

        if value is None:
            return bool(
                DEFAULT_STAGE_ENABLED[
                    stage
                ]
            )

        return bool(
            value
        )

    def source(
        self,
        stage: str,
    ) -> str:

        if stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Unknown pipeline stage: {stage}"
            )

        return (
            "default"
            if getattr(
                self,
                stage,
            ) is None
            else "user"
        )

    def enabled_stages(
        self,
    ) -> list[str]:

        return [
            stage
            for stage in PIPELINE_STAGES
            if self.is_enabled(
                stage
            )
        ]

    def disabled_stages(
        self,
    ) -> list[str]:

        return [
            stage
            for stage in PIPELINE_STAGES
            if not self.is_enabled(
                stage
            )
        ]

    def configuration_rows(
        self,
    ) -> list[dict]:

        rows = []

        for stage in PIPELINE_STAGES:

            rows.append(
                {
                    "section":
                        "Pipeline.Run",
                    "parameter":
                        stage,
                    "value":
                        self.is_enabled(
                            stage
                        ),
                    "source":
                        self.source(
                            stage
                        ),
                }
            )

        return rows

# ======================================================================
# YAML-SAFE CONFIG SERIALIZATION
# ======================================================================

def _yaml_safe_value(
    value,
):
    """
    Convert MorphoGlia configuration values to plain YAML-safe Python types.

    Real dataclass instance fields are serialized. ClassVar constants such as
    canonical metadata vocabularies are deliberately excluded. Annotation-only
    config containers such as StageRunConfig are supported as well.
    """

    if value is None or isinstance(
        value,
        (
            bool,
            int,
            float,
            str,
        ),
    ):
        return value


    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )


    if isinstance(
        value,
        dict,
    ):
        return {
            _yaml_safe_value(
                key
            ):
            _yaml_safe_value(
                item
            )
            for key, item in value.items()
        }


    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _yaml_safe_value(
                item
            )
            for item in value
        ]


    if isinstance(
        value,
        (
            set,
            frozenset,
        ),
    ):
        converted = [
            _yaml_safe_value(
                item
            )
            for item in value
        ]

        return sorted(
            converted,
            key=repr,
        )


    field_names = _configuration_field_names(
        value
    )


    if field_names:

        return {
            field_name:
            _yaml_safe_value(
                getattr(
                    value,
                    field_name,
                )
            )
            for field_name in field_names
        }


    item_method = getattr(
        value,
        "item",
        None,
    )


    if callable(
        item_method
    ):
        try:
            return _yaml_safe_value(
                item_method()
            )
        except Exception:
            pass


    raise TypeError(
        "MorphoGlia cannot serialize unsupported "
        f"configuration value of type "
        f"{type(value).__name__}: {value!r}"
    )




def _dataclass_yaml(
    value,
) -> dict:
    """
    Return one configuration section as a YAML-safe mapping.
    """

    result = _yaml_safe_value(
        value
    )


    if not isinstance(
        result,
        dict,
    ):
        raise TypeError(
            "MorphoGlia configuration sections must "
            "serialize to mappings."
        )


    return result


# ======================================================================
# YAML CONFIG FIELD INTROSPECTION / RECURSIVE ASSIGNMENT
# ======================================================================

def _configuration_field_names(
    value,
) -> tuple[str, ...]:
    """
    Return only declared configurable instance fields.

    Dataclasses use dataclasses.fields(), which excludes ClassVar and InitVar
    pseudo-fields. Annotation-only containers such as StageRunConfig use their
    declared annotations.
    """

    from dataclasses import (
        fields as dataclass_fields,
        is_dataclass,
    )


    if is_dataclass(
        value
    ):

        return tuple(
            field_info.name
            for field_info in dataclass_fields(
                value
            )
        )


    names: list[
        str
    ] = []


    for cls in reversed(
        type(
            value
        ).__mro__
    ):

        for name in getattr(
            cls,
            "__annotations__",
            {},
        ):

            name = str(
                name
            )

            if name not in names:
                names.append(
                    name
                )


    return tuple(
        names
    )


def _is_configuration_object(
    value,
) -> bool:
    """Return True only for structured MorphoGlia configuration containers."""

    if value is None or isinstance(
        value,
        (
            bool,
            int,
            float,
            str,
            Path,
            dict,
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return False


    return bool(
        _configuration_field_names(
            value
        )
    )


def _coerce_yaml_container(
    current,
    incoming,
):
    """
    Preserve an existing field's tuple/set semantics after yaml.safe_load().
    """

    if isinstance(
        current,
        set,
    ) and isinstance(
        incoming,
        (
            list,
            tuple,
            set,
        ),
    ):

        return set(
            incoming
        )


    if isinstance(
        current,
        frozenset,
    ) and isinstance(
        incoming,
        (
            list,
            tuple,
            set,
        ),
    ):

        return frozenset(
            incoming
        )


    if isinstance(
        current,
        tuple,
    ) and isinstance(
        incoming,
        (
            list,
            tuple,
        ),
    ):

        return tuple(
            incoming
        )


    return incoming


def _apply_mapping(
    target,
    values,
    *,
    section: str,
) -> None:
    """
    Recursively apply one YAML mapping to one MorphoGlia config object.

    Unknown keys are rejected. Nested config objects recurse rather than being
    replaced by ordinary dictionaries. Existing __post_init__ validators remain
    authoritative and are called later by PipelineConfig.from_yaml().
    """

    if not isinstance(
        values,
        dict,
    ):

        raise TypeError(
            f"Configuration section {section!r} must be a mapping."
        )


    field_names = _configuration_field_names(
        target
    )


    if not field_names:

        raise TypeError(
            f"Configuration section {section!r} does not expose "
            "declared configurable fields."
        )


    unknown = (
        set(
            values
        )
        - set(
            field_names
        )
    )


    if unknown:

        qualified = [
            f"{section}.{name}"
            for name in sorted(
                unknown
            )
        ]

        raise ValueError(
            "Unknown configuration option(s): "
            + ", ".join(
                qualified
            )
        )


    for name, incoming in values.items():

        current = getattr(
            target,
            name,
        )

        qualified_name = (
            f"{section}.{name}"
        )


        if _is_configuration_object(
            current
        ):

            if not isinstance(
                incoming,
                dict,
            ):

                raise TypeError(
                    f"Configuration option {qualified_name!r} "
                    "must be a mapping."
                )


            _apply_mapping(
                current,
                incoming,
                section=(
                    qualified_name
                ),
            )

            continue


        setattr(
            target,
            name,
            _coerce_yaml_container(
                current,
                incoming,
            ),
        )




# ======================================================================
@dataclass
class PipelineConfig:
    """
    Canonical MorphoGlia configuration.

    Normal users generally need only:

        input_dir
        preprocessing.input_mode
        category.category_fields
        qc.* True / False flags
        run.* True / False flags

    The remaining nested configuration objects expose advanced
    scientific and technical settings when needed.
    """

    # ==================================================================
    # PATHS / GLOBAL
    # ==================================================================

    input_dir: str | Path

    output_dir: str | Path | None = None

    random_seed: int = 24

    # MG_RESUME_CONFIG_V1
    # False preserves the historical recompute/overwrite behavior.
    resume: bool = False

    # Global computational resource policy.
    compute: ComputeConfig = field(
        default_factory=ComputeConfig
    )


    # ==================================================================
    # PIPELINE EXECUTION
    # ==================================================================

    run: StageRunConfig = field(
        default_factory=StageRunConfig
    )


    # ==================================================================
    # SCIENTIFIC STAGES
    # ==================================================================

    metadata: MetadataConfig = field(
        default_factory=MetadataConfig
    )


    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )

    segmentation: SegmentationConfig = field(
        default_factory=SegmentationConfig
    )


    instance_refinement: InstanceRefinementConfig = field(
        default_factory=InstanceRefinementConfig
    )


    morphometrics: MorphometricsConfig = field(
        default_factory=MorphometricsConfig
    )


    @property
    def qc(self):
        """Deprecated compatibility alias for instance_refinement."""
        return self.instance_refinement


    category: CategoryConfig = field(
        default_factory=CategoryConfig
    )


    feature_preparation: FeaturePreparationConfig = field(
        default_factory=FeaturePreparationConfig
    )


    dimensionality_reduction: DimensionalityReductionConfig = field(
        default_factory=DimensionalityReductionConfig
    )


    clustering: ClusteringConfig = field(
        default_factory=ClusteringConfig
    )


    mapping: MappingConfig = field(
        default_factory=MappingConfig
    )


    plots: PlotConfig = field(
        default_factory=PlotConfig
    )


    # ==================================================================
    # PATH REALIZATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:

        self.random_seed = int(
            self.random_seed
        )


        self.resume = bool(
            self.resume
        )


        self.input_dir = (
            Path(
                self.input_dir
            )
            .expanduser()
            .resolve()
        )


        if self.output_dir is None:

            self.output_dir = (
                self.input_dir
                / "_MorphoGlia"
            )

        else:

            self.output_dir = (
                Path(
                    self.output_dir
                )
                .expanduser()
                .resolve()
            )


        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )




    # ==================================================================
    # YAML
    # ==================================================================

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "PipelineConfig":
        """
        Load PipelineConfig from YAML.

        YAML mirrors the Python configuration structure directly.

        Example
        -------

        input_dir: /data

        preprocessing:
          input_mode: binary

        category:
          category_fields:
            - condition
            - region

        qc:
          small_objects: true
          large_objects: true
          tubular: false

        run:
          mapping: false
          plots: false
        """

        path = (
            Path(
                path
            )
            .expanduser()
            .resolve()
        )


        with path.open(
            "r",
        ) as file:

            data = yaml.safe_load(
                file
            )


        if data is None:

            data = {}


        if not isinstance(
            data,
            dict,
        ):

            raise TypeError(
                "MorphoGlia YAML root must be a mapping."
            )


        allowed_top_level = {
            "input_dir",
            "output_dir",
            "random_seed",
            "resume",
            "compute",

            "run",

            "metadata",
            "preprocessing",
            "segmentation",
            "instance_refinement",
            "morphometrics",
            "qc",
            "category",
            "feature_preparation",
            "dimensionality_reduction",
            "clustering",
            "mapping",
            "plots",

        }


        unknown = (
            set(
                data
            )
            - allowed_top_level
        )


        if unknown:

            raise ValueError(
                "Unknown top-level configuration "
                f"options: {sorted(unknown)}"
            )


        if "input_dir" not in data:

            raise ValueError(
                "MorphoGlia YAML requires input_dir."
            )


        config = cls(
            input_dir=(
                data[
                    "input_dir"
                ]
            ),

            output_dir=(
                data.get(
                    "output_dir"
                )
            ),

            random_seed=(
                data.get(
                    "random_seed",
                    24,
                )
            ),

            resume=bool(
                data.get(
                    "resume",
                    False,
                )
            ),
        )


        nested_sections = (
            "run",
            "compute",

            "metadata",
            "preprocessing",
            "segmentation",
            "qc",
            "instance_refinement",
            "morphometrics",
            "category",
            "feature_preparation",
            "dimensionality_reduction",
            "clustering",
            "mapping",
            "plots",

        )


        for section in nested_sections:

            if section not in data:
                continue


            _apply_mapping(
                getattr(
                    config,
                    section,
                ),
                data[
                    section
                ],
                section=section,
            )


        # Revalidate dataclasses whose values may have been changed by
        # recursive YAML assignment.
        for section in nested_sections:

            section_object = getattr(
                config,
                section,
            )


            validator = getattr(
                section_object,
                "__post_init__",
                None,
            )


            if callable(
                validator
            ):

                validator()


        config.__post_init__()


        return config


    def to_yaml(
        self,
        path: str | Path,
    ) -> Path:
        """
        Save a complete canonical configuration snapshot.

        Users may still write minimal YAML manually; omitted options are
        reconstructed from package defaults by from_yaml().
        """

        path = Path(
            path
        )


        data = {
            "input_dir": str(
                self.input_dir
            ),

            "output_dir": str(
                self.output_dir
            ),

            "random_seed": (
                self.random_seed
            ),

            "resume": bool(
                self.resume
            ),

            "run": _dataclass_yaml(
                self.run
            ),
            "compute": _dataclass_yaml(
                self.compute
            ),

            "metadata": _dataclass_yaml(
                self.metadata
            ),

            "preprocessing": _dataclass_yaml(
                self.preprocessing
            ),
            "segmentation": _dataclass_yaml(
                self.segmentation
            ),

            "instance_refinement": _dataclass_yaml(
                self.instance_refinement
            ),

            "morphometrics": _dataclass_yaml(
                self.morphometrics
            ),

            "category": _dataclass_yaml(
                self.category
            ),

            "feature_preparation": (
                _dataclass_yaml(
                    self.feature_preparation
                )
            ),

            "dimensionality_reduction": (
                _dataclass_yaml(
                    self.dimensionality_reduction
                )
            ),

            "clustering": _dataclass_yaml(
                self.clustering
            ),

            "mapping": _dataclass_yaml(
                self.mapping
            ),

            "plots": _dataclass_yaml(
                self.plots
            ),

        }


        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        with path.open(
            "w",
        ) as file:

            yaml.safe_dump(
                data,
                file,
                sort_keys=False,
            )


        return path


__all__ = [
    "PipelineConfig",
]
