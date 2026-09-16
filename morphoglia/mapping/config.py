from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MONTAGE_PADDING = 20
DEFAULT_FILL_ALPHA = 1.0


@dataclass
class MappingConfig:
    """
    Configuration for mapping the preferred morphology-state solution
    back to observed cells and source images.

    Canonical Mapping outputs
    -------------------------
    cell_labels.csv
        Preferred-resolution state assignment for every accepted cell.

    Cluster_Maps/
        Exact filled-mask tissue maps colored by the human-facing
        morphology-state identity C1 ... CK.

    _Prototypes/
        Six empirical prototype cells per state rendered at native pixel
        scale. selection_rank == 1 is the canonical observed prototype.

    mapping.csv
        Generated-asset and provenance manifest.

    component_match_errors.csv
        Exact-mask reconstruction failures.

    Architecture
    ------------
    - Mapping propagates only clustering.preferred_k.
    - Alternative robust resolutions remain analytical QC.
    - master_table.csv remains the cell-level source of truth.
    - Internal matching uses canonical image/cell/map identity.
    - Raw filenames are retained for traceability.
    - Cluster maps use exact filled component masks.
    - Prototype montages preserve native pixel scale.
    - UMAP is never used to select prototypes.
    """

    save_mapping_manifest: bool = True

    save_cluster_maps: bool = True

    save_prototypes: bool = True

    save_selection_montages: bool = True

    montage_padding: int | None = None

    fill_alpha: float | None = None
    show_ids: bool | None = None


    # ==================================================================
    # FIXED ARCHITECTURE
    # ==================================================================

    @property
    def cell_table(
        self,
    ) -> str:

        return "master_table.csv"


    @property
    def mapping_manifest(
        self,
    ) -> str:

        return "mapping.csv"


    @property
    def internal_identity(
        self,
    ) -> tuple[str, str, str]:

        return (
            "image_id",
            "cell_id",
            "map_label",
        )


    @property
    def traceability_field(
        self,
    ) -> str:

        return "raw_filename"


    @property
    def render_mode(
        self,
    ) -> str:

        return "filled_mask"


    @property
    def background(
        self,
    ) -> str:

        return "transparent"


    @property
    def montage_scale(
        self,
    ) -> str:

        return "native_pixel_scale"


    @property
    def prototype_center_method(
        self,
    ) -> str:

        return "coordinate_median"






    # ==================================================================
    # EFFECTIVE VALUES
    # ==================================================================



    @property
    def effective_montage_padding(
        self,
    ) -> int:

        value = (
            DEFAULT_MONTAGE_PADDING
            if self.montage_padding is None
            else int(
                self.montage_padding
            )
        )

        if value < 0:

            raise ValueError(
                "montage_padding must be >= 0."
            )

        return value


    @property
    def effective_show_ids(
        self,
    ) -> bool:

        return (
            False
            if self.show_ids is None
            else bool(
                self.show_ids
            )
        )


    @property
    def effective_fill_alpha(
        self,
    ) -> float:

        value = (
            DEFAULT_FILL_ALPHA
            if self.fill_alpha is None
            else float(
                self.fill_alpha
            )
        )

        if not 0.0 <= value <= 1.0:

            raise ValueError(
                "fill_alpha must be between 0.0 and 1.0."
            )

        return value


    # ==================================================================
    # TECHNICAL RECORD
    # ==================================================================

    def configuration_rows(
        self,
    ) -> list[dict]:

        return [
            {
                "section": "Mapping",
                "parameter": "mapping_manifest",
                "value": self.mapping_manifest,
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "internal_identity",
                "value": list(
                    self.internal_identity
                ),
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "traceability_field",
                "value": self.traceability_field,
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "render_mode",
                "value": self.render_mode,
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "background",
                "value": self.background,
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "montage_scale",
                "value": self.montage_scale,
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "prototype_center_method",
                "value": self.prototype_center_method,
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "save_mapping_manifest",
                "value": self.save_mapping_manifest,
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "save_cluster_maps",
                "value": self.save_cluster_maps,
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "save_prototypes",
                "value": self.save_prototypes,
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "save_selection_montages",
                "value": self.save_selection_montages,
                "source": "default",
            },
            {
                "section": "Mapping",
                "parameter": "montage_padding",
                "value": self.effective_montage_padding,
                "source": (
                    "default"
                    if self.montage_padding is None
                    else "user"
                ),
            },
            {
                "section": "Mapping",
                "parameter": "fill_alpha",
                "value": self.effective_fill_alpha,
                "source": (
                    "default"
                    if self.fill_alpha is None
                    else "user"
                ),
            },
            {
                "section": "Mapping",
                "parameter": "show_ids",
                "value": self.effective_show_ids,
                "source": (
                    "default"
                    if self.show_ids is None
                    else "user"
                ),
            },
        ]
