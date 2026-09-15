from __future__ import annotations

# MG_RESUME_MAPPING_V1

from ..checkpoint import (
    CheckpointJournal,
    atomic_write_bytes,
    file_identity,
    fingerprint,
    source_digest,
)

from ..core.instance_map import (
    validate_instance_map,
)

from ..plots.style import (
    cluster_display_order,
)

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from collections.abc import Mapping

import cv2
import numpy as np
import pandas as pd
import tifffile

from .config import MappingConfig

from .selection import (
    select_cluster_prototypes,
)

from .rendering import (
    render_selection_montage,
    cluster_rgb,
    get_map_label,
    extract_component_crop,
)


# ======================================================================
# RESULT
# ======================================================================

@dataclass
class MappingResult:
    """
    Output of the MorphoGlia Mapping stage.

    Mapping contains only the preferred clustering resolution.

    Attributes
    ----------
    manifest
        mapping.csv contents.

    cell_labels
        Cell-level assignment table keyed by preferred K.

    selections
        Canonical prototype selection keyed by preferred K.

    component_errors
        Exact-mask reconstruction failures.

    preferred_k
        Resolution selected by the clustering decision layer.

    output_dir
        Mapping output root.
    """

    manifest: pd.DataFrame

    cell_labels: dict[
        int,
        pd.DataFrame,
    ]

    selections: dict[
        int,
        dict[
            str,
            pd.DataFrame,
        ],
    ]

    component_errors: pd.DataFrame

    preferred_k: int

    output_dir: Path


# ======================================================================
# STAGE
# ======================================================================

class MappingStage:
    """
    Map the preferred clustering solution back to observed cells and
    source images.

    Scientific role
    ---------------
    Mapping does not discover morphology states.

    The explicit resolution-decision layer first selects preferred_k.
    Mapping then reconnects that single biological interpretation with
    the real segmented objects.

    Canonical outputs
    -----------------
    1. preferred-resolution cell labels
    2. six empirical prototypes per morphology state
    3. rank-1 canonical prototype masks
    4. native-scale 3 x 2 prototype montage
    5. exact filled-mask cluster maps
    6. provenance and component-error tables

    Alternative robust K values remain analytical QC and do not generate
    parallel Mapping products.

    The prototype is always an observed cell.
    UMAP is never used to select prototypes.
    """

    def __init__(
        self,
        config: MappingConfig,
    ):

        self.config = config


    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def run(
        self,
        master: pd.DataFrame,
        clustering,
        instance_dir: str | Path,
        output_dir: str | Path,
        label_columns: Sequence[str] = (),
        resume: bool = False,
    ) -> MappingResult:
        """
        Map the preferred clustering solution back to observed cells
        and source images.

        Mapping is a final interpretive stage.

        Alternative robust resolutions remain analytical QC and do not
        generate parallel tissue maps.
        """

        master = (
            master
            .copy()
            .reset_index(
                drop=True
            )
        )


        if isinstance(
            instance_dir,
            Mapping,
        ):

            instance_source = {
                str(
                    image_id
                ): Path(
                    path
                )
                for image_id, path in instance_dir.items()
            }

        else:

            instance_source = Path(
                instance_dir
            )


        output_dir = Path(
            output_dir
        )


        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        self._validate_master(
            master
        )


        # ==============================================================
        # PREFERRED CLUSTERING RESOLUTION
        # ==============================================================

        preferred_k = getattr(
            clustering,
            "preferred_k",
            None,
        )


        if preferred_k is None:

            raise ValueError(
                "Mapping requires preferred_k from "
                "the clustering decision layer."
            )


        preferred_k = int(
            preferred_k
        )


        selected_solutions = getattr(
            clustering,
            "selected_solutions",
            None,
        )


        if (
            selected_solutions is None
            or preferred_k
            not in selected_solutions
        ):

            raise RuntimeError(
                "Preferred clustering solution "
                f"K={preferred_k} is unavailable."
            )


        solution = (
            selected_solutions[
                preferred_k
            ]
        )


        d = int(
            solution.pca_dimensions
        )


        # ==============================================================
        # ATTACH PREFERRED SOLUTION
        # ==============================================================

        data = (
            self._attach_solution(
                master=master,
                solution=solution,
            )
        )


        # ==============================================================
        # HUMAN-FACING C1 ... CK IDENTITY
        #
        # Raw GMM labels remain unchanged in "cluster".
        # display_cluster controls user-facing labels and colors.
        # ==============================================================

        (
            _,
            display_rank,
        ) = cluster_display_order(
            master=master,
            labels=np.asarray(
                solution.labels,
                dtype=int,
            ),
            complexity_column="Cell_area",
        )


        data[
            "display_cluster"
        ] = (
            data[
                "cluster"
            ]
            .astype(int)
            .map(
                display_rank
            )
            .astype(int)
        )


        # ==============================================================
        # LOAD INSTANCE MAPS ONCE
        # ==============================================================

        needs_images = (
            self.config.save_cluster_maps
            or self.config.save_prototypes
            or self.config.save_selection_montages
        )


        if needs_images:

            image_cache = (
                self._build_image_cache(
                    master=master,
                    instance_dir=instance_source,
                )
            )

        else:

            image_cache = {}


        manifest_rows = []

        component_error_rows = []


        # ==============================================================
        # CELL LABELS
        # ==============================================================

        label_table = (
            self._cell_label_export(
                data=data,
                label_columns=(
                    label_columns
                ),
            )
        )


        label_path = (
            output_dir
            / "cell_labels.csv"
        )


        label_table.to_csv(
            label_path,
            index=False,
        )


        manifest_rows.append(
            {
                "mapping_type":
                    "cell_labels",

                "k":
                    preferred_k,

                "pca_dimensions":
                    d,

                "image_id":
                    "",

                "output_path":
                    str(
                        label_path
                    ),
            }
        )


        # ==============================================================
        # PROTOTYPES
        #
        # Six observed prototype cells per cluster.
        #
        # selection_rank == 1 remains the canonical prototype used by
        # PCA / UMAP morphology-state plots.
        # ==============================================================

        selections = {}


        if self.config.save_prototypes:

            prototypes_dir = (
                output_dir
                / "_Prototypes"
            )


            prototypes_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            prototype_cells = (
                select_cluster_prototypes(
                    data,
                    pca_dimensions=d,
                    n_per_cluster=6,
                )
            )


            prototype_cells[
                "display_cluster"
            ] = (
                prototype_cells[
                    "cluster"
                ]
                .astype(int)
                .map(
                    display_rank
                )
                .astype(int)
            )


            canonical_prototypes = (
                prototype_cells[
                    prototype_cells[
                        "selection_rank"
                    ]
                    == 1
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )


            selections[
                "prototype"
            ] = canonical_prototypes


            # ----------------------------------------------------------
            # SIX PROTOTYPES PER CLUSTER — CSV
            # ----------------------------------------------------------

            prototype_export = (
                self._selection_export(
                    prototype_cells,
                    label_columns=(
                        label_columns
                    ),
                )
            )


            prototype_csv_path = (
                prototypes_dir
                / "prototype_cells.csv"
            )


            prototype_export.to_csv(
                prototype_csv_path,
                index=False,
            )


            manifest_rows.append(
                {
                    "mapping_type":
                        "prototype_table",

                    "k":
                        preferred_k,

                    "pca_dimensions":
                        d,

                    "image_id":
                        "",

                    "output_path":
                        str(
                            prototype_csv_path
                        ),
                }
            )


            # ----------------------------------------------------------
            # PROTOTYPE MASKS FOR MORPHOLOGY-STATE FIGURES
            #
            # Export ranks 1-4 for the 2 x 2 prototype panel.
            #
            # Ranked files:
            #
            #     cluster_X_rank_1.png
            #     cluster_X_rank_2.png
            #     cluster_X_rank_3.png
            #     cluster_X_rank_4.png
            #
            # The historical:
            #
            #     cluster_X.png
            #
            # remains the canonical rank-1 alias.
            # ----------------------------------------------------------

            prototype_masks_dir = (
                prototypes_dir
                / "Prototype_Masks"
            )


            prototype_masks_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            figure_prototypes = (
                prototype_cells[
                    prototype_cells[
                        "selection_rank"
                    ]
                    .astype(int)
                    .le(6)
                ]
                .copy()
                .sort_values(
                    [
                        "cluster",
                        "selection_rank",
                    ],
                    kind="mergesort",
                )
                .reset_index(
                    drop=True
                )
            )


            for (
                _,
                prototype_cell,
            ) in figure_prototypes.iterrows():

                prototype_mask = (
                    extract_component_crop(
                        prototype_cell,
                        image_cache,
                    )
                )


                if prototype_mask is None:

                    raise RuntimeError(
                        "Could not recover exact "
                        "prototype component."
                    )


                raw_cluster = int(
                    prototype_cell[
                        "cluster"
                    ]
                )


                selection_rank = int(
                    prototype_cell[
                        "selection_rank"
                    ]
                )


                ranked_mask_path = (
                    prototype_masks_dir
                    / (
                        f"cluster_"
                        f"{raw_cluster}"
                        f"_rank_"
                        f"{selection_rank}.png"
                    )
                )


                success = cv2.imwrite(
                    str(
                        ranked_mask_path
                    ),
                    prototype_mask,
                )


                if not success:

                    raise RuntimeError(
                        "Could not save ranked "
                        "prototype mask: "
                        f"{ranked_mask_path}"
                    )


                manifest_rows.append(
                    {
                        "mapping_type":
                            "prototype_mask_ranked",

                        "k":
                            preferred_k,

                        "pca_dimensions":
                            d,

                        "image_id":
                            str(
                                prototype_cell[
                                    "image_id"
                                ]
                            ),

                        "output_path":
                            str(
                                ranked_mask_path
                            ),
                    }
                )


                # ------------------------------------------------------
                # CANONICAL RANK-1 ALIAS
                # ------------------------------------------------------

                if selection_rank == 1:

                    canonical_mask_path = (
                        prototype_masks_dir
                        / (
                            f"cluster_"
                            f"{raw_cluster}.png"
                        )
                    )


                    success = cv2.imwrite(
                        str(
                            canonical_mask_path
                        ),
                        prototype_mask,
                    )


                    if not success:

                        raise RuntimeError(
                            "Could not save canonical "
                            "prototype mask: "
                            f"{canonical_mask_path}"
                        )


                    manifest_rows.append(
                        {
                            "mapping_type":
                                "prototype_mask",

                            "k":
                                preferred_k,

                            "pca_dimensions":
                                d,

                            "image_id":
                                str(
                                    prototype_cell[
                                        "image_id"
                                    ]
                                ),

                            "output_path":
                                str(
                                    canonical_mask_path
                                ),
                        }
                    )


            # ----------------------------------------------------------
            # CANONICAL 3 × 2 PROTOTYPE GRID
            # ----------------------------------------------------------

            if (
                self.config
                .save_selection_montages
            ):

                montage_path = (
                    prototypes_dir
                    / "Prototype_Cells.png"
                )


                render_selection_montage(
                    prototype_cells,
                    image_cache=image_cache,
                    output_path=(
                        montage_path
                    ),
                    title=(
                        f"Prototype cells — "
                        f"d={d}, {preferred_k} morphology states"
                    ),
                    label_columns=(),
                    padding=(
                        self.config
                        .effective_montage_padding
                    ),
                    ncols=3,
                    max_cells_per_cluster=6,
                    show_full_id=False,
                )


                manifest_rows.append(
                    {
                        "mapping_type":
                            "prototype_montage",

                        "k":
                            preferred_k,

                        "pca_dimensions":
                            d,

                        "image_id":
                            "",

                        "output_path":
                            str(
                                montage_path
                            ),
                    }
                )


        # ==============================================================
        # CLUSTER MAPS
        # ==============================================================

        if self.config.save_cluster_maps:

            cluster_maps_dir = (
                output_dir
                / "Cluster_Maps"
            )


            cluster_maps_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            (
                map_manifest_rows,
                map_error_rows,
            ) = self._save_cluster_maps(
                data=data,
                image_cache=image_cache,
                output_dir=(
                    cluster_maps_dir
                ),
                k=preferred_k,
                d=d,
                instance_source=instance_source,
                checkpoint_root=output_dir.parent,
                resume=resume,
            )


            manifest_rows.extend(
                map_manifest_rows
            )


            component_error_rows.extend(
                map_error_rows
            )


        # ==============================================================
        # GLOBAL TABLES
        # ==============================================================

        manifest = pd.DataFrame(
            manifest_rows
        )


        component_errors = pd.DataFrame(
            component_error_rows,
            columns=[
                "k",
                "pca_dimensions",
                "image_id",
                "cell_id",
            ],
        )


        if (
            self.config
            .save_mapping_manifest
        ):

            manifest.to_csv(
                output_dir
                / self.config.mapping_manifest,
                index=False,
            )


        component_errors.to_csv(
            output_dir
            / "component_match_errors.csv",
            index=False,
        )


        return MappingResult(
            manifest=manifest,

            cell_labels={
                preferred_k:
                    data,
            },

            selections={
                preferred_k:
                    selections,
            },

            component_errors=(
                component_errors
            ),

            preferred_k=(
                preferred_k
            ),

            output_dir=(
                output_dir
            ),
        )


    # ==================================================================
    # MASTER VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_master(
        master: pd.DataFrame,
    ) -> None:

        required = [
            "image_id",
            "cell_id",
            "map_label",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
        ]


        missing = [
            column
            for column in required
            if column not in master.columns
        ]


        if missing:

            raise ValueError(
                "Mapping master table is missing "
                f"required columns: {missing}"
            )


        if master[
            [
                "image_id",
                "cell_id",
            ]
        ].duplicated().any():

            raise ValueError(
                "Mapping requires unique "
                "(image_id, cell_id) identity."
            )


    # ==================================================================
    # ATTACH SELECTED CLUSTERING SOLUTION
    # ==================================================================

    @staticmethod
    def _attach_solution(
        master: pd.DataFrame,
        solution,
    ) -> pd.DataFrame:

        n_cells = len(
            master
        )


        arrays = {
            "labels":
                np.asarray(
                    solution.labels
                ),

            "cluster_probability":
                np.asarray(
                    solution
                    .cluster_probability
                ),

            "consensus_reliability":
                np.asarray(
                    solution
                    .consensus_reliability
                ),
        }


        for name, values in (
            arrays.items()
        ):

            if len(
                values
            ) != n_cells:

                raise ValueError(
                    f"Mapping solution {name} has "
                    f"{len(values)} rows but master "
                    f"contains {n_cells} cells."
                )


        data = master.copy()


        data[
            "cluster"
        ] = arrays[
            "labels"
        ].astype(
            int
        )


        data[
            "cluster_probability"
        ] = arrays[
            "cluster_probability"
        ].astype(
            float
        )


        data[
            "consensus_reliability"
        ] = arrays[
            "consensus_reliability"
        ].astype(
            float
        )


        data[
            "pca_dimensions"
        ] = int(
            solution.pca_dimensions
        )


        data[
            "covariance_type"
        ] = str(
            solution.covariance_type
        )


        return data


    # ==================================================================
    # EXACT COMPONENT CACHE
    # ==================================================================

    @staticmethod
    def _read_instance_map(
        path: Path,
    ) -> np.ndarray:
        """
        Load one canonical 2D instance map.
        """

        if not path.exists():

            raise FileNotFoundError(
                f"Could not find instance map: {path}"
            )


        instance_map = np.asarray(
            tifffile.imread(
                str(path)
            )
        )


        instance_map = (
            validate_instance_map(
                instance_map
            )
        )


        if instance_map.ndim != 2:

            raise ValueError(
                "Current Mapping implementation requires 2D "
                f"instance maps, but {path.name!r} has shape "
                f"{instance_map.shape}."
            )


        return instance_map


    def _build_image_cache(
        self,
        master: pd.DataFrame,
        instance_dir,
    ) -> dict:
        """Lazy effective instance-map cache; load only requested images."""

        allowed = set(
            master[
                "image_id"
            ].astype(str).unique()
        )

        stage = self

        if isinstance(
            instance_dir,
            Mapping,
        ):

            paths_by_image = {
                str(
                    image_id
                ): Path(
                    path
                )
                for image_id, path in instance_dir.items()
            }

            root = None

        else:

            paths_by_image = None
            root = Path(
                instance_dir
            )

        class _LazyCache(dict):

            def __missing__(
                self,
                image_id,
            ):

                image_id = str(
                    image_id
                )

                if image_id not in allowed:
                    raise KeyError(
                        "Unknown image_id requested by Mapping: "
                        f"{image_id}"
                    )

                if paths_by_image is not None:

                    if image_id not in paths_by_image:
                        raise FileNotFoundError(
                            "Effective instance resolver has no path for "
                            f"image_id={image_id!r}."
                        )

                    path = paths_by_image[
                        image_id
                    ]

                else:

                    path = (
                        root
                        / f"{image_id}.tif"
                    )

                instance_map = stage._read_instance_map(
                    path
                )

                value = {
                    "instance_map": (
                        instance_map
                    )
                }

                self[
                    image_id
                ] = value

                return value

        return _LazyCache()


    # ==================================================================
    # EXACT CLUSTER MAPS
    # ==================================================================

    def _save_cluster_maps(
        self,
        data: pd.DataFrame,
        image_cache: dict,
        output_dir: Path,
        k: int,
        d: int,
        instance_source,
        checkpoint_root: Path,
        resume: bool = False,
    ) -> tuple[
        list[dict],
        list[dict],
    ]:

        manifest_rows = []

        error_rows = []


        alpha_value = int(
            round(
                255
                * self.config
                .effective_fill_alpha
            )
        )


        stage_signature = fingerprint(
            "mapping_cluster_maps_resume_v1",
            self.config,
            {"k": int(k), "pca_dimensions": int(d)},
            source_digest(Path(__file__)),
            source_digest(Path(__file__).with_name("rendering.py")),
        )
        journal = CheckpointJournal(
            output_dir=checkpoint_root,
            stage="mapping_cluster_maps",
            resume=resume,
            stage_signature=stage_signature,
        )

        grouped_images = data.groupby(
            "image_id",
            sort=True,
        )


        total_images = grouped_images.ngroups


        for (
            image_index,
            (
                image_id,
                image_cells,
            ),
        ) in enumerate(
            grouped_images,
            start=1,
        ):

            image_id = str(
                image_id
            )

            map_path = output_dir / f"{image_id}.png"
            if isinstance(instance_source, Mapping):
                if image_id not in instance_source:
                    raise FileNotFoundError(
                        "Effective instance resolver has no path for "
                        f"image_id={image_id!r}."
                    )
                source_path = Path(instance_source[image_id])
            else:
                source_path = Path(instance_source) / f"{image_id}.tif"

            assignment_payload = [
                (str(row["cell_id"]), get_map_label(row), int(row["display_cluster"]))
                for _, row in image_cells.iterrows()
            ]
            item_signature = fingerprint(
                stage_signature,
                file_identity(source_path),
                assignment_payload,
            )
            reusable = journal.reusable_record(
                item_id=image_id,
                item_signature=item_signature,
                outputs=[map_path],
            )
            if reusable is not None:
                print(f"[{image_index}/{total_images}] RESUME {image_id}")
                saved_errors = reusable.get("metadata", {}).get("component_errors", [])
                if isinstance(saved_errors, list):
                    error_rows.extend(saved_errors)
                manifest_rows.append({
                    "mapping_type": "cluster_map",
                    "k": k,
                    "pca_dimensions": d,
                    "image_id": image_id,
                    "output_path": str(map_path),
                })
                continue

            print(
                f"[{image_index}/{total_images}] "
                f"{image_id}"
            )
            error_start = len(error_rows)


            cached = image_cache[
                image_id
            ]


            instance_map = cached[
                "instance_map"
            ]


            h, w = (
                instance_map.shape[
                    :2
                ]
            )


            rgba = np.zeros(
                (
                    h,
                    w,
                    4,
                ),
                dtype=np.uint8,
            )


            for (
                _,
                row,
            ) in image_cells.iterrows():

                map_label = (
                    get_map_label(
                        row
                    )
                )


                if map_label is None:

                    error_rows.append(
                        {
                            "k":
                                k,

                            "pca_dimensions":
                                d,

                            "image_id":
                                image_id,

                            "cell_id":
                                row[
                                    "cell_id"
                                ],
                        }
                    )

                    continue


                mask = (
                    instance_map
                    == map_label
                )


                if not np.any(
                    mask
                ):

                    error_rows.append(
                        {
                            "k":
                                k,

                            "pca_dimensions":
                                d,

                            "image_id":
                                image_id,

                            "cell_id":
                                row[
                                    "cell_id"
                                ],
                        }
                    )

                    continue


                # cluster_rgb() expects the human-facing
                # 1-based C1 ... CK display identity.
                display_cluster = int(
                    row[
                        "display_cluster"
                    ]
                )


                rgb = cluster_rgb(display_cluster, n_colors=k)


                rgba[
                    mask,
                    0,
                ] = rgb[
                    0
                ]


                rgba[
                    mask,
                    1,
                ] = rgb[
                    1
                ]


                rgba[
                    mask,
                    2,
                ] = rgb[
                    2
                ]


                rgba[
                    mask,
                    3,
                ] = alpha_value


            bgra = cv2.cvtColor(
                rgba,
                cv2.COLOR_RGBA2BGRA,
            )

            success, encoded = cv2.imencode(
                ".png",
                bgra,
            )

            if not success:
                raise RuntimeError(
                    "Failed to encode cluster map PNG: "
                    f"{map_path}"
                )

            try:
                atomic_write_bytes(map_path, encoded.tobytes())
            except Exception as exc:
                raise RuntimeError(
                    "Failed to write cluster map PNG: "
                    f"{map_path}"
                ) from exc


            manifest_rows.append(
                {
                    "mapping_type":
                        "cluster_map",

                    "k":
                        k,

                    "pca_dimensions":
                        d,

                    "image_id":
                        image_id,

                    "output_path":
                        str(
                            map_path
                        ),
                }
            )

            journal.commit(
                item_id=image_id,
                item_signature=item_signature,
                outputs=[map_path],
                metadata={"component_errors": error_rows[error_start:]},
            )


        return (
            manifest_rows,
            error_rows,
        )


    # ==================================================================
    # CELL-LABEL EXPORT
    # ==================================================================

    @staticmethod
    def _cell_label_export(
        data: pd.DataFrame,
        label_columns: Sequence[str],
    ) -> pd.DataFrame:

        preferred = [
            "raw_filename",
            "canonical_filename",
            "image_id",
            "cell_id",
        ]


        preferred.extend(
            column
            for column in label_columns
            if column not in preferred
        )


        preferred.extend(
            [
                "cluster",
                "display_cluster",
                "cluster_probability",
                "consensus_reliability",
                "pca_dimensions",
                "covariance_type",
            ]
        )


        columns = [
            column
            for column in preferred
            if column in data.columns
        ]


        return (
            data[
                columns
            ]
            .copy()
        )


    # ==================================================================
    # SELECTION EXPORT
    # ==================================================================

    @staticmethod
    def _selection_export(
        selection: pd.DataFrame,
        label_columns: Sequence[str],
    ) -> pd.DataFrame:

        preferred = [
            "raw_filename",
            "canonical_filename",
            "image_id",
            "cell_id",
        ]


        preferred.extend(
            column
            for column in label_columns
            if column not in preferred
        )


        preferred.extend(
            [
                "cluster",
                "display_cluster",
                "selection_type",
                "selection_rank",
                "cluster_probability",
                "consensus_reliability",
                "prototype_distance",
                "prototype_center_method",
                "prototype_pca_dimensions",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
            ]
        )


        columns = [
            column
            for column in preferred
            if column in selection.columns
        ]


        return (
            selection[
                columns
            ]
            .copy()
        )
