from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegmentationConfig:
    """
    Configuration for canonical instance establishment.

    connectivity
        Connected-component connectivity used for internally generated
        instances.

        None uses full spatial connectivity:

            2D -> 8-connected
            3D -> 26-connected

        This setting applies to raw- and binary-derived masks.

        External label maps preserve their supplied identities and do
        not undergo connected-component labeling.
    """

    connectivity: int | None = None

    def __post_init__(
        self,
    ) -> None:

        if self.connectivity is None:
            return

        self.connectivity = int(
            self.connectivity
        )

        if self.connectivity < 1:
            raise ValueError(
                "Segmentation connectivity must be >= 1."
            )
