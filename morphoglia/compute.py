from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from time import perf_counter

import csv
import os

from threadpoolctl import (
    threadpool_info,
    threadpool_limits,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

@dataclass
class ComputeConfig:
    """
    Global MorphoGlia compute-resource policy.

    cpu_fraction
        Fraction of the machine's logical CPUs that MorphoGlia may use.

        Default = 0.80.

    max_cpus
        Optional hard upper bound.

        None
            Use only cpu_fraction.

        Integer
            Never exceed this number of logical CPUs.

    print_summary
        Print the resolved compute budget at pipeline start.
    """

    cpu_fraction: float = 0.80

    max_cpus: int | None = None

    print_summary: bool = True


    def __post_init__(
        self,
    ) -> None:

        self.cpu_fraction = float(
            self.cpu_fraction
        )


        if not (
            0.0
            < self.cpu_fraction
            <= 1.0
        ):

            raise ValueError(
                "compute.cpu_fraction must be in "
                "the interval (0, 1]."
            )


        if self.max_cpus is not None:

            self.max_cpus = int(
                self.max_cpus
            )


            if self.max_cpus < 1:

                raise ValueError(
                    "compute.max_cpus must be >= 1 "
                    "or None."
                )


        self.print_summary = bool(
            self.print_summary
        )


    def resolve_cpu_budget(
        self,
        logical_cpus: int | None = None,
    ) -> int:
        """
        Resolve the actual logical-CPU budget.

        We floor rather than round so the requested fraction is never
        exceeded.

        On multicore machines, at least one logical CPU is left outside
        MorphoGlia whenever possible.
        """

        logical = int(
            logical_cpus
            or os.cpu_count()
            or 1
        )


        budget = max(
            1,
            int(
                logical
                * self.cpu_fraction
            ),
        )


        if logical > 1:

            budget = min(
                budget,
                logical - 1,
            )


        if self.max_cpus is not None:

            budget = min(
                budget,
                self.max_cpus,
            )


        return max(
            1,
            budget,
        )


# ======================================================================
# RUNTIME INFORMATION
# ======================================================================

@dataclass
class ComputeRuntimeInfo:

    logical_cpus: int

    cpu_fraction: float

    max_cpus: int | None

    cpu_budget: int

    runtime_seconds: float = 0.0

    pools: tuple[
        dict,
        ...,
    ] = ()


# ======================================================================
# TECHNICAL RECORD
# ======================================================================

def _save_compute_record(
    config,
    info: ComputeRuntimeInfo,
) -> Path | None:

    output_dir = getattr(
        config,
        "output_dir",
        None,
    )


    if output_dir is None:

        return None


    technical_dir = (
        Path(
            output_dir
        )
        / "Technical_Record"
    )


    technical_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    path = (
        technical_dir
        / "compute_resources.csv"
    )


    pool_description = "; ".join(
        (
            f"{pool.get('internal_api', 'unknown')}:"
            f"{pool.get('prefix', 'unknown')}:"
            f"{pool.get('num_threads', 'unknown')}"
        )
        for pool in info.pools
    )


    row = {
        "logical_cpus":
            info.logical_cpus,

        "requested_cpu_fraction":
            info.cpu_fraction,

        "max_cpus":
            (
                ""
                if info.max_cpus is None
                else info.max_cpus
            ),

        "cpu_budget":
            info.cpu_budget,

        "effective_fraction":
            (
                info.cpu_budget
                / info.logical_cpus
            ),

        "runtime_seconds":
            info.runtime_seconds,

        "thread_pools_under_limit":
            pool_description,
    }


    with path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row
            ),
        )

        writer.writeheader()

        writer.writerow(
            row
        )


    return path


# ======================================================================
# COMPUTE CONTEXT
# ======================================================================

@contextmanager
def compute_context(
    compute_config: ComputeConfig,
):

    compute_config.__post_init__()


    logical_cpus = int(
        os.cpu_count()
        or 1
    )


    cpu_budget = (
        compute_config
        .resolve_cpu_budget(
            logical_cpus
        )
    )


    started = perf_counter()


    # ------------------------------------------------------------------
    # threadpool_limits dynamically controls already-loaded BLAS/OpenMP
    # libraries including OpenBLAS and libomp.
    #
    # This is preferable to relying on shell environment variables
    # because MorphoGlia may be run from:
    #
    #     Terminal
    #     Spyder
    #     notebooks
    #     GUI
    #     CLI
    # ------------------------------------------------------------------

    with threadpool_limits(
        limits=cpu_budget,
    ):

        pools = tuple(
            threadpool_info()
        )


        info = ComputeRuntimeInfo(
            logical_cpus=(
                logical_cpus
            ),
            cpu_fraction=(
                compute_config
                .cpu_fraction
            ),
            max_cpus=(
                compute_config
                .max_cpus
            ),
            cpu_budget=(
                cpu_budget
            ),
            pools=(
                pools
            ),
        )


        yield info


    info.runtime_seconds = float(
        perf_counter()
        - started
    )


# ======================================================================
# PIPELINE DECORATOR
# ======================================================================

def with_compute_limits(
    function,
):
    """
    Execute the complete pipeline under MorphoGlia's compute budget.
    """

    @wraps(
        function
    )
    def wrapped(
        config,
        *args,
        **kwargs,
    ):

        compute_config = getattr(
            config,
            "compute",
            ComputeConfig(),
        )


        with compute_context(
            compute_config
        ) as runtime:

            if (
                compute_config
                .print_summary
            ):
                """
                print()
                print("=" * 72)
                print("COMPUTE RESOURCES")
                print("=" * 72)
                print()

                print(
                    "Logical CPUs:",
                    runtime.logical_cpus,
                )

                print(
                    "CPU target:",
                    f"{100 * runtime.cpu_fraction:.0f}%",
                )

                print(
                    "CPU budget:",
                    (
                        f"{runtime.cpu_budget} / "
                        f"{runtime.logical_cpus}"
                    ),
                )

                print()
                
                """


            result = function(
                config,
                *args,
                **kwargs,
            )


        _save_compute_record(
            config,
            runtime,
        )


        return result


    return wrapped


__all__ = [
    "ComputeConfig",
    "ComputeRuntimeInfo",
    "compute_context",
    "with_compute_limits",
]
