from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline
from .config import PIPELINE_STAGES


# ======================================================================
# PARSER
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="morphoglia",
        description=(
            "MorphoGlia morphology-analysis pipeline."
        ),
    )


    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )


    # ==================================================================
    # RUN
    # ==================================================================

    run_parser = subparsers.add_parser(
        "run",
        help="Run the MorphoGlia pipeline.",
    )


    # ------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------

    run_parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        help=(
            "Input image directory. "
            "Do not provide this when using --config."
        ),
    )


    run_parser.add_argument(
        "--config",
        type=Path,
        help=(
            "YAML configuration file. "
            "When provided, input_dir is read from YAML."
        ),
    )


    run_parser.add_argument(
        "--input-mode",
        choices=(
            "binary",
            "labels",
            "raw",
        ),
        default=None,
        help=(
            "Input type. If omitted, use the "
            "PipelineConfig default or YAML value."
        ),
    )


    # ------------------------------------------------------------------
    # MORPHOLOGY-STATE INTERPRETATION
    # ------------------------------------------------------------------

    run_parser.add_argument(
        "--number-of-morphology-states",
        type=int,
        default=None,
        metavar="COUNT",
        help=(
            "Choose one stable/reproducible state count from a previous "
            "Dimensionality Reduction and Clustering estimation. If omitted, "
            "MorphoGlia applies and reports its automatic data-driven criterion."
        ),
    )


    # ------------------------------------------------------------------
    # CATEGORY
    # ------------------------------------------------------------------

    run_parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        metavar="FIELD",
        help=(
            "Filename metadata fields defining experimental "
            "categories, for example: "
            "--category condition region"
        ),
    )


    # ------------------------------------------------------------------
    # OBJECT QC
    # ------------------------------------------------------------------

    qc_group = run_parser.add_argument_group(
        "Object QC"
    )


    qc_group.add_argument(
        "--small-objects",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply or disable small-object exclusion."
        ),
    )


    qc_group.add_argument(
        "--large-objects",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply or disable large-object exclusion."
        ),
    )


    qc_group.add_argument(
        "--tubular",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply or disable tubular-object exclusion."
        ),
    )


    # ------------------------------------------------------------------
    # PIPELINE STAGES
    # ------------------------------------------------------------------

    stage_group = run_parser.add_argument_group(
        "Pipeline stages"
    )


    for stage in PIPELINE_STAGES:

        option = (
            "--stage-"
            + stage.replace(
                "_",
                "-",
            )
        )


        stage_group.add_argument(
            option,
            dest=stage,
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                f"Enable or disable the "
                f"{stage.replace('_', ' ')} stage."
            ),
        )


    return parser


# ======================================================================
# CONFIGURATION
# ======================================================================

def _build_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> PipelineConfig:

    # ==================================================================
    # CONFIG SOURCE
    # ==================================================================

    if (
        args.config is not None
        and args.input_dir is not None
    ):

        parser.error(
            "Use either an input directory or --config, "
            "not both."
        )


    if args.config is not None:

        config = PipelineConfig.from_yaml(
            args.config
        )


    elif args.input_dir is not None:

        config = PipelineConfig(
            input_dir=args.input_dir
        )


    else:

        parser.error(
            "Provide an input directory or --config."
        )


    # ==================================================================
    # SIMPLE CLI OVERRIDES
    # ==================================================================

    if args.input_mode is not None:

        config.preprocessing.input_mode = (
            args.input_mode
        )


    if args.category is not None:

        config.category.category_fields = list(
            args.category
        )


    if args.number_of_morphology_states is not None:

        config.clustering.number_of_morphology_states = int(
            args.number_of_morphology_states
        )


    qc_overrides = {
        "small_objects":
            args.small_objects,

        "large_objects":
            args.large_objects,

        "tubular":
            args.tubular,
    }


    for (
        name,
        value,
    ) in qc_overrides.items():

        if value is None:
            continue


        setattr(
            config.qc,
            name,
            value,
        )


    # ==================================================================
    # STAGE OVERRIDES
    # ==================================================================

    for stage in PIPELINE_STAGES:

        value = getattr(
            args,
            stage,
        )


        if value is None:
            continue


        setattr(
            config.run,
            stage,
            value,
        )


    return config


# ======================================================================
# MAIN
# ======================================================================

def main(
    argv=None,
) -> int:

    parser = _build_parser()

    args = parser.parse_args(
        argv
    )


    if args.command == "run":

        config = _build_config(
            args,
            parser,
        )


        run_pipeline(
            config
        )


        return 0


    parser.error(
        f"Unknown command: {args.command}"
    )


    return 2


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
