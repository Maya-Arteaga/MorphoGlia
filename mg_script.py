from pathlib import Path

from morphoglia import PipelineConfig, run_pipeline

# ======================================================================
# MORPHOGLIA 2.0.0 — SCRIPT INTERFACE
# ======================================================================
# Run this file FROM THE MorphoGlia_2.0.0 FOLDER.
#
# macOS / Linux:
#     pixi run python mg_script.py
#
# Windows with global Pixi:
#     pixi run python .\mg_script.py
#
# Windows when the installer created local Pixi:
#     .\.pixi-home\bin\pixi.exe run python .\mg_script.py
# ======================================================================

INPUT_DIR = Path("/path/to/directory")

if str(INPUT_DIR) == "/path/to/directory":
    raise SystemExit("\nEdit INPUT_DIR in mg_script.py before running MorphoGlia.\n")

config = PipelineConfig(input_dir=INPUT_DIR)

# INPUT ----------------------------------------------------------------
config.resume = True
config.metadata.microns_per_pixel = 1.0  # CHANGE to your microscope calibration.
config.preprocessing.spatial_dimension = "2d"  # "2d", "3d"
config.preprocessing.input_mode = "binary"      # "raw", "binary", "labels"
config.preprocessing.invert = False
config.preprocessing.save_intermediate = False

# Raw only: "mg1", "mg2", "mg3", "mg4"
config.preprocessing.preset = "mg1"

# RUN STAGES -----------------------------------------------------------
# True = recompute. False = do not recompute.
# Compatible prerequisite outputs may still be reused.
config.run.metadata = True
config.run.preprocessing = True
config.run.segmentation = True
config.run.instance_refinement = True
config.run.morphometrics = True
config.run.category = False  # Enable only after defining category_fields.
config.run.dim_reduction_clustering = True
config.run.mapping = True
config.run.spatial_analysis = False  # Not public in 2.0.0.
config.run.plots = True

# INSTANCE POSTPROCESSING ---------------------------------------------
config.instance_refinement.remove_small = True
config.instance_refinement.tubular_action = "reconnect"  # keep/remove/reconnect
config.instance_refinement.large_action = "split"        # keep/remove/split

# MORPHOLOGY STATES ---------------------------------------------------
# None = estimate supported K and use the automatic preferred solution.
config.clustering.number_of_morphology_states = None

# NOMENCLATURE --------------------------------------------------------
# Positions are 1-based. Leave absent fields as None.
n = config.metadata.nomenclature
n.source_separator = "_"
n.subject = None
n.genotype = None
n.sex = None
n.cell_type = None
n.condition = None
n.treatment = None
n.eye = None
n.layer = None
n.quadrant = None
n.region = None
n.hemisphere = None
n.tissue = None
n.sample = None
n.section = None
n.spatial_bin = None
n.replicate = None
n.session = None
n.timepoint = None
n.date = None
n.stain = None
n.objective = None
n.channel = None
n.acquisition = None
n.run = None

# CATEGORY ------------------------------------------------------------
# Example:
# config.run.category = True
# config.category.category_fields = ["condition", "layer"]
config.category.category_fields = []

# PLOTS ---------------------------------------------------------------
config.plots.dpi = 300
config.plots.morphology_state_palette = "MG1"
config.plots.metadata_order = {}

# RUN -----------------------------------------------------------------
result = run_pipeline(config)
