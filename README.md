# MorphoGlia


<img width="1161" height="260" alt="LOGO_MG3" src="https://github.com/user-attachments/assets/5f2376df-bfb2-413e-b89f-d29b77152570" />



**MorphoGlia** is a cell-morphology analysis pipeline for image preprocessing,
instance segmentation/postprocessing, morphometrics, morphology-state
estimation, tissue mapping, and visualization.

MorphoGlia 2.0.0 provides two interfaces over the same analysis backend:

- **GUI** — user-friendly version.
- **`mg_script.py`** — explicit, reproducible configuration in Python, same logic as the GUI menu.

The release uses a locked **Pixi** environment. Users do not need to manually
assemble a Python or Conda environment.

**License:** Apache License 2.0.

## Citation

If you use **MorphoGlia** in your research, please cite:

> Maya-Arteaga JP, Martínez-Orozco H, Diaz-Cintra S (2024).  
> **MorphoGlia, an interactive method to identify and map microglia morphologies, demonstrates differences in hippocampal subregions of an Alzheimer's disease mouse model.**  
> *Frontiers in Cellular Neuroscience* 18:1505048.  
> https://doi.org/10.3389/fncel.2024.1505048

BibTeX:

```bibtex
@article{MayaArteaga2024MorphoGlia,
  author  = {Maya-Arteaga, Juan Pablo and Martínez-Orozco, Humberto and Diaz-Cintra, Sofía},
  title   = {MorphoGlia, an interactive method to identify and map microglia morphologies, demonstrates differences in hippocampal subregions of an Alzheimer's disease mouse model},
  journal = {Frontiers in Cellular Neuroscience},
  volume  = {18},
  pages   = {1505048},
  year    = {2024},
  doi     = {10.3389/fncel.2024.1505048}
}
```

## Implementation

MorphoGlia implements an end-to-end, provenance-aware workflow in which image
processing, instance establishment/refinement, morphometric measurement,
latent-space estimation, multiresolution morphology-state inference, mapping,
and visualization are performed through a common configuration object. The GUI
and `mg_script.py` therefore execute the same scientific backend rather than
separate analysis implementations. Stage outputs are checkpointed and can be
reused when their inputs and configuration remain compatible, while the
Technical Record preserves the effective analytical settings used for each run.

The analytical workflow is deliberately unsupervised with respect to biological
group labels. Experimental categories are used for organization and downstream
comparison/visualization, but they are not used to discover the morphology
states.

### Methods-ready description

The following text is intended as a **methods template**. It is written in a
paper-style narrative so that users can copy it into a Methods section and
replace the bracketed fields according to the analysis performed.

> **A concise methods description of the MorphoGlia pipeline is provided below.
> Replace the bracketed fields according to the analysis performed.**
>
> Cell morphology was analyzed using MorphoGlia (v2.0.0; Maya-Arteaga et al.,
> 2024). Images were provided as **[raw fluorescence images / binary masks /
> pre-labeled instance images]**. **[For raw images: preprocessing was performed
> using the MG1/MG2/MG3/MG4 preset; insert the corresponding preprocessing
> description.]** **[If used: instances were further refined to remove small
> objects, reconnect tubular fragments, and/or separate candidate overlapping
> cells.]** Each cell was represented by a multidimensional morphometric profile
> integrating whole-cell geometry, convex-hull and soma measurements, skeleton
> and branch topology, Sholl analysis, branch order, and tortuosity-related
> features.
>
> Cells from all experimental groups were then pooled to construct a **shared
> latent morphological space**, without using experimental labels to define its
> structure. PCA was applied to the morphometric matrix, and plausible
> dimensionalities were determined using Parallel Analysis (500 permutations;
> 95th-percentile threshold) and the Broken-Stick criterion, with Two-NN
> intrinsic dimensionality retained as an independent diagnostic. Within this
> latent space, morphological states were identified using a **multiresolution
> Gaussian mixture modeling framework**. Candidate numbers of states, PCA
> dimensionalities, and covariance structures were systematically evaluated,
> with covariance models selected by Bayesian information criterion (BIC).
> Partition robustness was assessed across 50 stratified subsamples containing
> 80% of the cells using adjusted Rand index (ARI), cross-dimensional agreement,
> membership probability, and minimum state size. This stability landscape was
> used to identify reproducible morphological resolutions independently of
> experimental labels. **[The automatically selected supported solution / a
> supported solution with K = ___ morphological states]** was retained for
> downstream analysis, with cell-level consensus reliability estimated across
> 100 iterations. Only after morphological states had been defined were
> **[experimental categories / conditions / treatments / regions]** introduced
> for statistical comparison, thereby separating morphology-state discovery
> from biological interpretation.

For raw images, replace the preprocessing placeholder with the sentence
corresponding to the preset that was actually used:

- **MG1:** Raw fluorescence images were robustly intensity-rescaled, corrected
  by percentile background subtraction, rescaled, corrected by Gaussian
  background subtraction, enhanced using directional grayscale opening,
  rescaled again, binarized using Li thresholding, and filtered to remove small
  noise and compact objects.
- **MG2:** Raw fluorescence images were robustly intensity-rescaled, corrected
  by percentile and Gaussian background subtraction, enhanced using directional
  grayscale opening, rescaled, binarized using Li thresholding, and filtered to
  remove small noise.
- **MG3:** Raw fluorescence images were robustly intensity-rescaled, corrected
  by Gaussian background subtraction, binarized using Li thresholding, and
  filtered to remove small noise.
- **MG4:** Raw fluorescence images were corrected by percentile background
  subtraction, robustly intensity-rescaled, binarized using Li thresholding, and
  filtered to remove small noise.

Users should report the preprocessing, refinement, scaling, clustering,
morphology-state, and plotting settings used in their analysis. The run-specific Technical Record is intended to provide the exact
configuration needed to complete or verify the Methods description.

## Download

Download `MorphoGlia_2.0.0.zip` from the **Releases** section of:

https://github.com/Maya-Arteaga/MorphoGlia

Extract the complete folder before running MorphoGlia. Do not run files from
inside the ZIP.

A convenient location is:

```text
macOS:   ~/Desktop/MorphoGlia_2.0.0
Windows: C:\Users\YOUR_NAME\Desktop\MorphoGlia_2.0.0
```

Keep the release folder together.

## GUI installation

### macOS

**First use:** double-click `Install MorphoGlia.command`.

The installer finds or installs Pixi, installs the locked environment, runs the
compatibility checks, creates `MorphoGlia.app`, and opens the GUI.

If macOS blocks the unsigned installer, right-click it, choose **Open**, then
confirm **Open**.

**Normal use:** double-click `MorphoGlia.app`.

`MorphoGlia.command` is included as a fallback launcher.

### Windows

MorphoGlia 2.0.0 currently targets **Windows x86-64** (Intel/AMD).

**First use:** double-click `Install MorphoGlia.bat`.

The installer can keep Pixi locally inside the MorphoGlia folder, so users do
not need a separate Python or Conda installation.

**Normal use:** double-click `MorphoGlia.bat`.

### Linux

From a terminal inside the extracted release folder:

```bash
chmod +x install_morphoglia.sh
./install_morphoglia.sh
```

Later:

```bash
pixi run gui
```

## Basic GUI workflow

1. Choose the folder containing the TIFF images.
2. Enter the microscope pixel size in µm/pixel.
3. Choose 2D or 3D biological image geometry.
4. Choose **Raw**, **Binary**, or **Labels** input.
5. For Raw input, choose preprocessing preset **MG1–MG4**.
6. Define filename nomenclature.
7. Enable the stages that should be recomputed.
8. If Category is enabled, choose the metadata fields defining categories.
9. After Dimensionality Reduction and Clustering has been estimated, use the
   automatic preferred morphology-state solution or choose one of the supported
   resolutions.
10. Choose plot DPI and morphology-state palette.
11. Press **Run**.

Outputs are written under:

```text
YOUR_DATASET/
└── _MorphoGlia/
```

Original source images are not renamed or overwritten.

## Resume

**Resume checked:** reuse compatible completed work when possible.

**Resume unchecked:** recompute enabled stages and replace their generated
outputs.

A downstream stage may reuse a compatible prerequisite from an earlier run. If
a required prerequisite is unavailable, that stage is skipped.

## Raw preprocessing presets

- **MG1** — full MorphoGlia fluorescence workflow.
- **MG2** — background subtraction plus directional enhancement workflow.
- **MG3** — robust-rescale/Gaussian-subtraction/Li-threshold workflow.
- **MG4** — percentile-background/rescale/Li-threshold workflow.

MG1 is the default full workflow. Preset suitability is dataset-dependent;
inspect preprocessing output/QC rather than treating a preset as a biological
assumption.



## Script interface: `mg_script.py`

`mg_script.py` provides the same MorphoGlia analysis backend as the GUI, but
allows the analysis configuration to be defined explicitly in Python.

You do **not** need to install Python, Conda, NumPy, scikit-image, or the
individual MorphoGlia dependencies manually. **Pixi** creates and manages the
complete environment from the included `pixi.toml` and `pixi.lock` files.

First download and extract `MorphoGlia_2.0.0.zip` as described in the
[Download](#download) section above.

### Quick overview

```text
FIRST TIME
Download and extract MorphoGlia
        ↓
Install Pixi / prepare the environment
        ↓
Edit mg_script.py
        ↓
Run the analysis

EVERY LATER ANALYSIS
Edit mg_script.py
        ↓
pixi run python mg_script.py
```

Always run the script **from inside the extracted `MorphoGlia_2.0.0` folder**.

### macOS

#### First-time setup

The easiest option is to use the included installer:

```text
Install MorphoGlia.command
```

Double-click `Install MorphoGlia.command`.

This installs Pixi if necessary, prepares the locked MorphoGlia environment,
runs the compatibility checks, creates the GUI application, and opens MorphoGlia.

If you already installed MorphoGlia successfully through the GUI installer,
**you do not need to install anything else for the script interface**.

#### Run the script

Open Terminal and enter the MorphoGlia folder:

```bash
cd ~/Desktop/MorphoGlia_2.0.0
```

Then run:

```bash
pixi run python mg_script.py
```

If `pixi` is not available on your PATH:

```bash
~/.pixi/bin/pixi run python mg_script.py
```

#### Manual Pixi installation

Only use this if you did not use `Install MorphoGlia.command`.

Install Pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Open a new Terminal window and check that Pixi is available:

```bash
pixi --version
```

Then enter the MorphoGlia folder and prepare the shipped environment once:

```bash
cd ~/Desktop/MorphoGlia_2.0.0
pixi install --locked
```

After that, run MorphoGlia normally with:

```bash
pixi run python mg_script.py
```

### Windows PowerShell

#### First-time setup

The easiest option is to use the included installer:

```text
Install MorphoGlia.bat
```

Double-click `Install MorphoGlia.bat`.

The installer can install Pixi locally inside the MorphoGlia folder and prepare
the complete locked environment. You do not need to install Python or Conda
separately.

If you already installed MorphoGlia successfully through the Windows installer,
**you do not need to install anything else for the script interface**.

#### Run the script

Open PowerShell and enter the MorphoGlia folder:

```powershell
cd "$HOME\Desktop\MorphoGlia_2.0.0"
```

If Pixi is installed globally, run:

```powershell
pixi run python .\mg_script.py
```

If you are using the local Pixi installation created by the MorphoGlia
installer, run:

```powershell
.\.pixi-home\bin\pixi.exe run python .\mg_script.py
```

#### Manual environment preparation

If Pixi is installed globally and the environment has not yet been prepared,
run once:

```powershell
pixi install --locked
```

If you are using the local Pixi installation created by MorphoGlia:

```powershell
.\.pixi-home\bin\pixi.exe install --locked
```

After that, use the normal run command shown above.

### Linux

#### First-time setup

Install Pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Open a new terminal and verify:

```bash
pixi --version
```

Enter the extracted MorphoGlia folder:

```bash
cd ~/Desktop/MorphoGlia_2.0.0
```

Prepare the locked environment once:

```bash
pixi install --locked
```

#### Run the script

```bash
pixi run python mg_script.py
```

### Optional shortcut

The included Pixi configuration also provides:

```bash
pixi run script
```

This is equivalent to launching the script through the predefined Pixi task.
The explicit command

```bash
pixi run python mg_script.py
```

is recommended in the documentation because it makes clear which file is being
executed.

## Editing `mg_script.py`

Change:

```python
INPUT_DIR = Path("/path/to/directory")
```

to your dataset directory, then set the real microscope calibration:

```python
config.metadata.microns_per_pixel = 0.227
```

Choose input type:

```python
config.preprocessing.input_mode = "binary"
```

Valid values are `"raw"`, `"binary"`, and `"labels"`.

For Raw input:

```python
config.preprocessing.preset = "mg1"
```

with `mg1`, `mg2`, `mg3`, or `mg4`.

A stage setting such as:

```python
config.run.segmentation = True
```

means **recompute that stage**. `False` means do not recompute it; compatible
saved prerequisites may still be reused.

Spatial Analysis is intentionally not part of the public 2.0.0 release.

## Filename nomenclature

Positions are **1-based**. For:

```text
M13_HTN_OI_GCL_N_R3_rep1.tif
```

you could use:

```python
n.source_separator = "_"
n.subject = 1
n.condition = 2
n.eye = 3
n.layer = 4
n.quadrant = 5
n.spatial_bin = 6
n.replicate = 7
```

Fields absent from the filename should remain `None`.

## Category definition

```python
config.run.category = True
config.category.category_fields = [
    "condition",
    "layer",
]
```

The active category definition is reflected in the plot-output directory so
alternative categorical analyses can coexist.

## Morphology-state resolution

On the first DRC run:

```python
config.clustering.number_of_morphology_states = None
```

MorphoGlia estimates supported state counts and records the automatic preferred
interpretation. On later runs, you can choose one of the supported K values.

## Plot settings

```python
config.plots.dpi = 300
config.plots.morphology_state_palette = "MG1"
```

Public palette names:

```text
MG1, MG2, MG3, MG4,
plasma, tab10, tab20,
hls, rocket, flare, magma, Spectral
```

## Reproducibility

MorphoGlia 2.0.0 ships with `pixi.toml` and `pixi.lock`.

Supported Pixi targets:

- macOS Intel (`osx-64`)
- macOS Apple Silicon (`osx-arm64`)
- Windows x86-64 (`win-64`)
- Linux x86-64 (`linux-64`)

Check an installation with:

```bash
pixi run doctor
```

Do not assume historical MorphoGlia output folders are interchangeable with the
2.0.0 architecture. For a clean major-version analysis, use a fresh output
directory or dataset copy.

## Troubleshooting

**`pixi: command not found` on macOS**

```bash
~/.pixi/bin/pixi --version
```

**macOS blocks `Install MorphoGlia.command`**

Right-click it and choose **Open**.

**Windows installation fails**

The installer writes `MorphoGlia_install_log.txt` in the release directory.

**A later stage is skipped**

Check whether its required upstream output exists. Run the necessary earlier
stage once, then rerun the downstream stage.


## Contact and support

For bug reports or unexpected behavior, please open a
[GitHub issue](https://github.com/Maya-Arteaga/MorphoGlia/issues) and notify and send files to
**Email:**  [juan.mayaarteaga@ist.ac.at](mailto:juan.mayaarteaga@ist.ac.at)** 

To help reproduce and diagnose the problem, please briefly describe the issue
and the steps that led to it. When available, please attach the following files
from the `Technical_Record` folder:

- `configuration.csv`
- `compute_resources.csv`
- `analysis_summary.json`
- `errors.log`

For issues related to instance refinement, please also include the relevant
`instance_refinement_*` files.

### Scientific questions and contact

**Juan Pablo Maya Arteaga**  
Institute of Science and Technology Austria (ISTA)  
**Email:**  [juan.mayaarteaga@ist.ac.at](mailto:juan.mayaarteaga@ist.ac.at)

Collaboration inquiries are also welcome, particularly for projects involving MorphoGlia or related quantitative imaging challenges, including cell culture, cell migration, multi-label imaging, and further characterization of biological phenotypes.

## Version

**MorphoGlia 2.0.0** — Apache License 2.0.
