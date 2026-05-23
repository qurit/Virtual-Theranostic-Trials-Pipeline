# PyTheraTwin Pipeline

This pipeline creates patient-specific **theranostic digital twins** by combining CT-based anatomy/segmentation with PBPK kinetics and physics-based SPECT simulation/reconstruction, supporting research in diagnosis and therapy planning for **Virtual Theranostic Trials**.

---

## Overview

**Theranostics** is a "diagnose and treat" approach that uses the same biological target to both detect disease and guide targeted therapy.

**Radiopharmaceuticals (RPTs)** couple a targeting molecule with a radionuclide that accumulates in tissues expressing a biomarker (e.g., tumors). As the radionuclide decays, emitted particles can deliver therapy while emitted photons enable quantitative imaging. For example, **¹⁷⁷Lu-PSMA** targets PSMA-expressing prostate cancer and supports post-therapy SPECT imaging.

The **PyTheraTwin Pipeline** is a quantitative software framework that uses real patient CT data to build end-to-end digital twins for theranostics research. It integrates:

- **Patient-specific anatomy** from clinical CT scans
- **Organ/tumor segmentation** (TotalSegmentator-based workflows)
- **Pharmacokinetic (PBPK) modeling** to generate time-activity behavior
- **Monte Carlo SPECT simulation + reconstruction** (SIMIND/PyTomography) to produce quantitative images
- **Monte Carlo dosimetry simulation** (OpenGATE/Geant4) to generate organ-level dose maps

Because uptake and dose can vary substantially between patients, PyTheraTwin supports personalized evaluation of therapy strategies by enabling controlled, repeatable experiments across anatomy, kinetics, and imaging physics.

![PyTheraTwin Pipeline Overview](figures/pipeline_overview.png)

---

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| **Phase 1: Digital Twin & Ground Truth** | Segmentation + ROI unification (`--segmentation`) · Synthetic lesion generation (`--synthetic_lesions`, requires specs in config) · PBPK TAC generation (`--pbpk`) |
| **Phase 2: Simulations** | SIMIND SPECT simulation (`--spect`) · OpenGATE dosimetry (`--dosimetry`) |
| **Phase 3: Post-Processing** | SPECT post-processing (`--postprocess` + `--spect`) · Dosimetry post-processing (`--postprocess` + `--dosimetry`) |

---

## Installation

### Requirements
- Conda (Miniconda/Anaconda)
- A working C/C++ build toolchain for compiling certain Python dependencies (varies by OS)
- **SIMIND** installed separately (see Step 2)

#### Recommended
- Linux for full pipeline runs
- Sufficient disk for intermediate SIMIND outputs (can be large depending on photons / frames / ROIs)

### 1) Create the conda environment

```bash
conda env create -f environment.yml
conda activate pytheratwin
```

> This environment includes all required Python dependencies (TotalSegmentator, PyTomography, OpenGATE, PyCNO, etc.).

### 2) Install SIMIND (external)

**SIMIND is an external dependency** and must be installed separately.

#### Step 1 — Download and install SIMIND
```text
https://www.msf.lu.se/en/research/simind-monte-carlo-program/downloads
```

#### Step 2 — Point the pipeline to SIMIND in pipeline_paths.json
In your paths JSON found at src/data/pipeline_paths.json, set:
- `input_paths.SIMINDDirectory` = directory containing the `simind` executable

(Optional sanity check)
```bash
which simind
echo $SMC_DIR
```

---

## Running the Pipeline

Two interfaces are available — both call the same `main.py` entry point and number patients from `CT_1` upward.

| | Web UI | Command Line (CLI) |
|---|---|---|
| **Best for** | New users, interactive setup | Scripting, batch runs, remote headless servers |
| **Config** | Auto-generated form with tooltips | Hand-edited JSON file |
| **CT input** | Folder picker dialog or absolute path | `--input_ct_dir` flag |
| **Flags** | Toggle buttons in the browser | CLI flags (`--segmentation`, `--spect`, …) |
| **Start** | `python run_server.py` | `python main.py …` |

---

### Web UI

#### Launch

```bash
conda activate pytheratwin
python run_server.py
```

#### Step-by-step

| Page | What to do |
|------|------------|
| **1-Intro** | Read the pipeline overview and prerequisite checklist, then click **Get Started**. |
| **2 — CT Input** | Click **Choose CT Folder** to open a native folder picker, or type an absolute path into the text field and click **Scan**. Patients are detected automatically — DICOM subdirectories and `.nii` / `.nii.gz` files both work. Axial, coronal, and sagittal quick previews are generated for each CT mid-slices. |
| **3 — Configure** | Enter a project name (determines the parent output folder). Toggle pipeline stages: **Segmentation**, **PBPK**, **Synthetic Lesions**, **SPECT**, **Dosimetry**, **Post-Processing**. Choose **PRODUCTION** or **DEBUG** mode. Expand each patient card to adjust per-patient parameters. Synthetic lesions are configured per-patient by adding organ specs in Phase 1. If profiling is enabled, the UI also exposes a sampling-interval dropdown. If output directories from a previous run are detected, the UI compares the selected CT and effective stage config against saved rerun metadata before allowing a rerun. |
| **4 — Run** | Review the per-patient configuration summary, then click **Run Pipeline**. |

---

### Command Line (CLI)

#### 1) Create your patient config

```bash
cp config_template.json inputs/config.json
# edit config.json
```

#### 2) CT Input

Place CT data in any directory:

```bash
mkdir -p inputs/ct_input
# add DICOM subfolders and/or .nii / .nii.gz files
```

Each top-level DICOM folder or NIfTI file is treated as one patient. Other unsupported top-level entries are ignored in batch mode.

CLI preflight validates inputs before launching any patient run:
- `src/tests/validate_ct.py` checks single CT paths and batch-directory entries.
- `src/tests/validate_config.py` checks the run configuration after developer paths are injected.

#### 3) Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config_file PATH` | required | Path to your JSON config |
| `--input_ct_dir PATH` | required | Directory containing CT inputs (NIfTI files or DICOM folders) |
| `--mode {DEBUG,PRODUCTION}` | `PRODUCTION` | Verbosity and intermediate file cleanup |
| `--segmentation` | off | Run TotalSegmentator segmentation + ROI unification (Phase 1) |
| `--pbpk` | off | Run PBPK TAC generation (Phase 1) |
| `--synthetic_lesions` | off | Run synthetic lesion generation (Phase 1, requires specs in config) |
| `--spect` | off | Run SIMIND SPECT projection simulation (Phase 2) |
| `--dosimetry` | off | Run OpenGATE dosimetry simulation (Phase 2) |
| `--postprocess` | off | Run post-processing for whichever simulations ran (Phase 3) |
| `--profile INTERVAL_S` | off | Enable per-stage CPU/RAM profiling; pass sampling interval in seconds (`0.1`–`3.0`) |

#### 4) Run

**Phase 1 only (segmentation + TACs):**
```bash
python -u main.py \
  --config_file inputs/my_config.json \
  --input_ct_dir inputs/ct_input \
  --segmentation --pbpk
```

**Phase 1 with synthetic lesions:**
```bash
python -u main.py \
  --config_file inputs/my_config.json \
  --input_ct_dir inputs/ct_input \
  --segmentation --synthetic_lesions --pbpk
```

**Full SPECT pipeline:**
```bash
python -u main.py \
  --config_file inputs/my_config.json \
  --input_ct_dir inputs/ct_input \
  --segmentation --pbpk --spect --postprocess
```

**Full dosimetry pipeline:**
```bash
python -u main.py \
  --config_file inputs/my_config.json \
  --input_ct_dir inputs/ct_input \
  --segmentation --pbpk --dosimetry --postprocess
```

**Run everything (with profiling):**
```bash
python -u main.py \
  --config_file inputs/my_config.json \
  --input_ct_dir inputs/ct_input \
  --mode DEBUG \
  --segmentation --pbpk --synthetic_lesions \
  --spect --dosimetry --postprocess \
  --profile 2.0
```

---

## Outputs

Each CT input generates a subfolder `CT_<index>/` inside the project folder `<output_folder_title>/`, with subfolders per phase.

### Example Structure

```
test_run/
  CT_1/
  pipeline_metadata/                         <- persistent rerun guard metadata (created on first run)
    ct_input.json                           <- saved CT identity / provenance
    segmentation_stage.json                 <- stage-level rerun guard snapshot
    pbpk_tac_stage.json
    simind_simulation_stage.json
    opengate_simulation_stage.json
    spect_postprocess_stage.json
    dosemap_postprocess_stage.json
  digital_twin/                              <- Phase 1
    ct.nii.gz                                <- standardized CT handoff
    digital_twin.nii.gz                      <- unified PyTheraTwin multilabel segmentation handoff
    segmentation_stage/
    synthetic_lesions_stage/                  <- (if synthetic lesions enabled)
    pbpk_tac_stage/
      pbpk_tacs.json                         <- human-readable TAC metadata
      pbpk_tacs.npz                          <- full-resolution TAC arrays
  simulations/                               <- Phase 2
    simind_simulation/
      preprocess/                            <- SIMIND preprocessing outputs
      headers/                               <- SIMIND headers (survives PRODUCTION cleanup)
      work_dir/                              <- per-core SIMIND outputs
    <prefix>_tot_w1/w2/w3.a00                <- summed projection totals (all ROIs)
    calib.res                                <- SIMIND calibration (sensitivity)
    <prefix>_dose_sum.nii.gz                 <- summed dose map (Gy/decay, if --dosimetry)
    <prefix>_dose_sum_unc.nii.gz             <- summed relative uncertainty map (if --dosimetry)
    opengate_simulation/                     <- (if --dosimetry)
      <prefix>_dose_<roi>.nii.gz             <- per-ROI dose maps (Gy/decay)
      <prefix>_unc_<roi>.nii.gz              <- final per-ROI relative uncertainty maps
      <prefix>_material_labels.nii.gz        <- OpenGATE material labels
      work_dir/
        source_masks/
        resampled_inputs/
        <roi>/batches/batch_000001/          <- batch dose/uncertainty scratch outputs
  post_processing/                           <- Phase 3
    spect_postprocess/                       <- (if --spect --postprocess)
      <prefix>_<t_hr>_tot_w1/w2/w3.nii.gz   <- PBPK-weighted projections per frame
    reconstructed_SPECT_<t_hr>.nii.gz        <- reconstructed SPECT image per frame
    dosemap_postprocess/                     <- (if --dosimetry --postprocess)
      work_dir/                              <- per-ROI dose contributions / scratch outputs
    <prefix>_total_dose.nii.gz               <- total absorbed dose map (Gy)
  profiling_CT_1.json                        <- (if --profile) per-stage CPU/RAM samples
  logging_file_CT_1.log                      <- per-CT pipeline log with timings + key stage summaries
```

### Profiling Output

If `--profile INTERVAL_S` is provided, each patient output folder also contains `profiling_CT_<index>.json`.

- Sampling cadence is the interval you pass (0.1–3.0 s), set the same way in both the CLI and the web UI.
- `cpu_pct_samples` is the summed CPU usage of the pipeline process tree. `100` means one fully used logical core, `800` means roughly eight fully used cores.
- `ram_mb_samples` is the summed RSS memory of the pipeline process tree in MiB.
- `system_cpu_pct_samples` is whole-machine CPU usage for context.
- `process_count_samples` is the number of live processes in the sampled pipeline process tree.
- `sample_times_s` gives the x-axis for plotting within each stage.

This makes the profiler suitable for parameter sweeps where you want to compare how stage runtime, CPU load, and memory footprint change as you vary photon counts, histories, voxel spacing, reconstruction settings, or ROI selections which affects results/resolution.

---

## ROI Configuration

The pipeline uses two data files to manage regions of interest:

| File | Purpose |
|------|---------|
| `pipeline_roi_naming_map.json` | Complete technical definition of every PyTheraTwin ROI — label IDs, TotalSegmentator task/label mappings, PBPK VOI names, UI categories, and parent/child group relationships. Update this file when changing the underlying segmentor or PBPK model. |
| `pipeline_options.json` | Developer-curated user-selectable subset. The web UI and config validator both derive available choices from this file. Remove an entry here to disable an option without touching the label map. |

### Parent/child ROI groups

Some ROIs in `pipeline_roi_naming_map.json` are marked with a `parent_group` field. These are individual anatomical sub-structures whose TotalSegmentator labels are fully covered by the named parent ROI. The parent is the user-selectable entry; children are retained in `PyTheraTwin_allowed_rois` so a developer can re-enable them individually via `pipeline_options.json` if needed.

| Parent (selectable) | Children (grouped under parent) |
|---------------------|---------------------------------|
| `bone` | `spine`, `skull`, `ribs`, `humerus`, `scapula`, `clavicula`, `femur`, `hip` |
| `muscle` | `gluteus_maximus`, `gluteus_medius`, `gluteus_minimus`, `autochthon`, `iliopsoas` |
| `gi_tract` | `stomach`, `small_bowel`, `duodenum`, `colon` |

### Bilateral TotalSegmentator labels vs PyTheraTwin ROIs

`parent_group` is a **PyTheraTwin ROI-level** concept. It is distinct from TotalSegmentator's bilateral label convention, where many structures are segmented as left/right pairs and then merged into a single PyTheraTwin ROI (e.g. `kidney` combines `kidney_left` + `kidney_right`; `nasal_cavity` combines `nasal_cavity_right` + `nasal_cavity_left`). These bilateral labels are `totseg_rois` implementation details — they do not get their own PyTheraTwin label ID or `PyTheraTwin_to_totseg` entry, and `parent_group` does not apply to them.

`parent_group` only applies where both the parent **and** the child have their own `PyTheraTwin_Pipeline` label ID and `PyTheraTwin_to_totseg` entry.

### PBPK patient height and weight

Two optional config fields control body-composition scaling in the PBPK model:

```json
"pbpk_tac_stage": {
  "height_m":  null,
  "weight_kg": null
}
```

**Priority:**
1. If `height_m` / `weight_kg` is a positive number, that value is used directly.
2. If `null` or `0`, the pipeline tries to extract height/weight from the DICOM header (only possible when the CT input is a DICOM directory).
3. If neither source provides a value, the PBPK model uses its built-in population-average default.

The web UI exposes these fields on the PBPK stage card. They are also shown in the review summary on Page 4.

### PBPK model changes

The `pbpk_voi` field is independent of `parent_group`. When the active PBPK model no longer provides a dedicated observable for a ROI, set `pbpk_voi` to `null` on that entry in `pipeline_roi_naming_map.json`. This demotes the ROI from simulation-eligible to anatomy-only without any other changes required.

---

## Source Layout

```
src/
  stages/
    segmentation_stage.py           <- TotalSegmentator + ROI unification
    synthetic_lesions_stage.py      <- Optional synthetic lesion generation
    pbpk_tac_stage.py               <- PBPK TAC generation
    simind_simulation_stage.py      <- SIMIND preprocessing + Monte Carlo simulation
    opengate_simulation_stage.py    <- OpenGATE preprocessing + Monte Carlo simulation
    spect_postprocess_stage.py      <- TAC weighting + Poisson noise + OSEM reconstruction
    dosemap_postprocess_stage.py    <- Per-ROI integrated TAC-weighted dose map

  io/
    context.py                      <- Shared Context object; inter-stage state handoff
    runtime_config.py               <- Load + validate user JSON config
    config_paths.py                 <- Load pipeline_paths.json; inject developer fields
    pipeline_logging.py             <- Terminal + file logging: banners, summaries, progress
    profiler.py                     <- Background CPU/RAM sampler; JSON output per stage
    stage_metadata.py               <- Persistent per-stage rerun metadata (JSON)
    rerun_guard.py                  <- Rerun safety: compare config/CT/upstream digests
    rerun_snapshots.py              <- Per-stage rerun snapshot builders
    rerun_fingerprints.py           <- File fingerprinting (SHA256) + JSON digest helpers

  utils/
    nifti_utils.py                  <- NIfTI I/O helpers; axis transposition; spacing/volume
    label_utils.py                  <- ROI/PBPK metadata I/O; isotope config loader; class maps; ROI seg filtering
    dicom_utils.py                  <- Patient height/weight extraction from DICOM tags
    tac_utils.py                    <- TAC/PBPK helpers: cumulated activity
    resize_utils.py                 <- Simulation grid resolution; voxel spacing validation
    lesion_utils.py                 <- Synthetic lesion geometry: placement, overlap, labelmap
    opengate_utils.py               <- SimpleITK helpers: centering, resampling, HU tables
    simind_runtime_utils.py         <- SIMIND subprocess: env setup, aggregation, calibration
    simind_projection_utils.py      <- SIMIND file I/O: header parsing, projection read/write

  tests/
    validate_config.py              <- Config validation (run before pipeline launch)
    validate_ct.py                  <- CT input validation (single + batch mode)

  data/
    isotope_config.json             <- Isotope data: half-lives, attenuation coefficients, nuclear params
    pipeline_roi_naming_map.json    <- Complete ROI definition table: PyTheraTwin label IDs, TotalSegmentator
                                       task/label mappings, PBPK VOI names, PBPK observables, UI categories,
                                       and parent_group relationships (bone/muscle/gi_tract hierarchies).
                                       Updated when the underlying segmentor or PBPK model changes.
    pipeline_paths.json             <- Developer-facing path configuration (SIMIND dir, output root)
    pipeline_options.json           <- User-selectable ROI subset and stage option lists (dropdowns).
                                       Narrows pipeline_roi_naming_map.json to what is exposed in the
                                       web UI and accepted by config validation. Remove an entry here to
                                       disable an option without editing the underlying label map.
                                       ROIs with a parent_group must be selected as a unit via their
                                       parent (e.g. select "bone" not individual "spine" or "ribs").
    smc.smc / scattwin.win          <- SIMIND template files
    jaszak.smc                      <- SIMIND Jaszczak calibration template
```

---

## Analysis Notebooks

| Notebook | Description |
|----------|-------------|
| `scripts/output_analysis.ipynb` | Multi-patient output inspection, stage visualisation, PBPK vs reconstructed-SPECT TAC comparisons, profiling summaries |

---

## Contact

Maintainer: Peter Yazdi
Email: pyazdi@bccrc.ca
