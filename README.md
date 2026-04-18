# Virtual Theranostic Trials (VTT) Pipeline

This pipeline creates patient-specific **theranostic digital twins** by combining CT-based anatomy/segmentation with PBPK kinetics and physics-based SPECT simulation/reconstruction, supporting research in diagnosis and therapy planning.

---

## Overview

**Theranostics** is a "diagnose and treat" approach that uses the same biological target to both detect disease and guide targeted therapy.

**Radiopharmaceuticals (RPTs)** couple a targeting molecule with a radionuclide that accumulates in tissues expressing a biomarker (e.g., tumors). As the radionuclide decays, emitted particles can deliver therapy while emitted photons enable quantitative imaging. For example, **¹⁷⁷Lu-PSMA** targets PSMA-expressing prostate cancer and supports post-therapy SPECT imaging.

The **Virtual Theranostic Trials (VTT) Pipeline** is a quantitative software framework that uses real patient CT data to build end-to-end digital twins for theranostics research. It integrates:

- **Patient-specific anatomy** from clinical CT scans
- **Organ/tumor segmentation** (TotalSegmentator-based workflows)
- **Pharmacokinetic (PBPK) modeling** to generate time-activity behavior
- **Monte Carlo SPECT simulation + reconstruction** (SIMIND/PyTomography) to produce quantitative images
- **Monte Carlo dosimetry simulation** (OpenGATE/Geant4) to generate organ-level dose maps

Because uptake and dose can vary substantially between patients, VTT support personalized evaluation of therapy strategies by enabling controlled, repeatable experiments across anatomy, kinetics, and imaging physics. A key objective is demonstrating agreement with patient measurements to support reliability and validation; longer-term, this work supports **Virtual Theranostic Trials (VTTs)** and patient-specific dosimetry prediction.

![VTT Pipeline Overview](figures/pipeline_overview.png)

---

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| **Phase 1: Digital Twin & Ground Truth** | CT → TotalSegmentator + ROI unification → (optional) synthetic lesion generation → PBPK TAC generation |
| **Phase 2: Simulations** | SIMIND SPECT simulation (optional, `--spect`) · OpenGATE dosimetry (optional, `--dosimetry`) |
| **Phase 3: Post-Processing** | SPECT post-processing (optional, `--postprocess` + `--spect`) · Dosimetry post-processing (optional, `--postprocess` + `--dosimetry`) |

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
conda activate vtt
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

Two interfaces are available — both call the same `main.py` entry point and now number patients from `CT_1` upward by default.

| | Web UI | Command Line (CLI) |
|---|---|---|
| **Best for** | New users, interactive setup | Scripting, batch runs, remote headless servers |
| **Config** | Auto-generated form with tooltips | Hand-edited JSON file |
| **CT input** | Folder picker dialog or absolute path | `--input_ct_dir` flag |
| **Flags** | Toggle buttons in the browser | CLI flags (`--spect`, `--dosimetry`, …) |
| **Start** | `python run_server.py` | `python main.py …` |

---

### Web UI

#### Launch

```bash
conda activate vtt
python run_server.py
```

The browser opens automatically at **http://localhost:8766**.  
Pass `--port PORT` to change the port, or `--no-browser` to skip auto-open.

#### Step-by-step

| Page | What to do |
|------|------------|
| **1-Intro** | Read the pipeline overview and prerequisite checklist, then click **Get Started**. |
| **2 — CT Input** | Click **Choose CT Folder** to open a native folder picker, or type an absolute path into the text field and click **Scan** (useful on remote/Azure servers where a local desktop picker is unavailable). Patients are detected automatically — DICOM subdirectories and `.nii` / `.nii.gz` files both work. Axial, coronal, and sagittal previews are generated for each CT. |
| **3 — Configure** | Enter a project name (determines the output folder). Toggle pipeline stages: **SPECT**, **Dosimetry**, **Post-Processing**. Choose **PRODUCTION** or **DEBUG** mode. Expand each patient card to adjust per-patient parameters — every field has a tooltip and changes auto-save. Synthetic lesions are configured per-patient by adding organ specs in Phase 1 — the stage runs automatically for any patient whose config includes specs, and is skipped for patients without specs. The synthetic lesion editor supports automatic radii, manual per-lesion radii, and fully user-defined centres. If output directories from a previous run are detected, the UI compares the selected CT and effective stage config against saved rerun metadata before allowing a rerun. |
| **4 — Run** | Review the per-patient configuration summary, then click **Run Pipeline**. |

#### Web UI dependencies

The web UI requires a few packages beyond the base conda environment:

```bash
pip install "fastapi>=0.111" "uvicorn[standard]>=0.30" python-multipart pillow nibabel pydicom
```

> `run_server.py` checks for these on startup and installs any that are missing automatically.

---

### Command Line (CLI)

#### 1) Create your run config

```bash
cp config_template.json inputs/my_config.json
# edit my_config.json
```

**Must update (most users)**
- `phase_1.segmentation_stage.roi_subset` — list of ROIs to segment. `remaining_body` (the body outline minus all named organs) is always added automatically — do not include it explicitly.
- `phase_2.simind_stage.roi_subset` — list of ROIs for SIMIND simulation.
- `phase_2.opengate_stage.roi_subset` — list of ROIs for OpenGATE dosimetry.
- `phase_3.spect_postprocess_stage.FrameStartTimes` and `FrameDurations` — frame timing for SPECT reconstruction.
- `src/data/pipeline_paths.json` → `input_paths.SIMINDDirectory` — path to your SIMIND install (set once, shared across all runs).

**Synthetic lesions (optional, per-patient)**
- `phase_1.synthetic_lesions_stage.specs` — add an organ-keyed dict to enable synthetic lesion generation for a patient. The stage runs automatically when specs are present and is skipped when `specs` is absent or `null`. This means different patients in the same batch can have different lesion configurations.

**Common tweaks (runtime / quality)**
- `phase_2.simind_stage.NumPhotons`, `NumProjections`, `EnergyWindowWidth` — simulation fidelity vs. runtime. `NumProjections` must be ≥ 64; fewer projections causes angular undersampling and streak artifacts in OSEM reconstruction. The constraint `Subsets × Iterations ≤ NumProjections` must also hold.
- `phase_2.simind_stage.num_cpu` — CPU cores for parallel SIMIND (`0` = use all available).
- `phase_3.spect_postprocess_stage.Iterations`, `Subsets` — OSEM reconstruction settings. `Subsets × Iterations` must not exceed `NumProjections`.
- `phase_2.simind_stage.xyz_spacing_mm` — target voxel spacing in mm as `[sx, sy, sz]`. Each value must be ≥ the native CT spacing on that axis (only coarser grids allowed). Voxel counts are derived automatically from the physical CT extent. The native CT spacing is logged at runtime. `null` = native CT resolution.
- `phase_2.opengate_stage.xyz_spacing_mm` — same as above for dosimetry. Dose maps are always resampled back to native CT resolution after simulation.
- `phase_2.opengate_stage.gate.total_histories` — Monte Carlo histories for dosimetry.
- `phase_2.opengate_stage.gate.num_cpu` — OpenGATE threads (`0` = use all available).

#### 2) CT Input

Place CT data in any directory:

```bash
mkdir -p inputs/ct_input
# add DICOM subfolders and/or .nii / .nii.gz files
```

Each top-level item is treated as one patient — a DICOM folder or a NIfTI file.

#### 3) Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config_file PATH` | required | Path to your JSON config |
| `--input_ct_dir PATH` | required* | Directory containing CT inputs (NIfTI files or DICOM folders) |
| `--input_ct PATH` | required* | Single CT input — a NIfTI file or a DICOM directory |
| `--mode {DEBUG,PRODUCTION}` | `PRODUCTION` | Verbosity and intermediate file cleanup |
| `--logging_on / --no-logging_on` | on | Write per-CT log file |
| `--ct_index_start N` | `1` | Starting index for output folder naming (e.g. `2` → `_CT_2`, `_CT_3`, …) |
| `--spect` | off | Run SIMIND SPECT projection simulation |
| `--dosimetry` | off | Run OpenGATE dosimetry simulation |
| `--postprocess` | off | Run post-processing for whichever simulations ran |
| `--save_config` | off | Copy config JSON into each CT output folder |

> \* `--input_ct_dir` and `--input_ct` are mutually exclusive; exactly one is required.

#### 4) Run

**Phase 1 only (digital twin + TACs):**
```bash
python -u main.py \
  --config_file inputs/my_config.json \
  --input_ct_dir inputs/ct_input
```

**Full SPECT pipeline:**
```bash
python -u main.py \
  --config_file inputs/my_config.json \
  --input_ct_dir inputs/ct_input \
  --spect --postprocess
```

**Full dosimetry pipeline:**
```bash
python -u main.py \
  --config_file inputs/my_config.json \
  --input_ct_dir inputs/ct_input \
  --dosimetry --postprocess
```

**Run everything (synthetic lesions auto-enabled when specs are in config):**
```bash
python -u main.py \
  --config_file inputs/my_config.json \
  --input_ct_dir inputs/ct_input \
  --mode DEBUG \
  --logging_on \
  --save_config \
  --spect --dosimetry --postprocess
```

---

## Outputs

Each CT input generates an output folder under `<output_folder_title>_CT_<index>/` with subfolders per phase.

### Example Structure

```
test_run_CT_1/
  pipeline_metadata/                         <- persistent rerun metadata (survives work_dir cleanup)
    ct_input.json                           <- saved CT identity / provenance
    segmentation_stage.json                 <- stage-level rerun guard + debug metadata
    pbpk_tac_stage.json
    simind_simulation_stage.json
    opengate_simulation_stage.json
    spect_postprocess_stage.json
    dosemap_postprocess_stage.json
  digital_twin/                              <- Phase 1
    ct.nii.gz                                <- standardized CT handoff
    digital_twin.nii.gz                      <- unified VTT multilabel segmentation handoff
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
    opengate_simulation/                     <- (if --dosimetry)
      work_dir/
        source_masks/
        resampled_inputs/
        <prefix>_dose_<roi>.nii.gz           <- per-ROI dose maps (Gy/decay)
      <prefix>_dose_sum.nii.gz               <- summed dose map (Gy/decay)
  post_processing/                           <- Phase 3
    spect_postprocess/                       <- (if --spect --postprocess)
      <prefix>_<t_hr>_tot_w1/w2/w3.nii.gz   <- PBPK-weighted projections per frame
    reconstructed_SPECT_<t_hr>.nii.gz        <- reconstructed SPECT image per frame
    dosemap_postprocess/                     <- (if --dosimetry --postprocess)
      work_dir/                              <- per-ROI dose contributions / scratch outputs
    <prefix>_total_dose.nii.gz               <- total absorbed dose map (Gy)
  logging_file_CT_1.log                      <- per-CT pipeline log
```

**Notes:**
- `*_tot_w1/w2/w3.a00` are SIMIND energy-window projection totals (lower / photopeak / upper).
- `calib.res` is produced by SIMIND Jaszczak calibration and converts counts -> activity.
- All dose maps are saved in native CT resolution. If `xyz_spacing_mm` was set for OpenGATE, the simulation runs on the coarser grid and outputs are resampled back automatically.
- The total dose map is computed using per-ROI weighting: each ROI's dose-per-decay map is multiplied by that ROI's own cumulated activity (TAC integrated from t=0 over 10x the isotope half-life, capturing >99.9% of all decays), then summed across ROIs. This avoids unphysical cross-terms.
- PBPK TAC simulation length is derived automatically from the configured isotope half-life (10x multiplier). If any SPECT frame time extends beyond this, the TAC is extended to cover it.
- In `PRODUCTION` mode, SIMIND `work_dir` is deleted after post-processing to save disk space, but rerun metadata is preserved in `pipeline_metadata/`.
- Existing outputs are reused only when the saved CT provenance and the relevant stage config still match the persistent metadata. If they do not match, delete the old stage outputs for that patient before rerunning.
- SIMIND header files are preserved in `headers/` to support reconstruction.
- PBPK TACs are saved as JSON + npz in Phase 1 and reused by both SPECT and dosimetry post-processing.

---

## Stage Files

```
src/stages/
  segmentation_stage.py         <- TotalSegmentator + ROI unification
  synthetic_lesions_stage.py    <- Optional synthetic lesion generation
  pbpk_tac_stage.py             <- PBPK TAC generation (isotope-aware stop time)
  simind_simulation_stage.py    <- SIMIND preprocessing + Monte Carlo simulation
  opengate_simulation_stage.py  <- OpenGATE voxel-source dosimetry
  spect_postprocess_stage.py    <- TAC weighting + Poisson noise + OSEM reconstruction
  dosemap_postprocess_stage.py  <- Per-ROI TAC-weighted total absorbed dose map
```

---

## Contact

Maintainer: Peter Yazdi
Email: pyazdi@bccrc.ca
