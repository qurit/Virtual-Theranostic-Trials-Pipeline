# Theranostic Digital Twins (TDT) Pipeline

This pipeline creates patient-specific **theranostic digital twins** by combining CT-based anatomy/segmentation with PBPK kinetics and physics-based SPECT simulation/reconstruction, supporting research in diagnosis and therapy planning.

---

## Overview

**Theranostics** is a "diagnose and treat" approach that uses the same biological target to both detect disease and guide targeted therapy.

**Radiopharmaceuticals (RPTs)** couple a targeting molecule with a radionuclide that accumulates in tissues expressing a biomarker (e.g., tumors). As the radionuclide decays, emitted particles can deliver therapy while emitted photons enable quantitative imaging. For example, **¹⁷⁷Lu-PSMA** targets PSMA-expressing prostate cancer and supports post-therapy SPECT imaging.

The **Theranostic Digital Twins (TDT) Pipeline** is a quantitative software framework that uses real patient CT data to build end-to-end digital twins for theranostics research. It integrates:

- **Patient-specific anatomy** from clinical CT scans
- **Organ/tumor segmentation** (TotalSegmentator-based workflows)
- **Pharmacokinetic (PBPK) modeling** to generate time-activity behavior
- **Physics-based SPECT simulation + reconstruction** to produce quantitative images
- **Monte Carlo dosimetry** (OpenGATE/Geant4) to generate voxel-level dose maps

Because uptake and dose can vary substantially between patients, TDTs support personalized evaluation of therapy strategies by enabling controlled, repeatable experiments across anatomy, kinetics, and imaging physics. A key objective is demonstrating agreement with patient measurements to support reliability and validation; longer-term, this work supports **Virtual Theranostic Trials (VTTs)** and patient-specific dosimetry prediction.

![TDT Pipeline Overview](figures/pipeline_overview.png)

---

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| **Phase 1: Digital Twin** | CT → TotalSegmentator → ROI unification → (optional) synthetic lesion generation |
| **Phase 2: SPECT Simulation** | SIMIND preprocessing → Monte Carlo projection simulation |
| **Phase 3: SPECT Post-Processing** | PBPK TAC generation → projection weighting → OSEM + TEW reconstruction |
| **Phase 4: Dosimetry** | OpenGATE voxel-source Monte Carlo dose maps (Lu-177 decay physics) |

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
conda activate TDT_env
```

> This environment includes all required Python dependencies (TotalSegmentator, PyTomography, OpenGATE, PyCNO, etc.).

### 2) Install SIMIND (external)

**SIMIND is an external dependency** and must be installed separately.

#### Step 1 — Download and install SIMIND
```text
https://www.msf.lu.se/en/research/simind-monte-carlo-program/downloads
```

#### Step 2 — Point the pipeline config to SIMIND
In your JSON config, set:
- `phase_2.simind_stage.SIMINDDirectory` = directory containing the `simind` executable

(Optional sanity check)
```bash
which simind
echo $SMC_DIR
```

---

## Usage

> This repo is run via `main.py` using a user-editable JSON config and CT inputs placed under `inputs/`.
> Install required dependencies and external tools (SIMIND) before running.

### 1) Create your run config

```bash
cp inputs/config_default.json inputs/config.json
```

Then edit `inputs/config.json`.

#### Must update (most users)
- `phase_2.simind_stage.SIMINDDirectory` — path to your SIMIND install.
- `phase_1.segmentation_stage.roi_subset` — list of ROIs to segment and simulate.
- `phase_3.pbpk_stage.FrameStartTimes` and `FrameDurations` — frame timing for PBPK → simulation → reconstruction.
- `phase_1.unification_stage.label_map_path` — path to `tdt_map.json` label map file.

#### Common tweaks (runtime / quality)
- `phase_2.preprocess_simind_stage.xy_dim` — in-plane resize for SIMIND inputs (smaller = faster).
- `phase_2.simind_stage.NumPhotons`, `NumProjections`, `EnergyWindowWidth` — simulation fidelity vs. runtime.
- `phase_2.simind_stage.NumCores` — CPU cores for parallel SIMIND (`0` = use all available).
- `phase_3.reconstruction_stage.Iterations`, `Subsets` — OSEM reconstruction settings.
- `phase_4.opengate_simulation_stage.xy_dim` — downsample CT/seg before dosimetry simulation (e.g. `128` for fast validation, `null` for native resolution). Output dose maps are upsampled back to native CT space automatically.
- `phase_4.opengate_simulation_stage.gate.total_histories` — Monte Carlo histories for dosimetry.
- `phase_4.opengate_simulation_stage.gate.num_threads` — OpenGATE threads (up to your CPU count).

#### Dosimetry source options (Phase 4)
Two modes are supported in `phase_4.opengate_simulation_stage.source`:

**Option A — Lu-177 radioactive decay (recommended):**
```json
"source": { "isotope": "lu177" }
```
All decay products (betas, gammas, IC electrons) are simulated automatically with correct branching ratios via Geant4 RadioactiveDecay physics.

**Option B — Explicit monoenergetic components (legacy / other isotopes):**
```json
"source": {
  "components": [
    { "particle": "gamma", "energy_kev": 208, "relative_weight": 0.11 },
    { "particle": "e-",    "energy_kev": 133, "relative_weight": 0.89 }
  ]
}
```

#### Variance reduction (Phase 4)
Set `"variance_reduction": true` (default) in `opengate_simulation_stage` to enable Geant4 production cuts for electrons, reducing runtime with minimal effect on dose map quality. The cut range is controlled by `physics.electron_production_cut_mm` (default: 0.1 mm).

#### Quick validation settings (Phase 4 — recommended for first run)
```json
"xy_dim": 64,
"variance_reduction": true,
"gate": {
  "total_histories": 2000000,
  "num_threads": 20
}
```
Expected runtime: 15–30 minutes per ROI on a 112-core machine.

### 2) CT Input

Place your CT data under `inputs/ct_input/`:

```bash
mkdir -p inputs/ct_input
```

You can put **any mix** of the following inside `inputs/ct_input/`:
- A **single NIfTI CT** file (`.nii` or `.nii.gz`)
- One or more **DICOM folders** (each folder containing a CT DICOM series)
- Multiple CTs (multiple NIfTIs and/or multiple DICOM folders)

### 3) Command-line Interface

**Required arguments:**
- `--config_file` : Path to your JSON config file
- `--input_ct_dir` : Directory containing CT inputs

**Optional arguments:**
- `--mode {DEBUG,PRODUCTION}` : Controls verbosity and intermediate file cleanup (default: PRODUCTION)
- `--logging_on / --no-logging_on` : Enable/disable per-CT log file writing (default: enabled)
- `--save_ct_scan / --no-save_ct_scan` : Copy the CT input into the output folder for provenance (default: disabled)
- `--save_config / --no-save_config` : Copy the config JSON into each CT output folder (default: disabled)
- `--synthetic_lesions / --no-synthetic_lesions` : Run synthetic lesion generation (default: disabled; requires `phase_1.synthetic_lesions_stage.specs` to be set in config)

### 4) Run

**Basic run:**
```bash
python -u main.py \
  --config_file inputs/config.json \
  --input_ct_dir inputs/ct_input
```

**Run with all options:**
```bash
python -u main.py \
  --config_file inputs/config.json \
  --input_ct_dir inputs/ct_input \
  --mode PRODUCTION \
  --logging_on \
  --save_ct_scan \
  --save_config \
  --synthetic_lesions
```

---

## Outputs

Each CT input generates an output folder under `<output_folder_title>_CT_<index>/` with subfolders per phase.

### Example Structure

```
testing_dosimetry_CT_0/
  digital_twin/
    ct.nii.gz                          <- standardized CT handoff
    digital_twin.nii.gz                <- unified TDT multilabel segmentation handoff
    segmentation_stage/
    unification_stage/
    synthetic_lesions_stage/           <- (if synthetic lesions enabled)
  spect_simulation/
    preprocess_simind/
      <prefix>_atn_av.bin              <- attenuation binary for SIMIND
      <prefix>_<roi>_act_av.bin        <- per-ROI source maps
    simind_simulation/
      work_dir/                        <- SIMIND headers, COR files, per-core outputs
    <prefix>_<roi>_tot_w1/w2/w3.a00   <- per-organ energy window projection totals
    calib.res                          <- SIMIND calibration (sensitivity)
  spect_post_process/
    pbpk_stage/
      work_dir/                        <- TAC binaries, per-frame activity maps
    <prefix>_<t_hr>_tot_w1/w2/w3.nii.gz  <- PBPK-weighted projections per frame
    reconstructed_SPECT_<t_hr>.nii.gz    <- reconstructed SPECT image per frame
    reconstruction_stage/
      recon_atn_img.nii.gz             <- attenuation map in reconstruction grid
  dosimetry/
    opengate_simulation/
      source_masks/                    <- per-ROI binary source mask NIfTIs
      work_dir/                        <- per-ROI OpenGATE outputs + metadata
      resampled_inputs/                <- downsampled CT/seg (if xy_dim set)
    <prefix>_dose_raw_<roi>.nii.gz     <- per-ROI dose map (native CT space)
    <prefix>_dose_uncertainty_raw_<roi>.nii.gz
    <prefix>_dose_raw_sum.nii.gz       <- summed dose map across all ROIs
    <prefix>_material_labels.nii.gz    <- Geant4 material label image
  logging_file_CT_0.log                <- per-CT pipeline log
```

**Notes:**
- `*_tot_w1/w2/w3.a00` are SIMIND energy-window projection totals (lower / photopeak / upper).
- `calib.res` is produced by SIMIND Jaszczak calibration and converts counts → activity.
- All dose maps in the dosimetry output are in native CT resolution (upsampled back if `xy_dim` was set).
- In `PRODUCTION` mode, the SIMIND `work_dir` is deleted after reconstruction to save disk space.

---

## Contact

Maintainer: Peter Yazdi
Email: pyazdi@bccrc.ca