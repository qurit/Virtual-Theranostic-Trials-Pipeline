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
| **3 — Configure** | Enter a project name (determines the output folder). Toggle pipeline stages: **SPECT**, **Dosimetry**, **Post-Processing**. Choose **PRODUCTION** or **DEBUG** mode. Expand each patient card to adjust per-patient parameters — every field has a tooltip and changes auto-save. The voxel-spacing control is split into **XY** and **Z**: XY is applied to both X and Y, must stay square in-plane, and must be greater than or equal to the native CT in-plane spacing; Z must be greater than or equal to the native CT Z spacing. ROI pickers show PBPK and lesion badges, prevent grouped/child ROI overlap, and block invalid simulation selections before the Run page. Synthetic lesions are configured per-patient by adding organ specs in Phase 1 — the stage runs automatically for any patient whose config includes specs, and is skipped for patients without specs. The synthetic lesion editor supports automatic radii, manual per-lesion radii, and fully user-defined centres. If profiling is enabled, the UI also exposes a sampling-interval dropdown (0.1-3.0 s, default 2.0 s). If output directories from a previous run are detected, the UI compares the selected CT and effective stage config against saved rerun metadata before allowing a rerun. |
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
- `phase_2.simind_stage.roi_subset` — list of PBPK-compatible, non-Rest ROIs for SIMIND simulation. Every ROI must also be present in `phase_1.segmentation_stage.roi_subset`.
- `phase_2.opengate_stage.roi_subset` — list of PBPK-compatible, non-Rest ROIs for OpenGATE dosimetry. Every ROI must also be present in `phase_1.segmentation_stage.roi_subset`.
- `phase_3.spect_postprocess_stage.FrameStartTimes` and `FrameDurations` — frame timing for SPECT reconstruction.
- `src/data/pipeline_paths.json` → `input_paths.SIMINDDirectory` — path to your SIMIND install (set once, shared across all runs).

**Synthetic lesions (optional, per-patient)**
- `phase_1.synthetic_lesions_stage.specs` — add an organ-keyed dict to enable synthetic lesion generation for a patient. Each lesion host must be a Phase 1 ROI with a dedicated non-Rest PBPK VOI. The stage runs automatically when specs are present and is skipped when `specs` is absent or `null`. If SIMIND or OpenGATE selects any ROI that has synthetic lesions, that same simulation stage must select all lesion-host ROIs; the internal `synthetic_lesion` / `Tumor1` source is added automatically.

**Common tweaks (runtime / quality)**
- `phase_2.simind_stage.NumPhotons`, `NumProjections`, `EnergyWindowWidth` — simulation fidelity vs. runtime. `NumProjections` must be ≥ 64; fewer projections causes angular undersampling and streak artifacts in OSEM reconstruction. The constraint `Subsets × Iterations ≤ NumProjections` must also hold.
- `phase_2.simind_stage.num_cpu` — CPU cores for parallel SIMIND (`0` = use all available).
- `phase_3.spect_postprocess_stage.Iterations`, `Subsets` — OSEM reconstruction settings. `Subsets × Iterations` must not exceed `NumProjections`.
- `phase_2.simind_stage.xyz_spacing_mm` — target voxel spacing in mm as `[sxy, sxy, sz]`. X and Y must match. The shared XY value must be ≥ both native in-plane spacings; Z must be ≥ the native CT Z spacing (only coarser grids allowed). Voxel counts are derived automatically from the physical CT extent. The native CT spacing is logged at runtime. `null` = native CT resolution.
- `phase_2.opengate_stage.xyz_spacing_mm` — same rule as above for dosimetry. Dose maps are saved on the selected simulation grid; use `null` to keep native CT resolution.
- `phase_2.opengate_stage.gate.total_histories_per_batch` — Monte Carlo histories per adaptive OpenGATE batch.
- `phase_2.opengate_stage.gate.num_threads` — OpenGATE threads (`0` = use all available).
- `phase_2.opengate_stage.gate.target_uncertainty_percent` — ROI stopping target as a percent. Each ROI stops once the 95th percentile of its voxel uncertainty is below this value, or logs a warning and saves the best result after the developer batch cap.

When `xyz_spacing_mm` is used, the pipeline preserves the physical CT extent and derives integer voxel counts from that extent. This means the realised spacing can be very close to, but not mathematically identical to, the requested value when the extent is not exactly divisible by the target spacing. In the web UI this is edited as `XY` + `Z`; in the JSON config it is still stored as `[sxy, sxy, sz]`.

#### 2) CT Input

Place CT data in any directory:

```bash
mkdir -p inputs/ct_input
# add DICOM subfolders and/or .nii / .nii.gz files
```

Each top-level DICOM folder or NIfTI file is treated as one patient. Other unsupported top-level entries are ignored in batch mode.

CLI preflight now validates CT inputs before launching any patient run:
- `src/tests/validate_ct.py` checks single CT paths and batch-directory entries.
- `src/tests/validate_config.py` checks the run configuration after developer paths are injected.

#### 3) Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config_file PATH` | required | Path to your JSON config |
| `--input_ct_dir PATH` | required* | Directory containing CT inputs (NIfTI files or DICOM folders) |
| `--input_ct PATH` | required* | Single CT input — a NIfTI file or a DICOM directory |
| `--mode {DEBUG,PRODUCTION}` | `PRODUCTION` | Verbosity and intermediate file cleanup |
| `--logging_on / --no-logging_on` | on | Write per-CT log file |
| `--ct_index_start N` | `1` | Starting index for CT subfolder naming (e.g. `2` → `CT_2`, `CT_3`, …) |
| `--spect` | off | Run SIMIND SPECT projection simulation |
| `--dosimetry` | off | Run OpenGATE dosimetry simulation |
| `--postprocess` | off | Run post-processing for whichever simulations ran |
| `--save_config` | off | Copy config JSON into each CT output folder |
| `--profile` | off | Write `profiling_CT_<index>.json` with per-stage pipeline CPU/RAM samples |
| `--profile_interval_s FLOAT` | `2.0` | Profiler sampling interval in seconds when `--profile` is enabled. Allowed range: `0.1` to `3.0` |

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

**Notes:**
- `*_tot_w1/w2/w3.a00` are SIMIND energy-window projection totals (lower / photopeak / upper).
- `calib.res` is produced by SIMIND Jaszczak calibration and converts counts -> activity.
- OpenGATE dose maps are saved on the selected simulation grid. `xyz_spacing_mm: null` keeps native CT resolution; otherwise outputs stay on the coarser dosimetry grid.
- OpenGATE adaptive batches are independent Monte Carlo runs. Raw batch dose is accumulated and normalized by total simulated histories, so the final saved dose remains Gy/decay. Relative uncertainty improves approximately as `1 / sqrt(N)` as more histories are accumulated.

### Profiling Output

If `--profile` is enabled, each patient output folder also contains `profiling_CT_<index>.json`.

- Sampling cadence is 2 s by default and can be set in both the CLI (`--profile_interval_s`) and the web UI (0.1-3.0 s).
- `cpu_pct_samples` is the summed CPU usage of the pipeline process tree. `100` means one fully used logical core, `800` means roughly eight fully used cores.
- `ram_mb_samples` is the summed RSS memory of the pipeline process tree in MiB.
- `system_cpu_pct_samples` is whole-machine CPU usage for context.
- `process_count_samples` is the number of live processes in the sampled pipeline process tree.
- `sample_times_s` gives the x-axis for plotting within each stage.
- JSON is kept intentionally because it can hold the raw time series and effective config provenance in one portable file. If you later want CSV for plotting, it can be derived from this file without losing detail.

This makes the profiler suitable for parameter sweeps where you want to compare how stage runtime, CPU load, and memory footprint change as you vary photon counts, histories, voxel spacing, reconstruction settings, or ROI selections.
- The total dose map is computed using per-ROI weighting: each ROI's dose-per-decay map is multiplied by that ROI's own cumulated activity (TAC integrated from t=0 over 10x the isotope half-life, capturing >99.9% of all decays), then summed across ROIs. This avoids unphysical cross-terms.
- PBPK TAC simulation length is derived automatically from the configured isotope half-life (10x multiplier). If any SPECT frame time extends beyond this, the TAC is extended to cover it.
- In `PRODUCTION` mode, SIMIND and OpenGATE `work_dir` folders are deleted after the pipeline finishes to save disk space, but final outputs and rerun metadata are preserved.
- Existing outputs are reused only when the saved CT provenance and the relevant stage config still match the persistent metadata. If they do not match, delete the old stage outputs for that patient before rerunning.
- SIMIND header files are preserved in `headers/` to support reconstruction.
- PBPK TACs are saved as JSON + npz in Phase 1 and reused by both SPECT and dosimetry post-processing.

---

## Source Layout

```
src/
  stages/
    segmentation_stage.py           <- TotalSegmentator + ROI unification
    synthetic_lesions_stage.py      <- Optional synthetic lesion generation
    pbpk_tac_stage.py               <- PBPK TAC generation (isotope-aware stop time)
    simind_simulation_stage.py      <- SIMIND preprocessing + Monte Carlo simulation
    opengate_simulation_stage.py    <- OpenGATE voxel-source dosimetry
    spect_postprocess_stage.py      <- TAC weighting + Poisson noise + OSEM reconstruction
    dosemap_postprocess_stage.py    <- Per-ROI TAC-weighted total absorbed dose map

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
    pipeline_roi_naming_map.json    <- Full ROI definition table: VTT label IDs, TotalSegmentator task mappings, PBPK VOI names, PBPK observables, UI categories
    pipeline_paths.json             <- Developer-facing path configuration (SIMIND dir, output root)
    pipeline_options.json           <- Stage defaults and UI option lists
    smc.smc / scattwin.win          <- SIMIND template files
    jaszak.smc                      <- SIMIND Jaszczak calibration template
```

---

## Analysis Notebooks

| Notebook | Description |
|----------|-------------|
| `scripts/output_analysis.ipynb` | Multi-patient output inspection, stage visualisation, PBPK vs reconstructed-SPECT TAC comparisons, profiling summaries, and synthetic-lesion views. |

---

## Contact

Maintainer: Peter Yazdi
Email: pyazdi@bccrc.ca
