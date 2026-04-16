"""
FastAPI backend for the Virtual Theranostic Trials web UI.

Serves the single-page web app and exposes the API used to scan CT inputs,
build per-patient configs, preview CT data, and stream logs from pipeline
subprocess runs.

Endpoints
---------
GET  /                              Serve the React SPA (index.html)
GET  /api/system-info               CPU count and other system metadata
GET  /api/input-paths               Read input_paths section from pipeline_paths.json
GET  /api/config-template           Return parsed config template + field metadata
POST /api/scan-directory            Scan a local CT input directory → patient list
POST /api/upload-ct                 Accept uploaded directory files → temp path + patient list
POST /api/preview-ct                Generate axial/coronal/sagittal PNG previews for one CT
POST /api/pick-directory            Open a native OS folder-picker dialog
POST /api/create-output-dirs        Create per-patient output dirs and write base config.json
POST /api/save-patient-config       Inject developer fields and save config to an output dir
GET  /api/load-patient-config       Load existing config.json for rerun auto-fill
GET  /api/check-completion          Check which pipeline stages have already produced output
POST /api/run                       Start a pipeline run → run_id
POST /api/runs/{run_id}/stop        Kill a running pipeline run
GET  /api/runs/{run_id}             Poll run status / timing
WS   /ws/{run_id}                   Stream live logs for a run
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import shutil
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager, suppress
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.requests import ClientDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.io.config_paths import inject_pipeline_paths, load_pipeline_paths, strip_developer_fields

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.resolve()
WEB_DIR   = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
CONFIG_TEMPLATE   = REPO_ROOT / "config_template.json"
MAIN_PY           = REPO_ROOT / "main.py"
VTT_MAP           = REPO_ROOT / "src" / "data" / "vtt_map.json"
PIPELINE_OPTIONS  = REPO_ROOT / "src" / "data" / "pipeline_options.json"

RUN_TTL_SECONDS = 6 * 60 * 60
RUN_PRUNE_INTERVAL_SECONDS = 10 * 60
MAX_FINISHED_RUNS = 200
UPLOAD_TTL_SECONDS = 7 * 24 * 60 * 60   # delete upload dirs older than 7 days

# In-memory registry: run_id → run dict
_runs: Dict[str, Dict[str, Any]] = {}


def _prune_finished_runs(now: float | None = None) -> None:
    """Drop expired finished runs and cap retained history."""
    now = time.time() if now is None else now
    removable_ids: List[str] = []
    finished_runs: List[tuple[float, str]] = []

    for run_id, run in _runs.items():
        if run.get("status") not in {"done", "error"}:
            continue

        end_time = float(run.get("end_time") or run.get("start_time") or now)
        if now - end_time > RUN_TTL_SECONDS:
            removable_ids.append(run_id)
            continue

        finished_runs.append((end_time, run_id))

    for run_id in removable_ids:
        _runs.pop(run_id, None)

    if len(finished_runs) <= MAX_FINISHED_RUNS:
        return

    finished_runs.sort(key=lambda item: item[0])
    overflow = len(finished_runs) - MAX_FINISHED_RUNS
    for _, run_id in finished_runs[:overflow]:
        _runs.pop(run_id, None)


def _prune_old_uploads(now: float | None = None) -> None:
    """Delete upload temp dirs that are older than UPLOAD_TTL_SECONDS."""
    now = time.time() if now is None else now
    uploads_root = REPO_ROOT / "uploads"
    if not uploads_root.is_dir():
        return
    for entry in uploads_root.iterdir():
        if not entry.is_dir() or not entry.name.startswith("vtt_ct_upload_"):
            continue
        try:
            age = now - entry.stat().st_mtime
            if age > UPLOAD_TTL_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
        except Exception:
            pass


async def _run_registry_janitor() -> None:
    """Periodically prune expired finished runs and stale upload dirs."""
    while True:
        await asyncio.sleep(RUN_PRUNE_INTERVAL_SECONDS)
        _prune_finished_runs()
        _prune_old_uploads()

# ── App ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    janitor_task = asyncio.create_task(_run_registry_janitor())
    try:
        yield
    finally:
        janitor_task.cancel()
        with suppress(asyncio.CancelledError):
            await janitor_task

app = FastAPI(title="Virtual Theranostic Trials", lifespan=lifespan)


# ── Static / SPA ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=500)
    return HTMLResponse(index.read_text(encoding="utf-8"))


# ── System info ───────────────────────────────────────────────────────────────
@app.get("/api/system-info")
async def get_system_info() -> Dict:
    """Return basic system info (e.g. CPU count) for the UI to display."""
    return {"cpu_count": os.cpu_count() or 1}


# ── Pipeline input paths (read-only) ──────────────────────────────────────────
@app.get("/api/input-paths")
async def get_input_paths() -> Dict:
    """Return the current input_paths section from pipeline_paths.json."""
    try:
        pp = load_pipeline_paths(REPO_ROOT)
        return pp.get("input_paths", {})
    except Exception as e:
        raise HTTPException(500, str(e))



# ── Field descriptions (used by the frontend to render tooltips) ───────────────
FIELD_DESCRIPTIONS: Dict[str, str] = {
    "output_folder_title": "Name for the output folder created under the repo root. Each CT gets its own subfolder: <title>_CT_<index>/",
    "roi_subset": "Which organs to segment and include in downstream simulation stages",
    "default_seed": "Global random seed for reproducible lesion placement; 0 = non-reproducible",
    "auto_shrink_factor": "If a lesion can't be placed, reduce its radius by this factor (e.g. 0.85 = shrink 15%) and retry",
    "auto_max_shrink_iters": "Maximum number of radius reductions before abandoning placement of a lesion",
    "auto_start_frac": "Initial lesion radius as a fraction of the maximum physically possible radius (0–1)",
    "max_lesion_placement_attempts": "Maximum random placement attempts per lesion before giving up",
    "model_type": "PBPK pharmacokinetic model type. Currently only 'PSMA' (Lu-177 PSMA) is supported",
    "isotope": "Radionuclide for PBPK and simulation. Currently only 'lu177' (Lutetium-177) is supported",
    "VOIs": "PBPK volumes of interest to model. These are PyCNO observable names. Must cover all segmented ROIs",
    "Randomization_Kidney_SG_Para": "Randomize kidney and salivary-gland PBPK parameters via lognormal sampling — simulates patient-to-patient variability",
    "xyz_dim": "Target voxel counts [x, y, z] for downsampling CT/seg before simulation. Each value must be smaller than the corresponding CT dimension. Null = use native CT resolution (no downsampling).",
    "Collimator": "SIMIND collimator code (e.g. 'si-me' = Siemens medium-energy parallel-hole). Must match your SIMIND install",
    "Isotope": "Isotope code for SIMIND collimator lookup (e.g. 'lu177')",
    "NumProjections": "Number of SPECT angular projection views (typical range: 60–128)",
    "NumPhotons": "Total photon histories simulated per run. Higher = lower noise, longer runtime (e.g. 1e8)",
    "EnergyWindowWidth": "Width of the photopeak energy window as a percentage of the peak energy",
    "DetectorDistance": "Source-to-collimator-face distance in cm. Negative lets SIMIND contour to the patient",
    "DetectorWidth": "Detector crystal width in cm (e.g. 53.3 for a large-FOV Siemens camera)",
    "DetectorLength": "Detector crystal length in cm. 0 = infer from CT axial extent",
    "OutputImgSize": "Output projection image size in pixels per side",
    "OutputPixelWidth": "Output pixel pitch in cm (e.g. 0.5 = 5 mm)",
    "OutputSliceWidth": "Output slice thickness in cm",
    "num_cpu": "CPU cores/threads to use (applies to both SIMIND and OpenGATE). 0 = use all available cores on this system.",
    "save_per_roi_dose_maps": "Save a separate dose map NIfTI file for each segmented organ/ROI",
    "save_summed_dose_map": "Save a single NIfTI with the total dose summed across all ROIs",
    "save_uncertainty_map": "Save Monte Carlo statistical uncertainty maps alongside dose maps",
    "save_material_label_image": "Save the Schneider HU→material composition label image used in simulation",
    "write_mhd_outputs": "Also write outputs as MetaImage (.mhd/.raw) in addition to NIfTI",
    "variance_reduction": "Enable forced-detection variance reduction to accelerate Monte Carlo (see config comments for Lu-177 caveats)",
    "total_histories": "Total Monte Carlo particle histories for dosimetry (e.g. 1e7). Higher = more accurate, slower",
    "random_seed": "Monte Carlo random seed. 'auto' = unique each run; integer = reproducible",
    "start_new_process": "Launch OpenGATE in a fresh subprocess (recommended for memory isolation)",
    "density_tolerance_gcm3": "HU-to-material grouping tolerance in g/cm³. Smaller = more distinct materials, larger = faster",
    "world_margin_scale": "Scale factor applied to the simulation bounding box around the patient volume",
    "world_min_size_mm": "Minimum simulation world size in mm regardless of patient dimensions",
    "electron_production_cut_mm": "Secondary electron tracking cutoff length in mm. Affects beta dose accuracy",
    "apply_tac": "Weight SIMIND projections by interpolated PBPK time-activity curves",
    "apply_poisson_noise": "Add Poisson counting noise to simulate realistic scanner count statistics",
    "apply_reconstruction": "Run PyTomography OSEM+TEW reconstruction on the weighted projections",
    "apply_frame_duration": "Scale projection counts by frame acquisition duration",
    "FrameStartTimes": "Acquisition frame start times in minutes post-injection (e.g. [120, 1440, 2880])",
    "FrameDurations": "Duration of each acquisition frame in seconds — one value per frame (e.g. [10, 10, 10])",
    "ReconstructionAlgorithm": "Iterative reconstruction algorithm. Currently only 'OSEM' (with TEW scatter correction) is supported",
    "Iterations": "Number of OSEM reconstruction iterations",
    "Subsets": "Number of OSEM ordered subsets. Higher subsets → faster convergence but may be less stable",
    "n_lesions": "Number of synthetic lesions to place in this organ",
    "prob": "Lesion centre placement distribution: 'uniform' (random in organ), 'gaussian' (centroid-weighted), or 'user_defined'",
    "specs": "Per-organ synthetic lesion settings. Each configured organ gets its own lesion placement block.",
    "radii_mm": "Optional manual lesion radii in millimetres. Omit this list to let the stage choose radii automatically.",
    "user_centers_zyx": "Manual lesion centres in voxel coordinates ordered as [z, y, x]. Used only when prob = 'user_defined'.",
    "sigma_mm": "Standard deviation in mm for Gaussian lesion placement — controls spread from organ centroid",
    "margin_mm": "Minimum gap in mm between the lesion surface and the organ boundary (and other lesions)",
    "seed": "Random seed for this organ's lesion placement. 0 = non-reproducible",
}

ROI_CHOICES = ["kidney", "liver", "prostate", "spleen", "heart", "salivary_glands"]


# ── Config template ────────────────────────────────────────────────────────────
@app.get("/api/config-template")
async def get_config_template() -> Dict:
    """Return the parsed config template with field descriptions for form generation."""
    try:
        from json_minify import json_minify
    except ImportError:
        raise HTTPException(500, "json_minify is not installed. Activate your conda environment.")

    try:
        raw = CONFIG_TEMPLATE.read_text(encoding="utf-8")
        data = json.loads(json_minify(raw))
    except Exception as e:
        raise HTTPException(500, f"Could not parse config_template.json: {e}")

    # Strip all developer-controlled fields (sub_dir_name, file_prefix, label_map_path,
    # SIMINDDirectory, etc.) — these come from pipeline_paths.json, not from the user.
    data = strip_developer_fields(data)

    # Load ROI choices from the shared label map (TDT_Pipeline section, excluding reserved labels)
    _RESERVED = {"background", "remaining_body", "synthetic_lesion"}
    try:
        tdt_raw = VTT_MAP.read_text(encoding="utf-8")
        tdt_map = json.loads(json_minify(tdt_raw))
        roi_choices = [
            name for name in tdt_map.get("TDT_Pipeline", {}).values()
            if name not in _RESERVED
        ]
    except Exception:
        roi_choices = ROI_CHOICES  # fallback to hardcoded list

    # Load pipeline options (dropdowns) — supports JSONC comments via json_minify
    pipeline_options: Dict = {}
    try:
        pipeline_options = json.loads(json_minify(PIPELINE_OPTIONS.read_text(encoding="utf-8")))
    except Exception:
        pass

    # Add PascalCase / config-key aliases so the frontend can look up options
    # directly by the field name as it appears in config_template.json.
    _aliases: Dict[str, str] = {
        "Collimator":             "collimator",
        "Isotope":                "isotope",
        "ReconstructionAlgorithm": "reconstruction_algorithm",
        "NumProjections":         "num_projections",
        "Subsets":                "num_subsets",
        "Iterations":             "num_iterations",
    }
    for cfg_key, opt_key in _aliases.items():
        if opt_key in pipeline_options and cfg_key not in pipeline_options:
            pipeline_options[cfg_key] = pipeline_options[opt_key]

    return {
        "template": data,
        "roi_choices": roi_choices,
        "field_descriptions": FIELD_DESCRIPTIONS,
        "pipeline_options": pipeline_options,
    }


# ── CT directory scan ──────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    path: str


def _discover_patients(ct_dir: str) -> List[Dict]:
    """Mirror main.py's CT discovery logic exactly."""
    p = Path(ct_dir)
    if not p.exists():
        raise ValueError(f"Path does not exist: {ct_dir}")
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: {ct_dir}")

    # Matches main.py: sorted, no hidden files
    items = sorted(e for e in p.iterdir() if not e.name.startswith("."))

    patients: List[Dict] = []
    for e in items:
        if e.is_file() and (e.name.endswith(".nii") or e.name.endswith(".nii.gz")):
            patients.append({"name": e.name, "path": str(e), "type": "nifti"})
        elif e.is_dir():
            # Shallow count only — rglob over large DICOM series is very slow
            try:
                file_count = sum(1 for f in e.iterdir() if f.is_file())
            except PermissionError:
                file_count = None
            patients.append({
                "name": e.name,
                "path": str(e),
                "type": "dicom",
                "file_count": file_count,
            })
        # Other files (e.g. .json, .txt) are skipped — same as main.py
    return patients


@app.post("/api/scan-directory")
async def scan_directory(req: ScanRequest) -> Dict:
    try:
        loop = asyncio.get_running_loop()
        patients = await loop.run_in_executor(None, _discover_patients, req.path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not patients:
        raise HTTPException(
            400,
            "No patient CTs found. The directory must contain .nii / .nii.gz files "
            "or subdirectories (treated as DICOM series).",
        )
    return {"patients": patients, "count": len(patients), "path": req.path}


# ── Native folder picker ───────────────────────────────────────────────────────
@app.post("/api/pick-directory")
async def pick_directory() -> Dict:
    """Open a native OS folder-picker dialog and return the selected path."""
    script = (
        "import tkinter as tk; from tkinter import filedialog; "
        "root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1); "
        "path = filedialog.askdirectory(title='Select CT Folder'); "
        "print(path, end='')"
    )
    try:
        import subprocess as _sp
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _sp.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=120
            )
        )
        path = result.stdout.strip()
        if not path:
            return {"path": None, "cancelled": True}
        return {"path": path, "cancelled": False}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Create per-patient output directories ─────────────────────────────────────
class CreateDirsRequest(BaseModel):
    project_name: str
    patients: List[Dict[str, Any]]


@app.post("/api/create-output-dirs")
async def create_output_dirs(req: CreateDirsRequest) -> Dict:
    """
    Create one output directory per patient under REPO_ROOT:
      {project_name}_CT_1 / {project_name}_CT_2 / …
    Write a base config.json (user-facing fields only) into each dir.

    Returns `existed`: a per-patient flag indicating whether the directory
    already contained pipeline outputs (subdirectories) before this call.
    The frontend uses this to warn the user that completed stages will be skipped.
    """
    name = req.project_name.strip()
    if not name:
        raise HTTPException(400, "Project name is required.")

    try:
        from json_minify import json_minify as _jm
        base_cfg: dict = json.loads(_jm(CONFIG_TEMPLATE.read_text(encoding="utf-8")))
    except Exception:
        base_cfg = {}

    base_cfg = strip_developer_fields(base_cfg)
    base_cfg["output_folder_title"] = name

    # Do not pre-populate synthetic lesion specs — they start empty and are added
    # by the user through the UI only when the synthetic lesions flag is enabled.
    base_cfg.get("phase_1", {}).get("synthetic_lesions_stage", {}).pop("specs", None)

    created: Dict[str, str] = {}
    existed: Dict[str, bool] = {}
    existing_configs: Dict[str, Any] = {}   # patient_name → stripped user config (when prior run exists)

    for i, patient in enumerate(req.patients, start=1):
        out_dir = REPO_ROOT / f"{name}_CT_{i}"
        # Check for prior pipeline outputs (any subdirectory) before touching the dir.
        has_prior = out_dir.exists() and any(
            e.is_dir() for e in out_dir.iterdir() if not e.name.startswith(".")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "config.json"
        if not cfg_path.exists():
            full_cfg = inject_pipeline_paths(base_cfg, repo_root=REPO_ROOT, include_input_paths=False)
            cfg_path.write_text(json.dumps(full_cfg, indent=2))
        elif has_prior:
            # Return the existing config so the UI can pre-populate the form.
            try:
                from json_minify import json_minify as _jm
                existing_configs[patient["name"]] = strip_developer_fields(
                    json.loads(_jm(cfg_path.read_text(encoding="utf-8")))
                )
            except Exception:
                pass
        created[patient["name"]] = str(out_dir)
        existed[patient["name"]] = has_prior

    return {"dirs": created, "project_name": name, "existed": existed,
            "existing_configs": existing_configs}


# ── Real-time per-patient config save ─────────────────────────────────────────
class SavePatientConfigRequest(BaseModel):
    output_dir: str
    config: Dict[str, Any]


@app.post("/api/save-patient-config")
async def save_patient_config(req: SavePatientConfigRequest) -> Dict:
    """Inject developer fields and save config to a patient's output directory."""
    out_dir = Path(req.output_dir)
    if not out_dir.is_dir():
        raise HTTPException(400, f"Output directory not found: {out_dir}")
    full_cfg = inject_pipeline_paths(req.config, repo_root=REPO_ROOT, include_input_paths=False)
    (out_dir / "config.json").write_text(json.dumps(full_cfg, indent=2))
    return {"ok": True}


# ── Load existing patient config (for rerun auto-fill) ────────────────────────
@app.get("/api/load-patient-config")
async def load_patient_config(output_dir: str) -> Dict:
    """
    Read config.json from an existing output directory and return the
    user-facing fields (developer fields stripped).  Used by the UI to
    auto-populate the config form when rerunning a previous pipeline.
    """
    out_dir = Path(output_dir)
    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        raise HTTPException(404, f"No config.json found in {output_dir}")
    try:
        from json_minify import json_minify as _jm
        cfg = json.loads(_jm(cfg_path.read_text(encoding="utf-8")))
    except Exception as e:
        raise HTTPException(500, f"Could not parse config.json: {e}")
    return {"config": strip_developer_fields(cfg)}


# ── Phase completion check (for rerun warnings) ────────────────────────────────
@app.get("/api/check-completion")
async def check_completion(output_dir: str) -> Dict:
    """
    Inspect an output directory and return which pipeline phases / sub-stages
    have already produced output files on disk.

    Used by the UI to warn when a config change would conflict with already-
    completed work that would be skipped on the next run.

    Returns
    -------
    {
      "phase1_segmentation": bool,
      "phase1_synthetic_lesions": bool,
      "phase1_pbpk": bool,
      "phase2_simind": bool,
      "phase2_opengate": bool,
      "phase3_spect": bool,
      "phase3_dosemap": bool,
    }
    """
    out_dir = Path(output_dir)
    if not out_dir.is_dir():
        return {k: False for k in [
            "phase1_segmentation", "phase1_synthetic_lesions", "phase1_pbpk",
            "phase2_simind", "phase2_opengate", "phase3_spect", "phase3_dosemap",
        ]}

    def _any_file(glob_pattern: str) -> bool:
        return any(True for _ in out_dir.glob(glob_pattern))

    return {
        # Phase 1 segmentation: unified label map written
        "phase1_segmentation": _any_file("digital_twin/segmentation_stage/unified_labels*.nii*"),
        # Phase 1 synthetic lesions stage output
        "phase1_synthetic_lesions": _any_file("digital_twin/synthetic_lesions_stage/*.nii*"),
        # Phase 1 PBPK TAC
        "phase1_pbpk": _any_file("digital_twin/pbpk_tac_stage/*.json"),
        # Phase 2 SIMIND simulation outputs
        "phase2_simind": (
            _any_file("simulations/*.a00")
            or _any_file("simulations/simind_simulation/**/*.a00")
        ),
        # Phase 2 OpenGATE simulation outputs
        "phase2_opengate": _any_file("simulations/opengate_simulation/**/*.nii*"),
        # Phase 3 SPECT post-processing
        "phase3_spect": (
            _any_file("post_processing/spect_postprocess/*.nii*")
            or _any_file("post_processing/reconstructed_SPECT_*.nii*")
        ),
        # Phase 3 dose-map post-processing
        "phase3_dosemap": (
            _any_file("post_processing/*_total_dose.nii*")
            or _any_file("post_processing/dosemap_postprocess/**/*.nii*")
        ),
    }


# ── CT file upload ─────────────────────────────────────────────────────────────
@app.post("/api/upload-ct")
async def upload_ct(request: Request) -> Dict:
    """
    Accept files uploaded via <input type='file' webkitdirectory>.
    Each file.filename is the webkitRelativePath, e.g. 'folder_name/patient/001.dcm'.
    The server reconstructs the directory tree under a temp directory.

    Uses request.form() directly with a very high limit so large cohorts
    (thousands of DICOM slices) are not rejected by python-multipart's default
    max_files=1000 cap.
    """
    try:
        form = await request.form(max_files=1_000_000, max_fields=1_000_000)
    except ClientDisconnect:
        raise HTTPException(499, "Upload cancelled by client.")
    files = [v for _, v in form.multi_items() if hasattr(v, "read")]
    if not files:
        raise HTTPException(400, "No files received.")

    # Save uploads to a permanent directory so the path stays valid across reruns.
    uploads_root = REPO_ROOT / "uploads"
    uploads_root.mkdir(exist_ok=True)
    upload_dir = Path(tempfile.mkdtemp(prefix="vtt_ct_upload_", dir=str(uploads_root)))
    try:
        for f in files:
            # Normalise path separators (browser may send \ on Windows)
            rel = (f.filename or "").replace("\\", "/").lstrip("/")
            if not rel:
                continue
            dest = upload_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(await f.read())

        # The top-level component of the first file's path is the dropped folder name
        first_rel = (files[0].filename or "").replace("\\", "/").lstrip("/")
        top = first_rel.split("/")[0] if "/" in first_rel else ""
        ct_dir = str(upload_dir / top) if top else str(upload_dir)

        patients = _discover_patients(ct_dir)
        return {"path": ct_dir, "patients": patients, "count": len(patients)}
    except ValueError as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(400, str(e))
    except Exception as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(500, str(e))


# ── CT slice preview ───────────────────────────────────────────────────────────
class PreviewRequest(BaseModel):
    path: str
    ct_type: str   # "nifti" or "dicom"


def _extract_height_weight(path: str, ct_type: str) -> Dict:
    """
    Try to read PatientSize (0010,1020) and PatientWeight (0010,1030) from DICOM tags.

    Returns a dict with keys:
      height_m   : float or None
      weight_kg  : float or None
      missing    : list[str]   fields that could not be found
      source     : "dicom" | "not_dicom" | "no_files"
    """
    if ct_type != "dicom":
        return {"height_m": None, "weight_kg": None,
                "missing": ["height", "weight"], "source": "not_dicom"}

    import pydicom

    # Walk the directory tree — DICOM series are often nested several levels deep
    # (e.g. patient/study/series/*.dcm).  os.listdir only sees the top level, so
    # we use rglob to find actual files regardless of nesting depth.
    candidates: List[str] = []
    try:
        for fp in sorted(Path(path).rglob("*")):
            if fp.is_file():
                candidates.append(str(fp))
            if len(candidates) >= 200:
                break
    except Exception:
        pass

    if not candidates:
        return {"height_m": None, "weight_kg": None,
                "missing": ["height", "weight"], "source": "no_files"}

    height: Optional[float] = None
    weight: Optional[float] = None

    for fp in candidates:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            h = getattr(ds, "PatientSize", None)
            w = getattr(ds, "PatientWeight", None)
            h = float(h) if h not in (None, "", " ") else None
            w = float(w) if w not in (None, "", " ") else None
            if h is not None and h <= 0:
                h = None
            if w is not None and w <= 0:
                w = None
            if h is not None:
                height = h
            if w is not None:
                weight = w
            if height is not None and weight is not None:
                break
        except Exception:
            continue

    missing = []
    if height is None:
        missing.append("height")
    if weight is None:
        missing.append("weight")

    return {"height_m": height, "weight_kg": weight,
            "missing": missing, "source": "dicom"}


def _arr_to_png_b64(arr: np.ndarray, max_px: int = 512) -> str:
    """Normalise a 2-D array and encode as base64 PNG.

    The long axis is capped at *max_px* pixels (LANCZOS resampling) so the
    server never sends giant images and the browser never has to upscale.
    """
    from PIL import Image
    arr = arr.astype(np.float32)
    p_lo = float(np.percentile(arr, 2))
    p_hi = float(np.percentile(arr, 98))
    arr = np.clip(arr, p_lo, p_hi)
    if p_hi > p_lo:
        arr = (arr - p_lo) / (p_hi - p_lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    img = Image.fromarray(arr.astype(np.uint8))
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.LANCZOS,
        )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resize_to_physical(arr: np.ndarray, width_mm: float, height_mm: float) -> np.ndarray:
    """
    Resize a 2-D float array so its pixel aspect ratio matches the physical
    dimensions (width_mm × height_mm).  Uses vectorised bilinear row
    interpolation — no extra dependencies beyond numpy.
    """
    curr_h, curr_w = arr.shape
    if width_mm <= 0 or height_mm <= 0 or curr_w == 0 or curr_h == 0:
        return arr
    target_h = max(2, int(round(curr_w * height_mm / width_mm)))
    if target_h == curr_h:
        return arr
    arr_f = arr.astype(np.float32)
    new_rows = np.linspace(0.0, curr_h - 1.0, target_h, dtype=np.float32)
    low  = np.clip(np.floor(new_rows).astype(np.int32), 0, curr_h - 2)
    high = low + 1
    frac = (new_rows - low).reshape(-1, 1)   # (target_h, 1) → broadcasts over columns
    return arr_f[low] * (1.0 - frac) + arr_f[high] * frac


def _load_preview_slices(path: str, ct_type: str) -> Dict:
    """
    Load ONLY the three middle slices needed for preview — never the full volume.

    Returns {"axial": 2d-array, "coronal": 2d-array, "sagittal": 2d-array,
             "shape": [nx, ny, nz]}

    NIfTI: uses nibabel's lazy proxy so only the needed z/y/x slabs are
    actually decompressed from disk.

    DICOM: sorts files by z-position and reads only the minimum number
    needed — one file for axial (middle z), all files but using pixel
    array extraction for coronal/sagittal (unavoidable for those views,
    but we read with SimpleITK which is significantly faster than pydicom
    for bulk slice reading; pydicom is the fallback).
    """
    if ct_type == "nifti":
        import nibabel as nib
        img = nib.load(path)
        sh = img.shape           # (nx, ny, nz) or (nx, ny, nz, t)
        nx, ny, nz = sh[0], sh[1], sh[2]
        proxy = img.dataobj      # lazy proxy — decompresses slices on access

        try:
            zooms = img.header.get_zooms()[:3]
            dx, dy, dz = float(zooms[0]), float(zooms[1]), float(zooms[2])
        except Exception:
            dx = dy = dz = 1.0

        axial    = np.flipud(np.asarray(proxy[:, :, nz // 2]).T)   # (ny, nx)
        coronal  = np.flipud(np.asarray(proxy[:, ny // 2, :]).T)   # (nz, nx)
        sagittal = np.flipud(np.asarray(proxy[nx // 2, :, :]).T)   # (nz, ny)

        # Resize to physical aspect ratio (corrects squishing from anisotropic spacing)
        coronal  = _resize_to_physical(coronal,  nx * dx, nz * dz)
        sagittal = _resize_to_physical(sagittal, ny * dy, nz * dz)

        return {"axial": axial, "coronal": coronal, "sagittal": sagittal,
                "shape": [nx, ny, nz]}

    # ── DICOM ──────────────────────────────────────────────────────────────
    # Try SimpleITK — it uses a streaming reader and is much faster than
    # reading all DICOM files individually with pydicom.
    try:
        import SimpleITK as sitk
        reader = sitk.ImageSeriesReader()
        # Suppress the C++ ITK/GDCM "No Series can be found" stderr warning that
        # fires before GetGDCMSeriesFileNames returns an empty list.
        sitk.ProcessObject.SetGlobalWarningDisplay(False)
        try:
            fnames = reader.GetGDCMSeriesFileNames(path)
        finally:
            sitk.ProcessObject.SetGlobalWarningDisplay(True)
        if not fnames:
            raise RuntimeError("No DICOM series found")
        nz = len(fnames)

        # --- axial: read only the middle file --------------------------------
        single = sitk.ReadImage(fnames[nz // 2])
        arr2d  = sitk.GetArrayFromImage(single)  # (1, ny, nx) or (ny, nx)
        if arr2d.ndim == 3:
            arr2d = arr2d[0]
        ny_px, nx_px = arr2d.shape[0], arr2d.shape[1]
        axial = np.flipud(arr2d)
        try:
            sx = float(single.GetSpacing()[0])   # x pixel size mm
            sy = float(single.GetSpacing()[1])   # y pixel size mm
        except Exception:
            sx = sy = 1.0

        # --- coronal + sagittal: subsampled stack ----------------------------
        N = max(1, nz // 64)
        sparse_fnames = fnames[::N]
        reader.SetFileNames(sparse_fnames)
        itk_img = reader.Execute()
        vol = sitk.GetArrayFromImage(itk_img)   # (nz_s, ny, nx)
        nz_s = vol.shape[0]
        try:
            sz_eff = float(itk_img.GetSpacing()[2])  # z spacing of sparse volume
        except Exception:
            sz_eff = 1.0

        coronal  = np.flipud(vol[:, ny_px // 2, :])   # (nz_s, nx)
        sagittal = np.flipud(vol[:, :, nx_px // 2])   # (nz_s, ny)

        coronal  = _resize_to_physical(coronal,  nx_px * sx, nz_s * sz_eff)
        sagittal = _resize_to_physical(sagittal, ny_px * sy, nz_s * sz_eff)

        return {"axial": axial, "coronal": coronal, "sagittal": sagittal,
                "shape": [nx_px, ny_px, nz]}
    except Exception:
        pass

    # ── pydicom fallback ────────────────────────────────────────────────────
    import glob, pydicom
    patterns = [os.path.join(path, "**", ext)
                for ext in ("*.dcm", "*.DCM", "*.IMA")]
    ffiles: List[str] = sorted(set(f for pat in patterns
                                   for f in glob.glob(pat, recursive=True)))
    if not ffiles:
        raise RuntimeError(f"No DICOM files found in {path}")

    nz = len(ffiles)

    def _z(f):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        return float(getattr(ds, "ImagePositionPatient", [0, 0, 0])[2])
    ffiles.sort(key=_z)

    # Axial: only the middle file
    mid_ds = pydicom.dcmread(ffiles[nz // 2])
    arr2d  = mid_ds.pixel_array
    ny_px, nx_px = arr2d.shape
    axial = np.flipud(arr2d)

    # Pixel spacing from header
    try:
        ps = mid_ds.PixelSpacing
        sx, sy = float(ps[1]), float(ps[0])   # [row_spacing, col_spacing] → (x, y)
    except Exception:
        sx = sy = 1.0
    try:
        sz_orig = float(mid_ds.SliceThickness)
    except Exception:
        sz_orig = 1.0

    # Coronal + sagittal: subsample (max 64 slices)
    N = max(1, nz // 64)
    sparse = ffiles[::N]
    stack = np.stack([pydicom.dcmread(f).pixel_array for f in sparse], axis=0)
    nz_s = stack.shape[0]
    sz_eff = sz_orig * N

    coronal  = np.flipud(stack[:, ny_px // 2, :])   # (nz_s, nx)
    sagittal = np.flipud(stack[:, :, nx_px // 2])   # (nz_s, ny)

    coronal  = _resize_to_physical(coronal,  nx_px * sx, nz_s * sz_eff)
    sagittal = _resize_to_physical(sagittal, ny_px * sy, nz_s * sz_eff)

    return {"axial": axial, "coronal": coronal, "sagittal": sagittal,
            "shape": [nx_px, ny_px, nz]}


@app.post("/api/preview-ct")
async def preview_ct(req: PreviewRequest) -> Dict:
    try:
        loop = asyncio.get_running_loop()
        slices, hw = await asyncio.gather(
            loop.run_in_executor(None, _load_preview_slices, req.path, req.ct_type),
            loop.run_in_executor(None, _extract_height_weight, req.path, req.ct_type),
        )
        return {
            "axial":     _arr_to_png_b64(slices["axial"]),
            "coronal":   _arr_to_png_b64(slices["coronal"]),
            "sagittal":  _arr_to_png_b64(slices["sagittal"]),
            "shape":     slices["shape"],
            "height_m":  hw["height_m"],
            "weight_kg": hw["weight_kg"],
            "missing":   hw["missing"],
            "hw_source": hw["source"],
        }
    except Exception as e:
        raise HTTPException(500, f"Could not load CT preview: {e}")


# ── Pipeline run ───────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    """New-style run request: one pre-created output dir per patient."""
    project_name: str
    patients: List[Dict[str, Any]]        # [{name, type, path}] — original CT paths
    project_dirs: Dict[str, str]           # {patient_name: output_dir_path}
    flags: Dict[str, bool]
    mode: str = "PRODUCTION"


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts: List[str] = []
    if h:
        parts.append(f"{h} hr{'s' if h != 1 else ''}")
    if m:
        parts.append(f"{m} min")
    parts.append(f"{s} sec")
    return " ".join(parts)

@app.post("/api/run")
async def start_run(req: RunRequest) -> Dict:
    """
    Launch the pipeline for each patient using pre-created output directories.

    Each patient gets its own main.py call with:
    - A single-patient temp symlink dir as --input_ct_dir
    - The patient's config.json from its output dir as --config_file
    - --ct_index_start {i}  so output folder naming matches what the web UI created
    """
    if not req.patients:
        raise HTTPException(400, "No patients provided.")

    _prune_finished_runs()
    run_id = str(uuid.uuid4())[:8]

    # Build one job per patient — pass the CT path directly, no temp dirs or symlinks.
    jobs = []
    for i, patient in enumerate(req.patients, start=1):
        nm = patient["name"]
        out_dir = Path(req.project_dirs.get(nm, ""))
        if not out_dir.is_dir():
            raise HTTPException(400, f"Output directory for {nm!r} not found: {out_dir}")

        ct_path = Path(patient["path"])
        if not ct_path.exists():
            # Fall back to the CT copy saved in the output folder on a previous run.
            candidate = out_dir / ct_path.name
            if candidate.exists():
                ct_path = candidate
            else:
                raise HTTPException(
                    400,
                    f"CT source for '{nm}' not found at '{patient['path']}' "
                    f"and no saved copy exists in '{out_dir}'. "
                    "Please re-upload or re-scan the CT directory.",
                )

        jobs.append({
            "ct_path": str(ct_path),
            "config": str(out_dir / "config.json"),
            "ct_index_start": i,
            "patient_name": nm,
        })

    _runs[run_id] = {
        "status": "pending",
        "jobs": jobs,
        "flags": req.flags,
        "mode": req.mode,
        "logs": [],
        "start_time": time.time(),
        "end_time": None,
        "total_patients": len(req.patients),
        "patient_times": {},
        "output_title": req.project_name,
        "output_root": str(REPO_ROOT),
    }

    asyncio.create_task(_execute_run(run_id))
    return {"run_id": run_id}


async def _execute_run(run_id: str) -> None:
    run = _runs[run_id]
    run["status"] = "running"

    def _emit(msg: str) -> None:
        run["logs"].append(msg)
        sys.stdout.write(msg)
        sys.stdout.flush()

    # Build flag args from the flags dict
    flag_map = {
        "spect": "--spect",
        "dosimetry": "--dosimetry",
        "postprocess": "--postprocess",
        "synthetic_lesions": "--synthetic_lesions",
        "logging_on": "--logging_on",
        "save_config": "--save_config",
    }
    flag_args: List[str] = []
    for key, cli_flag in flag_map.items():
        val = run["flags"].get(key, False)
        if val:
            flag_args.append(cli_flag)
        elif key in ("logging_on",):
            flag_args.append(f"--no-{cli_flag.lstrip('-')}")

    total = run["total_patients"]
    processed = 0
    any_failed = False

    try:
        for job in run["jobs"]:
            nm = job["patient_name"]
            ct_idx = job["ct_index_start"]
            processed += 1
            t0 = time.time()
            run["patient_times"][nm] = {"start": t0}
            _emit(f"[VTT] ── Processing CT_{ct_idx} ({nm}) — {processed}/{total} ──\n")

            cmd = [
                sys.executable,
                "-u",
                str(MAIN_PY),
                "--config_file", job["config"],
                "--input_ct", job["ct_path"],
                "--mode", run["mode"],
                "--ct_index_start", str(ct_idx),
                "--launched_via", "web_ui",
            ] + flag_args

            _emit(f"[VTT] Command: {' '.join(cmd)}\n\n")

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=env,
                start_new_session=True,
            )
            run["proc"] = proc

            async for line in proc.stdout:
                _emit(line.decode(errors="replace"))

            await proc.wait()
            rc = proc.returncode

            if rc != 0:
                any_failed = True

            t_end = time.time()
            pt = run["patient_times"].get(nm, {})
            pt["end"] = t_end
            run["patient_times"][nm] = pt
            dt = t_end - pt.get("start", t_end)
            status_str = "OK" if rc == 0 else f"FAILED (exit code {rc})"
            _emit(f"\n[VTT] ── Finished CT_{ct_idx} ({nm}) — {status_str} in {_fmt_duration(dt)} ──\n\n")

        run["status"] = "error" if any_failed else "done"
    except Exception as exc:
        _emit(f"\n[VTT ERROR] {exc}\n")
        run["status"] = "error"
    finally:
        run["end_time"] = time.time()
        _prune_finished_runs(run["end_time"])


# ── WebSocket log streaming ────────────────────────────────────────────────────
@app.websocket("/ws/{run_id}")
async def ws_logs(ws: WebSocket, run_id: str) -> None:
    await ws.accept()
    _prune_finished_runs()

    if run_id not in _runs:
        await ws.send_text(json.dumps({"error": "Run not found"}))
        await ws.close()
        return

    run = _runs[run_id]
    sent = 0   # index into run["logs"] of the next unsent entry

    try:
        while True:
            logs = run["logs"]

            if sent < len(logs):
                chunk = "".join(logs[sent:])
                await ws.send_text(json.dumps({"type": "log", "data": chunk}))
                sent = len(logs)

            if run["status"] in ("done", "error"):
                elapsed = (run["end_time"] or time.time()) - run["start_time"]
                patient_times_out = {
                    name: {
                        **times,
                        "duration": _fmt_duration(times["end"] - times["start"])
                        if "end" in times else None,
                    }
                    for name, times in run["patient_times"].items()
                }
                await ws.send_text(json.dumps({
                    "type": "done",
                    "status": run["status"],
                    "elapsed": elapsed,
                    "elapsed_str": _fmt_duration(elapsed),
                    "patient_times": patient_times_out,
                    "output_title": run.get("output_title", ""),
                    "output_root": run.get("output_root", ""),
                }))
                break

            await asyncio.sleep(0.25)

    except WebSocketDisconnect:
        pass


# ── Stop a running run ────────────────────────────────────────────────────────
@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> Dict:
    _prune_finished_runs()
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    if run["status"] != "running":
        return {"status": run["status"]}
    proc = run.get("proc")
    if proc and proc.returncode is None:
        try:
            # Kill the entire process group so SIMIND/OpenGATE grandchildren
            # are also terminated (start_new_session=True gives them their own pgid).
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            with suppress(Exception):
                proc.kill()
    run["logs"].append("\n[VTT] Run stopped by user.\n")
    run["status"] = "error"
    run["end_time"] = time.time()
    _prune_finished_runs(run["end_time"])
    return {"status": "stopped"}


# ── Run status (for polling fallback) ─────────────────────────────────────────
@app.get("/api/runs/{run_id}")
async def get_run_status(run_id: str) -> Dict:
    _prune_finished_runs()
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    elapsed = time.time() - run["start_time"]
    return {
        "run_id": run_id,
        "status": run["status"],
        "elapsed": elapsed,
        "elapsed_str": _fmt_duration(elapsed),
        "total_patients": run["total_patients"],
        "patient_times": run["patient_times"],
    }
