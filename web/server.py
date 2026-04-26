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
POST /api/run                       Save configs, write pending-jobs file, shut down server
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
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket
from starlette.requests import ClientDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.io.config_paths import inject_pipeline_paths, load_pipeline_paths, strip_developer_fields
from src.io.profiler import validate_profile_interval_s
from src.io.rerun_fingerprints import flatten_paths
from src.io.rerun_guard import (
    STAGE_LABELS,
    build_ct_identity,
    build_dosemap_rerun_snapshot,
    build_opengate_rerun_snapshot,
    build_pbpk_rerun_snapshot,
    build_segmentation_rerun_snapshot,
    build_spect_rerun_snapshot,
    build_synthetic_lesions_rerun_snapshot,
    build_simind_rerun_snapshot,
    compare_stage_rerun_state,
    ct_input_metadata_path,
    fingerprint_optional_file,
    fingerprint_path,
    fingerprints_equal,
    load_json,
    stage_metadata_path,
    synthetic_lesions_enabled,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.resolve()
WEB_DIR   = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
CONFIG_TEMPLATE   = REPO_ROOT / "config_template.json"
MAIN_PY           = REPO_ROOT / "main.py"
VTT_MAP           = REPO_ROOT / "src" / "data" / "vtt_map.json"
PIPELINE_OPTIONS  = REPO_ROOT / "src" / "data" / "pipeline_options.json"

UPLOAD_TTL_SECONDS = 7 * 24 * 60 * 60

# Written by /api/run; read by run_server.py after uvicorn exits to exec main.py.
_PENDING_FILE = REPO_ROOT / ".vtt_pending_run.json"


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


# ── App ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    _prune_old_uploads()
    yield

app = FastAPI(title="Virtual Theranostic Trials", lifespan=lifespan)


# ── Static / SPA ──────────────────────────────────────────────────────────────
STATIC_DIR.mkdir(parents=True, exist_ok=True)
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
    "output_folder_title": "Name for the project folder created under the repo root. Each CT gets its own subfolder: <title>/CT_<index>/",
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
    "xyz_spacing_mm": "Target voxel spacing in mm as [sxy, sxy, sz]. X and Y must match. XY must be >= both native in-plane spacings; Z must be >= the native CT z-spacing (only coarser grids allowed). The pipeline logs the native CT spacing at runtime. Null = native CT resolution (no downsampling).",
    "Collimator": "SIMIND collimator code (e.g. 'si-me' = Siemens medium-energy parallel-hole). Must match your SIMIND install",
    "Isotope": "Isotope code for SIMIND collimator lookup (e.g. 'lu177')",
    "NumProjections": "Number of SPECT angular projection views. Minimum: 64 — fewer causes angular undersampling and streak artifacts in OSEM reconstruction. Rule: Subsets × Iterations must not exceed NumProjections.",
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
    "FrameStartTimes": "Acquisition frame start times in minutes post-injection (e.g. [120, 1440, 2880])",
    "FrameDurations": "Duration of each acquisition frame in seconds. This is always applied when projections are converted to counts.",
    "ReconstructionAlgorithm": "Iterative reconstruction algorithm. Currently only 'OSEM' (with TEW scatter correction) is supported",
    "Iterations": "Number of OSEM reconstruction iterations. Subsets × Iterations must not exceed NumProjections.",
    "Subsets": "Number of OSEM ordered subsets. Higher subsets accelerate convergence but may reduce stability. Subsets × Iterations must not exceed NumProjections.",
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

    # Load ROI choices from the shared label map (VTT_Pipeline section, excluding reserved labels)
    _RESERVED = {"background", "remaining_body", "synthetic_lesion"}
    try:
        tdt_raw = VTT_MAP.read_text(encoding="utf-8")
        tdt_map = json.loads(json_minify(tdt_raw))
        roi_choices = [
            name for name in tdt_map.get("VTT_Pipeline", tdt_map.get("TDT_Pipeline", {})).values()
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
    Create a project directory under REPO_ROOT and one CT subfolder per patient:
      {project_name}/CT_1 / {project_name}/CT_2 / …
    Write a base config.json (user-facing fields only) into each CT dir.

    Returns `existed`: a per-patient flag indicating whether the directory
    already contained pipeline outputs (subdirectories) before this call.
    The frontend uses this to warn the user that prior outputs exist and will be
    reused only if the CT and effective stage config still match the saved metadata.
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

    project_dir = REPO_ROOT / name
    project_dir.mkdir(parents=True, exist_ok=True)

    created: Dict[str, str] = {}
    existed: Dict[str, bool] = {}
    existing_configs: Dict[str, Any] = {}   # patient_name → stripped user config (when prior run exists)

    for i, patient in enumerate(req.patients, start=1):
        out_dir = project_dir / f"CT_{i}"
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
        # Phase 1 synthetic lesions: global outputs now live in the stage dir (not work_dir)
        "phase1_synthetic_lesions": (
            _any_file("digital_twin/synthetic_lesions_stage/*_all_lesions_binary.nii*")
            or _any_file("pipeline_metadata/synthetic_lesions_stage.json")
        ),
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


def _resolve_patient_ct_path(output_dir: Path, patient: Dict[str, Any]) -> Path:
    """Resolve the live CT path for a patient, falling back to the saved copy on rerun."""
    ct_path = Path(str(patient.get("path", "")))
    if ct_path.exists():
        return ct_path
    candidate = output_dir / ct_path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"CT source for '{patient.get('name', 'unknown')}' not found at '{patient.get('path')}' "
        f"and no saved copy exists in '{output_dir}'."
    )


def _load_effective_patient_config(
    output_dir: Path,
    user_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the effective full config for one patient with developer fields injected."""
    if user_config is not None:
        return inject_pipeline_paths(user_config, repo_root=REPO_ROOT, include_input_paths=True)

    cfg_path = output_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json found in {output_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return inject_pipeline_paths(cfg, repo_root=REPO_ROOT, include_input_paths=True)


def _format_rerun_stage_message(stage_name: str, reasons: List[str], metadata_path: Path) -> str:
    """Format a user-facing rerun-validation message for one stage."""
    label = STAGE_LABELS.get(stage_name, stage_name)
    lines = [f"{label}: existing outputs are not safe to reuse."]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.append(f"Delete the old stage outputs for this patient (and '{metadata_path}') and rerun.")
    return "\n".join(lines)


def _compare_ct_provenance(output_dir: Path, ct_path: Path) -> tuple[Dict[str, Any], List[str]]:
    """Return the current CT identity plus any provenance mismatches for this output folder."""
    current_fp = fingerprint_path(ct_path)
    saved_copy_path = output_dir / ct_path.name
    reasons: List[str] = []

    if saved_copy_path.exists():
        saved_fp = fingerprint_path(saved_copy_path)
        if not fingerprints_equal(current_fp, saved_fp):
            reasons.append("the selected CT does not match the CT copy already saved in this output folder")

    meta_path = Path(ct_input_metadata_path(output_dir))
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            stored_fp = meta.get("ct_identity", {}).get("fingerprint")
            if stored_fp and not fingerprints_equal(stored_fp, current_fp):
                reasons.append("the selected CT does not match the recorded CT provenance for this output folder")
        except Exception as exc:
            reasons.append(f"the CT provenance metadata could not be read: {exc}")

    ct_identity = build_ct_identity(
        current_input_path=str(ct_path),
        saved_copy_path=str(saved_copy_path),
        fingerprint=current_fp,
        input_type="dicom" if ct_path.is_dir() else "nii",
    )
    return ct_identity, reasons


def _validate_stage_rerun_state(
    *,
    stage_name: str,
    output_dir: Path,
    required_outputs: Any,
    config_snapshot: Dict[str, Any],
    ct_identity: Dict[str, Any],
    dependency_fingerprints: Dict[str, Any],
) -> Optional[str]:
    """Return a user-facing conflict message when a stage cache is unsafe to reuse."""
    output_paths = flatten_paths(required_outputs)
    existing_paths = [path for path in output_paths if Path(path).exists()]
    if not existing_paths:
        return None

    metadata_path = Path(stage_metadata_path(output_dir, stage_name))
    missing_paths = [path for path in output_paths if not Path(path).exists()]
    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        if len(missing_paths) > 5:
            preview += f", ... ({len(missing_paths)} missing total)"
        return _format_rerun_stage_message(
            stage_name,
            [f"cached outputs are incomplete; missing required output(s): {preview}"],
            metadata_path,
        )

    if not metadata_path.exists():
        return _format_rerun_stage_message(
            stage_name,
            ["cached outputs exist but the persistent stage metadata file is missing"],
            metadata_path,
        )

    try:
        reasons = compare_stage_rerun_state(
            stage_name=stage_name,
            metadata_path=metadata_path,
            current_config_snapshot=config_snapshot,
            current_ct_identity=ct_identity,
            current_upstream_fingerprints=dependency_fingerprints,
        )
    except Exception as exc:
        reasons = [f"the stage metadata could not be validated: {exc}"]

    if reasons:
        return _format_rerun_stage_message(stage_name, reasons, metadata_path)
    return None


def _validate_patient_rerun(
    *,
    patient: Dict[str, Any],
    output_dir: Path,
    config_full: Dict[str, Any],
    flags: Dict[str, bool],
    ct_path: Path,
) -> List[Dict[str, str]]:
    """Return rerun conflicts for one patient/output directory."""
    conflicts: List[Dict[str, str]] = []
    ct_identity, ct_reasons = _compare_ct_provenance(output_dir, ct_path)
    if ct_reasons:
        conflicts.append(
            {
                "stage": "ct_input",
                "message": "\n".join(["CT provenance mismatch:"] + [f"- {r}" for r in ct_reasons]),
            }
        )

    synthetic_enabled = synthetic_lesions_enabled(config_full)

    phase1_dir = output_dir / config_full["phase_1"]["sub_dir_name"]
    phase2_dir = output_dir / config_full["phase_2"]["sub_dir_name"]
    phase3_dir = output_dir / config_full["phase_3"]["sub_dir_name"]

    seg_cfg = config_full["phase_1"]["segmentation_stage"]
    seg_prefix = seg_cfg["file_prefix"]
    seg_stage_dir = phase1_dir / "segmentation_stage"
    expected_vtt_handoff = phase1_dir / "digital_twin.nii.gz"
    seg_markers = [
        seg_stage_dir / f"{seg_prefix}_body_ml.nii.gz",
        seg_stage_dir / f"{seg_cfg['unification_prefix']}.nii.gz",
        phase1_dir / "digital_twin.nii.gz",
    ]
    seg_message = _validate_stage_rerun_state(
        stage_name="segmentation_stage",
        output_dir=output_dir,
        required_outputs=seg_markers,
        config_snapshot=build_segmentation_rerun_snapshot(config_full),
        ct_identity=ct_identity,
        dependency_fingerprints={
            "label_map_json": fingerprint_optional_file(seg_cfg["label_map_path"]),
        },
    )
    if seg_message:
        conflicts.append({"stage": "segmentation_stage", "message": seg_message})

    syn_cfg = config_full["phase_1"]["synthetic_lesions_stage"]
    syn_prefix = syn_cfg["file_prefix"]
    syn_stage_dir = phase1_dir / "synthetic_lesions_stage" / "work_dir"
    syn_pre_lesion_path = syn_stage_dir / f"{syn_prefix}_pre_lesions.nii.gz"
    syn_markers = [
        syn_pre_lesion_path,
        syn_stage_dir / f"{syn_prefix}_all_lesions_binary.nii.gz",
        syn_stage_dir / f"{syn_prefix}_all_lesions_labels.nii.gz",
    ]
    if synthetic_enabled:
        syn_message = _validate_stage_rerun_state(
            stage_name="synthetic_lesions_stage",
            output_dir=output_dir,
            required_outputs=syn_markers,
            config_snapshot=build_synthetic_lesions_rerun_snapshot(config_full),
            ct_identity=ct_identity,
            dependency_fingerprints={
                "segmentation_stage_metadata": fingerprint_optional_file(
                    stage_metadata_path(output_dir, "segmentation_stage")
                ),
                "label_map_json": fingerprint_optional_file(seg_cfg["label_map_path"]),
                "pre_lesion_seg_handoff": fingerprint_optional_file(
                    syn_pre_lesion_path if syn_pre_lesion_path.exists() else phase1_dir / "digital_twin.nii.gz"
                ),
            },
        )
        if syn_message:
            conflicts.append({"stage": "synthetic_lesions_stage", "message": syn_message})

    pbpk_cfg = config_full["phase_1"]["pbpk_tac_stage"]
    pbpk_stage_dir = phase1_dir / pbpk_cfg.get("sub_dir_name", "pbpk_tac_stage")
    pbpk_markers = [
        pbpk_stage_dir / f"{pbpk_cfg['file_prefix']}_tacs.json",
        pbpk_stage_dir / f"{pbpk_cfg['file_prefix']}_tacs.npz",
    ]
    pbpk_message = _validate_stage_rerun_state(
        stage_name="pbpk_tac_stage",
        output_dir=output_dir,
        required_outputs=pbpk_markers,
        config_snapshot=build_pbpk_rerun_snapshot(config_full, synthetic_enabled=synthetic_enabled),
        ct_identity=ct_identity,
        dependency_fingerprints={},
    )
    if pbpk_message:
        conflicts.append({"stage": "pbpk_tac_stage", "message": pbpk_message})

    if flags.get("spect", False):
        simind_cfg = config_full["phase_2"]["simind_stage"]
        simind_prefix = simind_cfg["file_prefix"]
        simind_stage_dir = phase2_dir / simind_cfg.get("sub_dir_name", "simind_simulation")
        simind_pre = simind_stage_dir / "preprocess"
        simind_work = simind_stage_dir / "work_dir"
        simind_snapshot = build_simind_rerun_snapshot(config_full, synthetic_enabled=synthetic_enabled)
        simind_markers = [
            simind_pre / f"{simind_prefix}_preprocess_meta.json",
            simind_pre / f"{simind_prefix}_atn_av.bin",
            simind_pre / f"{simind_prefix}_body_seg.bin",
            simind_pre / f"{simind_prefix}_roi_body_seg.bin",
            phase2_dir / "calib.res",
            phase2_dir / f"{simind_prefix}_tot_w1.a00",
            phase2_dir / f"{simind_prefix}_tot_w2.a00",
            phase2_dir / f"{simind_prefix}_tot_w3.a00",
        ]
        for roi_name in ["remaining_body", *simind_snapshot["resolved_roi_subset"]]:
            simind_markers.extend(
                [
                    simind_work / f"{simind_prefix}_{roi_name}_tot_w1.a00",
                    simind_work / f"{simind_prefix}_{roi_name}_tot_w2.a00",
                    simind_work / f"{simind_prefix}_{roi_name}_tot_w3.a00",
                ]
            )
        simind_deps = {
            "ct_nii": fingerprint_optional_file(phase1_dir / "ct.nii.gz"),
            "vtt_roi_seg": fingerprint_optional_file(expected_vtt_handoff),
            "label_map_json": fingerprint_optional_file(seg_cfg["label_map_path"]),
            "segmentation_stage_metadata": fingerprint_optional_file(
                stage_metadata_path(output_dir, "segmentation_stage")
            ),
        }
        if synthetic_enabled:
            simind_deps["synthetic_lesions_stage_metadata"] = fingerprint_optional_file(
                stage_metadata_path(output_dir, "synthetic_lesions_stage")
            )
        simind_message = _validate_stage_rerun_state(
            stage_name="simind_simulation_stage",
            output_dir=output_dir,
            required_outputs=simind_markers,
            config_snapshot=simind_snapshot,
            ct_identity=ct_identity,
            dependency_fingerprints=simind_deps,
        )
        if simind_message:
            conflicts.append({"stage": "simind_simulation_stage", "message": simind_message})

    if flags.get("dosimetry", False):
        og_cfg = config_full["phase_2"]["opengate_stage"]
        og_prefix = og_cfg["file_prefix"]
        og_stage_dir = phase2_dir / og_cfg.get("sub_dir_name", "opengate_simulation")
        og_work = og_stage_dir / "work_dir"
        og_snapshot = build_opengate_rerun_snapshot(config_full, synthetic_enabled=synthetic_enabled)
        og_markers: List[Path] = []
        if og_cfg.get("save_summed_dose_map", True):
            og_markers.append(og_stage_dir / f"{og_prefix}_dose_sum.nii.gz")
        if og_cfg.get("save_material_label_image", True):
            og_markers.append(og_work / f"{og_prefix}_material_labels.nii.gz")
        og_deps = {
            "ct_nii": fingerprint_optional_file(phase1_dir / "ct.nii.gz"),
            "vtt_roi_seg": fingerprint_optional_file(expected_vtt_handoff),
            "label_map_json": fingerprint_optional_file(seg_cfg["label_map_path"]),
            "segmentation_stage_metadata": fingerprint_optional_file(
                stage_metadata_path(output_dir, "segmentation_stage")
            ),
        }
        if synthetic_enabled:
            og_deps["synthetic_lesions_stage_metadata"] = fingerprint_optional_file(
                stage_metadata_path(output_dir, "synthetic_lesions_stage")
            )
        og_message = _validate_stage_rerun_state(
            stage_name="opengate_simulation_stage",
            output_dir=output_dir,
            required_outputs=og_markers,
            config_snapshot=og_snapshot,
            ct_identity=ct_identity,
            dependency_fingerprints=og_deps,
        )
        if og_message:
            conflicts.append({"stage": "opengate_simulation_stage", "message": og_message})

    if flags.get("spect", False) and flags.get("postprocess", False):
        spect_cfg = config_full["phase_3"]["spect_postprocess_stage"]
        spect_prefix = spect_cfg["file_prefix"]
        spect_stage_dir = phase3_dir / spect_cfg.get("sub_dir_name", "spect_postprocess")
        spect_markers: List[Path] = []
        for frame in spect_cfg.get("FrameStartTimes", []) or []:
            frame_label = f"{float(frame)/60.0:.6f}".rstrip("0").rstrip(".")
            spect_markers.extend(
                [
                    spect_stage_dir / f"{spect_prefix}_{frame_label}_tot_w1.nii.gz",
                    spect_stage_dir / f"{spect_prefix}_{frame_label}_tot_w2.nii.gz",
                    spect_stage_dir / f"{spect_prefix}_{frame_label}_tot_w3.nii.gz",
                ]
            )
            if spect_cfg.get("apply_reconstruction", True):
                spect_markers.append(phase3_dir / f"reconstructed_SPECT_{frame_label}.nii.gz")
        if spect_cfg.get("apply_reconstruction", True):
            spect_markers.append(spect_stage_dir / "recon_atn_img.nii.gz")
        spect_message = _validate_stage_rerun_state(
            stage_name="spect_postprocess_stage",
            output_dir=output_dir,
            required_outputs=spect_markers,
            config_snapshot=build_spect_rerun_snapshot(config_full),
            ct_identity=ct_identity,
            dependency_fingerprints={
                "simind_stage_metadata": fingerprint_optional_file(
                    stage_metadata_path(output_dir, "simind_simulation_stage")
                ),
                "pbpk_tac_json": fingerprint_optional_file(pbpk_markers[0]),
                "pbpk_tac_npz": fingerprint_optional_file(pbpk_markers[1]),
            },
        )
        if spect_message:
            conflicts.append({"stage": "spect_postprocess_stage", "message": spect_message})

    if flags.get("dosimetry", False) and flags.get("postprocess", False):
        dose_cfg = config_full["phase_3"]["dosemap_postprocess_stage"]
        dose_markers = [
            phase3_dir / f"{dose_cfg['file_prefix']}_total_dose.nii.gz",
        ]
        dose_message = _validate_stage_rerun_state(
            stage_name="dosemap_postprocess_stage",
            output_dir=output_dir,
            required_outputs=dose_markers,
            config_snapshot=build_dosemap_rerun_snapshot(config_full),
            ct_identity=ct_identity,
            dependency_fingerprints={
                "opengate_stage_metadata": fingerprint_optional_file(
                    stage_metadata_path(output_dir, "opengate_simulation_stage")
                ),
                "pbpk_tac_json": fingerprint_optional_file(pbpk_markers[0]),
                "pbpk_tac_npz": fingerprint_optional_file(pbpk_markers[1]),
            },
        )
        if dose_message:
            conflicts.append({"stage": "dosemap_postprocess_stage", "message": dose_message})

    return conflicts


def _validate_reruns_for_request(
    *,
    patients: List[Dict[str, Any]],
    project_dirs: Dict[str, str],
    flags: Dict[str, bool],
    patient_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """Validate rerun compatibility for every patient in a request payload."""
    conflicts_by_patient: Dict[str, List[Dict[str, str]]] = {}
    patient_configs = patient_configs or {}

    for patient in patients:
        patient_name = patient["name"]
        output_dir = Path(project_dirs.get(patient_name, ""))
        if not output_dir.is_dir():
            conflicts_by_patient[patient_name] = [
                {
                    "stage": "setup",
                    "message": f"Output directory not found: {output_dir}",
                }
            ]
            continue

        try:
            ct_path = _resolve_patient_ct_path(output_dir, patient)
            config_full = _load_effective_patient_config(output_dir, patient_configs.get(patient_name))
            conflicts = _validate_patient_rerun(
                patient=patient,
                output_dir=output_dir,
                config_full=config_full,
                flags=flags,
                ct_path=ct_path,
            )
        except Exception as exc:
            conflicts = [{"stage": "setup", "message": str(exc)}]

        if conflicts:
            conflicts_by_patient[patient_name] = conflicts

    return conflicts_by_patient


class ValidateRerunRequest(BaseModel):
    patients: List[Dict[str, Any]]
    project_dirs: Dict[str, str]
    flags: Dict[str, bool]
    patient_configs: Optional[Dict[str, Dict[str, Any]]] = None


@app.post("/api/validate-rerun")
async def validate_rerun(req: ValidateRerunRequest) -> Dict:
    """Check whether existing patient outputs can be safely reused on rerun."""
    conflicts_by_patient = _validate_reruns_for_request(
        patients=req.patients,
        project_dirs=req.project_dirs,
        flags=req.flags,
        patient_configs=req.patient_configs,
    )
    return {
        "ok": not conflicts_by_patient,
        "conflicts_by_patient": conflicts_by_patient,
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
                "shape": [nx, ny, nz], "spacing": [dx, dy, dz]}

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

        sz_actual = sz_eff / max(1, N)
        return {"axial": axial, "coronal": coronal, "sagittal": sagittal,
                "shape": [nx_px, ny_px, nz], "spacing": [sx, sy, sz_actual]}
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

    header_pairs = [(f, pydicom.dcmread(f, stop_before_pixels=True)) for f in ffiles]

    def _header_z(ds):
        try:
            return float(getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])[2])
        except Exception:
            return 0.0

    header_pairs.sort(key=lambda pair: _header_z(pair[1]))
    ffiles = [f for f, _ in header_pairs]
    headers = [ds for _, ds in header_pairs]
    nz = len(ffiles)

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

    sz_orig = None
    try:
        z_positions = [_header_z(ds) for ds in headers]
        z_diffs = [abs(b - a) for a, b in zip(z_positions, z_positions[1:]) if abs(b - a) > 1e-6]
        if z_diffs:
            sz_orig = float(np.median(z_diffs))
    except Exception:
        sz_orig = None
    if not sz_orig or sz_orig <= 0:
        for attr in ("SpacingBetweenSlices", "SliceThickness"):
            try:
                sz_orig = float(getattr(mid_ds, attr))
            except Exception:
                continue
            if sz_orig > 0:
                break
    if not sz_orig or sz_orig <= 0:
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
            "shape": [nx_px, ny_px, nz], "spacing": [sx, sy, sz_orig]}


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
            "spacing":   slices.get("spacing"),
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
    patient_configs: Optional[Dict[str, Dict[str, Any]]] = None
    mode: str = "PRODUCTION"
    profile_interval_s: float = 2.0


async def _shutdown_after_delay() -> None:
    await asyncio.sleep(0.5)
    os.kill(os.getpid(), signal.SIGINT)


@app.post("/api/run")
async def start_run(req: RunRequest) -> Dict:
    """
    Save configs, write a pending-jobs file, then shut down the server.
    run_server.py picks up the pending file after uvicorn exits and runs
    each patient's main.py command directly in the terminal with inherited
    stdio — identical to the CLI experience.
    """
    if not req.patients:
        raise HTTPException(400, "No patients provided.")

    # Save patient configs sent by the UI (in case /api/save-patient-config wasn't called).
    if req.patient_configs:
        for patient in req.patients:
            nm = patient["name"]
            cfg = req.patient_configs.get(nm)
            out_dir = Path(req.project_dirs.get(nm, ""))
            if cfg and out_dir.is_dir():
                full_cfg = inject_pipeline_paths(cfg, repo_root=REPO_ROOT, include_input_paths=False)
                (out_dir / "config.json").write_text(json.dumps(full_cfg, indent=2))

    conflicts_by_patient = _validate_reruns_for_request(
        patients=req.patients,
        project_dirs=req.project_dirs,
        flags=req.flags,
        patient_configs=req.patient_configs,
    )
    if conflicts_by_patient:
        summary_lines: List[str] = []
        for patient_name, conflicts in conflicts_by_patient.items():
            summary_lines.append(f"{patient_name}:")
            summary_lines.extend(f"  - {c['message'].splitlines()[0]}" for c in conflicts)
        raise HTTPException(
            400,
            "Rerun validation failed. Delete the conflicting old outputs before running again.\n"
            + "\n".join(summary_lines),
        )

    if req.flags.get("profile", False):
        try:
            validate_profile_interval_s(float(req.profile_interval_s))
        except ValueError:
            raise HTTPException(
                400,
                "profile_interval_s must be between 0.1 and 3.0 seconds when profiling is enabled.",
            )

    flag_map = {
        "spect": "--spect",
        "dosimetry": "--dosimetry",
        "postprocess": "--postprocess",
        "logging_on": "--logging_on",
        "save_config": "--save_config",
        "profile": "--profile",
    }
    flag_args: List[str] = []
    for key, cli_flag in flag_map.items():
        val = req.flags.get(key, False)
        if val:
            flag_args.append(cli_flag)
        elif key == "logging_on":
            flag_args.append("--no-logging_on")

    jobs = []
    for i, patient in enumerate(req.patients, start=1):
        nm = patient["name"]
        out_dir = Path(req.project_dirs.get(nm, ""))
        if not out_dir.is_dir():
            raise HTTPException(400, f"Output directory for {nm!r} not found: {out_dir}")

        ct_path = Path(patient["path"])
        if not ct_path.exists():
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

        cmd = [
            sys.executable, "-u", str(MAIN_PY),
            "--config_file", str(out_dir / "config.json"),
            "--input_ct", str(ct_path),
            "--mode", req.mode,
            "--ct_index_start", str(i),
            "--launched_via", "web_ui",
        ] + flag_args
        if req.flags.get("profile", False):
            cmd.extend(["--profile_interval_s", str(float(req.profile_interval_s))])
        jobs.append({"cmd": cmd, "patient_name": nm})

    _PENDING_FILE.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    asyncio.create_task(_shutdown_after_delay())
    return {"ok": True}
