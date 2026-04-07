"""
FastAPI backend for the Virtual Theranostic Trials web UI.

Endpoints
---------
GET  /                          Serve the React SPA (index.html)
GET  /api/config-template       Return parsed config template + field metadata
POST /api/scan-directory        Scan a local CT input directory → patient list
POST /api/upload-ct             Accept uploaded directory files → temp path + patient list
POST /api/preview-ct            Generate axial/coronal/sagittal PNG previews for one CT
POST /api/run                   Start a pipeline run → run_id
WS   /ws/{run_id}               Stream live logs for a run
GET  /api/runs/{run_id}         Poll run status / timing
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from starlette.requests import ClientDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.resolve()
WEB_DIR   = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
CONFIG_TEMPLATE   = REPO_ROOT / "config_template.json"
MAIN_PY           = REPO_ROOT / "main.py"
TDT_MAP           = REPO_ROOT / "src" / "data" / "tdt_map.json"
PIPELINE_OPTIONS  = REPO_ROOT / "src" / "data" / "pipeline_options.json"
PIPELINE_PATHS    = REPO_ROOT / "src" / "data" / "pipeline_paths.json"

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Virtual Theranostic Trials")

# In-memory registry: run_id → run dict
_runs: Dict[str, Dict] = {}


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ── Static / SPA ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=500)
    return HTMLResponse(index.read_text(encoding="utf-8"))


# ── Pipeline input paths (SIMINDDirectory, label_map_path) ────────────────────
@app.get("/api/input-paths")
async def get_input_paths() -> Dict:
    """Return the current input_paths section from pipeline_paths.json."""
    try:
        from json_minify import json_minify as _jm
        pp = json.loads(_jm(PIPELINE_PATHS.read_text()))
        return pp.get("input_paths", {})
    except Exception as e:
        raise HTTPException(500, str(e))


class InputPathsUpdate(BaseModel):
    label_map_path: str = ""
    SIMINDDirectory: str = ""


@app.post("/api/input-paths")
async def set_input_paths(req: InputPathsUpdate) -> Dict:
    """Overwrite input_paths in pipeline_paths.json."""
    try:
        from json_minify import json_minify as _jm
        pp = json.loads(_jm(PIPELINE_PATHS.read_text()))
        pp.setdefault("input_paths", {})
        pp["input_paths"]["label_map_path"] = req.label_map_path
        pp["input_paths"]["SIMINDDirectory"] = req.SIMINDDirectory
        PIPELINE_PATHS.write_text(json.dumps(pp, indent=2))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Field descriptions (used by the frontend to render tooltips) ───────────────
FIELD_DESCRIPTIONS: Dict[str, str] = {
    "output_folder_title": "Name for the output folder created under the repo root. Each CT gets its own subfolder: <title>_CT_<index>/",
    "sub_dir_name": "Internal subdirectory name for this pipeline phase (advanced)",
    "file_prefix": "Filename prefix for outputs from this stage (advanced)",
    "roi_subset": "Which organs to segment and include in downstream simulation stages",
    "unification_prefix": "Prefix for the combined multi-label segmentation file (advanced)",
    "label_map_path": "Absolute path to tdt_map.json — maps TDT ROI names to integer label IDs",
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
    "SIMINDDirectory": "Absolute path to the directory containing the SIMIND executable (e.g. /home/user/simind/simind)",
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
    "NumCores": "CPU cores for parallel SIMIND simulation. 0 = auto-detect and use all available cores",
    "save_per_roi_dose_maps": "Save a separate dose map NIfTI file for each segmented organ/ROI",
    "save_summed_dose_map": "Save a single NIfTI with the total dose summed across all ROIs",
    "save_uncertainty_map": "Save Monte Carlo statistical uncertainty maps alongside dose maps",
    "save_material_label_image": "Save the Schneider HU→material composition label image used in simulation",
    "write_mhd_outputs": "Also write outputs as MetaImage (.mhd/.raw) in addition to NIfTI",
    "output_dose_spacing_mm": "Resample dose maps to this isotropic voxel spacing in mm. Null = match CT",
    "variance_reduction": "Enable forced-detection variance reduction to accelerate Monte Carlo (see config comments for Lu-177 caveats)",
    "total_histories": "Total Monte Carlo particle histories for dosimetry (e.g. 1e7). Higher = more accurate, slower",
    "num_threads": "Number of parallel OpenGATE simulation threads. More threads = faster but needs more RAM",
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
    "sigma_mm": "Standard deviation in mm for Gaussian lesion placement — controls spread from organ centroid",
    "margin_mm": "Minimum gap in mm between the lesion surface and the organ boundary (and other lesions)",
    "seed": "Random seed for this organ's lesion placement. 0 = non-reproducible",
    "Randomization_Kidney_SG_Para": "If true, kidney and salivary-gland PBPK parameters are randomly sampled from lognormal distributions",
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
    data = _strip_developer_fields(data)

    # Load TDT ROI choices from tdt_map.json (TDT_Pipeline section, excluding reserved labels)
    _RESERVED = {"background", "body", "synthetic_lesion"}
    try:
        tdt_raw = TDT_MAP.read_text(encoding="utf-8")
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
        loop = asyncio.get_event_loop()
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
        loop = asyncio.get_event_loop()
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
    """
    name = req.project_name.strip()
    if not name:
        raise HTTPException(400, "Project name is required.")

    try:
        from json_minify import json_minify as _jm
        base_cfg: dict = json.loads(_jm(CONFIG_TEMPLATE.read_text(encoding="utf-8")))
    except Exception:
        base_cfg = {}

    base_cfg = _strip_developer_fields(base_cfg)
    base_cfg["output_folder_title"] = name

    created: Dict[str, str] = {}
    for i, patient in enumerate(req.patients, start=1):
        out_dir = REPO_ROOT / f"{name}_CT_{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "config.json"
        if not cfg_path.exists():
            full_cfg = _inject_paths_into_config(base_cfg)
            cfg_path.write_text(json.dumps(full_cfg, indent=2))
        created[patient["name"]] = str(out_dir)

    return {"dirs": created, "project_name": name}


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
    full_cfg = _inject_paths_into_config(req.config)
    (out_dir / "config.json").write_text(json.dumps(full_cfg, indent=2))
    return {"ok": True}


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

    tmp = Path(tempfile.mkdtemp(prefix="vtt_ct_upload_"))
    try:
        for f in files:
            # Normalise path separators (browser may send \ on Windows)
            rel = (f.filename or "").replace("\\", "/").lstrip("/")
            if not rel:
                continue
            dest = tmp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(await f.read())

        # The top-level component of the first file's path is the dropped folder name
        first_rel = (files[0].filename or "").replace("\\", "/").lstrip("/")
        top = first_rel.split("/")[0] if "/" in first_rel else ""
        ct_dir = str(tmp / top) if top else str(tmp)

        patients = _discover_patients(ct_dir)
        return {"path": ct_dir, "patients": patients, "count": len(patients), "tmp": True}
    except ValueError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(400, str(e))
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(500, str(e))


# ── CT slice preview ───────────────────────────────────────────────────────────
class PreviewRequest(BaseModel):
    path: str
    ct_type: str   # "nifti" or "dicom"


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
        proxy = img.dataobj      # lazy — only decompresses what you index

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
        fnames = reader.GetGDCMSeriesFileNames(path)
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
        loop = asyncio.get_event_loop()
        slices = await loop.run_in_executor(
            None, _load_preview_slices, req.path, req.ct_type
        )
        return {
            "axial":    _arr_to_png_b64(slices["axial"]),
            "coronal":  _arr_to_png_b64(slices["coronal"]),
            "sagittal": _arr_to_png_b64(slices["sagittal"]),
            "shape":    slices["shape"],
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


# Developer-controlled fields that live in pipeline_paths.json and must NOT
# appear in user-facing config files shown or edited via the web UI.
_DEVELOPER_FIELDS = {"sub_dir_name", "file_prefix", "unification_prefix", "label_map_path", "SIMINDDirectory"}


def _strip_developer_fields(cfg: dict) -> dict:
    """Remove developer-controlled keys from a config dict (deep copy, non-destructive)."""
    import copy as _copy
    cfg = _copy.deepcopy(cfg)

    def _strip(obj: Any) -> None:
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k in _DEVELOPER_FIELDS:
                    del obj[k]
                else:
                    _strip(obj[k])

    _strip(cfg)
    return cfg


def _inject_paths_into_config(cfg: dict) -> dict:
    """
    Inject ALL developer-controlled fields from pipeline_paths.json into *cfg*.

    This covers:
    - input_paths.label_map_path  → phase_1.segmentation_stage.label_map_path
    - input_paths.SIMINDDirectory → phase_2.simind_stage.SIMINDDirectory
    - phase_*/sub_dir_name         → each phase's sub_dir_name
    - phase_*/*/file_prefix        → each stage's file_prefix / unification_prefix

    Returns a new dict with all fields filled in.
    """
    import copy as _copy
    cfg = _copy.deepcopy(cfg)

    try:
        from json_minify import json_minify as _jm
        pipeline_paths = json.loads(_jm(PIPELINE_PATHS.read_text(encoding="utf-8")))
    except Exception:
        pipeline_paths = {}

    input_paths = pipeline_paths.get("input_paths", {})

    # label_map_path
    lmp = input_paths.get("label_map_path", "").strip()
    if not lmp:
        lmp = str(TDT_MAP)
    cfg.setdefault("phase_1", {}).setdefault("segmentation_stage", {})["label_map_path"] = lmp

    # SIMINDDirectory
    sdir = input_paths.get("SIMINDDirectory", "").strip()
    if sdir:
        cfg.setdefault("phase_2", {}).setdefault("simind_stage", {})["SIMINDDirectory"] = sdir

    # sub_dir_name and file_prefix from pipeline_paths.json phases
    _STAGE_FIELDS = ("file_prefix", "unification_prefix", "sub_dir_name")
    for phase_key in ("phase_1", "phase_2", "phase_3"):
        pp_phase = pipeline_paths.get(phase_key, {})
        if not isinstance(pp_phase, dict):
            continue
        cfg_phase = cfg.setdefault(phase_key, {})
        if "sub_dir_name" in pp_phase:
            cfg_phase["sub_dir_name"] = pp_phase["sub_dir_name"]
        for stage_key, stage_data in pp_phase.items():
            if stage_key == "sub_dir_name" or not isinstance(stage_data, dict):
                continue
            cfg_stage = cfg_phase.setdefault(stage_key, {})
            for field in _STAGE_FIELDS:
                if field in stage_data:
                    cfg_stage[field] = stage_data[field]

    return cfg


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

    run_id = str(uuid.uuid4())[:8]
    tmp = Path(tempfile.mkdtemp(prefix=f"vtt_run_{run_id}_"))

    # Build one job per patient
    jobs = []
    for i, patient in enumerate(req.patients, start=1):
        nm = patient["name"]
        out_dir = Path(req.project_dirs.get(nm, ""))
        if not out_dir.is_dir():
            raise HTTPException(400, f"Output directory for {nm!r} not found: {out_dir}")

        # Single-patient temp dir with symlink to the original CT
        pt_dir = tmp / f"ct_{i}"
        pt_dir.mkdir()
        src = Path(patient["path"])
        link = pt_dir / src.name
        link.symlink_to(src)

        jobs.append({
            "ct_dir": str(pt_dir),
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
        "tmp_dir": str(tmp),
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
                "--input_ct_dir", job["ct_dir"],
                "--mode", run["mode"],
                "--ct_index_start", str(ct_idx),
                "--launched_via", "web_ui",
            ] + flag_args

            _emit(f"[VTT] Command: {' '.join(cmd)}\n\n")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=env,
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
        # Clean up the temporary run directory (config JSONs, symlinks)
        tmp = run.get("tmp_dir")
        if tmp and Path(tmp).exists():
            shutil.rmtree(tmp, ignore_errors=True)


# ── WebSocket log streaming ────────────────────────────────────────────────────
@app.websocket("/ws/{run_id}")
async def ws_logs(ws: WebSocket, run_id: str) -> None:
    await ws.accept()

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
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    if run["status"] != "running":
        return {"status": run["status"]}
    proc = run.get("proc")
    if proc and proc.returncode is None:
        try:
            proc.kill()   # SIGKILL — immediate, no cleanup
        except ProcessLookupError:
            pass
    run["logs"].append("\n[VTT] Run stopped by user.\n")
    run["status"] = "error"
    run["end_time"] = time.time()
    return {"status": "stopped"}


# ── Run status (for polling fallback) ─────────────────────────────────────────
@app.get("/api/runs/{run_id}")
async def get_run_status(run_id: str) -> Dict:
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
