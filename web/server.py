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
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.resolve()
WEB_DIR   = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
CONFIG_TEMPLATE = REPO_ROOT / "config_template.json"
MAIN_PY = REPO_ROOT / "main.py"
TDT_MAP = REPO_ROOT / "src" / "data" / "tdt_map.json"

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
    "xy_dim": "Resize CT/seg in-plane to this dimension (pixels) before simulation. Smaller = faster. Null = use native CT grid",
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

ROI_CHOICES = ["body", "kidney", "liver", "prostate", "spleen", "heart", "salivary_glands"]


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

    # Auto-fill the label_map_path with the actual repo path
    try:
        data["phase_1"]["segmentation_stage"]["label_map_path"] = str(TDT_MAP)
    except KeyError:
        pass

    return {
        "template": data,
        "roi_choices": ROI_CHOICES,
        "field_descriptions": FIELD_DESCRIPTIONS,
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


# ── CT file upload ─────────────────────────────────────────────────────────────
@app.post("/api/upload-ct")
async def upload_ct(files: List[UploadFile] = File(...)) -> Dict:
    """
    Accept files uploaded via <input type='file' webkitdirectory>.
    Each file.filename is the webkitRelativePath, e.g. 'folder_name/patient/001.dcm'.
    The server reconstructs the directory tree under a temp directory.
    """
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


def _arr_to_png_b64(arr: np.ndarray) -> str:
    """Normalise a 2-D array and encode as base64 PNG."""
    from PIL import Image
    p_lo = float(np.percentile(arr, 2))
    p_hi = float(np.percentile(arr, 98))
    arr = np.clip(arr, p_lo, p_hi).astype(np.float32)
    if p_hi > p_lo:
        arr = (arr - p_lo) / (p_hi - p_lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    img = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_volume(path: str, ct_type: str) -> np.ndarray:
    """Load a CT volume and return it as a (x, y, z) numpy array."""
    if ct_type == "nifti":
        import nibabel as nib
        img = nib.load(path)
        return np.asarray(img.dataobj)

    # DICOM — try SimpleITK first, fall back to pydicom
    try:
        import SimpleITK as sitk
        reader = sitk.ImageSeriesReader()
        fnames = reader.GetGDCMSeriesFileNames(path)
        if not fnames:
            raise RuntimeError("No DICOM series found")
        reader.SetFileNames(fnames)
        itk_img = reader.Execute()
        vol = sitk.GetArrayFromImage(itk_img)   # (z, y, x)
        return np.transpose(vol, (2, 1, 0))     # → (x, y, z)
    except Exception:
        pass

    import glob
    import pydicom
    patterns = [
        os.path.join(path, "**", "*.dcm"),
        os.path.join(path, "**", "*.DCM"),
        os.path.join(path, "**", "*.IMA"),
    ]
    ffiles: List[str] = []
    for pat in patterns:
        ffiles.extend(glob.glob(pat, recursive=True))
    ffiles = sorted(set(ffiles))
    if not ffiles:
        raise RuntimeError(f"No DICOM files found in {path}")

    slices = [pydicom.dcmread(f, stop_before_pixels=False) for f in ffiles]
    slices.sort(key=lambda s: float(
        getattr(s, "ImagePositionPatient", [0, 0, 0])[2]
    ))
    return np.stack([s.pixel_array for s in slices], axis=-1)  # (x, y, z)


@app.post("/api/preview-ct")
async def preview_ct(req: PreviewRequest) -> Dict:
    try:
        loop = asyncio.get_event_loop()
        vol = await loop.run_in_executor(None, _load_volume, req.path, req.ct_type)
        nx, ny, nz = vol.shape

        # Middle slice along each axis; flip so superior is up
        axial    = np.flipud(vol[:, :, nz // 2].T)    # (y, x)
        coronal  = np.flipud(vol[:, ny // 2, :].T)    # (z, x)
        sagittal = np.flipud(vol[nx // 2, :, :].T)    # (z, y)

        return {
            "axial":    _arr_to_png_b64(axial),
            "coronal":  _arr_to_png_b64(coronal),
            "sagittal": _arr_to_png_b64(sagittal),
            "shape":    [nx, ny, nz],
        }
    except Exception as e:
        raise HTTPException(500, f"Could not load CT preview: {e}")


# ── Pipeline run ───────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    ct_dir: str
    config: Dict[str, Any]
    flags: Dict[str, bool]
    mode: str = "PRODUCTION"
    per_patient_configs: Optional[Dict[str, Dict[str, Any]]] = None


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
    try:
        patients = _discover_patients(req.ct_dir)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not patients:
        raise HTTPException(400, "No patients found in the specified CT directory.")

    run_id = str(uuid.uuid4())[:8]
    tmp = Path(tempfile.mkdtemp(prefix=f"vtt_run_{run_id}_"))

    if req.per_patient_configs is None:
        # Same config for all patients — one job covers the whole directory
        cfg_path = tmp / "config.json"
        cfg_path.write_text(json.dumps(req.config, indent=2))
        jobs = [{"ct_dir": req.ct_dir, "config": str(cfg_path), "patients": patients}]
    else:
        # Per-patient config — one job per patient using a symlinked single-patient dir
        jobs = []
        for patient in patients:
            nm = patient["name"]
            pcfg = req.per_patient_configs.get(nm, req.config)
            cfg_path = tmp / f"config_{nm}.json"
            cfg_path.write_text(json.dumps(pcfg, indent=2))

            p_dir = tmp / f"ct_{nm}"
            p_dir.mkdir(exist_ok=True)
            src = Path(patient["path"])
            link = p_dir / src.name
            link.symlink_to(src)

            jobs.append({
                "ct_dir": str(p_dir),
                "config": str(cfg_path),
                "patients": [patient],
            })

    output_title = req.config.get("output_folder_title", "output")
    _runs[run_id] = {
        "status": "pending",
        "jobs": jobs,
        "flags": req.flags,
        "mode": req.mode,
        "logs": [],
        "start_time": time.time(),
        "end_time": None,
        "tmp_dir": str(tmp),
        "total_patients": len(patients),
        "patient_times": {},
        "output_title": output_title,
        "output_root": str(REPO_ROOT),
    }

    asyncio.create_task(_execute_run(run_id))
    return {"run_id": run_id}


async def _execute_run(run_id: str) -> None:
    run = _runs[run_id]
    run["status"] = "running"

    def _emit(msg: str) -> None:
        run["logs"].append(msg)

    # Build flag args from the flags dict
    flag_map = {
        "spect": "--spect",
        "dosimetry": "--dosimetry",
        "postprocess": "--postprocess",
        "synthetic_lesions": "--synthetic_lesions",
        "logging_on": "--logging_on",
        "save_ct_scan": "--save_ct_scan",
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
            for patient in job["patients"]:
                processed += 1
                t0 = time.time()
                run["patient_times"][patient["name"]] = {"start": t0}
                _emit(
                    f"[VTT] ── Processing CT {processed}/{total}: "
                    f"{patient['name']} ──\n"
                )

            cmd = [
                sys.executable,
                "-u",          # unbuffered stdout/stderr → real-time log streaming
                str(MAIN_PY),
                "--config_file", job["config"],
                "--input_ct_dir", job["ct_dir"],
                "--mode", run["mode"],
            ] + flag_args

            _emit(f"[VTT] Command: {' '.join(cmd)}\n\n")

            # Ensure src/ imports resolve: add repo root to PYTHONPATH
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=env,
            )
            run["proc"] = proc   # expose so /stop can kill it

            async for line in proc.stdout:
                _emit(line.decode(errors="replace"))

            await proc.wait()
            rc = proc.returncode

            if rc != 0:
                any_failed = True

            t_end = time.time()
            for patient in job["patients"]:
                pt = run["patient_times"].get(patient["name"], {})
                pt["end"] = t_end
                run["patient_times"][patient["name"]] = pt
                dt = t_end - pt.get("start", t_end)
                status_str = "OK" if rc == 0 else f"FAILED (exit code {rc})"
                _emit(
                    f"\n[VTT] ── Finished {patient['name']} "
                    f"({status_str}) in {_fmt_duration(dt)} ──\n\n"
                )

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
