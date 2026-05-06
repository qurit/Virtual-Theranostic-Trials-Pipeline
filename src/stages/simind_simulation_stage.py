"""
SIMIND simulation for the PyTheraTwin pipeline.

This stage prepares inputs for SIMIND and runs Monte Carlo SPECT projection simulations.

Preprocessing:
- Converting CT + segmentation NIfTIs into the SIMIND grid convention (z, y, x with y-flip).
- Optionally resizing to a target in-plane dimension via isotropic zoom.
- Building ROI masks and a label->name class map from the unified PyTheraTwin multilabel segmentation.
- Writing binary files used by SIMIND (attenuation map, body mask, per-ROI binary source maps).

Simulation:
- Validate required context fields (labels, spacing, files).
- Configure SIMIND environment variables (SMC_DIR, PATH).
- Copy SIMIND template files into a per-CT work directory.
- For each ROI:
  - Copy the source map binary into the work directory.
  - Run SIMIND in parallel using `num_cpu` processes (each with a unique /rr: seed).
  - Aggregate per-core projection totals into a single per-organ file per energy window.
- Run a Jaszczak-based calibration (if not already present) to produce `calib.res`.
- Sum per-organ projections into total projections per energy window.
- Copy SIMIND headers to the phase output directory for downstream reconstruction.

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.config["phase_2"]["simind_stage"] : dict (including roi_subset, xyz_spacing_mm)
- label map loaded from ``src/data/pipeline_paths.json`` input_paths.label_map_path
- context.subdir_paths["phase_2"] : str
- context.mode : str  ("DEBUG" or "PRODUCTION")
- context.ct_nii_path : str
- context.pytheratwin_roi_seg_path : str  (unified PyTheraTwin ROI segmentation NIfTI)
- context.downstream_roi_subset : list[str] | None

On success, this stage sets:
- context.body_seg_arr, context.roi_body_seg_arr, context.mask_roi_body, context.class_seg
- context.atn_av_path, context.binary_roi_act_map_paths
- context.arr_px_spacing_cm, context.arr_shape_new
- context.spect_sim_output_dir, context.simind_stage_output_dir
- context.simind_work_dir, context.simind_metadata_path
- context.simind_calibration_path, context.simind_projection_paths
- context.simind_summed_projection_paths
- context.simind_num_cpu, context.simind_geometry
- context.simind_total_num_voxels, context.simind_scale_factor
- context.simind_switches_by_organ
- context.simind_header_dir
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from src.io.config_paths import get_label_map_path
from src.io.rerun_guard import (
    assert_stage_rerun_safe,
    build_stage_metadata,
    build_simind_rerun_snapshot,
    fingerprint_optional_file,
    load_json,
    stage_metadata_path,
    write_json,
)
from src.utils.label_utils import load_pytheratwin_label_map, build_class_map, build_label_masks, filter_roi_seg_to_subset, load_isotope_config
from src.utils.resize_utils import resolve_simulation_grid
from src.utils.simind_runtime_utils import (
    aggregate_core_projection_totals,
    build_projection_path_dict,
    build_simind_switches,
    configure_simind_environment,
    copy_simind_headers_to_dir,
    copy_simind_templates,
    derive_input_geometry,
    get_projection_paths_for_organ,
    get_summed_projection_paths,
    hu_to_mu,
    organ_headers_exist,
    organ_totals_exist,
    run_jaszczak_calibration,
    sum_projections_across_organs,
)


class _SimindPreprocessor:                                                             
    """
    Internal helper: prepare CT + ROI masks for SIMIND simulation.

    Grid convention
    ---------------
    Arrays are transposed to (z, y, x) and flipped in `y` to match SIMIND's expected
    orientation before any resizing or binary export.

    Resizing
    --------
    If a resize target is provided, an isotropic scale factor is derived from
    the in-plane dimension and applied to all three axes via scipy.ndimage.zoom.
    """

    def __init__(
        self,
        ct_nii_path: str,
        pytheratwin_roi_seg_path: str,
        pytheratwin_name2id: Dict[str, int],
        roi_subset: Sequence[str],
        output_dir: str,
        prefix: str,
        xyz_spacing_mm: Optional[list],
        mu_water: float,
        mu_bone: float,
        debug: bool = False,
    ) -> None:
        self.ct_nii_path = ct_nii_path
        self.pytheratwin_roi_seg_path = pytheratwin_roi_seg_path
        self.pytheratwin_name2id = pytheratwin_name2id
        self.roi_subset = roi_subset
        self.output_dir = output_dir
        self.prefix = prefix
        self.xyz_spacing_mm = xyz_spacing_mm
        self.mu_water = mu_water
        self.mu_bone = mu_bone
        self.debug = debug

    def _write_attenuation_bin(
        self,
        ct_arr: np.ndarray,
        body_seg_arr: np.ndarray,
        pixel_size_cm: float,
        filename: str,
    ) -> str:
        """
        Compute the attenuation map and write it as a flat float32 binary for SIMIND.

        The mu map is masked to the body so air outside the patient is zeroed.
        """
        mu_map = hu_to_mu(np.asarray(ct_arr, dtype=np.float32), pixel_size_cm, self.mu_water, self.mu_bone)
        mu_map *= body_seg_arr  # zero outside body
        out_path = os.path.join(self.output_dir, filename)
        mu_map.tofile(out_path)
        return out_path

    @staticmethod
    def _to_simind_grid(
        nii_obj: nib.Nifti1Image,
        xyz_spacing_mm: Optional[list] = None,
        transpose_tuple: Tuple[int, int, int] = (2, 1, 0),
        zoom_order: int = 0,
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Convert a NIfTI object to SIMIND grid format with optional resampling.

        Convention: transpose to (z, y, x), then flip y.

        Parameters
        ----------
        nii_obj : nib.Nifti1Image
        xyz_spacing_mm : list[float] | None
            Target voxel spacing in mm as ``[sx, sy, sz]``.  Each value must be
            >= the native CT spacing on that axis.  ``None`` = no resampling.
        transpose_tuple : tuple[int,int,int]
        zoom_order : int  0 = nearest (seg), 1 = linear (CT)

        Returns
        -------
        (array_zyx, scale_x, scale_y, scale_z)
        """
        arr = np.array(nii_obj.get_fdata(dtype=np.float32))
        arr = np.transpose(arr, transpose_tuple)[:, ::-1, :]  # now (z, y, x)

        # arr is (sz, sy, sx); present as (sx, sy, sz) for resolve_simulation_grid
        sz, sy, sx = arr.shape
        zooms_nii_mm = np.array(nii_obj.header.get_zooms()[:3], dtype=float)  # (x, y, z)
        result = resolve_simulation_grid(xyz_spacing_mm, (sx, sy, sz), tuple(zooms_nii_mm))

        scale_x = scale_y = scale_z = 1.0
        if result is not None:
            (nx, ny, nz), (scale_x, scale_y, scale_z) = result
            arr = zoom(arr, (scale_z, scale_y, scale_x), order=zoom_order)

        return arr, scale_x, scale_y, scale_z

    def _write_binary_roi_maps(
        self,
        roi_body_arr: np.ndarray,
        class_seg: Dict[str, int],
        arr_px_spacing_cm: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, str]:
        """
        Write one flat float32 binary 0/1 source map per ROI for SIMIND.

        Also saves a NIfTI copy of each mask (same grid, same spacing) alongside
        the .bin file so the SIMIND-grid source maps are human-inspectable without
        special binary readers — mirroring what OpenGATE writes to its work_dir.

        Returns
        -------
        dict[str, str]  {roi_name: path}  (.bin paths; NIfTIs are saved as a side-effect)
        """
        out_paths: Dict[str, str] = {}
        for roi_name, lab in class_seg.items():
            roi_mask = (roi_body_arr == lab).astype(np.float32)
            out_path = os.path.join(self.output_dir, f"{self.prefix}_{roi_name}_act_av.bin")
            roi_mask.tofile(out_path)
            out_paths[roi_name] = out_path

            # NIfTI companion for visual inspection.
            nii_path = os.path.join(self.output_dir, f"{self.prefix}_{roi_name}_source_mask.nii.gz")
            affine = np.eye(4)
            if arr_px_spacing_cm is not None:
                # spacing is (z, y, x) in cm → convert to mm and embed in affine diagonal
                sz, sy, sx = [v * 10.0 for v in arr_px_spacing_cm]
                affine[0, 0] = sx
                affine[1, 1] = sy
                affine[2, 2] = sz
            nib.save(nib.Nifti1Image(roi_mask, affine), nii_path)

        return out_paths

    def _load_preprocess_cache(self, meta_path: str) -> Dict[str, Any]:
        """
        Reload preprocessing outputs from disk when all binaries already exist.

        Returns the same dict as run() without reprocessing CT/seg.
        """
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        shape = tuple(meta["shape"])
        arr_px_spacing_cm = tuple(float(x) * 0.1 for x in meta["zooms_mm"])

        atn_path = os.path.join(self.output_dir, f"{self.prefix}_atn_av.bin")
        body_path = os.path.join(self.output_dir, f"{self.prefix}_body_seg.bin")
        roi_body_path = os.path.join(self.output_dir, f"{self.prefix}_roi_body_seg.bin")

        body_seg_arr = np.fromfile(body_path, dtype=np.float32).reshape(shape)
        roi_body_arr = np.fromfile(roi_body_path, dtype=np.float32).reshape(shape).astype(np.int16)

        id_to_name = {v: k for k, v in self.pytheratwin_name2id.items()}
        class_seg = build_class_map(roi_body_arr, id_to_name)
        masks = build_label_masks(roi_body_arr)

        binary_roi_act_map_paths: Dict[str, str] = {}
        for roi_name in class_seg:
            p = os.path.join(self.output_dir, f"{self.prefix}_{roi_name}_act_av.bin")
            if os.path.exists(p):
                binary_roi_act_map_paths[roi_name] = p

        return {
            "body_seg_arr": body_seg_arr,
            "roi_body_seg_arr": roi_body_arr,
            "masks": masks,
            "class_seg": class_seg,
            "atn_av_path": atn_path,
            "binary_roi_act_map_paths": binary_roi_act_map_paths,
            "arr_px_spacing_cm": arr_px_spacing_cm,
            "arr_shape_new": shape,
        }

    def run(self) -> Dict[str, Any]:
        """
        Execute preprocessing and return results dict instead of setting context.

        Skips reprocessing if all output binaries and a cache meta file already exist
        (allowing reruns to skip the expensive CT/seg grid conversion).

        Returns
        -------
        dict with keys: body_seg_arr, roi_body_seg_arr, masks, class_seg,
                        atn_av_path, binary_roi_act_map_paths, arr_px_spacing_cm,
                        arr_shape_new
        """
        if self.ct_nii_path is None or not os.path.exists(self.ct_nii_path):
            raise FileNotFoundError(f"ct_nii_path not found: {self.ct_nii_path}")
        if self.pytheratwin_roi_seg_path is None or not os.path.exists(self.pytheratwin_roi_seg_path):
            raise FileNotFoundError(f"Unified PyTheraTwin ROI seg not found: {self.pytheratwin_roi_seg_path}")
        if not self.roi_subset:
            raise ValueError("No ROI subset provided for SIMIND preprocessing.")

        # Skip reprocessing if all key outputs already exist AND every ROI in the
        # current roi_subset has its binary source map on disk.  If the roi_subset
        # was enlarged since the last run, the new organ's map would be missing and
        # the simulation would silently drop it — so we must re-run preprocessing.
        meta_path     = os.path.join(self.output_dir, f"{self.prefix}_preprocess_meta.json")
        atn_path      = os.path.join(self.output_dir, f"{self.prefix}_atn_av.bin")
        body_path     = os.path.join(self.output_dir, f"{self.prefix}_body_seg.bin")
        roi_body_path = os.path.join(self.output_dir, f"{self.prefix}_roi_body_seg.bin")
        base_ok = all(os.path.exists(p) for p in (meta_path, atn_path, body_path, roi_body_path))
        if base_ok:
            # Also verify that every non-remaining_body ROI has a binary map cached.
            roi_maps_ok = all(
                os.path.exists(os.path.join(self.output_dir, f"{self.prefix}_{r}_act_av.bin"))
                for r in self.roi_subset
                if r != "remaining_body"
            )
            if roi_maps_ok:
                if self.debug:
                    print("[SimindPreprocessor] Preprocessing outputs already exist, skipping.")
                return self._load_preprocess_cache(meta_path)
            elif self.debug:
                missing = [
                    r for r in self.roi_subset
                    if r != "remaining_body"
                    and not os.path.exists(
                        os.path.join(self.output_dir, f"{self.prefix}_{r}_act_av.bin")
                    )
                ]
                print(
                    f"[SimindPreprocessor] ROI binary maps missing for {missing}; "
                    "re-running preprocessing."
                )

        ct_nii = nib.load(self.ct_nii_path)
        roi_nii = nib.load(self.pytheratwin_roi_seg_path)

        # Convert to SIMIND grid (z, y, x) with optional resampling.
        # CT uses linear interpolation; seg uses nearest-neighbour.
        native_zooms_mm = tuple(float(v) for v in ct_nii.header.get_zooms()[:3])  # (x, y, z)
        if self.xyz_spacing_mm is not None:
            print(
                f"[SimindPreprocessor] Native CT spacing (x,y,z): "
                f"({native_zooms_mm[0]:.3f}, {native_zooms_mm[1]:.3f}, {native_zooms_mm[2]:.3f}) mm  →  "
                f"target spacing: {tuple(self.xyz_spacing_mm)} mm"
            )
        ct_arr, scale_x, scale_y, scale_z = self._to_simind_grid(ct_nii, xyz_spacing_mm=self.xyz_spacing_mm, zoom_order=1)
        roi_arr_full, _, _, _ = self._to_simind_grid(roi_nii, xyz_spacing_mm=self.xyz_spacing_mm, zoom_order=0)
        roi_arr_full = roi_arr_full.astype(np.int16)

        body_label = self.pytheratwin_name2id.get("remaining_body")
        if body_label is None:
            raise ValueError("PyTheraTwin label map does not contain a 'remaining_body' label.")
        if not np.any(roi_arr_full != 0):
            raise ValueError("Unified PyTheraTwin segmentation is empty (no labelled voxels).")

        # Body mask: all non-zero voxels in the unified seg (patient boundary).
        body_mask = (roi_arr_full != 0).astype(np.float32)

        roi_arr = filter_roi_seg_to_subset(roi_arr_full, self.roi_subset, self.pytheratwin_name2id)
        # Mask ROI labels to body to prevent out-of-body artifacts.
        roi_body_arr = (roi_arr * body_mask).astype(np.int16)

        masks = build_label_masks(roi_body_arr)
        id_to_name = {v: k for k, v in self.pytheratwin_name2id.items()}
        class_seg = build_class_map(roi_body_arr, id_to_name)

        # Spacing: original NIfTI zooms (x, y, z) in mm, each divided by its own
        # per-axis scale factor so the physical voxel size is exact for every axis.
        zooms_nii_mm = np.array(ct_nii.header.get_zooms()[:3], dtype=float)  # (x, y, z)
        zooms_mm = np.array([
            zooms_nii_mm[2] / scale_z,  # z-axis → arr index 0
            zooms_nii_mm[1] / scale_y,  # y-axis → arr index 1
            zooms_nii_mm[0] / scale_x,  # x-axis → arr index 2
        ])
        arr_px_spacing_cm = tuple(float(x) * 0.1 for x in zooms_mm)

        # HU->mu conversion uses mean in-plane (y, x) spacing.
        pixel_size_cm = (arr_px_spacing_cm[1] + arr_px_spacing_cm[2]) / 2.0

        atn_av_path = self._write_attenuation_bin(
            ct_arr,
            body_mask,
            pixel_size_cm=pixel_size_cm,
            filename=f"{self.prefix}_atn_av.bin",
        )

        # Write auxiliary SIMIND binary files.
        roi_arr.astype(np.float32).tofile(os.path.join(self.output_dir, f"{self.prefix}_roi_seg.bin"))
        body_mask.astype(np.float32).tofile(os.path.join(self.output_dir, f"{self.prefix}_body_seg.bin"))
        roi_body_arr.astype(np.float32).tofile(os.path.join(self.output_dir, f"{self.prefix}_roi_body_seg.bin"))

        binary_roi_act_map_paths = self._write_binary_roi_maps(roi_body_arr, class_seg, arr_px_spacing_cm)

        # Save cache so future reruns can skip reprocessing.
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"shape": list(ct_arr.shape), "zooms_mm": [x * 10.0 for x in arr_px_spacing_cm]}, f)

        return {
            "body_seg_arr": body_mask,
            "roi_body_seg_arr": roi_body_arr,
            "masks": masks,
            "class_seg": class_seg,
            "atn_av_path": atn_av_path,
            "binary_roi_act_map_paths": binary_roi_act_map_paths,
            "arr_px_spacing_cm": arr_px_spacing_cm,
            "arr_shape_new": ct_arr.shape,
        }


class SimindSimulationStage:
    """
    Run SIMIND simulations (per-organ, parallel cores) and save per-organ projection totals.

    Parameters
    ----------
    context : Context-like
        Pipeline context containing config and phase-1 outputs.
"""

    def __init__(self, context: Any) -> None:
        context.require(
            "subdir_paths",
            "config",
            "ct_nii_path",
            "pytheratwin_roi_seg_path",
            "output_folder_path",
            "ct_input_identity",
        )
        self.context = context
        self.config: Dict[str, Any] = context.config
        self.ct_input_identity: Dict[str, Any] = context.ct_input_identity

        # src/ directory: one level above this stage file (src/stages/ -> src/).
        self.repo_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.phase_output_dir: str = context.subdir_paths["phase_2"]
        self.stage_cfg: Dict[str, Any] = context.config["phase_2"]["simind_stage"]
        self.output_dir: str = self.phase_output_dir
        self.stage_output_dir: str = os.path.join(
            self.phase_output_dir,
            self.stage_cfg.get("sub_dir_name", "simind_simulation"),
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.stage_output_dir, exist_ok=True)

        self.work_dir: str = os.path.join(self.stage_output_dir, "work_dir")
        os.makedirs(self.work_dir, exist_ok=True)

        # Preprocessing output dir (inside stage output) 
        self.preprocess_dir: str = os.path.join(self.stage_output_dir, "preprocess") 
        os.makedirs(self.preprocess_dir, exist_ok=True) 

        # Header copy dir (survives PRODUCTION cleanup) 
        self.header_dir: str = os.path.join(self.stage_output_dir, "headers") 
        os.makedirs(self.header_dir, exist_ok=True) 

        self.metadata_path: str = stage_metadata_path(context.output_folder_path, "simind_simulation_stage")
        self.calibration_path: str = os.path.join(self.output_dir, "calib.res")

        self.prefix: str = self.stage_cfg["file_prefix"]
        self.mode: str = self.context.mode
        self.debug: bool = self.mode == "DEBUG"                                        

        # SIMIND acquisition parameters from config
        self.collimator: str = self.stage_cfg["Collimator"]
        self.isotope: str = self.stage_cfg["isotope"]

        # Load attenuation coefficients for this isotope from isotope_config.json
        _iso_cfg = load_isotope_config()
        _mu_data = _iso_cfg["mu_table_cm_inv"].get(self.isotope, _iso_cfg["mu_table_cm_inv"]["lu177"])
        self._mu_water: float = _mu_data["water"]
        self._mu_bone: float = _mu_data["bone"]

        self.num_projections: int = self.stage_cfg["NumProjections"]
        self.detector_distance: float = self.stage_cfg["DetectorDistance"]
        self.output_img_size: int = self.stage_cfg["OutputImgSize"]
        self.output_pixel_width: float = self.stage_cfg["OutputPixelWidth"]
        self.output_slice_width: float = self.stage_cfg["OutputSliceWidth"]
        self.num_photons: float = self.stage_cfg["NumPhotons"]
        self.simind_dir: str = self.stage_cfg["SIMINDDirectory"]
        self.energy_window_width: float = self.stage_cfg["EnergyWindowWidth"]
        self.detector_width: float = self.stage_cfg["DetectorWidth"]
        self.detector_length: float = self.stage_cfg["DetectorLength"]

        # Preprocessing parameters — target voxel spacing in mm [x, y, z], or null for native CT
        self.xyz_spacing_mm = self.stage_cfg.get("xyz_spacing_mm")

        # SIMIND ROI subset (independent from phase_1 roi_subset) 
        using_config_roi_subset = self.stage_cfg.get("roi_subset") is not None
        simind_roi_subset = self.stage_cfg.get("roi_subset")                           
        if simind_roi_subset is None:                                                  
            simind_roi_subset = getattr(context, "downstream_roi_subset", [])          
        if isinstance(simind_roi_subset, str):                                         
            simind_roi_subset = [simind_roi_subset]                                    
        self.simind_roi_subset: List[str] = [str(r).strip() for r in simind_roi_subset if str(r).strip()] 

        lesion_specs = (
            self.context.config.get("phase_1", {})
            .get("synthetic_lesions_stage", {})
            .get("specs")
        )
        lesion_hosts = [str(r).strip() for r in lesion_specs] if isinstance(lesion_specs, dict) else []
        picked_lesion_hosts = [r for r in lesion_hosts if r in self.simind_roi_subset]
        missing_lesion_hosts = [r for r in lesion_hosts if r not in self.simind_roi_subset]
        if using_config_roi_subset and "synthetic_lesion" in self.simind_roi_subset:
            raise ValueError(
                "SIMIND roi_subset contains internal ROI 'synthetic_lesion'. "
                "Select lesion host ROIs instead; "
                "the synthetic_lesion source is added automatically."
            )
        if picked_lesion_hosts and missing_lesion_hosts:
            raise ValueError(
                "SIMIND roi_subset selects some synthetic-lesion host ROIs "
                f"({picked_lesion_hosts}) but not all ({missing_lesion_hosts}). "
                "Include all lesion host ROIs or none."
            )
        if picked_lesion_hosts and "synthetic_lesion" not in self.simind_roi_subset:
            self.simind_roi_subset.append("synthetic_lesion")

        # Validate SIMIND ROI subset against phase_1 segmented ROIs
        phase1_rois = set(getattr(context, "downstream_roi_subset", []) or [])
        _internal = {"remaining_body", "synthetic_lesion"}
        invalid_rois = [r for r in self.simind_roi_subset if r not in phase1_rois and r not in _internal]
        if invalid_rois:
            raise ValueError(
                f"SIMIND roi_subset contains ROIs not segmented in Phase 1: {invalid_rois}. "
                f"Available: {sorted(phase1_rois)}"
            )                                                                          

        # Load PyTheraTwin label map
        self.ts_map_path: str = get_label_map_path()
        self.pytheratwin_name2id: Dict[str, int] = load_pytheratwin_label_map(self.ts_map_path)

        # CPU count: 0 or invalid -> use all available cores
        _cfg_num_cpu = self.stage_cfg.get("num_cpu", 1)
        max_cores = os.cpu_count() or 1
        if isinstance(_cfg_num_cpu, bool) or not isinstance(_cfg_num_cpu, int) or _cfg_num_cpu < 0:
            self.num_cpu = max_cores
        elif _cfg_num_cpu == 0:
            self.num_cpu = max_cores
        else:
            self.num_cpu = min(_cfg_num_cpu, max_cores)

        # SIMIND executable (supports .exe suffix on Windows)
        simind_exe = os.path.join(self.simind_dir, "simind")
        if not os.path.exists(simind_exe) and os.path.exists(simind_exe + ".exe"):
            simind_exe = simind_exe + ".exe"
        self.simind_exe: str = simind_exe

        if not os.path.exists(self.simind_exe):
            raise FileNotFoundError(f"SIMIND executable not found: {self.simind_exe}")

    def _rerun_config_snapshot(self) -> Dict[str, Any]:
        """Return the SIMIND settings that must match for cached outputs to remain valid."""
        return build_simind_rerun_snapshot(
            self.context.config,
            synthetic_enabled=bool(getattr(self.context, "synthetic_lesions_enabled", False)),
        )

    def _current_dependency_fingerprints(self) -> Dict[str, Any]:
        """Return fingerprints for the current stage inputs that affect SIMIND outputs."""
        deps = {
            "ct_nii": fingerprint_optional_file(self.context.ct_nii_path),
            "pytheratwin_roi_seg": fingerprint_optional_file(self.context.pytheratwin_roi_seg_path),
            "label_map_json": fingerprint_optional_file(self.ts_map_path),
            "segmentation_stage_metadata": fingerprint_optional_file(
                stage_metadata_path(self.context.output_folder_path, "segmentation_stage")
            ),
        }
        if "synthetic_lesion" in self.simind_roi_subset:
            deps["synthetic_lesions_stage_metadata"] = fingerprint_optional_file(
                stage_metadata_path(self.context.output_folder_path, "synthetic_lesions_stage")
            )
        return deps

    def _cache_marker_paths(self) -> List[str]:
        """Return representative cache files whose presence means a rerun guard is needed."""
        markers = [
            os.path.join(self.preprocess_dir, f"{self.prefix}_preprocess_meta.json"),
            os.path.join(self.preprocess_dir, f"{self.prefix}_atn_av.bin"),
            os.path.join(self.preprocess_dir, f"{self.prefix}_body_seg.bin"),
            os.path.join(self.preprocess_dir, f"{self.prefix}_roi_body_seg.bin"),
            self.calibration_path,
        ]
        markers.extend(get_summed_projection_paths(self.output_dir, self.prefix).values())

        cached_roi_list = None
        if os.path.exists(self.metadata_path):
            try:
                meta = load_json(self.metadata_path)
            except (OSError, json.JSONDecodeError):
                meta = {}
            roi_list = meta.get("roi_list")
            if isinstance(roi_list, list) and roi_list:
                cached_roi_list = [str(r) for r in roi_list if str(r).strip()]

        marker_rois = cached_roi_list or ["remaining_body", *self.simind_roi_subset]
        for organ_name in marker_rois:
            markers.extend(get_projection_paths_for_organ(self.work_dir, self.prefix, organ_name).values())
        return markers

    def _run_simind_for_organ_cores(self, organ_name: str, simind_switches: str) -> None:
        """
        Launch one SIMIND process per core in parallel for a single organ.

        Each process uses a unique /rr:<core_id> random seed so contributions are
        statistically independent. Core 0 outputs to stdout; others are silenced.
        """
        processes: List[Tuple[int, str, subprocess.Popen]] = []
        for j in range(self.num_cpu):
            cmd = (
                f"{self.simind_exe} {self.prefix} {self.prefix}_{organ_name}_{j} "
                + simind_switches
                + f"/rr:{j}"
            )
            stdout = None if j == 0 else subprocess.DEVNULL
            process = subprocess.Popen(cmd, shell=True, cwd=self.work_dir, stdout=stdout)
            processes.append((j, cmd, process))

        failures: List[str] = []
        for core_id, cmd, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append(f"core {core_id} exited {return_code}: {cmd}")

        if failures:
            joined = "\n".join(failures)
            raise RuntimeError(
                f"SIMIND failed while simulating organ '{organ_name}'.\n{joined}"
            )

    def _save_stage_metadata(
        self,
        simind_projection_paths: Dict[str, Dict[str, str]],
        roi_list: List[str],
        geometry: Dict[str, float],
        total_num_voxels: int,
        scale_factor: float,
        simind_switches_by_organ: Dict[str, str],
        organ_act_paths: Dict[str, str],
        atn_av_path: str,
        preprocess_results: Dict[str, Any],                                            
        summed_projection_paths: Dict[str, str],                                       
    ) -> None:
        """Save stage-specific metadata for debugging / provenance."""
        outputs: Dict[str, Any] = {
            "simind_projection_paths": simind_projection_paths,
            "summed_projection_paths": summed_projection_paths,
            "calibration_path": self.calibration_path,
            "header_dir": self.header_dir,
            "preprocess_dir": self.preprocess_dir,
        }
        skipped_zero_voxel_rois = [
            roi for roi in self.simind_roi_subset
            if roi not in roi_list
        ]
        extra: Dict[str, Any] = {
            "stage": "simind_simulation_stage",
            "phase_output_dir": self.phase_output_dir,
            "output_dir": self.output_dir,
            "stage_output_dir": self.stage_output_dir,
            "work_dir": self.work_dir,
            "preprocess_dir": self.preprocess_dir,                                     
            "header_dir": self.header_dir,                                             
            "file_prefix": self.prefix,
            "simind_exe": self.simind_exe,
            "simind_dir": self.simind_dir,
            "collimator": self.collimator,
            "isotope": self.isotope,
            "num_projections": self.num_projections,
            "num_photons": self.num_photons,
            "detector_distance": self.detector_distance,
            "detector_width": self.detector_width,
            "detector_length": self.detector_length,
            "output_img_size": self.output_img_size,
            "output_pixel_width": self.output_pixel_width,
            "output_slice_width": self.output_slice_width,
            "energy_window_width": self.energy_window_width,
            "num_cpu": self.num_cpu,
            "simind_roi_subset": list(self.simind_roi_subset),                         
            "requested_roi_subset": ["remaining_body", *self.simind_roi_subset],
            "skipped_zero_voxel_rois": skipped_zero_voxel_rois,
            "xyz_spacing_mm": self.xyz_spacing_mm,
            "roi_list": roi_list,
            "geometry": geometry,
            "total_num_voxels": total_num_voxels,
            "scale_factor": scale_factor,
            "simind_switches_by_organ": simind_switches_by_organ,
            "binary_roi_act_map_paths": organ_act_paths,
            "atn_av_path": atn_av_path,
            "simind_projection_paths": simind_projection_paths,
            "summed_projection_paths": summed_projection_paths,                        
            "arr_px_spacing_cm": list(preprocess_results["arr_px_spacing_cm"]),         
            "arr_shape_new": list(preprocess_results["arr_shape_new"]),                 
        }
        metadata = build_stage_metadata(
            stage_name="simind_simulation_stage",
            config_snapshot=self._rerun_config_snapshot(),
            ct_identity=self.ct_input_identity,
            upstream_fingerprints=self._current_dependency_fingerprints(),
            outputs=outputs,
            extra=extra,
        )
        write_json(self.metadata_path, metadata)

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------
    def run(self) -> Any:
        """
        Execute preprocessing + SIMIND simulation for all organs and save projection totals.

        Returns
        -------
        context : Context-like
        """
        if self.debug:
            print(f"[SimindSimulationStage] Running preprocessing with xyz_spacing_mm={self.xyz_spacing_mm}")

        assert_stage_rerun_safe(
            stage_name="simind_simulation_stage",
            metadata_path=self.metadata_path,
            required_outputs=self._cache_marker_paths(),
            current_config_snapshot=self._rerun_config_snapshot(),
            current_ct_identity=self.ct_input_identity,
            current_upstream_fingerprints=self._current_dependency_fingerprints(),
            context=self.context,
        )
        if self.context.stage_skipped:
            self.context.spect_sim_output_dir = self.output_dir
            self.context.simind_stage_output_dir = self.stage_output_dir
            self.context.simind_work_dir = self.work_dir
            self.context.simind_metadata_path = self.metadata_path
            self.context.simind_calibration_path = self.calibration_path
            self.context.simind_header_dir = self.header_dir
            # Restore path dicts and scalars from metadata.
            meta = load_json(self.metadata_path)
            cached_roi_list = [
                str(r) for r in meta.get("roi_list", [])
                if str(r).strip()
            ]
            cached_projection_paths = meta.get("simind_projection_paths", {})
            if cached_roi_list and isinstance(cached_projection_paths, dict):
                self.context.simind_projection_paths = {
                    roi: cached_projection_paths[roi]
                    for roi in cached_roi_list
                    if roi in cached_projection_paths
                }
            else:
                self.context.simind_projection_paths = cached_projection_paths
            self.context.simind_summed_projection_paths = meta.get("summed_projection_paths", {})
            self.context.simind_num_cpu = self.num_cpu
            self.context.simind_geometry = meta.get("geometry")
            self.context.simind_total_num_voxels = meta.get("total_num_voxels")
            self.context.simind_scale_factor = meta.get("scale_factor")
            self.context.simind_switches_by_organ = meta.get("simind_switches_by_organ")
            # Re-run the deterministic preprocessor to rebuild in-memory numpy arrays
            # (body/ROI segmentation arrays and masks) needed by SpectPostprocessStage.
            preprocessor = _SimindPreprocessor(
                ct_nii_path=self.context.ct_nii_path,
                pytheratwin_roi_seg_path=self.context.pytheratwin_roi_seg_path,
                pytheratwin_name2id=self.pytheratwin_name2id,
                roi_subset=cached_roi_list or self.simind_roi_subset,
                output_dir=self.preprocess_dir,
                prefix=self.prefix,
                xyz_spacing_mm=self.xyz_spacing_mm,
                mu_water=self._mu_water,
                mu_bone=self._mu_bone,
                debug=self.debug,
            )
            preprocess_results = preprocessor.run()
            if cached_roi_list:
                roi_body_arr = filter_roi_seg_to_subset(
                    preprocess_results["roi_body_seg_arr"],
                    cached_roi_list,
                    self.pytheratwin_name2id,
                )
                id_to_name = {v: k for k, v in self.pytheratwin_name2id.items()}
                preprocess_results["roi_body_seg_arr"] = roi_body_arr
                preprocess_results["masks"] = build_label_masks(roi_body_arr)
                preprocess_results["class_seg"] = build_class_map(roi_body_arr, id_to_name)
            self.context.body_seg_arr = preprocess_results["body_seg_arr"]
            self.context.roi_body_seg_arr = preprocess_results["roi_body_seg_arr"]
            self.context.mask_roi_body = preprocess_results["masks"]
            self.context.class_seg = preprocess_results["class_seg"]
            self.context.atn_av_path = preprocess_results["atn_av_path"]
            self.context.binary_roi_act_map_paths = preprocess_results["binary_roi_act_map_paths"]
            self.context.arr_px_spacing_cm = preprocess_results["arr_px_spacing_cm"]
            self.context.arr_shape_new = preprocess_results["arr_shape_new"]
            requested_roi_subset = meta.get("requested_roi_subset", ["remaining_body", *self.simind_roi_subset])
            simulated_roi_names = list((self.context.simind_projection_paths or {}).keys())
            skipped_zero_voxel_rois = meta.get("skipped_zero_voxel_rois")
            if skipped_zero_voxel_rois is None:
                skipped_zero_voxel_rois = [
                    roi for roi in requested_roi_subset
                    if roi not in simulated_roi_names
                ]
                meta["requested_roi_subset"] = requested_roi_subset
                meta["skipped_zero_voxel_rois"] = skipped_zero_voxel_rois
                write_json(self.metadata_path, meta)
            self.context.extras["simind_simulation_stage"] = {
                "requested_roi_subset": requested_roi_subset,
                "simulated_roi_names": simulated_roi_names,
                "skipped_zero_voxel_rois": skipped_zero_voxel_rois,
                "metadata_path": self.metadata_path,
            }
            return self.context

        preprocessor = _SimindPreprocessor(
            ct_nii_path=self.context.ct_nii_path,
            pytheratwin_roi_seg_path=self.context.pytheratwin_roi_seg_path,
            pytheratwin_name2id=self.pytheratwin_name2id,
            roi_subset=self.simind_roi_subset,
            output_dir=self.preprocess_dir,
            prefix=self.prefix,
            xyz_spacing_mm=self.xyz_spacing_mm,
            mu_water=self._mu_water,
            mu_bone=self._mu_bone,
            debug=self.debug,
        )
        preprocess_results = preprocessor.run()

        self.context.body_seg_arr = preprocess_results["body_seg_arr"]
        self.context.roi_body_seg_arr = preprocess_results["roi_body_seg_arr"]
        self.context.mask_roi_body = preprocess_results["masks"]
        self.context.class_seg = preprocess_results["class_seg"]
        self.context.atn_av_path = preprocess_results["atn_av_path"]
        self.context.binary_roi_act_map_paths = preprocess_results["binary_roi_act_map_paths"]
        self.context.arr_px_spacing_cm = preprocess_results["arr_px_spacing_cm"]
        self.context.arr_shape_new = preprocess_results["arr_shape_new"]

        class_seg = preprocess_results["class_seg"]
        arr_shape = preprocess_results["arr_shape_new"]
        arr_px_spacing_cm = preprocess_results["arr_px_spacing_cm"]
        organ_act_paths = preprocess_results["binary_roi_act_map_paths"]
        masks = preprocess_results["masks"]
        atn_av_path = preprocess_results["atn_av_path"]


        if not os.path.exists(atn_av_path):
            raise FileNotFoundError(f"Attenuation map not found: {atn_av_path}")

        roi_list = [roi_name for roi_name in class_seg.keys() if roi_name in organ_act_paths]
        if not roi_list:
            raise ValueError("No ROI binary source maps found for SIMIND simulation.")

        simind_projection_paths = build_projection_path_dict(self.work_dir, self.prefix, roi_list)
        geometry = derive_input_geometry(
            arr_shape,
            arr_px_spacing_cm,
            output_slice_width=self.output_slice_width,
            detector_width=self.detector_width,
            detector_length=self.detector_length,
        )

        configure_simind_environment(self.simind_dir)
        copy_simind_templates(self.repo_root, self.work_dir, self.prefix)

        # Copy attenuation map into work_dir (SIMIND resolves input paths relative to cwd).
        # Always refresh so work_dir stays in sync with current preprocessing outputs.
        atn_work_name = f"{self.prefix}_atn_av.bin"
        atn_work_path = os.path.join(self.work_dir, atn_work_name)
        shutil.copyfile(atn_av_path, atn_work_path)

        # Scale factor: photons per source voxel per core.
        total_num_voxels = int(np.sum([np.sum(mask) for mask in masks.values()]))
        if total_num_voxels <= 0:
            raise ValueError("Total source voxels is zero; cannot compute SIMIND scale factor.")

        scale_factor = float(self.num_photons / total_num_voxels / self.num_cpu)
        if scale_factor < 1:
            print(f"Not enough photons for this patient/num_cpu. Requested: {self.num_photons}")
            print(f"Increasing to: {total_num_voxels * self.num_cpu}")
            scale_factor = 1.0

        simind_switches_by_organ: Dict[str, str] = {}

        for organ_name in roi_list:
            act_path = organ_act_paths[organ_name]
            act_work_name = f"{self.prefix}_{organ_name}_act_av.bin"
            act_work_path = os.path.join(self.work_dir, act_work_name)
            if not os.path.exists(act_path):
                raise FileNotFoundError(f"Binary ROI source map not found: {act_path}")

            if not os.path.exists(act_work_path):
                shutil.copyfile(act_path, act_work_path)

            simind_switches = build_simind_switches(
                atn_name=atn_work_name,
                act_name=act_work_name,
                arr_shape=arr_shape,
                geometry=geometry,
                scale_factor=scale_factor,
                collimator=self.collimator,
                isotope=self.isotope,
                energy_window_width=self.energy_window_width,
                output_pixel_width=self.output_pixel_width,
                num_projections=self.num_projections,
                detector_distance=self.detector_distance,
                output_img_size=self.output_img_size,
            )
            simind_switches_by_organ[organ_name] = simind_switches

            if organ_totals_exist(self.work_dir, self.prefix, organ_name) and organ_headers_exist(self.work_dir, self.prefix, organ_name):
                if self.debug: 
                    print(f"[SimindSimulationStage] Organ '{organ_name}' projections already exist, skipping.") 
                continue

            if self.debug: 
                print(f"[SimindSimulationStage] Simulating organ: {organ_name}") 
            self._run_simind_for_organ_cores(organ_name, simind_switches)
            aggregate_core_projection_totals(
                self.work_dir,
                self.prefix,
                organ_name,
                num_cpu=self.num_cpu,
                mode=self.mode,
            )

        run_jaszczak_calibration(
            self.output_dir,
            self.repo_root,
            self.simind_exe,
            self.isotope,
            self.collimator,
        )
        copy_simind_headers_to_dir(self.work_dir, self.header_dir, self.prefix, roi_list[0])
        summed_projection_paths = sum_projections_across_organs(
            self.work_dir,
            self.output_dir,
            self.prefix,
            roi_list,
        )

        self._save_stage_metadata(
            simind_projection_paths=simind_projection_paths,
            roi_list=roi_list,
            geometry=geometry,
            total_num_voxels=total_num_voxels,
            scale_factor=scale_factor,
            simind_switches_by_organ=simind_switches_by_organ,
            organ_act_paths=organ_act_paths,
            atn_av_path=atn_av_path,
            preprocess_results=preprocess_results,                                     
            summed_projection_paths=summed_projection_paths,                           
        )

        self.context.spect_sim_output_dir = self.output_dir
        self.context.simind_stage_output_dir = self.stage_output_dir
        self.context.simind_work_dir = self.work_dir
        self.context.simind_metadata_path = self.metadata_path
        self.context.simind_calibration_path = self.calibration_path
        self.context.simind_projection_paths = simind_projection_paths
        self.context.simind_summed_projection_paths = summed_projection_paths           
        self.context.simind_num_cpu = self.num_cpu
        self.context.simind_geometry = geometry
        self.context.simind_total_num_voxels = total_num_voxels
        self.context.simind_scale_factor = scale_factor
        self.context.simind_switches_by_organ = simind_switches_by_organ
        self.context.simind_header_dir = self.header_dir                               
        self.context.extras["simind_simulation_stage"] = {
            "requested_roi_subset": ["remaining_body", *self.simind_roi_subset],
            "simulated_roi_names": list(simind_projection_paths.keys()),
            "skipped_zero_voxel_rois": [
                roi for roi in self.simind_roi_subset
                if roi not in roi_list
            ],
            "metadata_path": self.metadata_path,
        }

        return self.context
