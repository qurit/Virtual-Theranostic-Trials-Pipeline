"""
SIMIND Simulation Stage for the TDT pipeline (includes preprocessing).

This stage prepares inputs for SIMIND and runs Monte Carlo SPECT projection simulations.

Preprocessing (merged from preprocessing_simind_stage.py):
- Converting CT + segmentation NIfTIs into the SIMIND grid convention (z, y, x with y-flip).
- Optionally resizing to a target in-plane dimension via isotropic zoom.
- Building ROI masks and a label->name class map from the unified TDT multilabel segmentation.
- Writing binary files used by SIMIND (attenuation map, body mask, per-ROI binary source maps).

Simulation:
- Validate required context fields (labels, spacing, files).
- Configure SIMIND environment variables (SMC_DIR, PATH).
- Copy SIMIND template files into a per-CT work directory.
- For each ROI:
  - Copy the source map binary into the work directory.
  - Run SIMIND in parallel using `num_cores` processes (each with a unique /rr: seed).
  - Aggregate per-core projection totals into a single per-organ file per energy window.
- Run a Jaszczak-based calibration (if not already present) to produce `calib.res`.
- Sum per-organ projections into total projections per energy window.
- Copy SIMIND headers to the phase output directory for downstream reconstruction.

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.config["phase_2"]["simind_stage"] : dict (including roi_subset, xy_dim)
- context.config["phase_1"]["segmentation_stage"]["label_map_path"] : str
- context.subdir_paths["phase_2"] : str
- context.mode : str  ("DEBUG" or "PRODUCTION")
- context.ct_nii_path : str
- context.tdt_roi_seg_path : str  (unified TDT ROI segmentation NIfTI)
- context.downstream_roi_subset : list[str] | None

On success, this stage sets:
- context.body_seg_arr, context.roi_body_seg_arr, context.mask_roi_body, context.class_seg
- context.atn_av_path, context.binary_roi_act_map_paths
- context.arr_px_spacing_cm, context.arr_shape_new
- context.spect_sim_output_dir, context.simind_stage_output_dir
- context.simind_work_dir, context.simind_metadata_path
- context.simind_calibration_path, context.simind_projection_paths
- context.simind_summed_projection_paths
- context.simind_num_cores, context.simind_geometry
- context.simind_total_num_voxels, context.simind_scale_factor
- context.simind_switches_by_organ
- context.simind_header_dir

Maintainer / contact: pyazdi@bccrc.ca
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
from json_minify import json_minify                                                    

# Linear attenuation coefficients at the Lu-177 photopeak (~208 keV), in 1/cm. 
MU_WATER_CM_INV: float = 0.1537                                                       
MU_BONE_CM_INV: float = 0.2234                                                        


class _SimindPreprocessor:                                                             
    """
    Internal helper: prepare CT + ROI masks for SIMIND simulation.

    Grid convention
    ---------------
    Arrays are transposed to (z, y, x) and flipped in `y` to match SIMIND's expected
    orientation before any resizing or binary export.

    Resizing
    --------
    If `xy_dim` is provided, an isotropic scale factor is derived from the in-plane
    dimension and applied to all three axes via scipy.ndimage.zoom.

    This class was previously the standalone `SimindPreprocessStage`.
    """

    def __init__(                                                                      
        self,
        ct_nii_path: str,
        tdt_roi_seg_path: str,
        tdt_name2id: Dict[str, int],
        roi_subset: Sequence[str],
        output_dir: str,
        prefix: str,
        resize: Optional[int],
        debug: bool = False,
    ) -> None:
        self.ct_nii_path = ct_nii_path                                                 
        self.tdt_roi_seg_path = tdt_roi_seg_path                                       
        self.tdt_name2id = tdt_name2id                                                 
        self.roi_subset = roi_subset                                                   
        self.output_dir = output_dir                                                   
        self.prefix = prefix                                                           
        self.resize = resize                                                           
        self.debug = debug                                                             
    @staticmethod
    def _build_class_map(seg_arr: np.ndarray, id_to_name: Dict[int, str]) -> Dict[str, int]:
        """
        Return {roi_name: label_id} for labels actually present in `seg_arr`.

        Parameters
        ----------
        seg_arr : np.ndarray
        id_to_name : dict[int, str]

        Returns
        -------
        dict[str, int]
        """
        class_map: Dict[str, int] = {}
        for lab in np.unique(seg_arr.astype(int)):
            if lab == 0:
                continue
            name = id_to_name.get(int(lab))
            if name is not None:
                class_map[name] = int(lab)
        return class_map

    @staticmethod
    def _build_label_masks(arr: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Build boolean masks for each non-zero label in a multilabel segmentation.

        Returns
        -------
        dict[int, np.ndarray]  (label_id -> bool mask)

        Raises
        ------
        ValueError  if the array contains only background (all zeros).
        """
        labels = np.unique(arr)
        labels = labels[labels != 0]
        if labels.size == 0:
            raise ValueError(
                "Segmentation has no non-zero labels. "
                "Segmentation likely failed or ROI subset is empty/mismatched."
            )
        return {int(lab): (arr == lab) for lab in labels}

    @staticmethod
    def _hu_to_mu(
        hu_arr: np.ndarray,
        pixel_size_cm: float,
        mu_water: float = MU_WATER_CM_INV,
        mu_bone: float = MU_BONE_CM_INV,
    ) -> np.ndarray:
        """
        Convert HU CT values to a linear attenuation map (mu) scaled to per-pixel units.

        Two-segment model:
          - HU <= 0  (soft tissue/air): mu = mu_water * (1 + HU/1000)
          - HU  > 0  (bone):           mu = mu_water + (HU/1000) * (mu_bone - mu_water)

        Parameters
        ----------
        hu_arr : np.ndarray
        pixel_size_cm : float  mean in-plane voxel size in cm
        mu_water : float  linear attenuation of water (1/cm) at ~208 keV
        mu_bone : float   linear attenuation of bone  (1/cm) at ~208 keV

        Returns
        -------
        np.ndarray  (float32)  mu per pixel (dimensionless)
        """
        mu_water_pixel = mu_water * pixel_size_cm
        mu_bone_pixel = mu_bone * pixel_size_cm

        mu_map = np.zeros_like(hu_arr, dtype=np.float32)
        soft = hu_arr <= 0
        bone = hu_arr > 0
        mu_map[soft] = mu_water_pixel * (1 + hu_arr[soft] / 1000.0)
        mu_map[bone] = mu_water_pixel + (hu_arr[bone] / 1000.0) * (mu_bone_pixel - mu_water_pixel)
        return mu_map

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
        mu_map = self._hu_to_mu(np.asarray(ct_arr, dtype=np.float32), pixel_size_cm)
        mu_map *= body_seg_arr  # zero outside body
        out_path = os.path.join(self.output_dir, filename)
        mu_map.tofile(out_path)
        return out_path

    def _filter_to_requested_rois(self, roi_seg_arr: np.ndarray) -> np.ndarray:
        """
        Zero out labels not in the requested ROI subset (body label is always kept).

        Raises
        ------
        ValueError  if a requested ROI name is not in the TDT label map.
        """
        requested = set(self.roi_subset)
        keep_ids: set = set()

        if "body" in self.tdt_name2id:
            keep_ids.add(self.tdt_name2id["body"])

        for name in requested:
            lab = self.tdt_name2id.get(name)
            if lab is None:
                raise ValueError(f"Requested ROI '{name}' not in TDT label map.")
            keep_ids.add(lab)

        keep_ids.discard(0)

        out = roi_seg_arr.copy()
        out[~np.isin(out, list(keep_ids))] = 0
        return out

    @staticmethod
    def _to_simind_grid(
        nii_obj: nib.Nifti1Image,
        resize: Optional[int] = None,
        transpose_tuple: Tuple[int, int, int] = (2, 1, 0),
        zoom_order: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """
        Convert a NIfTI object to SIMIND grid format with optional isotropic resizing.

        Convention: transpose to (z, y, x), then flip y.

        Parameters
        ----------
        nii_obj : nib.Nifti1Image
        resize : Optional[int | tuple[int,int,int]]
            If int: isotropic scale derived from that in-plane target (legacy behaviour).
            If (x, y, z) tuple: resize each axis independently.
            None: no resize.
        transpose_tuple : tuple[int,int,int]
        zoom_order : int  0 = nearest (seg), 1 = linear (CT)

        Returns
        -------
        (array_zyx, mean_xy_scale_factor)
        """
        arr = np.array(nii_obj.get_fdata(dtype=np.float32))
        arr = np.transpose(arr, transpose_tuple)[:, ::-1, :]  # now (z, y, x)

        scale = 1.0
        if resize is not None:
            if isinstance(resize, (list, tuple)):
                # xyz_dim = [x, y, z]  →  target shape (z, y, x)
                tx, ty, tz = int(resize[0]), int(resize[1]), int(resize[2])
                sz, sy, sx = arr.shape
                scale_z = tz / sz if sz > 0 else 1.0
                scale_y = ty / sy if sy > 0 else 1.0
                scale_x = tx / sx if sx > 0 else 1.0
                arr = zoom(arr, (scale_z, scale_y, scale_x), order=zoom_order)
                scale = (scale_x + scale_y) / 2.0  # representative in-plane scale for spacing
            else:
                # Legacy scalar: isotropic in-plane resize
                scale = resize / arr.shape[1]
                arr = zoom(arr, (scale, scale, scale), order=zoom_order)

        return arr, scale

    def _write_binary_roi_maps(self, roi_body_arr: np.ndarray, class_seg: Dict[str, int]) -> Dict[str, str]:
        """
        Write one flat float32 binary 0/1 source map per ROI for SIMIND.

        Returns
        -------
        dict[str, str]  {roi_name: path}  (excludes body; body is used only as a mask)
        """
        out_paths: Dict[str, str] = {}
        for roi_name, lab in class_seg.items():
            roi_mask = (roi_body_arr == lab).astype(np.float32)
            out_path = os.path.join(self.output_dir, f"{self.prefix}_{roi_name}_act_av.bin")
            roi_mask.tofile(out_path)
            out_paths[roi_name] = out_path
        return out_paths

    def run(self) -> Dict[str, Any]:                                                   
        """
        Execute preprocessing and return results dict instead of setting context.

        Returns
        -------
        dict with keys: body_seg_arr, roi_body_seg_arr, masks, class_seg,
                        atn_av_path, binary_roi_act_map_paths, arr_px_spacing_cm,
                        arr_shape_new
        """
        if self.ct_nii_path is None or not os.path.exists(self.ct_nii_path):
            raise FileNotFoundError(f"ct_nii_path not found: {self.ct_nii_path}")
        if self.tdt_roi_seg_path is None or not os.path.exists(self.tdt_roi_seg_path):
            raise FileNotFoundError(f"Unified TDT ROI seg not found: {self.tdt_roi_seg_path}")
        if not self.roi_subset:
            raise ValueError("No ROI subset provided for SIMIND preprocessing.")

        ct_nii = nib.load(self.ct_nii_path)
        roi_nii = nib.load(self.tdt_roi_seg_path)

        # Convert to SIMIND grid (z, y, x) with optional resize.
        # CT uses linear interpolation; seg uses nearest-neighbour.
        ct_arr, scale = self._to_simind_grid(ct_nii, resize=self.resize, zoom_order=1)
        roi_arr_full, _ = self._to_simind_grid(roi_nii, resize=self.resize, zoom_order=0)
        roi_arr_full = roi_arr_full.astype(np.int16)

        body_label = self.tdt_name2id.get("body")
        if body_label is None:
            raise ValueError("TDT label map does not contain a 'body' label.")
        if not np.any(roi_arr_full == body_label):
            raise ValueError("Unified TDT segmentation does not contain any 'body' voxels.")

        # Body mask: all non-zero voxels in the unified seg (patient boundary).
        body_mask = (roi_arr_full != 0).astype(np.float32)

        roi_arr = self._filter_to_requested_rois(roi_arr_full)
        # Mask ROI labels to body to prevent out-of-body artifacts.
        roi_body_arr = (roi_arr * body_mask).astype(np.int16)

        masks = self._build_label_masks(roi_body_arr)
        id_to_name = {v: k for k, v in self.tdt_name2id.items()}
        class_seg = self._build_class_map(roi_body_arr, id_to_name)

        # Spacing: original NIfTI zooms are in mm; after zoom, spacing shrinks by scale.
        zooms_mm = np.array(ct_nii.header.get_zooms()[:3], dtype=float) / scale
        zooms_mm = zooms_mm[[2, 1, 0]]  # reorder to (z, y, x) to match the transposed array
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

        binary_roi_act_map_paths = self._write_binary_roi_maps(roi_body_arr, class_seg)

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
    Includes preprocessing (previously separate stage).

    Parameters
    ----------
    context : Context-like
        Pipeline context containing config and phase-1 outputs.
    """

    def __init__(self, context: Any) -> None:
        context.require("subdir_paths", "config", "ct_nii_path", "tdt_roi_seg_path")   
        self.context = context
        self.config: Dict[str, Any] = context.config

        # Repository root: one level above this stage file (src/stages/ -> repo root).
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

        self.metadata_path: str = os.path.join(self.work_dir, "simind_metadata.json")
        self.calibration_path: str = os.path.join(self.output_dir, "calib.res")

        self.prefix: str = self.stage_cfg["file_prefix"]
        self.mode: str = self.context.mode
        self.debug: bool = self.mode == "DEBUG"                                        

        # SIMIND acquisition parameters from config
        self.collimator: str = self.stage_cfg["Collimator"]
        self.isotope: str = self.stage_cfg["Isotope"]
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

        # Preprocessing parameters — supports xyz_dim (list/tuple) or legacy xy_dim (int)
        xyz = self.stage_cfg.get("xyz_dim")
        self.resize = xyz if xyz is not None else self.stage_cfg.get("xy_dim")

        # SIMIND ROI subset (independent from phase_1 roi_subset) 
        simind_roi_subset = self.stage_cfg.get("roi_subset")                           
        if simind_roi_subset is None:                                                  
            simind_roi_subset = getattr(context, "downstream_roi_subset", [])          
        if isinstance(simind_roi_subset, str):                                         
            simind_roi_subset = [simind_roi_subset]                                    
        self.simind_roi_subset: List[str] = [str(r).strip() for r in simind_roi_subset if str(r).strip()] 

        # Validate SIMIND ROI subset against phase_1 segmented ROIs 
        phase1_rois = set(getattr(context, "downstream_roi_subset", []) or [])         
        invalid_rois = [r for r in self.simind_roi_subset if r not in phase1_rois and r != "body"] 
        if invalid_rois:                                                               
            raise ValueError(                                                          
                f"SIMIND roi_subset contains ROIs not segmented in Phase 1: {invalid_rois}. " 
                f"Available: {sorted(phase1_rois)}"                                    
            )                                                                          

        # Load TDT label map 
        self.ts_map_path: str = context.config["phase_1"]["segmentation_stage"]["label_map_path"] 
        if not os.path.exists(self.ts_map_path):                                       
            raise FileNotFoundError(f"Class map json not found: {self.ts_map_path}")   
        with open(self.ts_map_path, encoding="utf-8") as f:                            
            ts_map_json = json.loads(json_minify(f.read()))                             
        self.tdt_name2id: Dict[str, int] = {                                           
            name: int(lab) for lab, name in ts_map_json["TDT_Pipeline"].items()        
        }                                                                              

        # CPU count: 0 or invalid -> use all available cores
        num_cores = self.stage_cfg["NumCores"]
        max_cores = os.cpu_count() or 1
        if isinstance(num_cores, bool) or not isinstance(num_cores, int) or num_cores < 0 or num_cores > max_cores:
            self.num_cores = max_cores
        elif num_cores == 0:
            self.num_cores = max_cores
        else:
            self.num_cores = num_cores

        # SIMIND executable (supports .exe suffix on Windows)
        simind_exe = os.path.join(self.simind_dir, "simind")
        if not os.path.exists(simind_exe) and os.path.exists(simind_exe + ".exe"):
            simind_exe = simind_exe + ".exe"
        self.simind_exe: str = simind_exe

        if not os.path.exists(self.simind_exe):
            raise FileNotFoundError(f"SIMIND executable not found: {self.simind_exe}")

    # -----------------------------
    # helpers
    # -----------------------------

    def _set_simind_environment(self) -> None:
        """
        Set environment variables required by SIMIND at runtime.

        SMC_DIR must point to the `smc_dir` resource folder and end with os.sep
        (SIMIND requires this trailing separator).
        """
        smc_dir = os.path.join(self.simind_dir, "smc_dir")
        if not os.path.isdir(smc_dir):
            raise FileNotFoundError(f"SMC_DIR folder not found: {smc_dir}")
        if not smc_dir.endswith(os.sep):
            smc_dir += os.sep
        os.environ["SMC_DIR"] = smc_dir
        os.environ["PATH"] = self.simind_dir + os.pathsep + os.environ.get("PATH", "")

    def _copy_templates(self) -> None:
        """
        Copy SIMIND template files from <repo_root>/data into the work directory.

        Templates required:
        - scattwin.win  -> <work_dir>/<prefix>.win
        - smc.smc       -> <work_dir>/<prefix>.smc
        """
        shutil.copyfile(
            os.path.join(self.repo_root, "data", "scattwin.win"),
            os.path.join(self.work_dir, f"{self.prefix}.win"),
        )
        shutil.copyfile(
            os.path.join(self.repo_root, "data", "smc.smc"),
            os.path.join(self.work_dir, f"{self.prefix}.smc"),
        )

    def _get_projection_paths_for_organ(self, organ_name: str) -> Dict[str, str]:
        """Return output projection paths (w1/w2/w3) for a single organ in work_dir.""" 
        return {
            "w1": os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_tot_w1.a00"), 
            "w2": os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_tot_w2.a00"), 
            "w3": os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_tot_w3.a00"), 
        }

    def _get_summed_projection_paths(self) -> Dict[str, str]:                          
        """Return output paths for summed (across all ROIs) projection totals."""       
        return {                                                                       
            "w1": os.path.join(self.output_dir, f"{self.prefix}_tot_w1.a00"),          
            "w2": os.path.join(self.output_dir, f"{self.prefix}_tot_w2.a00"),          
            "w3": os.path.join(self.output_dir, f"{self.prefix}_tot_w3.a00"),          
        }                                                                              

    def _build_projection_path_dict(self, roi_list: List[str]) -> Dict[str, Dict[str, str]]:
        """Return output projection paths for all organs."""
        return {organ: self._get_projection_paths_for_organ(organ) for organ in roi_list}

    def _organ_totals_exist(self, organ_name: str) -> bool:
        """Return True if all three energy window total files exist for this organ."""
        return all(
            os.path.exists(p)
            for p in self._get_projection_paths_for_organ(organ_name).values()
        )

    def _organ_headers_exist(self, organ_name: str) -> bool:
        """Return True if the SIMIND header for core 0 exists (w2 photopeak window)."""
        return os.path.exists(
            os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_0_tot_w2.h00")
        )

    def _calibration_exists(self) -> bool:
        """Return True if `calib.res` exists in the output directory."""
        return os.path.exists(os.path.join(self.output_dir, "calib.res"))

    def _run_jaszczak_calibration(self) -> None:
        """
        Run Jaszczak calibration in SIMIND to produce `calib.res`.

        No-op if `calib.res` already exists.
        Requires `jaszak.smc` (note: SIMIND uses this spelling) in <repo_root>/data.
        """
        if self._calibration_exists():
            return

        jaszak_file = os.path.join(self.repo_root, "data", "jaszak.smc") 
        shutil.copyfile(jaszak_file, os.path.join(self.output_dir, "jaszak.smc"))

        cmd = (
            f"{self.simind_exe} jaszak calib"
            f"/fi:{self.isotope}"
            f"/cc:{self.collimator}"
            f"/29:1"
            f"/15:5"
            f"/fa:11"
            f"/fa:15"
            f"/fa:14"
        )
        subprocess.run(cmd, shell=True, cwd=self.output_dir, stdout=subprocess.DEVNULL)

    def _get_input_geometry(
        self,
        arr_shape: tuple,
        arr_px_spacing_cm: tuple,
    ) -> Dict[str, float]:
        """
        Derive SIMIND input/output geometry values from preprocessing outputs.

        All lengths in cm; lengths match the (z, y, x) array convention.
        """
        input_slice_width = float(arr_px_spacing_cm[0])
        input_pixel_width = float(arr_px_spacing_cm[1])
        input_half_length = float(input_slice_width * arr_shape[0] / 2.0)
        output_img_length = float(input_slice_width * arr_shape[0] / self.output_slice_width)
        detector_width_cm = float(self.detector_width)
        detector_length_cm = (
            float(arr_shape[0] * input_slice_width)
            if self.detector_length == 0
            else float(self.detector_length)
        )
        return {
            "input_slice_width": input_slice_width,
            "input_pixel_width": input_pixel_width,
            "input_half_length": input_half_length,
            "output_img_length": output_img_length,
            "detector_width_cm": detector_width_cm,
            "detector_length_cm": detector_length_cm,
        }

    def _build_simind_switches(
        self,
        atn_name: str,
        act_name: str,
        arr_shape: tuple,
        geometry: Dict[str, float],
        scale_factor: float,
    ) -> str:
        """
        Build the SIMIND command-line switch string for a single organ simulation.

        Key switches
        ------------
        /fd  attenuation map filename
        /fs  activity source map filename
        /nn  photons per voxel (scaled by num_cores)
        /cc  collimator
        /fi  isotope
        /02,/05  input half-length (z extent)
        /08,/10  detector length and width
        /14,/15  energy window lower bounds
        /20,/21  energy window widths (signed negative = % of photopeak)
        /28  output pixel width
        /29  number of projections
        /31  input pixel width
        /34  input z pixels
        /42  detector-to-patient distance
        /76  output image size
        /77  output image length
        /78,/79  output z and x pixels
        """
        return (
            f"/fd:{atn_name}"
            f"/fs:{act_name}"
            f"/in:x22,3x"
            f"/nn:{scale_factor}"
            f"/cc:{self.collimator}"
            f"/fi:{self.isotope}"
            f"/02:{geometry['input_half_length']}"
            f"/05:{geometry['input_half_length']}"
            f"/08:{geometry['detector_length_cm']:.2f}"
            f"/10:{geometry['detector_width_cm']:.2f}"
            f"/14:-7"
            f"/15:-7"
            f"/20:{-1 * self.energy_window_width}"
            f"/21:{-1 * self.energy_window_width}"
            f"/28:{self.output_pixel_width}"
            f"/29:{self.num_projections}"
            f"/31:{geometry['input_pixel_width']}"
            f"/34:{arr_shape[0]}"
            f"/42:{self.detector_distance}"
            f"/76:{self.output_img_size}"
            f"/77:{geometry['output_img_length']}"
            f"/78:{arr_shape[1]}"
            f"/79:{arr_shape[2]}"
        )

    def _run_simind_for_organ_cores(self, organ_name: str, simind_switches: str) -> None:
        """
        Launch one SIMIND process per core in parallel for a single organ.

        Each process uses a unique /rr:<core_id> random seed so contributions are
        statistically independent. Core 0 outputs to stdout; others are silenced.
        """
        processes: List[subprocess.Popen] = []
        for j in range(self.num_cores):
            cmd = (
                f"{self.simind_exe} {self.prefix} {self.prefix}_{organ_name}_{j} "
                + simind_switches
                + f"/rr:{j}"
            )
            stdout = None if j == 0 else subprocess.DEVNULL
            processes.append(subprocess.Popen(cmd, shell=True, cwd=self.work_dir, stdout=stdout))

        for p in processes:
            p.wait()

    def _aggregate_core_totals_for_organ(self, organ_name: str) -> None:
        """
        Average projection totals across `num_cores` SIMIND runs and write to work_dir.

        Reads per-core files:  <work_dir>/<prefix>_<organ>_<j>_tot_w{1,2,3}.a00
        Writes averaged totals: <work_dir>/<prefix>_<organ>_tot_w{1,2,3}.a00

        Units after averaging: counts/MB/s (SIMIND convention).
        In PRODUCTION mode, per-core files are deleted to save disk space.
        """
        xtot_w1 = xtot_w2 = xtot_w3 = 0.0

        for j in range(self.num_cores):
            p1 = os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_{j}_tot_w1.a00")
            p2 = os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_{j}_tot_w2.a00")
            p3 = os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_{j}_tot_w3.a00")

            xtot_w1 += np.fromfile(p1, dtype=np.float32)
            xtot_w2 += np.fromfile(p2, dtype=np.float32)
            xtot_w3 += np.fromfile(p3, dtype=np.float32)

            if self.mode == "PRODUCTION":
                for p in (p1, p2, p3):
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

        # Average across cores
        xtot_w1 /= self.num_cores
        xtot_w2 /= self.num_cores
        xtot_w3 /= self.num_cores

        organ_paths = self._get_projection_paths_for_organ(organ_name)
        np.asarray(xtot_w1, dtype=np.float32).tofile(organ_paths["w1"])
        np.asarray(xtot_w2, dtype=np.float32).tofile(organ_paths["w2"])
        np.asarray(xtot_w3, dtype=np.float32).tofile(organ_paths["w3"])

    def _copy_headers_to_header_dir(self, roi_list: List[str]) -> None:                
        """
        Copy SIMIND header files (.h00, .cor, .hct, .ict) from work_dir to header_dir.

        These files are needed by reconstruction and must survive PRODUCTION cleanup.
        Only copies from the first ROI (all ROIs share the same geometry).
        The .ict file is the attenuation binary referenced by the .hct header —
        PyTomography's simind.get_attenuation_map() reads .hct then loads .ict
        from the same directory.
        """
        first_roi = roi_list[0]                                                        
        extensions = [                                                                 
            f"_{first_roi}_0_tot_w1.h00",                                              
            f"_{first_roi}_0_tot_w2.h00",                                              
            f"_{first_roi}_0_tot_w3.h00",                                              
            f"_{first_roi}_0.cor",                                                     
            f"_{first_roi}_0.hct",                                                     
            f"_{first_roi}_0.ict",                                                     
        ]                                                                              
        for ext in extensions:                                                         
            src = os.path.join(self.work_dir, f"{self.prefix}{ext}")                   
            dst = os.path.join(self.header_dir, f"{self.prefix}{ext}")                 
            if os.path.exists(src) and not os.path.exists(dst):                        
                shutil.copyfile(src, dst)                                               

    def _sum_projections_across_organs(self, roi_list: List[str]) -> Dict[str, str]:   
        """
        Sum per-organ projection totals into a single total per energy window.

        Writes summed projections to the phase output directory.
        Per-organ projections remain in work_dir for post-processing.
        """
        summed_paths = self._get_summed_projection_paths()                             

        # Skip if summed projections already exist 
        if all(os.path.exists(p) for p in summed_paths.values()):                      
            return summed_paths                                                        

        sum_w1 = sum_w2 = sum_w3 = None                                                
        for organ_name in roi_list:                                                    
            organ_paths = self._get_projection_paths_for_organ(organ_name)             
            w1 = np.fromfile(organ_paths["w1"], dtype=np.float32)                      
            w2 = np.fromfile(organ_paths["w2"], dtype=np.float32)                      
            w3 = np.fromfile(organ_paths["w3"], dtype=np.float32)                      
            sum_w1 = w1.copy() if sum_w1 is None else sum_w1 + w1                      
            sum_w2 = w2.copy() if sum_w2 is None else sum_w2 + w2                      
            sum_w3 = w3.copy() if sum_w3 is None else sum_w3 + w3                      

        if sum_w1 is not None:                                                         
            np.asarray(sum_w1, dtype=np.float32).tofile(summed_paths["w1"])             
            np.asarray(sum_w2, dtype=np.float32).tofile(summed_paths["w2"])             
            np.asarray(sum_w3, dtype=np.float32).tofile(summed_paths["w3"])             

        return summed_paths                                                            

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
        metadata: Dict[str, Any] = {
            "stage": "simind_stage (includes preprocessing)",                          
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
            "num_cores": self.num_cores,
            "simind_roi_subset": list(self.simind_roi_subset),                         
            "xyz_dim": self.resize,
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
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    # -----------------------------
    # main
    # -----------------------------
    def run(self) -> Any:
        """
        Execute preprocessing + SIMIND simulation for all organs and save projection totals.

        Returns
        -------
        context : Context-like
        """
        # --- Step 1: Run preprocessing (merged from SimindPreprocessStage) --- 
        if self.debug: 
            print(f"[SimindSimulationStage] Running preprocessing with xyz_dim={self.resize}")

        preprocessor = _SimindPreprocessor( 
            ct_nii_path=self.context.ct_nii_path, 
            tdt_roi_seg_path=self.context.tdt_roi_seg_path, 
            tdt_name2id=self.tdt_name2id, 
            roi_subset=self.simind_roi_subset, 
            output_dir=self.preprocess_dir, 
            prefix=self.prefix, 
            resize=self.resize, 
            debug=self.debug, 
        ) 
        preprocess_results = preprocessor.run() 

        # Set context fields from preprocessing results 
        self.context.body_seg_arr = preprocess_results["body_seg_arr"] 
        self.context.roi_body_seg_arr = preprocess_results["roi_body_seg_arr"] 
        self.context.mask_roi_body = preprocess_results["masks"] 
        self.context.class_seg = preprocess_results["class_seg"] 
        self.context.atn_av_path = preprocess_results["atn_av_path"] 
        self.context.binary_roi_act_map_paths = preprocess_results["binary_roi_act_map_paths"] 
        self.context.arr_px_spacing_cm = preprocess_results["arr_px_spacing_cm"] 
        self.context.arr_shape_new = preprocess_results["arr_shape_new"] 

        # --- Step 2: Run SIMIND simulation --- 
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

        simind_projection_paths = self._build_projection_path_dict(roi_list)
        geometry = self._get_input_geometry(arr_shape, arr_px_spacing_cm)

        self._set_simind_environment()
        self._copy_templates()

        # Copy attenuation map into work_dir (SIMIND resolves input paths relative to cwd).
        atn_work_name = f"{self.prefix}_atn_av.bin"
        atn_work_path = os.path.join(self.work_dir, atn_work_name)
        if not os.path.exists(atn_work_path):
            shutil.copyfile(atn_av_path, atn_work_path)

        # Scale factor: photons per source voxel per core.
        total_num_voxels = int(np.sum([np.sum(mask) for mask in masks.values()]))
        if total_num_voxels <= 0:
            raise ValueError("Total source voxels is zero; cannot compute SIMIND scale factor.")

        scale_factor = float(self.num_photons / total_num_voxels / self.num_cores)
        if scale_factor < 1:
            print(f"Not enough photons for this patient/num_cores. Requested: {self.num_photons}")
            print(f"Increasing to: {total_num_voxels * self.num_cores}")
            scale_factor = 1.0

        simind_switches_by_organ: Dict[str, str] = {}

        for organ_name in roi_list:
            act_path = organ_act_paths[organ_name]
            act_work_name = f"{self.prefix}_{organ_name}_act_av.bin"
            act_work_path = os.path.join(self.work_dir, act_work_name)
            if not os.path.exists(act_work_path):
                shutil.copyfile(act_path, act_work_path)

            if not os.path.exists(act_path):
                raise FileNotFoundError(f"Binary ROI source map not found: {act_path}")

            simind_switches = self._build_simind_switches(
                atn_name=atn_work_name,
                act_name=act_work_name,
                arr_shape=arr_shape,
                geometry=geometry,
                scale_factor=scale_factor,
            )
            simind_switches_by_organ[organ_name] = simind_switches

            if self._organ_totals_exist(organ_name) and self._organ_headers_exist(organ_name):
                if self.debug: 
                    print(f"[SimindSimulationStage] Organ '{organ_name}' projections already exist, skipping.") 
                continue

            if self.debug: 
                print(f"[SimindSimulationStage] Simulating organ: {organ_name}") 
            self._run_simind_for_organ_cores(organ_name, simind_switches)
            self._aggregate_core_totals_for_organ(organ_name)

        self._run_jaszczak_calibration()

        # Copy headers to survive PRODUCTION cleanup 
        self._copy_headers_to_header_dir(roi_list)                                     

        # Sum per-organ projections into total projections 
        summed_projection_paths = self._sum_projections_across_organs(roi_list)         

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
        self.context.simind_num_cores = self.num_cores
        self.context.simind_geometry = geometry
        self.context.simind_total_num_voxels = total_num_voxels
        self.context.simind_scale_factor = scale_factor
        self.context.simind_switches_by_organ = simind_switches_by_organ
        self.context.simind_header_dir = self.header_dir                               

        return self.context