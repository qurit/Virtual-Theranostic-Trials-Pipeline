"""
PBPK Stage (PSMA model via PyCNO) for the TDT pipeline.

This stage generates time-activity curves (TACs) for user-specified VOIs and converts them
into PBPK-weighted projection data for reconstruction.

Core responsibilities
---------------------
- Validate PBPK configuration inputs (VOIs, frame start times, CT input path).
- Optionally randomize kidney/salivary-gland parameters (lognormal sampling).
- Optionally extract patient height/weight from a DICOM directory.
- Run the PyCNO PSMA model to obtain TACs (time, tacs).
- For each ROI present in the unified segmentation, map ROI -> VOI and:
    - Interpolate TAC values at configured frame start times.
    - Build a per-frame activity map (MBq/mL) distributed uniformly within the ROI mask.
    - Save first-frame per-organ activity map for debugging / provenance as NIfTI.
    - Save TAC time series + sampled values to .bin files.
    - Apply TAC-derived activity to the phase-2 SIMIND projection totals to create
      PBPK-weighted frame-wise projections for reconstruction.

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.subdir_paths["phase_3"] : str
- context.ct_input_path : str  (NIfTI path or DICOM directory for height/weight extraction)
- context.config["phase_3"]["pbpk_stage"] with keys:
    - "file_prefix" or "name" : str
    - "VOIs" : list[str]  (PyCNO observables, e.g. ["Kidney","Liver","Rest",...])
    - "FrameStartTimes" : list[float]  (minutes)
    - "FrameDurations" : list[float]  (seconds)
    - "Randomization_Kidney_SG_Para" : bool
- context.roi_body_seg_arr : np.ndarray  (z, y, x int labels)
- context.mask_roi_body : dict[int, np.ndarray[bool]]  (label_id -> mask)
- context.class_seg : dict[str, int]  (roi_name -> label_id)
- context.arr_px_spacing_cm : tuple[float, float, float]  (z, y, x) in cm
- context.simind_projection_paths : dict[str, dict[str, str]]

On success, this stage sets:
- context.activity_map_sum : np.ndarray  (n_frames,) total activity [MBq]
- context.activity_organ_sum : dict[str, np.ndarray]  per-ROI activity per frame [MBq]
- context.activity_map_paths_by_organ : list[str]
- context.pbpk_tacs_by_organ : dict[str, dict[str, str]]
- context.pbpk_frame_start_times_min : np.ndarray
- context.pbpk_frame_durations_s : np.ndarray
- context.pbpk_projection_paths : dict[str, dict[str, str]]
- context.pbpk_height_m : Optional[float]
- context.pbpk_weight_kg : Optional[float]
- context.pbpk_parameters : Optional[dict[str, float]]

Maintainer / contact: pyazdi@bccrc.ca
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pycno
import pydicom
import SimpleITK as sitk
from pytomography.io.shared import get_header_value


class PbpkStage:
    """
    Run PBPK (PyCNO PSMA) and generate activity maps for SIMIND projection weighting.

    Activity maps are uniform-concentration within each ROI mask:
        concentration(t) = TAC_interp(t) / (n_voxels * voxel_volume_ml)
    Units: MBq/mL if TAC is in MBq and voxel_volume_ml in mL.
    """

    def __init__(self, context: Any) -> None:
        self.context = context

        self.phase_output_dir: str = context.subdir_paths["phase_3"]
        self.stage_cfg: Dict[str, Any] = context.config["phase_3"]["pbpk_stage"]
        self.output_dir: str = os.path.join(
            self.phase_output_dir,
            self.stage_cfg.get("sub_dir_name", "pbpk_stage"),
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.work_dir: str = os.path.join(self.output_dir, "work_dir")
        os.makedirs(self.work_dir, exist_ok=True)
        self.metadata_path: str = os.path.join(self.work_dir, "pbpk_metadata.json")

        self.ct_input_path: str = context.ct_input_path

        self.prefix: str = self.stage_cfg.get("file_prefix", self.stage_cfg.get("name", "pbpk"))
        self.vois_pbpk: List[str] = list(self.stage_cfg["VOIs"])
        self.frame_start: Sequence[float] = self.stage_cfg["FrameStartTimes"]
        self.frame_durations_s: Sequence[float] = self.stage_cfg["FrameDurations"]
        self.randomize_kidney_sg_para: bool = self.stage_cfg["Randomization_Kidney_SG_Para"]
        self.frame_stop: float = float(max(self.frame_start)) if self.frame_start else 0.0

        self.simind_cfg: Dict[str, Any] = context.config["phase_2"]["simind_stage"]
        self.simind_num_projections: int = int(self.simind_cfg["NumProjections"])
        self.simind_output_img_size: int = int(self.simind_cfg["OutputImgSize"])
        self.simind_output_pixel_width_cm: float = float(self.simind_cfg["OutputPixelWidth"])
        self.simind_output_slice_width_cm: float = float(self.simind_cfg["OutputSliceWidth"])
        self.simind_work_dir: Optional[str] = getattr(context, "simind_work_dir", None)
        self.simind_prefix: str = str(self.simind_cfg["file_prefix"])
        self.proj_dim1: Optional[int] = None
        self.proj_dim2: Optional[int] = None
        self.num_proj: Optional[int] = None

        self.roi_body_seg_arr: np.ndarray = self.context.roi_body_seg_arr
        self.mask_roi_body: Dict[int, np.ndarray] = self.context.mask_roi_body

        self.height: Optional[float] = None
        self.weight: Optional[float] = None
        self.parameters: Dict[str, float] = {}
        self.roi_to_voi_used: Dict[str, str] = {}

    # -----------------------------
    # helpers
    # -----------------------------

    def _voxel_volume_ml(self, arr_px_spacing_cm: Sequence[float]) -> float:
        """Compute voxel volume in mL from (z, y, x) spacing in cm. (cm^3 == mL)"""
        return float(np.prod(np.asarray(arr_px_spacing_cm, dtype=float)))

    def _write_activity_map_nifti(self, arr_zyx: np.ndarray, out_path: str) -> None:
        """Save a SIMIND-grid activity map as NIfTI using SimpleITK."""
        img = sitk.GetImageFromArray(np.asarray(arr_zyx, dtype=np.float32))
        spacing_mm = tuple(float(x) * 10.0 for x in self.context.arr_px_spacing_cm[::-1])
        img.SetSpacing(spacing_mm)
        sitk.WriteImage(img, out_path, True)

    def _get_metadata_from_header(
        self, photopeak_path: str, lower_path: str, upper_path: str
    ) -> Tuple[int, int, int]:
        """Parse SIMIND header files and return (proj_dim1, proj_dim2, num_proj)."""
        for p in (photopeak_path, lower_path, upper_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing SIMIND header file: {p}")
        with open(photopeak_path, "r", encoding="utf-8") as f:
            headerdata = np.array(f.readlines())
        return (
            int(get_header_value(headerdata, "matrix size [1]", int)),
            int(get_header_value(headerdata, "matrix size [2]", int)),
            int(get_header_value(headerdata, "total number of images", int)),
        )

    def _read_projection_bin(self, proj_path: str) -> np.ndarray:
        """
        Read a SIMIND .a00 projection binary and convert to PyTomography convention (n_proj, x, y).
        """
        if self.proj_dim1 is None or self.proj_dim2 is None or self.num_proj is None:
            raise AttributeError("Projection metadata has not been initialized from SIMIND headers.")

        proj_arr = np.fromfile(proj_path, dtype=np.float32)
        expected_size = int(self.num_proj * self.proj_dim1 * self.proj_dim2)
        if proj_arr.size != expected_size:
            raise ValueError(
                f"Unexpected projection file size for {proj_path}. "
                f"Got {proj_arr.size}, expected {expected_size} "
                f"for shape ({self.num_proj}, {self.proj_dim1}, {self.proj_dim2})."
            )
        proj_arr = proj_arr.reshape((self.num_proj, self.proj_dim2, self.proj_dim1))
        return np.asarray(np.transpose(proj_arr[:, ::-1, :], (0, 2, 1)), dtype=np.float32)

    def _write_projection_nifti(self, arr_zyx: np.ndarray, out_path: str) -> None:
        """Save a PBPK-weighted projection volume as NIfTI (PyTomography convention: n_proj, x, y)."""
        img = sitk.GetImageFromArray(np.asarray(arr_zyx, dtype=np.float32))
        img.SetSpacing((
            float(self.simind_output_pixel_width_cm) * 10.0,
            float(self.simind_output_pixel_width_cm) * 10.0,
            1.0,
        ))
        sitk.WriteImage(img, out_path, True)

    def _sample_lognormal_from_mean_sd(self, mean: float, sd: float) -> float:
        """
        Sample from a lognormal distribution parameterized by desired mean and SD.

        If X ~ LogNormal(mu, sigma^2):
          sigma^2 = ln(1 + (sd/mean)^2)
          mu      = ln(mean) - sigma^2/2

        Parameters
        ----------
        mean, sd : float  (both must be > 0)
        """
        mean, sd = float(mean), float(sd)
        if mean <= 0 or sd <= 0:
            raise ValueError(f"mean and sd must be > 0 (got mean={mean}, sd={sd})")
        sigma2 = np.log(1.0 + (sd * sd) / (mean * mean))
        mu = np.log(mean) - 0.5 * sigma2
        return float(np.random.lognormal(mean=mu, sigma=np.sqrt(sigma2)))

    def _extract_height_weight_from_dicom_dir(self, dicom_dir: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Try to extract patient height (m) and weight (kg) from DICOM tags.

        Tags: PatientSize (0010,1020) -> meters; PatientWeight (0010,1030) -> kg.
        Scans up to 50 files; returns (None, None) if not found or unreadable.
        """
        if not os.path.isdir(dicom_dir):
            return None, None

        candidates: List[str] = []
        for name in sorted(os.listdir(dicom_dir)):
            path = os.path.join(dicom_dir, name)
            if os.path.isfile(path):
                candidates.append(path)
            if len(candidates) >= 50:
                break

        for path in candidates:
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                height = getattr(ds, "PatientSize", None)
                weight = getattr(ds, "PatientWeight", None)

                height = float(height) if height not in (None, "", " ") else None
                weight = float(weight) if weight not in (None, "", " ") else None

                if height is not None and height <= 0:
                    height = None
                if weight is not None and weight <= 0:
                    weight = None

                if height is not None or weight is not None:
                    return height, weight
            except Exception:
                continue

        return None, None

    def _parameter_check(self) -> Dict[str, float]:
        """
        Validate PBPK config inputs and build the PyCNO parameter override dict.

        Returns
        -------
        dict[str, float]  (may be empty if no overrides)

        Raises
        ------
        AttributeError, ValueError
        """
        if not os.path.exists(self.ct_input_path):
            raise ValueError(f"CT input path does not exist: {self.ct_input_path}")
        if not isinstance(self.vois_pbpk, (list, tuple)) or len(self.vois_pbpk) == 0:
            raise ValueError("PBPK VOIs must be a non-empty list/tuple")

        frame_start = np.asarray(self.frame_start, dtype=float)
        if not isinstance(self.frame_start, (list, tuple, np.ndarray)) or len(self.frame_start) == 0:
            raise ValueError("FrameStartTimes must be a non-empty list/tuple/array")
        if not np.all(np.isfinite(frame_start)):
            raise ValueError("FrameStartTimes contain non-finite values")
        if np.any(frame_start < 0):
            raise ValueError("FrameStartTimes must be >= 0")

        frame_durations_s = np.asarray(self.frame_durations_s, dtype=float)
        if not isinstance(self.frame_durations_s, (list, tuple, np.ndarray)) or len(self.frame_durations_s) == 0:
            raise ValueError("FrameDurations must be a non-empty list/tuple/array")
        if not np.all(np.isfinite(frame_durations_s)):
            raise ValueError("FrameDurations contain non-finite values")
        if np.any(frame_durations_s <= 0):
            raise ValueError("FrameDurations must be > 0")
        if len(frame_durations_s) != len(frame_start):
            raise ValueError("FrameStartTimes and FrameDurations must have the same length")

        if not isinstance(self.randomize_kidney_sg_para, bool):
            raise ValueError("Randomization_Kidney_SG_Para must be True or False")

        parameters: Dict[str, float] = {}

        vois_set = {str(v).strip().lower() for v in (self.vois_pbpk or [])}
        has_kidney = "kidney" in vois_set
        has_sg = "sg" in vois_set or any("salivary" in v for v in vois_set)

        if self.randomize_kidney_sg_para:
            if has_sg:
                parameters["Rden_SG"] = self._sample_lognormal_from_mean_sd(60.0, 20.0)
                parameters["lambdaRel_SG"] = self._sample_lognormal_from_mean_sd(3.9e-4, 0.63e-4)
            if has_kidney:
                parameters["Rden_Kidney"] = self._sample_lognormal_from_mean_sd(30.0, 10.0)
                parameters["lambdaRel_Kidney"] = self._sample_lognormal_from_mean_sd(2.88e-4, 0.55e-4)

        self.height = None
        self.weight = None
        if os.path.isdir(self.ct_input_path):
            self.height, self.weight = self._extract_height_weight_from_dicom_dir(self.ct_input_path)

        if self.height is not None:
            parameters["bodyHeight"] = float(self.height)
        if self.weight is not None:
            parameters["bodyWeight"] = float(self.weight)

        return parameters

    def _run_psma_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate parameters, run PyCNO PSMA PBPK model, and return (time, tacs).

        Returns
        -------
        time : np.ndarray  shape (T,)
        tacs : np.ndarray  typically shape (1, T, n_vois) from PyCNO
        """
        self.parameters = self._parameter_check()

        # Store provenance on context early in case of later failure.
        self.context.pbpk_height_m = self.height
        self.context.pbpk_weight_kg = self.weight
        self.context.pbpk_parameters = dict(self.parameters) if self.parameters else None

        model = pycno.Model(
            model_name="PSMA",
            hotamount=10,
            coldamount=100,
            parameters=self.parameters,
        )

        stop = int(self.frame_stop)
        steps = int(np.ceil(stop)) if stop > 0 else 1
        time, tacs = model.simulate(stop=stop, steps=steps, observables=self.vois_pbpk)
        return np.asarray(time, dtype=float), np.asarray(tacs, dtype=float)

    def _roi_to_voi(self, roi_name: str) -> Optional[str]:
        """
        Map a TDT ROI name to its PyCNO VOI observable name.

        Returns None if there is no explicit mapping (caller falls back to "Rest").
        """
        roi_to_voi = {
            "kidney":          "Kidney",
            "body":            "Rest",
            "liver":           "Liver",
            "prostate":        "Prostate",
            "heart":           "Heart",
            "spleen":          "Spleen",
            "salivary_glands": "SG",
            "synthetic_lesion":"Tumor1",
        }
        return roi_to_voi.get(roi_name, None)

    def _extract_tac_for_voi(self, tacs: np.ndarray, voi_index: int) -> np.ndarray:
        """Extract one VOI TAC from PyCNO output (handles 2D and 3D array shapes)."""
        if tacs.ndim == 3:
            return np.asarray(tacs[0, :, voi_index], dtype=float)
        if tacs.ndim == 2:
            return np.asarray(tacs[:, voi_index], dtype=float)
        raise ValueError(f"Unexpected TAC array shape from PyCNO: {tacs.shape}")

    def _save_tac_files(
        self,
        roi_name: str,
        time: np.ndarray,
        tac_voi: np.ndarray,
        tac_interp: np.ndarray,
    ) -> Dict[str, str]:
        """
        Save TAC full-resolution and frame-sampled arrays to binary files.

        Files written (float32):
        - <prefix>_<roi>_TAC_time.bin
        - <prefix>_<roi>_TAC_values.bin
        - <prefix>_<roi>_sample_times.bin
        - <prefix>_<roi>_sample_values.bin
        """
        roi_tag = roi_name.lower()
        paths = {
            "tac_time":     os.path.join(self.work_dir, f"{self.prefix}_{roi_tag}_TAC_time.bin"),
            "tac_values":   os.path.join(self.work_dir, f"{self.prefix}_{roi_tag}_TAC_values.bin"),
            "sample_time":  os.path.join(self.work_dir, f"{self.prefix}_{roi_tag}_sample_times.bin"),
            "sample_values":os.path.join(self.work_dir, f"{self.prefix}_{roi_tag}_sample_values.bin"),
        }
        np.asarray(time, dtype=np.float32).tofile(paths["tac_time"])
        np.asarray(tac_voi, dtype=np.float32).tofile(paths["tac_values"])
        np.asarray(self.frame_start, dtype=np.float32).tofile(paths["sample_time"])
        np.asarray(tac_interp, dtype=np.float32).tofile(paths["sample_values"])
        return paths

    def _generate_time_activity_arr_roi(
        self,
        roi_name: str,
        label_value: int,
        mask_roi_body: Dict[int, np.ndarray],
        roi_body_seg_arr: np.ndarray,
        voxel_vol_ml: float,
        time: np.ndarray,
        tacs: np.ndarray,
        saved_tacs: Dict[str, Dict[str, str]],
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Build per-frame activity map for a single ROI and save the first frame for provenance.

        Parameters
        ----------
        roi_name : str
        label_value : int  label ID in roi_body_seg_arr
        mask_roi_body : dict[int, np.ndarray]
        roi_body_seg_arr : np.ndarray  (z, y, x)
        voxel_vol_ml : float
        time : np.ndarray  (T,)
        tacs : np.ndarray  PyCNO output
        saved_tacs : dict  cache to avoid duplicate file writes for shared VOIs

        Returns
        -------
        activity_map_organ : np.ndarray  (n_frames, z, y, x) float32
        organ_sum : np.ndarray  (n_frames,) [MBq]
        organ_map_path : str  path to the first-frame NIfTI
        """
        voi_name = self._roi_to_voi(roi_name)

        # Fall back to "Rest" if no direct mapping or VOI not present in the model.
        if voi_name is None or voi_name not in self.vois_pbpk:
            if "Rest" in self.vois_pbpk:
                voi_name = "Rest"
            else:
                raise ValueError(
                    f"No VOI mapping for ROI '{roi_name}' and no 'Rest' VOI in PBPK model "
                    f"(available: {sorted(self.vois_pbpk)})."
                )

        self.roi_to_voi_used[roi_name] = voi_name
        voi_index = self.vois_pbpk.index(voi_name)

        mask = mask_roi_body[label_value]
        n_vox = int(np.sum(mask))
        if n_vox == 0:
            raise AssertionError(f"Mask for '{voi_name}' (roi='{roi_name}') is empty.")

        tac_voi = self._extract_tac_for_voi(tacs, voi_index)
        tac_interp = np.interp(np.asarray(self.frame_start, dtype=float), time, tac_voi)

        activity_map_organ = np.zeros((len(self.frame_start), *roi_body_seg_arr.shape), dtype=np.float32)
        activity_map_organ[:, mask] = tac_interp[:, None] / (n_vox * voxel_vol_ml)

        organ_map_path = os.path.join(self.work_dir, f"{self.prefix}_{roi_name}_act_av.nii.gz")
        self._write_activity_map_nifti(activity_map_organ[0], organ_map_path)

        organ_sum = np.sum(activity_map_organ, axis=(1, 2, 3)) * voxel_vol_ml

        if roi_name not in saved_tacs:
            saved_tacs[roi_name] = self._save_tac_files(roi_name, time, tac_voi, tac_interp)

        return activity_map_organ, organ_sum, organ_map_path

    def _build_pbpk_projection_paths(self) -> Dict[str, Dict[str, str]]:
        """Build the expected output paths for PBPK-weighted frame projections."""
        projection_paths: Dict[str, Dict[str, str]] = {}
        for frame in self.frame_start:
            # Label in hours (max 6 decimal places, trailing zeros stripped)
            frame_label = f"{float(frame)/60.0:.6f}".rstrip("0").rstrip(".")
            projection_paths[frame_label] = {
                "w1": os.path.join(self.output_dir, f"{self.prefix}_{frame_label}_tot_w1.nii.gz"),
                "w2": os.path.join(self.output_dir, f"{self.prefix}_{frame_label}_tot_w2.nii.gz"),
                "w3": os.path.join(self.output_dir, f"{self.prefix}_{frame_label}_tot_w3.nii.gz"),
            }
        return projection_paths

    def _save_stage_metadata(
        self,
        saved_tacs: Dict[str, Dict[str, str]],
        organ_paths: List[str],
        pbpk_projection_paths: Dict[str, Dict[str, str]],
        activity_organ_sum: Dict[str, np.ndarray],
        activity_map_sum: np.ndarray,
    ) -> None:
        """Save PBPK stage metadata for debugging / provenance."""
        metadata: Dict[str, Any] = {
            "stage": "pbpk_stage",
            "phase_output_dir": self.phase_output_dir,
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "file_prefix": self.prefix,
            "ct_input_path": self.ct_input_path,
            "vois_pbpk": list(self.vois_pbpk),
            "frame_start_times_min": np.asarray(self.frame_start, dtype=float).tolist(),
            "frame_durations_s": np.asarray(self.frame_durations_s, dtype=float).tolist(),
            "frame_labels_hours": [
                f"{float(t)/60.0:.6f}".rstrip("0").rstrip(".") for t in self.frame_start
            ],
            "randomize_kidney_sg_para": bool(self.randomize_kidney_sg_para),
            "pbpk_height_m": None if self.height is None else float(self.height),
            "pbpk_weight_kg": None if self.weight is None else float(self.weight),
            "pbpk_parameters": {k: float(v) for k, v in self.parameters.items()},
            "roi_to_voi_used": dict(self.roi_to_voi_used),
            "saved_tacs": saved_tacs,
            "activity_map_paths_by_organ": organ_paths,
            "activity_organ_sum_mbq": {
                k: np.asarray(v, dtype=float).tolist() for k, v in activity_organ_sum.items()
            },
            "activity_map_sum_mbq": np.asarray(activity_map_sum, dtype=float).tolist(),
            "simind_projection_paths": self.context.simind_projection_paths,
            "pbpk_projection_paths": pbpk_projection_paths,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    # -----------------------------
    # main
    # -----------------------------
    def run(self) -> Any:
        """
        Execute PBPK simulation and write PBPK-weighted projections.

        Returns
        -------
        context : Context-like
        """
        for k in ("roi_body_seg_arr", "mask_roi_body", "class_seg", "arr_px_spacing_cm",
                  "simind_projection_paths", "simind_work_dir"):
            if getattr(self.context, k, None) is None:
                raise AttributeError(f"Context missing required field: {k}")

        # Remove "background" label if present (not a real ROI).  
        class_seg = {k: v for k, v in self.context.class_seg.items() if k != "background"}
        voxel_vol_ml = self._voxel_volume_ml(self.context.arr_px_spacing_cm)

        roi_list = list(self.context.simind_projection_paths.keys())
        if not roi_list:
            raise ValueError("No phase-2 SIMIND projection paths found in context.")

        # Locate SIMIND headers from work_dir (phase 2) to get projection dimensions.
        photopeak_h = os.path.join(self.simind_work_dir, f"{self.simind_prefix}_{roi_list[0]}_0_tot_w2.h00")
        lower_h = os.path.join(self.simind_work_dir, f"{self.simind_prefix}_{roi_list[0]}_0_tot_w1.h00")
        upper_h = os.path.join(self.simind_work_dir, f"{self.simind_prefix}_{roi_list[0]}_0_tot_w3.h00")
        self.proj_dim1, self.proj_dim2, self.num_proj = self._get_metadata_from_header(
            photopeak_h, lower_h, upper_h
        )

        time, tacs = self._run_psma_model()

        n_frames = len(self.frame_start)
        activity_map = np.zeros((n_frames, *self.roi_body_seg_arr.shape), dtype=np.float32)

        activity_organ_sum: Dict[str, np.ndarray] = {}
        organ_paths: List[str] = []
        saved_tacs: Dict[str, Dict[str, str]] = {}
        pbpk_projection_paths = self._build_pbpk_projection_paths()
        frame_projection_sums: Dict[str, Dict[str, Optional[np.ndarray]]] = {
            label: {"w1": None, "w2": None, "w3": None} for label in pbpk_projection_paths
        }

        for roi_name, label_value in class_seg.items():
            activity_map_organ, organ_sum, organ_map_path = self._generate_time_activity_arr_roi(
                roi_name=roi_name,
                label_value=int(label_value),
                mask_roi_body=self.mask_roi_body,
                roi_body_seg_arr=self.roi_body_seg_arr,
                voxel_vol_ml=voxel_vol_ml,
                time=time,
                tacs=tacs,
                saved_tacs=saved_tacs,
            )

            organ_paths.append(organ_map_path)
            activity_organ_sum[roi_name] = organ_sum

            mask = self.mask_roi_body[int(label_value)]
            activity_map[:, mask] = activity_map_organ[:, mask]

            if roi_name not in self.context.simind_projection_paths:
                raise KeyError(f"Missing phase-2 SIMIND projections for ROI: {roi_name}")

            roi_projection_paths = self.context.simind_projection_paths[roi_name]
            for i in range(n_frames):
                frame_label = f"{float(self.frame_start[i])/60.0:.6f}".rstrip("0").rstrip(".")
                # frame_scale: counts = (counts/MBq/s) * MBq * s
                frame_scale = float(organ_sum[i] * float(self.frame_durations_s[i]))
                for window_key in ("w1", "w2", "w3"):
                    proj_path = roi_projection_paths[window_key]
                    if not os.path.exists(proj_path):
                        raise FileNotFoundError(f"SIMIND projection not found: {proj_path}")
                    proj_arr = self._read_projection_bin(proj_path)
                    weighted = np.asarray(proj_arr * frame_scale, dtype=np.float32)
                    if frame_projection_sums[frame_label][window_key] is None:
                        frame_projection_sums[frame_label][window_key] = weighted
                    else:
                        frame_projection_sums[frame_label][window_key] += weighted

        activity_map_sum = np.sum(activity_map, axis=(1, 2, 3)) * voxel_vol_ml

        # Save full activity maps per frame to work_dir for provenance.
        for i, frame in enumerate(activity_map):
            frame_label = f"{float(self.frame_start[i])/60.0:.6f}".rstrip("0").rstrip(".")
            self._write_activity_map_nifti(
                frame, os.path.join(self.work_dir, f"{self.prefix}_{frame_label}_act_map.nii.gz")
            )

        # Write PBPK-weighted projection NIfTIs for reconstruction.
        for frame_label, window_dict in frame_projection_sums.items():
            for window_key, proj_arr in window_dict.items():
                if proj_arr is None:
                    raise ValueError(
                        f"No PBPK-weighted projection generated for {frame_label} {window_key}"
                    )
                self._write_projection_nifti(proj_arr, pbpk_projection_paths[frame_label][window_key])

        self._save_stage_metadata(
            saved_tacs=saved_tacs,
            organ_paths=organ_paths,
            pbpk_projection_paths=pbpk_projection_paths,
            activity_organ_sum=activity_organ_sum,
            activity_map_sum=activity_map_sum,
        )

        self.context.activity_map_sum = activity_map_sum
        self.context.activity_organ_sum = activity_organ_sum
        self.context.activity_map_paths_by_organ = organ_paths
        self.context.pbpk_tacs_by_organ = saved_tacs
        self.context.pbpk_frame_start_times_min = np.asarray(self.frame_start, dtype=float)
        self.context.pbpk_frame_durations_s = np.asarray(self.frame_durations_s, dtype=float)
        self.context.pbpk_projection_paths = pbpk_projection_paths
        self.context.pbpk_height_m = self.height
        self.context.pbpk_weight_kg = self.weight
        self.context.pbpk_parameters = dict(self.parameters)

        return self.context