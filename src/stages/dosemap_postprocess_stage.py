"""
Dosimetry Post-Processing Stage for the TDT pipeline.

This stage applies PBPK time-activity curves to OpenGATE per-ROI dose maps to produce
cumulative dose maps in absolute units (Gy) at specified timepoints.

Core responsibilities
---------------------
- Load per-ROI dose maps from OpenGATE (Gy/decay), NOT the pre-summed map.
- Load PBPK TACs from Phase 1.
- For each configured cumulative timepoint, compute per-ROI cumulated activity
  (total decays) by integrating each ROI's TAC from t=0 to that timepoint.
- Multiply each ROI's dose_per_decay by its own cumulated activity, then sum
  across ROIs to get total cumulative absorbed dose (Gy).
- Save per-timepoint dose maps as NIfTI files to the phase output directory.

Dose unit conversion
--------------------
OpenGATE outputs dose in Gy/decay per ROI source. To convert to cumulative
absolute Gy at time T:

    dose_Gy(T) = SUM over ROIs [ dose_per_decay_roi [Gy/decay] * N_decays_roi(0->T) ]

where N_decays_roi(0->T) is the total number of decays from injection to time T
for that specific ROI:

    N_decays_roi = integral(A_roi(t), 0, T)  [decays]

A_roi(t) is in MBq = 10^6 decays/second, so:

    N_decays_roi = integral(A_roi(t), 0, T) * 1e6 * 60  [decays]
    (converting MBq*min to decays)

Why per-ROI weighting matters
-----------------------------
Each ROI dose map represents the dose field produced when decays occur ONLY in
that organ. The kidney dose map and the liver dose map must be scaled by their
respective cumulated activities, not by a combined total. Multiplying the summed
dose map by the summed cumulated activity creates unphysical cross-terms (e.g.,
kidney dose scaled by liver activity).

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.subdir_paths["phase_3"] : str
- context.config["phase_3"]["dosemap_postprocess_stage"] : dict
- context.dosimetry_raw_dose_paths : dict[str, str]  {roi_name: path to per-ROI dose NIfTI}
- context.pbpk_tac_time : np.ndarray  (time in minutes)
- context.pbpk_tac_values : dict[str, np.ndarray]  {roi_name: TAC array in MBq}
- context.ct_nii_path : str

On success, this stage sets:
- context.dosemap_postprocess_output_dir : str
- context.dosemap_postprocess_paths : dict[str, str]  {timepoint_label: path}

Maintainer / contact: pyazdi@bccrc.ca
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import SimpleITK as sitk

# NumPy 2.0 renamed np.trapz -> np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


class DosemapPostprocessStage:
    """
    Apply PBPK time-activity curves to OpenGATE per-ROI dose maps.

    Produces cumulative dose maps in absolute Gy by weighting each ROI's
    dose_per_decay map by that ROI's own cumulated activity, then summing
    across all ROIs.
    """

    def __init__(self, context: Any) -> None:
        context.require(
            "subdir_paths",
            "config",
            "dosimetry_raw_dose_paths",
            "pbpk_tac_time",
            "pbpk_tac_values",
            "ct_nii_path",
        )
        self.context = context
        self.debug: bool = getattr(context, "mode", "").upper() == "DEBUG"

        self.phase_output_dir: str = context.subdir_paths["phase_3"]
        self.stage_cfg: Dict[str, Any] = context.config["phase_3"]["dosemap_postprocess_stage"]

        self.stage_output_dir: str = os.path.join(
            self.phase_output_dir,
            self.stage_cfg.get("sub_dir_name", "dosemap_postprocess"),
        )
        os.makedirs(self.stage_output_dir, exist_ok=True)
        self.work_dir: str = os.path.join(self.stage_output_dir, "work_dir")
        os.makedirs(self.work_dir, exist_ok=True)
        self.metadata_path: str = os.path.join(self.work_dir, "dosemap_postprocess_metadata.json")

        self.prefix: str = self.stage_cfg.get("file_prefix", "dosemap_postprocess")

        # Post-processing control flags
        self.apply_tac: bool = bool(self.stage_cfg.get("apply_tac", True))

        # Cumulative timepoints in minutes (integrate TAC from 0 to each)
        self.cumulative_timepoints: List[float] = list(
            self.stage_cfg["CumulativeTimepoints"]
        )

    # -----------------------------
    # helpers
    # -----------------------------

    @staticmethod
    def _save_nii(ref: sitk.Image, arr: np.ndarray, path: str) -> str:
        """Save a numpy array as NIfTI using `ref` geometry."""
        img = sitk.GetImageFromArray(np.asarray(arr, dtype=np.float32))
        img.CopyInformation(ref)
        sitk.WriteImage(img, path, imageIO="NiftiImageIO")
        return path

    @staticmethod
    def _compute_roi_cumulated_activity(
        tac_time: np.ndarray,
        tac_roi: np.ndarray,
        t_end_min: float,
    ) -> float:
        """
        Compute cumulated activity (total decays) for a single ROI from t=0 to t_end_min.

        Uses trapezoidal integration of the ROI's TAC over [0, t_end_min].

        Parameters
        ----------
        tac_time : np.ndarray  (minutes)
        tac_roi : np.ndarray  (TAC in MBq for this specific ROI)
        t_end_min : float  (minutes)

        Returns
        -------
        float  total number of decays from t=0 to t_end_min for this ROI
        """
        tac_float = tac_roi.astype(np.float64)

        # Select points from t=0 to t_end_min
        mask = (tac_time >= 0) & (tac_time <= t_end_min)
        if not np.any(mask):
            # No TAC points in range; interpolate at boundaries
            t_pts = np.array([0.0, t_end_min])
            a_pts = np.interp(t_pts, tac_time, tac_float)
            integral_mbq_min = _trapezoid(a_pts, t_pts)
        else:
            t_window = tac_time[mask].copy()
            a_window = tac_float[mask].copy()

            # Ensure t=0 boundary is included
            if t_window[0] > 0.0:
                t_window = np.insert(t_window, 0, 0.0)
                a_window = np.insert(a_window, 0, np.interp(0.0, tac_time, tac_float))

            # Ensure t_end boundary is included
            if t_window[-1] < t_end_min:
                t_window = np.append(t_window, t_end_min)
                a_window = np.append(a_window, np.interp(t_end_min, tac_time, tac_float))

            integral_mbq_min = _trapezoid(a_window, t_window)

        # Convert MBq*min -> decays: 1 MBq = 1e6 decays/s, 1 min = 60 s
        n_decays = integral_mbq_min * 1e6 * 60.0
        return float(n_decays)

    def _save_stage_metadata(
        self,
        dose_paths: Dict[str, str],
        cumulated_activities: Dict[str, Dict[str, float]],
        roi_names_used: List[str],
    ) -> None:
        """Save post-processing metadata."""
        metadata: Dict[str, Any] = {
            "stage": "dosemap_postprocess_stage",
            "stage_output_dir": self.stage_output_dir,
            "work_dir": self.work_dir,
            "apply_tac": self.apply_tac,
            "cumulative_timepoints_min": self.cumulative_timepoints,
            "dose_input_paths": dict(self.context.dosimetry_raw_dose_paths),
            "roi_names_used": roi_names_used,
            "dose_paths": dose_paths,
            "cumulated_activities_per_roi_decays": cumulated_activities,
            "dose_input_units": "Gy/decay (per ROI source)",
            "dose_output_units": "Gy (cumulative from t=0, summed across ROI sources)"
            if self.apply_tac
            else "Gy/decay",
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    # -----------------------------
    # main
    # -----------------------------
    def run(self) -> Any:
        """
        Apply per-ROI PBPK TAC weighting to per-ROI dose maps and produce
        cumulative total dose maps.

        For each timepoint T:
            dose(T) = SUM_roi [ dose_per_decay_roi * integral(TAC_roi, 0, T) ]

        Returns
        -------
        context : Context-like
        """
        raw_dose_paths = self.context.dosimetry_raw_dose_paths
        if not raw_dose_paths:
            raise ValueError("No per-ROI dose map paths found in context.dosimetry_raw_dose_paths")

        # Load reference CT for geometry
        ref_ct = sitk.ReadImage(str(self.context.ct_nii_path))

        tac_time = self.context.pbpk_tac_time
        tac_values = self.context.pbpk_tac_values

        # Load per-ROI dose maps (Gy/decay) and match with TACs
        roi_dose_maps: Dict[str, np.ndarray] = {}
        roi_names_used: List[str] = []

        for roi_name, dose_path in raw_dose_paths.items():
            if not os.path.exists(dose_path):
                raise FileNotFoundError(f"Per-ROI dose map not found: {dose_path}")

            if roi_name not in tac_values:
                if self.debug:
                    print(
                        f"[DosemapPostprocessStage] WARNING: ROI '{roi_name}' has a dose map "
                        f"but no TAC — skipping this ROI in dose weighting."
                    )
                continue

            dose_arr = sitk.GetArrayFromImage(
                sitk.ReadImage(str(dose_path))
            ).astype(np.float64)
            roi_dose_maps[roi_name] = dose_arr
            roi_names_used.append(roi_name)

            if self.debug:
                print(
                    f"[DosemapPostprocessStage] Loaded dose map for '{roi_name}': "
                    f"shape={dose_arr.shape}, max={dose_arr.max():.4e} Gy/decay"
                )

        if not roi_dose_maps:
            raise ValueError(
                "No ROI dose maps could be matched with TACs. "
                f"Dose ROIs: {list(raw_dose_paths.keys())}, TAC ROIs: {list(tac_values.keys())}"
            )

        dose_paths: Dict[str, str] = {}
        cumulated_activities: Dict[str, Dict[str, float]] = {}

        for t_min in self.cumulative_timepoints:
            t_hr = t_min / 60.0
            timepoint_label = f"{t_hr:.6f}".rstrip("0").rstrip(".")

            # Final deliverable goes to phase output dir
            output_path = os.path.join(
                self.phase_output_dir,
                f"{self.prefix}_cumulative_dose_{timepoint_label}hr.nii.gz",
            )

            # Skip if output already exists
            if os.path.exists(output_path):
                if self.debug:
                    print(
                        f"[DosemapPostprocessStage] Cumulative dose at {timepoint_label}hr "
                        f"already exists, skipping."
                    )
                dose_paths[timepoint_label] = output_path
                continue

            if self.apply_tac:
                # Accumulate dose across ROIs: SUM[ dose_roi * CA_roi ]
                total_dose = None
                ca_this_timepoint: Dict[str, float] = {}

                for roi_name, dose_per_decay in roi_dose_maps.items():
                    n_decays = self._compute_roi_cumulated_activity(
                        tac_time, tac_values[roi_name], t_min
                    )
                    ca_this_timepoint[roi_name] = n_decays

                    roi_contribution = dose_per_decay * n_decays

                    if total_dose is None:
                        total_dose = roi_contribution.copy()
                    else:
                        total_dose += roi_contribution

                    if self.debug:
                        print(
                            f"[DosemapPostprocessStage]   {roi_name}: "
                            f"CA={n_decays:.2e} decays, "
                            f"max_dose_contribution={np.max(roi_contribution):.4e} Gy"
                        )

                cumulated_activities[timepoint_label] = ca_this_timepoint

                if self.debug:
                    print(
                        f"[DosemapPostprocessStage] Cumulative dose at {timepoint_label}hr: "
                        f"max_total_dose={np.max(total_dose):.4e} Gy"
                    )
            else:
                # No TAC applied — sum the per-ROI dose_per_decay maps as-is
                total_dose = None
                for dose_per_decay in roi_dose_maps.values():
                    if total_dose is None:
                        total_dose = dose_per_decay.copy()
                    else:
                        total_dose += dose_per_decay
                cumulated_activities[timepoint_label] = {r: 0.0 for r in roi_names_used}

            self._save_nii(ref_ct, total_dose.astype(np.float32), output_path)
            dose_paths[timepoint_label] = output_path

        self._save_stage_metadata(dose_paths, cumulated_activities, roi_names_used)

        self.context.dosemap_postprocess_output_dir = self.stage_output_dir
        self.context.dosemap_postprocess_paths = dose_paths

        self.context.extras["dosemap_postprocess_stage"] = {
            "stage_output_dir": self.stage_output_dir,
            "dose_paths": dose_paths,
            "metadata_path": self.metadata_path,
        }

        return self.context