"""
Dosimetry post-processing for the PyTheraTwin pipeline.

This stage applies PBPK time-activity curves to OpenGATE per-ROI dose maps to produce
a single total absorbed dose map in absolute units (Gy).

Core responsibilities
---------------------
- Load per-ROI dose maps from OpenGATE (Gy/decay).
- Load PBPK TACs from Phase 1.
- Fold absent dedicated non-Rest PBPK organ TACs into remaining_body so excluded
  organ activity is correctly accounted for in the dose weighting.
- Integrate each ROI's TAC from t=0 to the end of the TAC (which covers 10x the
  isotope half-life, capturing >99.9% of all decays) to get total cumulated activity.
- Multiply each ROI's dose_per_decay by its own cumulated activity, then sum
  across ROIs to get the total absorbed dose map in Gy.
- Save one dose map NIfTI to the phase output directory.

Dose unit conversion
--------------------
OpenGATE outputs dose in Gy/decay per ROI source. The final dose map is:

    dose_Gy = SUM over ROIs [ dose_per_decay_roi [Gy/decay] * N_decays_roi ]

where N_decays_roi is the total number of decays for that ROI over the full
treatment period:

    N_decays_roi = integral(A_roi(t), 0, T_end)  [decays]

A_roi(t) is in MBq = 10^6 decays/second, so:

    N_decays_roi = integral(A_roi(t), 0, T_end) * 1e6 * 60  [decays]
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
- context.dosemap_postprocess_dose_path : str  (path to single total dose map)
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import SimpleITK as sitk

from src.io.rerun_guard import (
    assert_stage_rerun_safe,
    build_stage_metadata,
    build_dosemap_rerun_snapshot,
    fingerprint_optional_file,
    stage_metadata_path,
    write_json,
)
from src.utils.body_mask_utils import body_mask_from_roi_mask_paths
from src.utils.nifti_utils import save_nii_sitk
from src.utils.tac_utils import compute_roi_cumulated_activity, get_pbpk_voi_name_for_roi


class DosemapPostprocessStage:
    """
    Apply PBPK time-activity curves to OpenGATE per-ROI dose maps.

    Produces a single total absorbed dose map in Gy by integrating each ROI's
    TAC over the full treatment period, multiplying by dose_per_decay, and
    summing across all ROIs.
    """

    def __init__(self, context: Any) -> None:
        context.require(
            "subdir_paths",
            "config",
            "dosimetry_raw_dose_paths",
            "pbpk_tac_time",
            "pbpk_tac_values",
            "ct_nii_path",
            "output_folder_path",
            "ct_input_identity",
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
        self.metadata_path: str = stage_metadata_path(context.output_folder_path, "dosemap_postprocess_stage")
        self.ct_input_identity: Dict[str, Any] = context.ct_input_identity

        self.prefix: str = self.stage_cfg.get("file_prefix", "dosemap_postprocess")

        # apply_tac is the only post-processing applied to dose maps. Without it,
        # OpenGATE outputs (Gy/decay) cannot be converted to absolute dose (Gy).
        if not self.stage_cfg.get("apply_tac", True):
            raise ValueError(
                "'phase_3.dosemap_postprocess_stage.apply_tac' must be true. "
                "TAC weighting is required to convert OpenGATE Gy/decay outputs to "
                "absolute dose (Gy). There is no other post-processing path for dose maps."
            )

    def _rerun_config_snapshot(self) -> Dict[str, Any]:
        """Return the dose post-processing settings that must match for reruns."""
        return build_dosemap_rerun_snapshot(self.context.config)

    def _current_dependency_fingerprints(self) -> Dict[str, Any]:
        """Return fingerprints for the upstream artifacts that feed this stage."""
        return {
            "opengate_stage_metadata": fingerprint_optional_file(self.context.dosimetry_metadata_path),
            "pbpk_tac_json": fingerprint_optional_file(self.context.pbpk_tac_json_path),
            "pbpk_tac_npz": fingerprint_optional_file(self.context.pbpk_tac_npz_path),
        }

    def _cache_marker_paths(self) -> List[str]:
        """Return representative output files whose presence means rerun metadata must match."""
        markers = [
            os.path.join(self.phase_output_dir, f"{self.prefix}_total_dose.nii.gz"),
        ]
        for roi_name in self.context.dosimetry_raw_dose_paths.keys():
            markers.append(os.path.join(self.work_dir, f"{self.prefix}_{roi_name}_dose_contribution.nii.gz"))
        return markers

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _same_image_grid(a: sitk.Image, b: sitk.Image, *, atol: float = 1e-5) -> bool:
        """Return True when two images share the same voxel grid geometry."""
        return (
            tuple(a.GetSize()) == tuple(b.GetSize())
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
        )

    def _apply_body_mask(self, dose_arr: np.ndarray) -> np.ndarray:
        """Zero out dose voxels outside the OpenGATE simulation body outline."""
        mask_paths = getattr(self.context, "dosimetry_mask_paths", None)
        mask = body_mask_from_roi_mask_paths(mask_paths, dose_arr.shape, debug=self.debug, stage_tag="DosemapPostprocessStage")
        if mask is None:
            return dose_arr
        return dose_arr * mask

    def _load_metadata_extra(self) -> Dict[str, Any]:
        """Load previously saved metadata extras when this stage is skipped."""
        if not os.path.exists(self.metadata_path):
            return {}
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            extra = meta.get("extra", {})
            return extra if isinstance(extra, dict) else {}
        except Exception:
            return {}

    def _save_stage_metadata(
        self,
        dose_path: str,
        cumulated_activities: Dict[str, float],
        roi_names_used: List[str],
        integration_end_min: float,
        dose_ref_img: sitk.Image,
        folded_remaining_body_decays: Optional[Dict[str, float]] = None,
        rest_owned_skipped_rois: Optional[List[str]] = None,
    ) -> None:
        """Save post-processing metadata."""
        roi_contribution_paths = {
            roi: os.path.join(self.work_dir, f"{self.prefix}_{roi}_dose_contribution.nii.gz")
            for roi in roi_names_used
        }
        outputs: Dict[str, Any] = {
            "dose_output_path": dose_path,
            "roi_contribution_paths": roi_contribution_paths,
        }
        extra: Dict[str, Any] = {
            "stage": "dosemap_postprocess_stage",
            "stage_output_dir": self.stage_output_dir,
            "work_dir": self.work_dir,
            "dose_input_paths": dict(self.context.dosimetry_raw_dose_paths),
            "roi_names_used": roi_names_used,
            "dose_output_path": dose_path,
            "roi_contribution_paths": roi_contribution_paths,
            "integration_range_min": [0.0, integration_end_min],
            "integration_range_hr": [0.0, integration_end_min / 60.0],
            "integration_range_days": [0.0, integration_end_min / 60.0 / 24.0],
            "cumulated_activities_per_roi_decays": cumulated_activities,
            "dose_input_units": "Gy/decay (per ROI source)",
            "dose_output_units": "Gy (total absorbed dose, summed across ROI sources)",
            "dose_output_grid": {
                "size_xyz": list(dose_ref_img.GetSize()),
                "spacing_xyz_mm": [float(v) for v in dose_ref_img.GetSpacing()],
                "origin_xyz_mm": [float(v) for v in dose_ref_img.GetOrigin()],
            },
        }
        if folded_remaining_body_decays:
            extra["remaining_body_folded_from_decays"] = folded_remaining_body_decays
            extra["remaining_body_folded_total_decays"] = float(sum(folded_remaining_body_decays.values()))
            extra["remaining_body_folded_rois"] = list(folded_remaining_body_decays.keys())
        if rest_owned_skipped_rois is not None:
            extra["rest_owned_skipped_rois"] = list(rest_owned_skipped_rois)
        metadata = build_stage_metadata(
            stage_name="dosemap_postprocess_stage",
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
        Apply per-ROI PBPK TAC weighting to per-ROI dose maps and produce
        a single total absorbed dose map.

        The TAC is integrated from t=0 to the end of the PBPK simulation
        (10x isotope half-life), capturing >99.9% of all decays:

            dose = SUM_roi [ dose_per_decay_roi * integral(TAC_roi, 0, T_end) ]

        Returns
        -------
        context : Context-like
        """
        raw_dose_paths = self.context.dosimetry_raw_dose_paths
        if not raw_dose_paths:
            raise ValueError("No per-ROI dose map paths found in context.dosimetry_raw_dose_paths")

        tac_time = self.context.pbpk_tac_time
        # Work on a local copy so we can safely patch remaining_body.
        tac_values: Dict[str, np.ndarray] = dict(self.context.pbpk_tac_values)
        integration_end_min = float(tac_time[-1])
        folded_remaining_body_decays: Dict[str, float] = {}
        rest_owned_skipped_rois: List[str] = []

        # Aggregate excluded dedicated-organ TACs into remaining_body.
        # OpenGATE's roi_subset may be smaller than Phase 1's roi_subset.  Any ROI
        # with a dedicated non-Rest PBPK VOI that is not present in the OpenGATE
        # dose maps is geometrically absorbed into OpenGATE's remaining_body mask.
        # ROIs mapped to Rest are already represented by remaining_body itself.
        simulation_rois: set = set(raw_dose_paths.keys()) - {"remaining_body"}
        if "remaining_body" in tac_values:
            effective_rb = tac_values["remaining_body"].copy().astype(np.float64)
            for _rn, _rt in self.context.pbpk_tac_values.items():
                if _rn != "remaining_body" and _rn not in simulation_rois:
                    voi_name = get_pbpk_voi_name_for_roi(self.context, _rn)
                    if isinstance(voi_name, str) and voi_name.strip().lower() == "rest":
                        rest_owned_skipped_rois.append(_rn)
                        continue
                    effective_rb += np.asarray(_rt, dtype=np.float64)
                    folded_remaining_body_decays[_rn] = compute_roi_cumulated_activity(tac_time, _rt)
            tac_values["remaining_body"] = effective_rb.astype(np.float32)
        if self.debug and (folded_remaining_body_decays or rest_owned_skipped_rois):
            print(
                "[DosemapPostprocessStage] Remaining-body TAC folding summary: "
                f"folded={len(folded_remaining_body_decays)} ROI(s) | "
                f"skipped_rest_owned={len(rest_owned_skipped_rois)} ROI(s). "
                f"Details: {self.metadata_path}"
            )

        # Final output path
        output_path = os.path.join(
            self.phase_output_dir,
            f"{self.prefix}_total_dose.nii.gz",
        )

        assert_stage_rerun_safe(
            stage_name="dosemap_postprocess_stage",
            metadata_path=self.metadata_path,
            required_outputs=self._cache_marker_paths(),
            current_config_snapshot=self._rerun_config_snapshot(),
            current_ct_identity=self.ct_input_identity,
            current_upstream_fingerprints=self._current_dependency_fingerprints(),
            context=self.context,
        )
        if self.context.stage_skipped or os.path.exists(output_path):
            if self.debug:
                print("[DosemapPostprocessStage] Total dose map already exists, skipping.")
            metadata_extra = self._load_metadata_extra()
            self.context.dosemap_postprocess_output_dir = self.stage_output_dir
            self.context.dosemap_postprocess_dose_path = output_path
            self.context.extras["dosemap_postprocess_stage"] = {
                "stage_output_dir": self.stage_output_dir,
                "dose_path": output_path,
                "metadata_path": self.metadata_path,
                "folded_remaining_body_rois": metadata_extra.get(
                    "remaining_body_folded_rois",
                    list(folded_remaining_body_decays.keys()),
                ),
                "rest_owned_skipped_rois": metadata_extra.get(
                    "rest_owned_skipped_rois",
                    rest_owned_skipped_rois,
                ),
            }
            return self.context

        # Load per-ROI dose maps and match with TACs
        roi_dose_maps: Dict[str, np.ndarray] = {}
        roi_names_used: List[str] = []
        dose_ref_img: Optional[sitk.Image] = None

        for roi_name, dose_path in raw_dose_paths.items():
            if not os.path.exists(dose_path):
                raise FileNotFoundError(f"Per-ROI dose map not found: {dose_path}")

            if roi_name not in tac_values:
                if self.debug:
                    print(
                        f"[DosemapPostprocessStage] WARNING: ROI '{roi_name}' has a dose map "
                        "but no TAC — skipping this ROI in dose weighting."
                    )
                continue

            dose_img = sitk.ReadImage(str(dose_path))
            if dose_ref_img is None:
                dose_ref_img = dose_img
            elif not self._same_image_grid(dose_img, dose_ref_img):
                raise ValueError(
                    f"Per-ROI dose map grid mismatch for '{roi_name}'. Dose maps must "
                    "all be on the same OpenGATE simulation grid before dose post-processing."
                )

            dose_arr = sitk.GetArrayFromImage(dose_img).astype(np.float64)
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
        if dose_ref_img is None:
            raise RuntimeError("Dose post-processing reference grid was not initialized.")

        # Accumulate total dose: SUM[ dose_roi * CA_roi ]
        total_dose = None
        cumulated_activities: Dict[str, float] = {}

        for roi_name, dose_per_decay in roi_dose_maps.items():
            n_decays = compute_roi_cumulated_activity(tac_time, tac_values[roi_name])
            cumulated_activities[roi_name] = n_decays

            roi_contribution = dose_per_decay * n_decays

            if total_dose is None:
                total_dose = roi_contribution.copy()
            else:
                total_dose += roi_contribution

            # Save per-ROI dose contribution NIfTI for inspection / provenance.
            roi_contrib_path = os.path.join(
                self.work_dir, f"{self.prefix}_{roi_name}_dose_contribution.nii.gz"
            )
            save_nii_sitk(dose_ref_img, roi_contribution.astype(np.float32), roi_contrib_path)

            if self.debug:
                print(
                    f"[DosemapPostprocessStage]   {roi_name}: "
                    f"CA={n_decays:.2e} decays, "
                    f"max_dose_contribution={np.max(roi_contribution):.4e} Gy"
                )

        if self.debug:
            print(
                "[DosemapPostprocessStage] Total absorbed dose: "
                f"max={np.max(total_dose):.4e} Gy | "
                f"TAC integrated from 0 to {integration_end_min:.0f} min "
                f"({integration_end_min / 60.0 / 24.0:.1f} days)"
            )

        # Apply body-outline mask: zero out voxels outside the patient boundary.
        total_dose = self._apply_body_mask(total_dose)

        save_nii_sitk(dose_ref_img, total_dose.astype(np.float32), output_path)

        self._save_stage_metadata(
            output_path,
            cumulated_activities,
            roi_names_used,
            integration_end_min,
            dose_ref_img,
            folded_remaining_body_decays,
            rest_owned_skipped_rois,
        )

        self.context.dosemap_postprocess_output_dir = self.stage_output_dir
        self.context.dosemap_postprocess_dose_path = output_path

        self.context.extras["dosemap_postprocess_stage"] = {
            "stage_output_dir": self.stage_output_dir,
            "dose_path": output_path,
            "metadata_path": self.metadata_path,
            "folded_remaining_body_rois": list(folded_remaining_body_decays.keys()),
            "rest_owned_skipped_rois": rest_owned_skipped_rois,
        }

        return self.context
