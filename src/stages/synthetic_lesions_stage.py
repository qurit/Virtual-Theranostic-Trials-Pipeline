"""
Synthetic lesion generation for the PyTheraTwin pipeline.

Goal
----
Generate synthetic spherical lesions inside user-specified organ ROIs from the
unified label map, then write them back into that segmentation as the
``synthetic_lesion`` label.

Key behavior
------------
Constraints:
- Center must be inside ROI.
- Sphere must remain inside ROI (enforced using distance transform boundary constraint).
- Lesions must not overlap (physical distance in mm).
Sampling:
- prob="uniform": uniform candidate sampling
- prob="gaussian": Gaussian weights centered at ROI centroid
- prob="user_defined": user provides centers_zyx explicitly (validated)

Outputs
-------
Writes into:
  <phase_1_output>/synthetic_lesions_stage/

Persistent (survive work_dir cleanup, used for rerun cache check):
  <file_prefix>_all_lesions_binary.nii.gz   (uint8 0/1)
  <file_prefix>_all_lesions_labels.nii.gz   (uint8 0=bg, 1..K=lesion id across ALL ROIs)
  <output_root>/pipeline_metadata/synthetic_lesions_stage.json

Debug / QC only (inside work_dir, safe to delete):
  work_dir/<file_prefix>_pre_lesions.nii.gz
  work_dir/<roi>/<roi>_lesions_labels.nii.gz
  work_dir/<roi>/<roi>_lesions_binary.nii.gz
  work_dir/<roi>/<roi>_organ_minus_lesions.nii.gz
  work_dir/<roi>/<roi>_lesion_metadata.json

Primary side effect
-------------------
Overwrites `context.pytheratwin_roi_seg_path` on disk so that:
- organ voxels remain their organ label
- lesion voxels become label = PyTheraTwin_Pipeline["synthetic_lesion"] (e.g. 8)

Expected Context interface
--------------------------
Incoming `context` must provide:
- context.subdir_paths["phase_1"]
- context.config["phase_1"]["synthetic_lesions_stage"] with:
    - "file_prefix": str
    - "specs": dict | None
- context.config["phase_1"]["segmentation_stage"]["roi_subset"]
- label map loaded from ``src/data/pipeline_paths.json`` input_paths.label_map_path
- context.pytheratwin_roi_seg_path: str (unified multilabel seg produced by SegmentationStage)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from src.io.config_paths import get_label_map_path
from src.io.rerun_guard import (
    assert_stage_rerun_safe,
    build_stage_metadata,
    build_synthetic_lesions_rerun_snapshot,
    fingerprint_optional_file,
    stage_metadata_path,
    write_json,
)
from src.utils.nifti_utils import xyz_to_zyx, zyx_to_xyz, get_spacing_zyx_mm, save_nifti_nib
from src.utils.label_utils import load_pytheratwin_label_map
from src.utils.lesion_utils import (
    auto_place_lesions,
    build_lesion_labelmap_zyx,
    compute_distance_to_boundary_mm,
    place_lesion_centers,
)


class SyntheticLesionsStage:
    """
    Generate synthetic spherical lesions inside organ ROIs of a unified PyTheraTwin segmentation,
    then overwrite `context.pytheratwin_roi_seg_path` by painting lesion voxels as the
    `synthetic_lesion` label.

    Notes on conventions
    --------------------
    - Computation is done in array order (Z, Y, X) == "zyx".
    - NIfTI storage is treated as (X, Y, Z) == "xyz" (nibabel convention).
    - All distances / radii / margins are in **millimeters (mm)**.
    """

    # ---------- Auto-radii defaults ----------
    AUTO_SHRINK_FACTOR: float = 0.85
    AUTO_MAX_SHRINK_ITERS: int = 30
    AUTO_START_FRAC: float = 0.60  # fraction used in r_start heuristic (volume-based)

    # Placement attempts
    MAX_LESION_PLACEMENT_ATTEMPTS: int = 4000

    # Voxel-scale radius floor to avoid empty spheres
    _EPS_RADIUS_VOX_FRAC: float = 0.50  # radius floor ~0.5 * min_voxel_size (epsilon radius vox fraction)

    def __init__(self, context: Any) -> None:
        context.require(
            "subdir_paths",
            "config",
            "pytheratwin_roi_seg_path",
            "output_folder_path",
            "ct_input_identity",
        )
        self.context = context

        # Output base directory for this stage (under phase 1)
        self.phase_output_dir: str = context.subdir_paths["phase_1"]
        _syn_subdir = context.config.get("phase_1", {}).get("synthetic_lesions_stage", {}).get("sub_dir_name", "synthetic_lesions_stage")
        self.output_dir: str = os.path.join(self.phase_output_dir, _syn_subdir)
        self.work_dir: str = os.path.join(self.output_dir, "work_dir")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)

        # Stage config block
        self.cfg: Dict[str, Any] = context.config.get("phase_1", {}).get("synthetic_lesions_stage", {})
        self.prefix: str = str(self.cfg.get("file_prefix", "synthetic_lesions"))
        self.specs: Optional[Dict[str, Dict[str, Any]]] = self.cfg.get("specs", None)
        self.ct_input_identity: Dict[str, Any] = context.ct_input_identity

        # Input unified segmentation path (multilabel) - will be overwritten on disk by this stage with lesions inserted
        self.pytheratwin_roi_seg_path: Optional[str] = getattr(context, "pytheratwin_roi_seg_path", None)

        # Keep ROI subset updated so downstream TAC can include synthetic_lesion if needed
        roi_subset = getattr(self.context, "downstream_roi_subset", None)  
        if roi_subset is None:  
            roi_subset = self.context.config["phase_1"]["segmentation_stage"]["roi_subset"]  
        if isinstance(roi_subset, str):
            roi_subset = [roi_subset]
        self.roi_subset: List[str] = [str(r).strip() for r in roi_subset if str(r).strip()]

        # Logger + tunables (stage-level)
        self.logger = getattr(self.context, "_logger", logging.getLogger(__name__))
        self.default_seed: int = int(self.cfg.get("default_seed", 0))

        self.auto_shrink_factor: float = float(self.cfg.get("auto_shrink_factor", self.AUTO_SHRINK_FACTOR))
        self.auto_max_shrink_iters: int = int(self.cfg.get("auto_max_shrink_iters", self.AUTO_MAX_SHRINK_ITERS))
        self.auto_start_frac: float = float(self.cfg.get("auto_start_frac", self.AUTO_START_FRAC))
        self.max_lesion_placement_attempts: int = int(
            self.cfg.get("max_lesion_placement_attempts", self.MAX_LESION_PLACEMENT_ATTEMPTS)
        )

        # Persistent primary outputs live directly in output_dir (survive work_dir cleanup).
        # Per-ROI QC files and the pre-lesion backup stay in work_dir (debug only).
        self.metadata_path = stage_metadata_path(context.output_folder_path, "synthetic_lesions_stage")
        self.global_bin_path = os.path.join(self.output_dir, f"{self.prefix}_all_lesions_binary.nii.gz")
        self.global_lbl_path = os.path.join(self.output_dir, f"{self.prefix}_all_lesions_labels.nii.gz")
        self.backup_path = os.path.join(self.work_dir, f"{self.prefix}_pre_lesions.nii.gz")

        # Load label map from pipeline_paths.json
        _label_map_path = get_label_map_path()
        self.pytheratwin_name2id = load_pytheratwin_label_map(_label_map_path)
        if "synthetic_lesion" not in self.pytheratwin_name2id:
            raise ValueError(
                "The label map is missing 'synthetic_lesion' in PyTheraTwin_Pipeline. "
                "Add it (e.g. \"8\": \"synthetic_lesion\")."
            )
        self.synthetic_lesion_id: int = int(self.pytheratwin_name2id["synthetic_lesion"])

    def _rerun_config_snapshot(self) -> Dict[str, Any]:
        """Return the config subset that must match for cached lesion outputs to remain valid."""
        return build_synthetic_lesions_rerun_snapshot(self.context.config)

    def _current_dependency_fingerprints(self) -> Dict[str, Any]:
        """Return fingerprints for dependencies that must remain unchanged on rerun."""
        pre_lesion_seg_path = self.backup_path if os.path.exists(self.backup_path) else self.pytheratwin_roi_seg_path
        return {
            "segmentation_stage_metadata": fingerprint_optional_file(
                stage_metadata_path(self.context.output_folder_path, "segmentation_stage")
            ),
            "label_map_json": fingerprint_optional_file(get_label_map_path()),
            "pre_lesion_seg_handoff": fingerprint_optional_file(pre_lesion_seg_path),
        }

    def _cleanup_failed_run(self) -> None:
        """Remove partial outputs after a failed fresh lesion-generation attempt."""
        shutil.rmtree(self.output_dir, ignore_errors=True)
        try:
            os.remove(self.metadata_path)
        except FileNotFoundError:
            pass

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _ensure_synthetic_lesion_in_roi_subset(self) -> None:
        """Ensure downstream ROI subset includes 'synthetic_lesion'."""
        self.roi_subset = [str(r).strip() for r in self.roi_subset if str(r).strip()]
        if "synthetic_lesion" not in self.roi_subset:
            self.roi_subset.append("synthetic_lesion")
        self.context.downstream_roi_subset = list(self.roi_subset)  

    def _load_unified_seg(self) -> Tuple[nib.Nifti1Image, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the unified multilabel segmentation.

        Returns
        -------
        seg_nii : nib.Nifti1Image
        seg_xyz : np.ndarray (X,Y,Z) int
        seg_zyx : np.ndarray (Z,Y,X) int
        spacing_zyx_mm : np.ndarray (3,) float64
        """
        if self.pytheratwin_roi_seg_path is None or (not os.path.exists(self.pytheratwin_roi_seg_path)):
            raise FileNotFoundError(f"Unified PyTheraTwin ROI seg not found: {self.pytheratwin_roi_seg_path}")

        seg_nii = nib.load(self.pytheratwin_roi_seg_path)
        seg_xyz = np.asanyarray(seg_nii.dataobj).astype(np.uint8, copy=False)
        seg_zyx = xyz_to_zyx(seg_xyz)
        spacing_zyx = get_spacing_zyx_mm(seg_nii)
        return seg_nii, seg_xyz, seg_zyx, spacing_zyx

    def _write_backup_seg(self, seg_nii: nib.Nifti1Image, seg_xyz: np.ndarray) -> str:
        """Save a pre-lesion backup of the unified seg; returns path."""
        os.makedirs(self.output_dir, exist_ok=True)
        save_nifti_nib(self.backup_path, seg_xyz, seg_nii, dtype=np.uint8)
        return self.backup_path

    def _save_stage_metadata(
        self,
        results: Dict[str, Any],
        backup_path: str,
        global_bin_path: str,
        global_lbl_path: str,
    ) -> None:
        """Save stage-specific metadata for debugging / provenance."""
        outputs = {
            "backup_seg_path": backup_path,
            "global_binary_path": global_bin_path,
            "global_labels_path": global_lbl_path,
            "pytheratwin_roi_seg_path": self.pytheratwin_roi_seg_path,
        }
        extra = {
            "stage": "synthetic_lesions_stage",
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "file_prefix": self.prefix,
            "pytheratwin_roi_seg_path": self.pytheratwin_roi_seg_path,
            "synthetic_lesion_id": int(self.synthetic_lesion_id),
            "default_seed": int(self.default_seed),
            "auto_shrink_factor": float(self.auto_shrink_factor),
            "auto_max_shrink_iters": int(self.auto_max_shrink_iters),
            "auto_start_frac": float(self.auto_start_frac),
            "max_lesion_placement_attempts": int(self.max_lesion_placement_attempts),
            "backup_seg_path": backup_path,
            "global_binary_path": global_bin_path,
            "global_labels_path": global_lbl_path,
            "results_summary": list(results.keys()),
            "results": results,
        }
        metadata = build_stage_metadata(
            stage_name="synthetic_lesions_stage",
            config_snapshot=self._rerun_config_snapshot(),
            ct_identity=self.ct_input_identity,
            upstream_fingerprints=self._current_dependency_fingerprints(),
            outputs=outputs,
            extra=extra,
        )
        write_json(self.metadata_path, metadata)

    # -------------------------------------------------------------------------
    # Spec parsing + ROI processing
    # -------------------------------------------------------------------------

    def _validate_roi_name(self, roi_name: str) -> None:
        """Validate ROI exists in label map and is not synthetic_lesion itself."""
        if roi_name not in self.pytheratwin_name2id:
            raise ValueError(f"ROI '{roi_name}' not found in PyTheraTwin_Pipeline label map.")
        if roi_name == "synthetic_lesion":
            raise ValueError("Do not specify lesions inside ROI='synthetic_lesion'.")

    def _parse_roi_spec(self, roi_name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse + validate a single ROI spec, returning a normalized dict.

        Returned keys:
        - n_lesions : int
        - prob : str
        - sigma_mm : Optional[float]
        - margin_mm : float
        - seed : int
        - auto_radii : bool
        - radii_mm : Optional[List[float]]
        - user_centers_zyx : Optional[List[Tuple[int,int,int]]]
        """
        if not isinstance(spec, dict):
            raise ValueError(f"[{roi_name}] spec must be a dict.")

        n_lesions_raw = spec.get("n_lesions", None)
        if not isinstance(n_lesions_raw, int) or n_lesions_raw <= 0:
            raise ValueError(f"[{roi_name}] n_lesions must be an int > 0 (got {n_lesions_raw}).")
        n_lesions = int(n_lesions_raw)

        prob = str(spec.get("prob", "uniform"))
        prob_l = prob.lower()

        sigma_mm = spec.get("sigma_mm", None)
        if prob_l == "gaussian":
            if sigma_mm is None:
                raise ValueError(f"[{roi_name}] prob='gaussian' requires sigma_mm in spec.")
            sigma_mm = float(sigma_mm)
        else:
            sigma_mm = float(sigma_mm) if sigma_mm is not None else None

        margin_mm = float(spec.get("margin_mm", 1.0))
        seed = int(spec.get("seed", self.default_seed))

        radii_raw = spec.get("radii_mm", None)
        auto_radii = ("radii_mm" not in spec) or (radii_raw is None)

        if prob_l == "user_defined" and auto_radii:
            raise ValueError(f"[{roi_name}] prob='user_defined' requires radii_mm (cannot be None).")

        radii_mm: Optional[List[float]] = None
        if not auto_radii:
            if not isinstance(radii_raw, list):
                raise ValueError(f"[{roi_name}] radii_mm must be a list.")
            if len(radii_raw) != n_lesions:
                raise ValueError(f"[{roi_name}] radii_mm length must match n_lesions ({n_lesions}).")
            radii_mm = [float(r) for r in radii_raw]
            if any((not np.isfinite(r)) or (r <= 0) for r in radii_mm):
                raise ValueError(f"[{roi_name}] all radii_mm must be finite and > 0.")

        user_centers_zyx: Optional[List[Tuple[int, int, int]]] = None
        if prob_l == "user_defined":
            user_centers_raw = spec.get("user_centers_zyx", None)
            if user_centers_raw is None or (not isinstance(user_centers_raw, list)):
                raise ValueError(f"[{roi_name}] prob='user_defined' requires user_centers_zyx as a list of [z,y,x].")
            if len(user_centers_raw) != n_lesions:
                raise ValueError(f"[{roi_name}] user_centers_zyx length must match n_lesions ({n_lesions}).")
            user_centers_zyx = [tuple(map(int, c)) for c in user_centers_raw]

        return {
            "n_lesions": n_lesions,
            "prob": prob,
            "sigma_mm": sigma_mm,
            "margin_mm": margin_mm,
            "seed": seed,
            "auto_radii": bool(auto_radii),
            "radii_mm": radii_mm,
            "user_centers_zyx": user_centers_zyx,
        }

    def _auto_place_roi_lesions(
        self,
        roi_name: str,
        organ_mask_zyx: np.ndarray,
        spacing_zyx_mm: np.ndarray,
        dist_mm: np.ndarray,
        n_lesions: int,
        prob: str,
        sigma_mm: Optional[float],
        margin_mm: float,
        seed: int,
    ) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Auto-place lesions with the shared shrink-and-retry utility."""
        return auto_place_lesions(
            roi_name=roi_name,
            organ_mask_zyx=organ_mask_zyx,
            spacing_zyx_mm=spacing_zyx_mm,
            dist_mm=dist_mm,
            n_lesions=n_lesions,
            prob=prob,
            sigma_mm=sigma_mm,
            margin_mm=margin_mm,
            seed=seed,
            auto_start_frac=self.auto_start_frac,
            auto_shrink_factor=self.auto_shrink_factor,
            auto_max_shrink_iters=self.auto_max_shrink_iters,
            eps_radius_vox_frac=self._EPS_RADIUS_VOX_FRAC,
            max_attempts_per_lesion=self.max_lesion_placement_attempts,
        )

    def _save_roi_outputs(
        self,
        roi_dir: str,
        roi_name: str,
        roi_lesion_labels_zyx: np.ndarray,
        roi_lesion_binary_zyx: np.ndarray,
        roi_organ_minus_lesions_zyx: np.ndarray,
        seg_nii: nib.Nifti1Image,
    ) -> Dict[str, str]:
        """Save per-ROI NIfTI outputs; returns dict of saved paths."""
        os.makedirs(roi_dir, exist_ok=True)

        labels_path = os.path.join(roi_dir, f"{roi_name}_lesions_labels.nii.gz")
        bin_path = os.path.join(roi_dir, f"{roi_name}_lesions_binary.nii.gz")
        minus_path = os.path.join(roi_dir, f"{roi_name}_organ_minus_lesions.nii.gz")

        save_nifti_nib(labels_path, zyx_to_xyz(roi_lesion_labels_zyx), seg_nii, dtype=np.uint16)  
        save_nifti_nib(bin_path, zyx_to_xyz(roi_lesion_binary_zyx), seg_nii, dtype=np.uint8)
        save_nifti_nib(minus_path, zyx_to_xyz(roi_organ_minus_lesions_zyx), seg_nii, dtype=np.uint8)

        return {
            "lesions_labels": labels_path,
            "lesions_binary": bin_path,
            "organ_minus_lesions": minus_path,
        }

    def _process_single_roi(
        self,
        roi_name: str,
        spec: Dict[str, Any],
        seg_zyx: np.ndarray,
        seg_nii: nib.Nifti1Image,
        spacing_zyx_mm: np.ndarray,
        global_next_id: int,
        global_lesion_binary_zyx: np.ndarray,
        global_lesion_labels_zyx: np.ndarray,
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Process a single ROI:
        - validate + parse spec
        - build organ mask and distance transform
        - place lesions (auto or manual)
        - build lesion labelmaps
        - write per-ROI outputs + metadata
        - update global lesion arrays and return updated global_next_id
        """
        self._validate_roi_name(roi_name) # fail hard if invalid
        parsed = self._parse_roi_spec(roi_name, spec) # checks + returns normalized spec dict; fail hard if invalid

        roi_id = int(self.pytheratwin_name2id[roi_name])
        organ_mask_zyx = (seg_zyx == roi_id) # boolean mask of the organ ROI in zyx order
        if int(organ_mask_zyx.sum()) == 0:
            raise ValueError(f"[{roi_name}] mask is empty in unified segmentation.")

        # Pre-compute distance-to-boundary (mm) for boundary constraint
        dist_mm = compute_distance_to_boundary_mm(organ_mask_zyx, spacing_zyx_mm)

        n_lesions = int(parsed["n_lesions"])
        prob = str(parsed["prob"])
        sigma_mm = parsed["sigma_mm"]
        margin_mm = float(parsed["margin_mm"])
        seed = int(parsed["seed"])

        if bool(parsed["auto_radii"]):
            centers_zyx, placed_radii = self._auto_place_roi_lesions(
                roi_name=roi_name,
                organ_mask_zyx=organ_mask_zyx,
                spacing_zyx_mm=spacing_zyx_mm,
                dist_mm=dist_mm,
                n_lesions=n_lesions,
                prob=prob,
                sigma_mm=sigma_mm,
                margin_mm=margin_mm,
                seed=seed,
            )
        else:
            # Manual radii (and optional user centers). Fail hard if impossible.
            centers_zyx, placed_radii = place_lesion_centers(
                mask_zyx=organ_mask_zyx,
                dist_mm=dist_mm,
                radii_mm=list(parsed["radii_mm"]),
                spacing_zyx_mm=spacing_zyx_mm,
                prob=prob,
                sigma_mm=sigma_mm,
                margin_mm=margin_mm,
                seed=seed,
                max_attempts_per_lesion=self.max_lesion_placement_attempts,
                tom_map_zyx=None,
                user_centers_zyx=parsed["user_centers_zyx"],
            )

        roi_lesion_labels_zyx = build_lesion_labelmap_zyx(
            mask_zyx=organ_mask_zyx,
            centers_zyx=centers_zyx,
            radii_mm=placed_radii,
            spacing_zyx_mm=spacing_zyx_mm,
        )
        roi_lesion_binary_zyx = (roi_lesion_labels_zyx > 0).astype(np.uint8)
        roi_organ_minus_lesions_zyx = (organ_mask_zyx & (roi_lesion_binary_zyx == 0)).astype(np.uint8)

        # Update global lesion binary
        global_lesion_binary_zyx |= roi_lesion_binary_zyx # lesion voxels are 1, so bitwise OR accumulates them across ROIs

        # Update global lesion labels by offsetting per-ROI local IDs
        roi_max = int(roi_lesion_labels_zyx.max())
        if roi_max > 0:
            offset = int(global_next_id) - 1
            m = roi_lesion_labels_zyx > 0
            global_lesion_labels_zyx[m] = (roi_lesion_labels_zyx[m].astype(np.uint16) + offset).astype(np.uint16)  
            global_next_id += roi_max

        # Save per-ROI QC outputs into work_dir (debug artifacts, not needed for reruns)
        roi_dir = os.path.join(self.work_dir, roi_name)
        paths = self._save_roi_outputs(
            roi_dir=roi_dir,
            roi_name=roi_name,
            roi_lesion_labels_zyx=roi_lesion_labels_zyx,
            roi_lesion_binary_zyx=roi_lesion_binary_zyx,
            roi_organ_minus_lesions_zyx=roi_organ_minus_lesions_zyx,
            seg_nii=seg_nii,
        )

        # Metadata
        meta_path = os.path.join(roi_dir, f"{roi_name}_lesion_metadata.json")
        meta = {
            "roi": roi_name,
            "roi_id": roi_id,
            "synthetic_lesion_id": int(self.synthetic_lesion_id),
            "prob": prob,
            "sigma_mm": float(sigma_mm) if sigma_mm is not None else None,
            "margin_mm": float(margin_mm),  # mm
            "seed": int(seed),
            "spacing_zyx_mm": [float(x) for x in spacing_zyx_mm.tolist()],  # mm
            "centers_zyx": [list(map(int, c)) for c in centers_zyx],
            "radii_mm": [float(r) for r in placed_radii],
            "dist_to_boundary_mm": [float(dist_mm[c]) for c in centers_zyx],
            "paths": paths,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta, global_next_id

    def _save_global_outputs(
        self,
        seg_nii: nib.Nifti1Image,
        global_lesion_binary_zyx: np.ndarray,
        global_lesion_labels_zyx: np.ndarray,
    ) -> Tuple[str, str]:
        """Save global lesion binary + label volumes to output_dir; returns (binary_path, labels_path)."""
        save_nifti_nib(self.global_bin_path, zyx_to_xyz(global_lesion_binary_zyx), seg_nii, dtype=np.uint8)
        save_nifti_nib(self.global_lbl_path, zyx_to_xyz(global_lesion_labels_zyx), seg_nii, dtype=np.uint16)
        return self.global_bin_path, self.global_lbl_path

    def _overwrite_unified_seg_with_lesions(
        self,
        seg_nii: nib.Nifti1Image,
        seg_zyx: np.ndarray,
        global_lesion_binary_zyx: np.ndarray,
    ) -> None:
        """
        Overwrite `context.pytheratwin_roi_seg_path` so that lesion voxels become synthetic_lesion_id.
        Saved as uint8 to avoid label truncation.
        """
        seg_zyx_mod = seg_zyx.copy()
        seg_zyx_mod[global_lesion_binary_zyx > 0] = int(self.synthetic_lesion_id)

        seg_xyz_mod = zyx_to_xyz(seg_zyx_mod).astype(np.uint8, copy=False)
        save_nifti_nib(self.pytheratwin_roi_seg_path, seg_xyz_mod, seg_nii, dtype=np.uint8)

    # -------------------------------------------------------------------------
    # Public entrypoint
    # -------------------------------------------------------------------------

    def run(self) -> Any:
        """
        Run the synthetic lesions stage.

        Behavior
        --------
        - If no specs are provided, stage is a no-op and context is returned unchanged
          (except for stage output fields being set to None/empty).
        - For each ROI in specs, lesions are generated and saved.
        - Global lesion masks are saved.
        - The unified segmentation on disk is overwritten to include synthetic_lesion voxels.
        """
        # Skip if no specs
        if not self.specs:
            self.logger.info("No synthetic lesion specs provided; skipping lesion generation.")
            self.context.synthetic_lesions_outdir = None
            self.context.synthetic_lesions_results = {}
            self.context.extras["synthetic_lesions_stage"] = {
                "output_dir": None,
                "work_dir": None,
                "metadata_path": None,
            }
            return self.context

        # Skip if outputs from a previous run already exist.
        cache_meta = assert_stage_rerun_safe(
            stage_name="synthetic_lesions_stage",
            metadata_path=self.metadata_path,
            required_outputs=[self.backup_path, self.global_bin_path, self.global_lbl_path],
            current_config_snapshot=self._rerun_config_snapshot(),
            current_ct_identity=self.ct_input_identity,
            current_upstream_fingerprints=self._current_dependency_fingerprints(),
            context=self.context,
        )

        if cache_meta is not None:
            self.logger.info("Synthetic lesions already exist, skipping generation.")
            self.context.synthetic_lesions_outdir = self.output_dir
            self.context.synthetic_lesions_results = cache_meta.get("results", {})
            self.context.synthetic_lesions_backup_seg_path = cache_meta.get("backup_seg_path")
            self.context.synthetic_lesions_global_binary_path = cache_meta.get("global_binary_path")
            self.context.synthetic_lesions_global_labels_path = cache_meta.get("global_labels_path")
            self.context.pytheratwin_roi_seg_path = self.pytheratwin_roi_seg_path
            self.context.extras["synthetic_lesions_stage"] = {
                "output_dir": self.output_dir,
                "work_dir": self.work_dir,
                "metadata_path": self.metadata_path,
            }
            self._ensure_synthetic_lesion_in_roi_subset()
            return self.context

        os.makedirs(self.output_dir, exist_ok=True)

        # Load segmentation + backup
        seg_nii, seg_xyz, seg_zyx, spacing_zyx_mm = self._load_unified_seg()
        backup_path = self._write_backup_seg(seg_nii, seg_xyz)

        # Global accumulators (zyx)
        global_lesion_binary_zyx = np.zeros(seg_zyx.shape, dtype=np.uint8)
        global_lesion_labels_zyx = np.zeros(seg_zyx.shape, dtype=np.uint16)  
        global_next_id = 1

        results: Dict[str, Any] = {}

        # Main ROI loop -> loop over rois in spec
        for roi_name, roi_spec in self.specs.items():
            try:
                meta, global_next_id = self._process_single_roi(
                    roi_name=roi_name,
                    spec=roi_spec,
                    seg_zyx=seg_zyx,
                    seg_nii=seg_nii,
                    spacing_zyx_mm=spacing_zyx_mm,
                    global_next_id=global_next_id,
                    global_lesion_binary_zyx=global_lesion_binary_zyx,
                    global_lesion_labels_zyx=global_lesion_labels_zyx,
                )
                if meta is not None:
                    results[roi_name] = meta
            except (RuntimeError, ValueError) as e:
                self._cleanup_failed_run()
                raise RuntimeError(
                    f"[{roi_name}] Synthetic lesion generation failed. "
                    "Delete the old synthetic-lesion outputs for this patient and rerun "
                    f"after fixing the config. Original error: {e}"
                ) from e

        if not results or int(global_lesion_binary_zyx.sum()) == 0:
            self._cleanup_failed_run()
            raise RuntimeError(
                "Synthetic lesion generation did not create any lesion voxels. "
                "Delete the old synthetic-lesion outputs for this patient and rerun "
                "after fixing the config."
            )

        # Save global masks
        global_bin_path, global_lbl_path = self._save_global_outputs(
            seg_nii=seg_nii,
            global_lesion_binary_zyx=global_lesion_binary_zyx,
            global_lesion_labels_zyx=global_lesion_labels_zyx,
        )

        # Overwrite unified seg on disk with synthetic lesion label
        self._overwrite_unified_seg_with_lesions(
            seg_nii=seg_nii,
            seg_zyx=seg_zyx,
            global_lesion_binary_zyx=global_lesion_binary_zyx,
        )

        self._save_stage_metadata(results, backup_path, global_bin_path, global_lbl_path)

        # Update context for downstream stages
        self._ensure_synthetic_lesion_in_roi_subset()
        self.context.synthetic_lesions_outdir = self.output_dir
        self.context.synthetic_lesions_results = results
        self.context.synthetic_lesions_backup_seg_path = backup_path
        self.context.synthetic_lesions_global_binary_path = global_bin_path
        self.context.synthetic_lesions_global_labels_path = global_lbl_path
        self.context.pytheratwin_roi_seg_path = self.pytheratwin_roi_seg_path
        self.context.extras["synthetic_lesions_stage"] = {
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "metadata_path": self.metadata_path,
        }

        return self.context
