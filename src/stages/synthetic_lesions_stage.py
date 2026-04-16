"""
Synthetic lesion generation for the VTT pipeline.

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

- Backup of unified seg BEFORE lesions:
    work_dir/<file_prefix>_pre_lesions.nii.gz  
- Global lesion masks:
    work_dir/<file_prefix>_all_lesions_binary.nii.gz   (uint8 0/1)  
    work_dir/<file_prefix>_all_lesions_labels.nii.gz   (uint8 0=bg, 1..K=lesion id across ALL ROIs)  
- Per-ROI outputs (for QC):
    work_dir/<roi>/<roi>_lesions_labels.nii.gz  
    work_dir/<roi>/<roi>_lesions_binary.nii.gz  
    work_dir/<roi>/<roi>_organ_minus_lesions.nii.gz  
    work_dir/<roi>/<roi>_lesion_metadata.json  
- Stage metadata / debug:
    work_dir/<file_prefix>_metadata.json

Primary side effect
-------------------
Overwrites `context.tdt_roi_seg_path` on disk so that:
- organ voxels remain their organ label
- lesion voxels become label = TDT_Pipeline["synthetic_lesion"] (e.g. 8)

Expected Context interface
--------------------------
Incoming `context` must provide:
- context.subdir_paths["phase_1"]
- context.config["phase_1"]["synthetic_lesions_stage"] with:
    - "file_prefix": str
    - "specs": dict | None
- context.config["phase_1"]["segmentation_stage"]["roi_subset"]
- context.config["phase_1"]["segmentation_stage"]["label_map_path"]
- context.tdt_roi_seg_path: str (unified multilabel seg produced by TdtRoiUnifyStage)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from json_minify import json_minify
from scipy.ndimage import distance_transform_edt

from src.utils.nifti_utils import xyz_to_zyx, zyx_to_xyz, get_spacing_zyx_mm, save_nifti_nib
from src.utils.label_utils import load_tdt_label_map


class SyntheticLesionsStage:
    """
    Generate synthetic spherical lesions inside organ ROIs of a unified TDT segmentation,
    then overwrite `context.tdt_roi_seg_path` by painting lesion voxels as the
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
        context.require("subdir_paths", "config", "tdt_roi_seg_path")  
        self.context = context

        # Output base directory for this stage (under phase 1)
        self.phase_output_dir: str = context.subdir_paths["phase_1"]
        self.output_dir: str = os.path.join(self.phase_output_dir, "synthetic_lesions_stage")
        self.work_dir: str = os.path.join(self.output_dir, "work_dir")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)

        # Stage config block
        self.cfg: Dict[str, Any] = context.config.get("phase_1", {}).get("synthetic_lesions_stage", {})
        self.prefix: str = str(self.cfg.get("file_prefix", "synthetic_lesions"))
        self.specs: Optional[Dict[str, Dict[str, Any]]] = self.cfg.get("specs", None)

        # Input unified segmentation path (multilabel) - will be overwritten on disk by this stage with lesions inserted
        self.tdt_roi_seg_path: Optional[str] = getattr(context, "tdt_roi_seg_path", None)

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

        # Stage outputs directory
        self.lesions_outdir = self.work_dir  
        self.metadata_path = os.path.join(self.work_dir, f"{self.prefix}_metadata.json")

        # Load label map from configured phase 1 unification stage path
        self.tdt_name2id = self._load_tdt_label_map()
        if "synthetic_lesion" not in self.tdt_name2id:
            raise ValueError(
                "The label map is missing 'synthetic_lesion' in TDT_Pipeline. "
                "Add it (e.g. \"8\": \"synthetic_lesion\")."
            )
        self.synthetic_lesion_id: int = int(self.tdt_name2id["synthetic_lesion"])

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _load_tdt_label_map(self) -> Dict[str, int]:
        """Load TDT_Pipeline label map from configured label_map_path."""
        ts_map_path = self.context.config["phase_1"]["segmentation_stage"]["label_map_path"]
        return load_tdt_label_map(ts_map_path)

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
        if self.tdt_roi_seg_path is None or (not os.path.exists(self.tdt_roi_seg_path)):
            raise FileNotFoundError(f"Unified TDT ROI seg not found: {self.tdt_roi_seg_path}")

        seg_nii = nib.load(self.tdt_roi_seg_path)
        seg_xyz = np.asanyarray(seg_nii.dataobj).astype(np.uint8, copy=False)
        seg_zyx = xyz_to_zyx(seg_xyz)
        spacing_zyx = get_spacing_zyx_mm(seg_nii)
        return seg_nii, seg_xyz, seg_zyx, spacing_zyx

    def _write_backup_seg(self, seg_nii: nib.Nifti1Image, seg_xyz: np.ndarray) -> str:
        """Save a pre-lesion backup of the unified seg; returns path."""
        os.makedirs(self.lesions_outdir, exist_ok=True)
        backup_path = os.path.join(self.lesions_outdir, f"{self.prefix}_pre_lesions.nii.gz")
        # Use uint8 to avoid truncating larger label IDs
        save_nifti_nib(backup_path, seg_xyz, seg_nii, dtype=np.uint8)
        return backup_path

    def _save_stage_metadata(
        self,
        results: Dict[str, Any],
        backup_path: str,
        global_bin_path: str,
        global_lbl_path: str,
    ) -> None:
        """Save stage-specific metadata for debugging / provenance."""
        metadata = {
            "stage": "synthetic_lesions_stage",
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "file_prefix": self.prefix,
            "tdt_roi_seg_path": self.tdt_roi_seg_path,
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
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # -------------------------------------------------------------------------
    # Geometry + sampling helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _phys_dist_mm(
        idx1_zyx: Sequence[int],
        idx2_zyx: Sequence[int],
        spacing_zyx_mm: np.ndarray,
    ) -> float:
        """Physical distance (mm) between two voxel indices in zyx indexing."""
        d = (np.array(idx1_zyx, dtype=np.float64) - np.array(idx2_zyx, dtype=np.float64)) * spacing_zyx_mm
        return float(np.sqrt(np.sum(d * d)))

    @staticmethod
    def _candidate_weights(
        mask_zyx: np.ndarray,
        cand_zyx: np.ndarray,
        spacing_zyx_mm: np.ndarray,
        prob: str,
        sigma_mm: Optional[float] = None,
        tom_map_zyx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return nonnegative weights for candidate center voxels.

        Parameters
        ----------
        mask_zyx : bool array, True inside organ ROI
        cand_zyx : (N,3) int indices in zyx order (subset of mask)
        spacing_zyx_mm : (3,) float voxel spacing in mm
        prob : {"uniform","gaussian","user_defined","tom"}
        sigma_mm : float, required if prob=="gaussian"
        tom_map_zyx : optional TOM probability map (not yet implemented)

        Returns
        -------
        w : (N,) float64 weights
        """
        prob = prob.lower()
        n = cand_zyx.shape[0]

        if prob in ("uniform", "user_defined"):
            return np.ones(n, dtype=np.float64) # if user_defined, weights are ignored since centers are fixed; return uniform for simplicity

        if prob == "gaussian":
            if sigma_mm is None:
                raise ValueError("sigma_mm required for prob='gaussian'")
            roi_pts = np.argwhere(mask_zyx)
            if roi_pts.size == 0:
                raise ValueError("ROI mask is empty; cannot compute gaussian centroid.")
            mu = roi_pts.mean(axis=0)

            dz = (cand_zyx[:, 0] - mu[0]) * spacing_zyx_mm[0] # mm physical distance from centroid in z
            dy = (cand_zyx[:, 1] - mu[1]) * spacing_zyx_mm[1] # mm physical distance from centroid in y
            dx = (cand_zyx[:, 2] - mu[2]) * spacing_zyx_mm[2] # mm physical distance from centroid in x
            r2 = dx * dx + dy * dy + dz * dz # mm^2 squared physical distance from centroid
            w = np.exp(-0.5 * r2 / (float(sigma_mm) ** 2)) # Gaussian weight based on physical distance from centroid
            return w.astype(np.float64)

        if prob == "tom":
            raise NotImplementedError(
                "prob='tom' selected, but TOM integration is TODO. "
                "Use prob='uniform' or prob='gaussian' for now."
            )

        raise ValueError(f"Unknown prob choice: {prob}")

    @staticmethod
    def _compute_distance_to_boundary_mm(mask_zyx: np.ndarray, spacing_zyx_mm: np.ndarray) -> np.ndarray:
        """
        Distance transform (mm) inside the ROI mask.

        dist_mm[z,y,x] = distance (mm) from that voxel to the nearest background voxel.
        Outside ROI, distance is 0.
        """
        # distance_transform_edt expects a binary array; sampling specifies spacing per axis
        return distance_transform_edt(mask_zyx.astype(np.uint8), sampling=spacing_zyx_mm)

    # Auto radius heuristics + sampling
    def _find_auto_radius_start_mm(
        self,
        mask_zyx: np.ndarray,
        spacing_zyx_mm: np.ndarray,
        n_lesions: int,
        dist_mm: np.ndarray,
        margin_mm: float,
    ) -> float:
        """
        Compute an initial auto radius guess r_start (mm).

        Uses:
        - distance-transform max (largest inscribed sphere radius)
        - volume-equivalent sphere radius, scaled by n_lesions^(1/3)

        Returns r_start >= 0.
        """
        # Largest feasible radius at any center given boundary margin
        max_r_mm = max(0.0, float(dist_mm.max()) - float(margin_mm))

        # Volume-equivalent sphere radius (mm) based on ROI volume
        roi_vox = int(mask_zyx.sum())
        roi_vol_mm3 = float(roi_vox) * float(np.prod(spacing_zyx_mm))  # mm^3
        r_eq = float((3.0 * roi_vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)) if roi_vol_mm3 > 0 else 0.0

        # More lesions => smaller expected radius per lesion, used n^(1/3) scaling heuristic based on volume partitioning
        scale = max(1.0, float(n_lesions) ** (1.0 / 3.0))
        r_start = min(max_r_mm, float(self.auto_start_frac) * (r_eq / scale))
        return max(0.0, float(r_start))

    def _sample_auto_radii_mm(
        self,
        r_start_mm: float,
        n_lesions: int,
        spacing_zyx_mm: np.ndarray,
        seed: int,
    ) -> List[float]:
        """
        Sample an initial set of radii for auto mode.

        No user-provided minimum radius is required:
        - We derive a tiny numerical floor from voxel spacing to avoid exactly-zero radii.
        - The shrink loop will reduce these radii further if placement fails.

        Returns radii sorted descending.
        """
        r_start = float(r_start_mm)
        if n_lesions <= 0:
            raise ValueError("n_lesions must be > 0")

        # Derived numerical floor (mm) ~ half the smallest voxel dimension
        eps_mm = float(self._EPS_RADIUS_VOX_FRAC * float(np.min(spacing_zyx_mm)))

        # If r_start is extremely small, just start with r_start (or eps) and let shrink logic handle it
        r_hi = max(r_start, 0.0)
        if r_hi <= 0.0:
            return [0.0 for _ in range(n_lesions)]

        rng = np.random.default_rng(int(seed)) # uniform sampling in [0,1) for each lesion

        # Bias smaller radii to improve placement success when multiple lesions are requested.
        # u^2 biases toward 0; then scale to [eps, r_hi].
        u = rng.random(n_lesions) ** 2 # will be in range [0,1], bias towards 0 (due to power of 2)
        radii = (eps_mm + (r_hi - eps_mm) * u).astype(np.float64)

        # Guard: if eps >= r_hi, collapse to a single radius slightly below r_hi
        if eps_mm >= r_hi:
            radii = np.full(n_lesions, max(0.5 * r_hi, 1e-6), dtype=np.float64)

        radii_list = sorted([float(r) for r in radii], reverse=True)
        return radii_list

    @staticmethod
    def _place_lesion_centers(
        mask_zyx: np.ndarray,
        dist_mm: np.ndarray,
        radii_mm: List[float],
        spacing_zyx_mm: np.ndarray,
        prob: str = "uniform",
        sigma_mm: Optional[float] = None,
        margin_mm: float = 1.0,
        seed: int = 0,
        max_attempts_per_lesion: int = 4000,
        tom_map_zyx: Optional[np.ndarray] = None,
        user_centers_zyx: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """
        Place lesion centers inside a mask, enforcing:
        - dist_to_boundary_mm(center) >= radius + margin_mm
        - pairwise distances between centers >= r_i + r_j + margin_mm

        Parameters
        ----------
        mask_zyx : bool array (Z,Y,X)
        dist_mm : float array (Z,Y,X), distance-to-boundary in mm (0 outside mask)
        radii_mm : list of radii (mm), one per lesion
        spacing_zyx_mm : voxel spacing (mm)
        prob : sampling strategy
        sigma_mm : gaussian width (mm) if prob="gaussian"
        margin_mm : clearance from organ boundary AND between lesions (mm)
        seed : RNG seed
        max_attempts_per_lesion : max random draws per lesion
        user_centers_zyx : only used if prob="user_defined"

        Returns
        -------
        centers_zyx, placed_radii_mm
        """
        prob_l = str(prob).lower()
        rng = np.random.default_rng(int(seed))

        centers: List[Tuple[int, int, int]] = []
        placed_r: List[float] = []

        if prob_l == "user_defined":
            if user_centers_zyx is None:
                raise ValueError("prob='user_defined' but user_centers_zyx=None")
            if len(user_centers_zyx) != len(radii_mm):
                raise ValueError("user_centers_zyx length must match radii_mm length")

            for c_in, r in zip(user_centers_zyx, radii_mm):
                c = tuple(map(int, c_in))
                r = float(r)

                if not mask_zyx[c]:
                    raise ValueError(f"User center {c} not inside ROI")
                if float(dist_mm[c]) < (r + float(margin_mm)):
                    raise ValueError(f"User center {c} too close to ROI boundary for radius {r} mm")

                for cj, rj in zip(centers, placed_r):
                    if SyntheticLesionsStage._phys_dist_mm(c, cj, spacing_zyx_mm) < (r + rj + float(margin_mm)):
                        raise ValueError(f"User center {c} overlaps existing lesion at {cj}")

                centers.append(c)
                placed_r.append(r)

            return centers, placed_r

        # Auto/uniform/gaussian placement (TOM will be implemented later)
        for i, r in enumerate(radii_mm, start=1):
            r = float(r)
            cand_mask = dist_mm >= (r + float(margin_mm))
            cand = np.argwhere(cand_mask)
            if cand.shape[0] == 0:
                raise RuntimeError(
                    f"No valid candidate centers for radius={r:.3f} mm (after margin). "
                    "Try smaller radii or smaller margin_mm."
                )

            w = SyntheticLesionsStage._candidate_weights(
                mask_zyx=mask_zyx,
                cand_zyx=cand,
                spacing_zyx_mm=spacing_zyx_mm,
                prob=prob_l,
                sigma_mm=sigma_mm,
                tom_map_zyx=tom_map_zyx,
            )
            w = np.maximum(w, 0.0)
            w_sum = float(w.sum())
            if w_sum <= 0.0:
                raise RuntimeError("All candidate weights are zero. Check gaussian sigma / probability map.")
            p = w / w_sum

            placed = False
            for _ in range(int(max_attempts_per_lesion)):
                k = int(rng.choice(cand.shape[0], p=p))
                c = tuple(map(int, cand[k]))

                ok = True
                for cj, rj in zip(centers, placed_r):
                    if SyntheticLesionsStage._phys_dist_mm(c, cj, spacing_zyx_mm) < (r + rj + float(margin_mm)):
                        # Too close to existing lesion; try another candidate
                        ok = False
                        break

                if ok:
                    centers.append(c)
                    placed_r.append(r)
                    placed = True
                    break

            if not placed:
                raise RuntimeError(
                    f"Failed to place lesion {i}/{len(radii_mm)} (r={r:.3f} mm) after {max_attempts_per_lesion} attempts. "
                    "Try reducing radii, margin_mm, or switching prob='uniform'."
                )

        return centers, placed_r

    @staticmethod
    def _build_lesion_labelmap_zyx(
        mask_zyx: np.ndarray,
        centers_zyx: List[Tuple[int, int, int]],
        radii_mm: List[float],
        spacing_zyx_mm: np.ndarray,
    ) -> np.ndarray:
        """
        Create a per-ROI lesion label map in zyx order:
        - 0 = background
        - 1..K = lesion id within this ROI

        Lesions are clipped to organ mask.
        """
        Z, Y, X = mask_zyx.shape
        labels = np.zeros((Z, Y, X), dtype=np.uint16)  

        for lbl, (c, r) in enumerate(zip(centers_zyx, radii_mm), start=1):
            z0, y0, x0 = c # centre in zyx order
            r = float(r) # radius in mm

            # radius in voxels, rounded up to ensure coverage; use physical radius divided by voxel size per axis
            rz = int(np.ceil(r / float(spacing_zyx_mm[0])))
            ry = int(np.ceil(r / float(spacing_zyx_mm[1])))
            rx = int(np.ceil(r / float(spacing_zyx_mm[2])))

            zmin, zmax = max(0, z0 - rz), min(Z, z0 + rz + 1)
            ymin, ymax = max(0, y0 - ry), min(Y, y0 + ry + 1)
            xmin, xmax = max(0, x0 - rx), min(X, x0 + rx + 1)

            zz, yy, xx = np.ogrid[zmin:zmax, ymin:ymax, xmin:xmax]
            dz = (zz - z0) * float(spacing_zyx_mm[0])
            dy = (yy - y0) * float(spacing_zyx_mm[1])
            dx = (xx - x0) * float(spacing_zyx_mm[2])

            sphere = (dx * dx + dy * dy + dz * dz) <= (r * r) # boolean mask of the sphere in the local bounding box
            sphere &= mask_zyx[zmin:zmax, ymin:ymax, xmin:xmax]

            labels[zmin:zmax, ymin:ymax, xmin:xmax][sphere] = np.uint16(lbl)  

        return labels

    # -------------------------------------------------------------------------
    # Spec parsing + ROI processing
    # -------------------------------------------------------------------------

    def _validate_roi_name(self, roi_name: str) -> None:
        """Validate ROI exists in label map and is not synthetic_lesion itself."""
        if roi_name not in self.tdt_name2id:
            raise ValueError(f"ROI '{roi_name}' not found in TDT_Pipeline label map.")
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
        """
        Auto mode:
        - compute r_start (mm)
        - sample radii
        - attempt placement; if fails, shrink radii and retry
        - if still fails after auto_max_shrink_iters, raise RuntimeError
        """
        r_start = self._find_auto_radius_start_mm(
            mask_zyx=organ_mask_zyx,
            spacing_zyx_mm=spacing_zyx_mm,
            n_lesions=n_lesions,
            dist_mm=dist_mm,
            margin_mm=margin_mm,
        )

        if r_start <= 0.0:
            raise RuntimeError(
                f"[{roi_name}] No room for lesions: r_start <= 0 (distance-to-boundary too small vs margin_mm)."
            )

        radii_try = self._sample_auto_radii_mm(
            r_start_mm=r_start,
            n_lesions=n_lesions,
            spacing_zyx_mm=spacing_zyx_mm,
            seed=seed,
        )

        eps_mm = float(self._EPS_RADIUS_VOX_FRAC * float(np.min(spacing_zyx_mm)))

        last_err: Optional[Exception] = None
        for shrink_i in range(int(self.auto_max_shrink_iters)):
            try: # if placement fails, catch error and shrink radii for next attempt
                centers, placed_r = self._place_lesion_centers(
                    mask_zyx=organ_mask_zyx,
                    dist_mm=dist_mm,
                    radii_mm=radii_try,
                    spacing_zyx_mm=spacing_zyx_mm,
                    prob=prob,
                    sigma_mm=sigma_mm,
                    margin_mm=margin_mm,
                    seed=seed + shrink_i,
                    max_attempts_per_lesion=self.max_lesion_placement_attempts,
                    tom_map_zyx=None,
                    user_centers_zyx=None,
                )
                return centers, placed_r
            except RuntimeError as e:
                last_err = e
                # shrink radii and keep descending order
                radii_try = sorted([float(r) * float(self.auto_shrink_factor) for r in radii_try], reverse=True)

                # If radii are effectively below a voxel-scale radius and still failing, stop.
                if float(max(radii_try)) <= eps_mm + 1e-9:
                    break

        raise RuntimeError(f"[{roi_name}] Auto placement failed after shrinking. Last error: {last_err}")

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

        roi_id = int(self.tdt_name2id[roi_name])
        organ_mask_zyx = (seg_zyx == roi_id) # boolean mask of the organ ROI in zyx order
        if int(organ_mask_zyx.sum()) == 0:
            raise ValueError(f"[{roi_name}] mask is empty in unified segmentation.")

        # Pre-compute distance-to-boundary (mm) for boundary constraint
        dist_mm = self._compute_distance_to_boundary_mm(organ_mask_zyx, spacing_zyx_mm) # mm distance from each voxel to nearest background voxel (0 outside mask)

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
            centers_zyx, placed_radii = self._place_lesion_centers(
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

        roi_lesion_labels_zyx = self._build_lesion_labelmap_zyx(
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

        # Save per-ROI outputs
        roi_dir = os.path.join(self.lesions_outdir, roi_name)
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
        """Save global lesion binary + label volumes; returns (binary_path, labels_path)."""
        global_bin_path = os.path.join(self.lesions_outdir, f"{self.prefix}_all_lesions_binary.nii.gz")
        global_lbl_path = os.path.join(self.lesions_outdir, f"{self.prefix}_all_lesions_labels.nii.gz")

        save_nifti_nib(global_bin_path, zyx_to_xyz(global_lesion_binary_zyx), seg_nii, dtype=np.uint8)
        save_nifti_nib(global_lbl_path, zyx_to_xyz(global_lesion_labels_zyx), seg_nii, dtype=np.uint16)  

        return global_bin_path, global_lbl_path

    def _overwrite_unified_seg_with_lesions(
        self,
        seg_nii: nib.Nifti1Image,
        seg_zyx: np.ndarray,
        global_lesion_binary_zyx: np.ndarray,
    ) -> None:
        """
        Overwrite `context.tdt_roi_seg_path` so that lesion voxels become synthetic_lesion_id.
        Saved as uint8 to avoid label truncation.
        """
        seg_zyx_mod = seg_zyx.copy()
        seg_zyx_mod[global_lesion_binary_zyx > 0] = int(self.synthetic_lesion_id)

        seg_xyz_mod = zyx_to_xyz(seg_zyx_mod).astype(np.uint8, copy=False)
        save_nifti_nib(self.tdt_roi_seg_path, seg_xyz_mod, seg_nii, dtype=np.uint8)

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
        if os.path.exists(self.metadata_path):
            self.logger.info("Synthetic lesions already exist, skipping generation.")
            with open(self.metadata_path, encoding="utf-8") as _f:
                _meta = json.load(_f)
            self.context.synthetic_lesions_outdir = self.lesions_outdir
            self.context.synthetic_lesions_results = {}
            self.context.synthetic_lesions_backup_seg_path = _meta.get("backup_seg_path")
            self.context.synthetic_lesions_global_binary_path = _meta.get("global_binary_path")
            self.context.synthetic_lesions_global_labels_path = _meta.get("global_labels_path")
            self.context.tdt_roi_seg_path = self.tdt_roi_seg_path
            self.context.extras["synthetic_lesions_stage"] = {
                "output_dir": self.output_dir,
                "work_dir": self.work_dir,
                "metadata_path": self.metadata_path,
            }
            self._ensure_synthetic_lesion_in_roi_subset()
            return self.context

        os.makedirs(self.lesions_outdir, exist_ok=True)
        self._ensure_synthetic_lesion_in_roi_subset() # ensures downstream ROI subset includes synthetic_lesion

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
                # Auto mode failures are expected sometimes; we skip that ROI but continue.
                # Manual mode failures also raise RuntimeError; you can change this to "raise" if preferred.
                self.logger.warning(f"[{roi_name}] Lesion generation failed; skipping ROI. Error: {e}")
                continue

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
        self.context.synthetic_lesions_outdir = self.lesions_outdir
        self.context.synthetic_lesions_results = results
        self.context.synthetic_lesions_backup_seg_path = backup_path
        self.context.synthetic_lesions_global_binary_path = global_bin_path
        self.context.synthetic_lesions_global_labels_path = global_lbl_path
        self.context.tdt_roi_seg_path = self.tdt_roi_seg_path
        self.context.extras["synthetic_lesions_stage"] = {
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "metadata_path": self.metadata_path,
        }

        return self.context
