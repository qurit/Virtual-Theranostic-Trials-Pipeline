"""
SIMIND Preprocessing Stage for the TDT pipeline.

This stage prepares inputs needed by SIMIND by:
- Converting CT + segmentation NIfTIs into the SIMIND grid convention (z, y, x with y-flip).
- Optionally resizing to a target in-plane dimension via isotropic zoom.
- Building ROI masks and a label->name class map from the unified TDT multilabel segmentation.
- Writing binary files used by SIMIND (attenuation map, body mask, per-ROI binary source maps).

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.subdir_paths["phase_2"] : str
- context.config["phase_2"]["preprocess_simind_stage"] : dict
- context.config["phase_1"]["unification_stage"]["label_map_path"] : str
- context.ct_nii_path : str
- context.tdt_roi_seg_path : str  (unified TDT ROI segmentation NIfTI)
- context.downstream_roi_subset : list[str] | None

On success, this stage sets:
- context.body_seg_arr : np.ndarray  (float32 body mask in SIMIND grid)
- context.roi_body_seg_arr : np.ndarray  (int16 labels; requested ROIs only, body-masked)
- context.mask_roi_body : dict[int, np.ndarray]  (label_id -> boolean mask)
- context.class_seg : dict[str, int]  (roi_name -> label_id)
- context.atn_av_path : str  (attenuation binary path)
- context.binary_roi_act_map_paths : dict[str, str]  ({roi_name: binary source map path})
- context.arr_px_spacing_cm : tuple[float, float, float]  ((z, y, x) spacing in cm)
- context.arr_shape_new : tuple[int, int, int]  ((z, y, x) array shape after optional resize)

Maintainer / contact: pyazdi@bccrc.ca
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from json_minify import json_minify

# Linear attenuation coefficients at the Lu-177 photopeak (~208 keV), in 1/cm.
MU_WATER_CM_INV: float = 0.1537
MU_BONE_CM_INV: float = 0.2234


class SimindPreprocessStage:
    """
    Prepare CT + ROI masks for SIMIND simulation.

    Grid convention
    ---------------
    Arrays are transposed to (z, y, x) and flipped in `y` to match SIMIND's expected
    orientation before any resizing or binary export.

    Resizing
    --------
    If `xy_dim` is provided, an isotropic scale factor is derived from the in-plane
    dimension and applied to all three axes via scipy.ndimage.zoom.
    """

    def __init__(self, context: Any) -> None:
        context.require("subdir_paths", "config", "ct_nii_path", "tdt_roi_seg_path")
        self.context = context

        self.phase_output_dir: str = context.subdir_paths["phase_2"]
        self.stage_cfg: Dict[str, Any] = context.config["phase_2"]["preprocess_simind_stage"]
        self.output_dir: str = os.path.join(
            self.phase_output_dir,
            self.stage_cfg.get("sub_dir_name", "preprocess_simind"),  #changed: consistent with other stages
        )
        self.work_dir: str = os.path.join(self.output_dir, "work_dir")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)

        self.prefix: str = str(self.stage_cfg["file_prefix"])
        self.resize: Optional[int] = self.stage_cfg.get("xy_dim")

        # downstream_roi_subset is set by main._context_setup and flows through all phases.
        roi_subset = getattr(context, "downstream_roi_subset", None)
        if roi_subset is None:
            roi_subset = self.stage_cfg.get("roi_subset", [])
        if isinstance(roi_subset, str):
            roi_subset = [roi_subset]
        self.roi_subset: Sequence[str] = [str(r).strip() for r in roi_subset if str(r).strip()]

        # Load only the TDT_Pipeline label map (name -> id).
        self.ts_map_path: str = context.config["phase_1"]["unification_stage"]["label_map_path"]
        if not os.path.exists(self.ts_map_path):
            raise FileNotFoundError(f"Class map json not found: {self.ts_map_path}")

        with open(self.ts_map_path, encoding="utf-8") as f:
            ts_map_json = json.loads(json_minify(f.read()))

        self.tdt_name2id: Dict[str, int] = {
            name: int(lab) for lab, name in ts_map_json["TDT_Pipeline"].items()
        }

        self.ct_nii_path: Optional[str] = context.ct_nii_path
        self.tdt_roi_seg_path: Optional[str] = context.tdt_roi_seg_path
        self.metadata_path: str = os.path.join(self.work_dir, f"{self.prefix}_metadata.json")

    # -----------------------------
    # helpers
    # -----------------------------

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
        resize : Optional[int]  target in-plane dimension (square xy assumed after transpose)
        transpose_tuple : tuple[int,int,int]
        zoom_order : int  0 = nearest (seg), 1 = linear (CT)

        Returns
        -------
        (array_zyx, scale_factor)
        """
        arr = np.array(nii_obj.get_fdata(dtype=np.float32))
        arr = np.transpose(arr, transpose_tuple)[:, ::-1, :]

        scale = 1.0
        if resize is not None:
            if arr.shape[1] != arr.shape[2]:
                raise ValueError("Resize parameter requires square in-plane dimensions (x=y).")
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

    def _save_stage_metadata(
        self,
        arr_px_spacing_cm: Tuple[float, float, float],
        arr_shape_new: Tuple[int, int, int],
        atn_av_path: str,
        binary_roi_act_map_paths: Dict[str, str],
        class_seg: Dict[str, int],
    ) -> None:
        """Save stage-specific metadata for debugging / provenance."""
        metadata: Dict[str, Any] = {
            "stage": "preprocess_simind",
            "ct_nii_path": self.ct_nii_path,
            "tdt_roi_seg_path": self.tdt_roi_seg_path,
            "ts_map_path": self.ts_map_path,
            "phase_output_dir": self.phase_output_dir,
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "file_prefix": self.prefix,
            "resize": self.resize,
            "roi_subset": list(self.roi_subset),
            "arr_px_spacing_cm": list(arr_px_spacing_cm),
            "arr_shape_new": list(arr_shape_new),
            "atn_av_path": atn_av_path,
            "binary_roi_act_map_paths": binary_roi_act_map_paths,
            "class_seg": class_seg,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    # -----------------------------
    # main
    # -----------------------------
    def run(self) -> Any:
        """
        Execute preprocessing and write SIMIND-ready binary files.

        Returns
        -------
        context : Context-like
        """
        if self.ct_nii_path is None or not os.path.exists(self.ct_nii_path):
            raise FileNotFoundError(f"ct_nii_path not found: {self.ct_nii_path}")
        if self.tdt_roi_seg_path is None or not os.path.exists(self.tdt_roi_seg_path):
            raise FileNotFoundError(f"Unified TDT ROI seg not found: {self.tdt_roi_seg_path}")
        if not self.roi_subset:
            raise ValueError("No downstream ROI subset provided for SIMIND preprocessing.")

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

        self._save_stage_metadata(
            arr_px_spacing_cm=arr_px_spacing_cm,
            arr_shape_new=ct_arr.shape,
            atn_av_path=atn_av_path,
            binary_roi_act_map_paths=binary_roi_act_map_paths,
            class_seg=class_seg,
        )

        self.context.body_seg_arr = body_mask
        self.context.roi_body_seg_arr = roi_body_arr
        self.context.mask_roi_body = masks
        self.context.class_seg = class_seg
        self.context.atn_av_path = atn_av_path
        self.context.binary_roi_act_map_paths = binary_roi_act_map_paths
        self.context.arr_px_spacing_cm = arr_px_spacing_cm
        self.context.arr_shape_new = ct_arr.shape
        self.context.extras["preprocess_simind_metadata_path"] = self.metadata_path

        return self.context