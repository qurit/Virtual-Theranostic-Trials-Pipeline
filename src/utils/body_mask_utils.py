"""
Shared body-masking utility for post-processing stages.

Both SPECT and dose-map post-processing use this to zero out voxels outside
the patient body outline.  The body is the union of all ROI masks that were
active in that specific simulation — no resampling, no file loading beyond
what is already available in context.

SPECT: body = context.roi_body_seg_arr != 0  (SIMIND-specific seg array)
Dosemap: body = OR of all context.dosimetry_mask_paths  (OpenGATE-specific masks)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import SimpleITK as sitk


def body_mask_from_seg_arr(
    seg_arr: Optional[np.ndarray],
    output_shape: tuple,
    *,
    debug: bool = False,
    stage_tag: str = "",
) -> Optional[np.ndarray]:
    """
    Return a boolean body mask from a labeled segmentation array.

    Any non-zero label = inside the patient body.  If *seg_arr* is None or its
    shape does not match *output_shape*, returns None (body masking is skipped).

    Used by SpectPostprocessStage with context.roi_body_seg_arr.
    """
    if seg_arr is None:
        if debug:
            print(f"[{stage_tag}] No segmentation array on context; skipping body masking.")
        return None
    if seg_arr.shape != output_shape:
        if debug:
            print(
                f"[{stage_tag}] Seg shape {seg_arr.shape} != output shape {output_shape}; "
                "skipping body masking."
            )
        return None
    return seg_arr != 0


def body_mask_from_roi_mask_paths(
    mask_paths: Optional[Dict[str, str]],
    output_shape: tuple,
    *,
    debug: bool = False,
    stage_tag: str = "",
) -> Optional[np.ndarray]:
    """
    Return a boolean body mask by OR-ing all per-ROI binary mask NIfTIs.

    Each mask file contains a binary (0/1) array in the simulation grid.
    The union of all ROI masks gives the patient body outline for that
    simulation's specific ROI subset.

    Used by DosemapPostprocessStage with context.dosimetry_mask_paths.
    """
    if not mask_paths:
        if debug:
            print(f"[{stage_tag}] No ROI mask paths on context; skipping body masking.")
        return None
    body: Optional[np.ndarray] = None
    for roi_name, path in mask_paths.items():
        try:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(bool)
        except Exception as exc:
            if debug:
                print(f"[{stage_tag}] Could not load mask for {roi_name!r} ({exc}); skipping.")
            continue
        if arr.shape != output_shape:
            if debug:
                print(
                    f"[{stage_tag}] Mask shape {arr.shape} for {roi_name!r} != "
                    f"output shape {output_shape}; skipping body masking."
                )
            return None
        body = arr if body is None else (body | arr)
    return body
