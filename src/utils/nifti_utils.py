"""
NIfTI / array utilities shared across pipeline stages.

Two save helpers exist because different stages use different imaging libraries:
  - save_nifti_nib   : nibabel-based, preserves exact NIfTI affine + header
  - save_nii_sitk    : SimpleITK-based, copies geometry from a reference sitk.Image
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Axis transposition helpers (NIfTI convention ↔ ZYX convention)
# ---------------------------------------------------------------------------

def xyz_to_zyx(arr_xyz: np.ndarray) -> np.ndarray:
    """Transpose an array from (X, Y, Z) to (Z, Y, X)."""
    return np.transpose(arr_xyz, (2, 1, 0))


def zyx_to_xyz(arr_zyx: np.ndarray) -> np.ndarray:
    """Transpose an array from (Z, Y, X) to (X, Y, Z)."""
    return np.transpose(arr_zyx, (2, 1, 0))


# ---------------------------------------------------------------------------
# Spacing / geometry helpers
# ---------------------------------------------------------------------------

def get_spacing_zyx_mm(nii: "nib.Nifti1Image") -> np.ndarray:  # type: ignore[name-defined]
    """
    Return voxel spacing in (Z, Y, X) order from a nibabel NIfTI image, in mm.
    NIfTI header zooms are stored in (X, Y, Z) order.
    """
    import nibabel as nib  # noqa: F401 — kept local so nibabel is optional elsewhere
    spacing_xyz = np.array(nii.header.get_zooms()[:3], dtype=np.float64)
    return np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float64)


def voxel_volume_ml(spacing_cm: Sequence[float]) -> float:
    """
    Compute voxel volume in mL from a spacing tuple in cm.
    (cm³ == mL, so this is just the product of the three spacings.)
    """
    return float(np.prod(np.asarray(spacing_cm, dtype=float)))


# ---------------------------------------------------------------------------
# NIfTI I/O — nibabel
# ---------------------------------------------------------------------------

def load_int_seg(path: str) -> np.ndarray:
    """Load a NIfTI segmentation and return it as int16 (sufficient for label IDs)."""
    import nibabel as nib
    return nib.load(path).get_fdata().astype(np.int16)


def save_nifti_nib(
    path: str,
    data_xyz: np.ndarray,
    ref_nii: "nib.Nifti1Image",  # type: ignore[name-defined]
    dtype: "np.dtype",
) -> None:
    """
    Save a numpy array as NIfTI using the affine + header from `ref_nii`.

    Parameters
    ----------
    path      : output file path
    data_xyz  : array shaped (X, Y, Z)
    ref_nii   : nibabel image whose affine and header are copied
    dtype     : output data type (e.g. np.uint8, np.uint16, np.float32)
    """
    import nibabel as nib
    out = nib.Nifti1Image(data_xyz.astype(dtype, copy=False), ref_nii.affine, ref_nii.header.copy())
    out.set_data_dtype(dtype)
    nib.save(out, path)


# ---------------------------------------------------------------------------
# NIfTI I/O — SimpleITK
# ---------------------------------------------------------------------------

def save_nii_sitk(
    ref: "sitk.Image",  # type: ignore[name-defined]
    arr: np.ndarray,
    path: str,
) -> str:
    """
    Save a numpy array as NIfTI, copying geometry from a SimpleITK reference image.

    Parameters
    ----------
    ref  : SimpleITK image whose origin, spacing, and direction are copied
    arr  : numpy array (will be cast to float32 if not already)
    path : output file path (must end in .nii or .nii.gz)

    Returns
    -------
    str  the output path (for chaining)
    """
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.asarray(arr, dtype=np.float32))
    img.CopyInformation(ref)
    sitk.WriteImage(img, str(path), imageIO="NiftiImageIO")
    return str(path)
