"""
SimpleITK/OpenGATE image-preparation helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import SimpleITK as sitk

def cleanup_mhd(path: Path) -> None:
    """Delete an MHD file and its paired raw/zraw payload if they exist."""
    for candidate in [path, path.with_suffix(".raw"), path.with_suffix(".zraw")]:
        try:
            candidate.unlink(missing_ok=True)
        except Exception:
            pass


def find_hu_tables(gate_module: Any) -> Tuple[Path, Path]:
    """Locate OpenGATE Schneider HU-to-material conversion tables."""
    pkg = Path(gate_module.__file__).resolve().parent
    for root in [pkg, pkg.parent, *list(pkg.parents[:4])]:
        for rel in [Path("tests/data"), Path("opengate/tests/data")]:
            mat_table = root / rel / "Schneider2000MaterialsTable.txt"
            den_table = root / rel / "Schneider2000DensitiesTable.txt"
            if mat_table.exists() and den_table.exists():
                return mat_table, den_table
    raise FileNotFoundError("Could not locate OpenGATE Schneider HU tables")


def to_centered_identity(img: sitk.Image, is_label: bool = False) -> Tuple[sitk.Image, List[int]]:
    """Convert an image to OpenGATE's centered identity-direction convention."""
    size = np.array(img.GetSize(), dtype=np.float64)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    direction = np.array(img.GetDirection()).reshape(3, 3)

    arr = sitk.GetArrayFromImage(img)
    if is_label:
        arr = arr.astype(np.int32)

    flipped_axes: List[int] = []
    col_to_npy = {0: 2, 1: 1, 2: 0}
    for col in range(3):
        if direction[col, col] < 0:
            np_axis = col_to_npy[col]
            arr = np.flip(arr, axis=np_axis).copy()
            flipped_axes.append(np_axis)

    out = sitk.GetImageFromArray(arr)
    out.SetOrigin((-(size * spacing) / 2.0 + spacing / 2.0).tolist())
    out.SetSpacing(spacing.tolist())
    out.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    return out, flipped_axes


def resample_image(
    img: sitk.Image,
    size: Tuple[int, ...],
    spacing: Tuple[float, ...],
    *,
    is_label: bool = False,
) -> sitk.Image:
    """Resample an image onto a new grid, using NN for labels and linear for CT."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(list(spacing))
    resampler.SetSize(list(size))
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(img)


def upsample_array_to_native(arr: np.ndarray, ref: sitk.Image, native: sitk.Image) -> np.ndarray:
    """Resample a simulation-space array back to the native CT grid."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.CopyInformation(ref)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(native)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(sitk.Transform())
    return sitk.GetArrayFromImage(resampler.Execute(img)).astype(np.float32)
