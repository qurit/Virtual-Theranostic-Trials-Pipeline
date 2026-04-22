"""
SIMIND projection/header I/O helpers shared by post-processing code.
"""

from __future__ import annotations

import os
from typing import Dict, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from pytomography.io.SPECT import simind
from pytomography.io.shared import get_header_value


def write_activity_map_nifti(arr_zyx: np.ndarray, spacing_cm_zyx: Sequence[float], out_path: str) -> None:
    """Save a SIMIND-grid activity map as NIfTI using SimpleITK."""
    img = sitk.GetImageFromArray(np.asarray(arr_zyx, dtype=np.float32))
    spacing_mm = tuple(float(x) * 10.0 for x in spacing_cm_zyx[::-1])
    img.SetSpacing(spacing_mm)
    sitk.WriteImage(img, out_path, True)


def read_simind_header_dims(photopeak_path: str, lower_path: str, upper_path: str) -> Tuple[int, int, int]:
    """Parse SIMIND headers and return ``(proj_dim1, proj_dim2, num_proj)``."""
    for path in (photopeak_path, lower_path, upper_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing SIMIND header file: {path}")

    with open(photopeak_path, "r", encoding="utf-8") as fh:
        headerdata = np.array(fh.readlines())

    return (
        int(get_header_value(headerdata, "matrix size [1]", int)),
        int(get_header_value(headerdata, "matrix size [2]", int)),
        int(get_header_value(headerdata, "total number of images", int)),
    )


def read_projection_bin(
    proj_path: str,
    *,
    proj_dim1: int,
    proj_dim2: int,
    num_proj: int,
) -> np.ndarray:
    """Read a SIMIND ``.a00`` projection binary into PyTomography convention."""
    proj_arr = np.fromfile(proj_path, dtype=np.float32)
    expected_size = int(num_proj * proj_dim1 * proj_dim2)
    if proj_arr.size != expected_size:
        raise ValueError(
            f"Unexpected projection file size for {proj_path}. "
            f"Got {proj_arr.size}, expected {expected_size} "
            f"for shape ({num_proj}, {proj_dim1}, {proj_dim2})."
        )
    proj_arr = proj_arr.reshape((num_proj, proj_dim2, proj_dim1))
    return np.asarray(np.transpose(proj_arr[:, ::-1, :], (0, 2, 1)), dtype=np.float32)


def write_projection_nifti(arr_zyx: np.ndarray, output_pixel_width_cm: float, out_path: str) -> None:
    """Save a PBPK-weighted projection volume as NIfTI."""
    img = sitk.GetImageFromArray(np.asarray(arr_zyx, dtype=np.float32))
    img.SetSpacing((
        float(output_pixel_width_cm) * 10.0,
        float(output_pixel_width_cm) * 10.0,
        1.0,
    ))
    sitk.WriteImage(img, out_path, True)


def read_sensitivity_from_calibration_file(calibration_file: str) -> float:
    """Read sensitivity (counts/s/MBq) from SIMIND ``calib.res``."""
    with open(calibration_file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    return float(lines[70].strip().split(":")[-1].strip().split()[0])


def read_header_dims_and_windows(
    photopeak_path: str,
    lower_path: str,
    upper_path: str,
) -> Tuple[int, int, int, float, float, float]:
    """Parse projection dimensions plus energy window widths from SIMIND headers."""
    with open(photopeak_path, "r", encoding="utf-8") as fh:
        headerdata = np.array(fh.readlines())

    proj_dim1 = int(get_header_value(headerdata, "matrix size [1]", int))
    proj_dim2 = int(get_header_value(headerdata, "matrix size [2]", int))
    num_proj = int(get_header_value(headerdata, "total number of images", int))
    ww_peak, ww_lower, ww_upper = [
        simind.get_energy_window_width(path)
        for path in (photopeak_path, lower_path, upper_path)
    ]
    return proj_dim1, proj_dim2, num_proj, ww_peak, ww_lower, ww_upper


def load_cor_data(cor_path: str) -> np.ndarray:
    """Load a COR file and sanitize 2D output back to 1D in place when needed."""
    cor_data = np.loadtxt(cor_path).astype(float)
    if cor_data.ndim == 2:
        cor_data = cor_data[:, 0]
        np.savetxt(cor_path, cor_data)
    return cor_data


def read_projection_nifti(projection_path: str) -> np.ndarray:
    """Read a projection NIfTI and return it as float32."""
    if not os.path.exists(projection_path):
        raise FileNotFoundError(f"Missing projection file: {projection_path}")
    return sitk.GetArrayFromImage(sitk.ReadImage(projection_path)).astype(np.float32)


def build_frame_projection_paths(
    output_dir: str,
    prefix: str,
    frame_start: Sequence[float],
) -> Dict[str, Dict[str, str]]:
    """Build the expected output paths for PBPK-weighted frame projections."""
    projection_paths: Dict[str, Dict[str, str]] = {}
    for frame in frame_start:
        frame_label = f"{float(frame)/60.0:.6f}".rstrip("0").rstrip(".")
        projection_paths[frame_label] = {
            "w1": os.path.join(output_dir, f"{prefix}_{frame_label}_tot_w1.nii.gz"),
            "w2": os.path.join(output_dir, f"{prefix}_{frame_label}_tot_w2.nii.gz"),
            "w3": os.path.join(output_dir, f"{prefix}_{frame_label}_tot_w3.nii.gz"),
        }
    return projection_paths
