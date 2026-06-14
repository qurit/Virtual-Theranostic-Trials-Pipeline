"""
Pre-flight CT input validation helpers for the PyTheraTwin pipeline.

These checks are intentionally lightweight and filesystem-focused. They are
used by ``main.py`` before any patient run starts so obvious input mistakes are
caught early, in the same spirit as ``validate_config``.
"""

from __future__ import annotations

import os
from typing import List, Literal, Tuple


CTInputType = Literal["nii", "dicom"]


def is_supported_nii_name(name: str) -> bool:
    """Return True when *name* is a supported NIfTI filename."""
    name_l = name.lower()
    return name_l.endswith(".nii") or name_l.endswith(".nii.gz")


def validate_ct_input_path(path: str) -> CTInputType:
    """
    Validate a single CT input path and return its detected type.

    Parameters
    ----------
    path : str
        Path to a NIfTI file or DICOM directory.

    Returns
    -------
    {"nii", "dicom"}
        Detected CT input type.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *path* is a file but not a supported NIfTI extension.
    """
    if os.path.isfile(path):
        if not is_supported_nii_name(path):
            raise ValueError(f"CT input file must be .nii or .nii.gz, got: {path}")
        return "nii"
    if os.path.isdir(path):
        return "dicom"
    raise FileNotFoundError(f"CT input not found: {path}")


def discover_ct_inputs(ct_inputs_dir: str) -> Tuple[List[str], List[str]]:
    """
    Discover supported batch CT inputs in *ct_inputs_dir*.

    Supported entries are:
    - top-level directories (treated as DICOM patients)
    - top-level ``.nii`` / ``.nii.gz`` files

    Hidden entries are ignored. Everything else is returned in ``skipped``.
    """
    items: List[str] = []
    skipped: List[str] = []
    for name in sorted(os.listdir(ct_inputs_dir)):
        if name.startswith("."):
            continue
        path = os.path.join(ct_inputs_dir, name)
        if os.path.isdir(path) or is_supported_nii_name(name):
            items.append(name)
        else:
            skipped.append(name)
    return items, skipped
