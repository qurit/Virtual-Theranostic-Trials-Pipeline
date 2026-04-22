"""
DICOM patient metadata utilities shared across pipeline stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

try:
    import pydicom
except ImportError:
    pydicom = None  # type: ignore[assignment]


def extract_height_weight(dicom_dir: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract patient height (m) and weight (kg) from DICOM tags in a directory.

    Reads PatientSize (0010,1020) → metres and PatientWeight (0010,1030) → kg.
    Scans all files recursively (series are often nested patient/study/series/*.dcm).
    Stops as soon as both values are found.

    Parameters
    ----------
    dicom_dir : str
        Path to a DICOM directory.

    Returns
    -------
    (height_m, weight_kg) — either or both may be None if not found or unreadable.
    """
    if pydicom is None:
        return None, None

    candidates = []
    try:
        for fp in sorted(Path(dicom_dir).rglob("*")):
            if fp.is_file():
                candidates.append(str(fp))
    except Exception:
        return None, None

    height: Optional[float] = None
    weight: Optional[float] = None

    for fp in candidates:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            h = getattr(ds, "PatientSize", None)
            w = getattr(ds, "PatientWeight", None)
            h = float(h) if h not in (None, "", " ") else None
            w = float(w) if w not in (None, "", " ") else None
            if h is not None and h <= 0:
                h = None
            if w is not None and w <= 0:
                w = None
            if h is not None:
                height = h
            if w is not None:
                weight = w
            if height is not None and weight is not None:
                break
        except Exception:
            continue

    return height, weight
