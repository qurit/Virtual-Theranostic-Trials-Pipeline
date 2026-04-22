"""
TAC / PBPK utilities shared across pipeline stages.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# NumPy 2.0 renamed np.trapz → np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# ROI ↔ VOI name mapping
# ---------------------------------------------------------------------------

# Mapping from VTT segmentation ROI names to PyCNO PSMA model VOI observable names.
# Used by PbpkTacStage (to select VOIs to simulate) and by any downstream stage
# that needs to look up a TAC by VOI name.
# ROIs not in this map should fall back to "Rest" in the caller.
_ROI_TO_VOI: dict[str, str] = {
    "kidney":           "Kidney",
    "remaining_body":   "Rest",
    "liver":            "Liver",
    "prostate":         "Prostate",
    "heart":            "Heart",
    "spleen":           "Spleen",
    "salivary_glands":  "SG",
    "synthetic_lesion": "Tumor1",
}


def roi_to_voi(roi_name: str) -> Optional[str]:
    """
    Map a VTT ROI name to its PyCNO PSMA VOI observable name.

    Returns None if there is no explicit mapping (callers should fall back to "Rest").
    """
    return _ROI_TO_VOI.get(roi_name, None)


# ---------------------------------------------------------------------------
# TAC integration
# ---------------------------------------------------------------------------

def compute_roi_cumulated_activity(
    tac_time: np.ndarray,
    tac_roi: np.ndarray,
) -> float:
    """
    Compute total cumulated activity (total number of decays) for one ROI.

    Uses trapezoidal integration of the TAC from t=0 to the last timepoint.
    The PBPK simulation runs for 10× the isotope half-life, so the TAC is
    effectively zero at the endpoint and the integral captures >99.9% of
    all decays — no extrapolation to infinity is needed.

    Parameters
    ----------
    tac_time : np.ndarray  (minutes)
    tac_roi  : np.ndarray  (activity in MBq for this ROI)

    Returns
    -------
    float  total number of decays over the full TAC period
    """
    integral_mbq_min = _trapezoid(
        tac_roi.astype(np.float64),
        tac_time.astype(np.float64),
    )
    # 1 MBq = 1e6 decays/s;  1 min = 60 s  →  MBq·min × 1e6 × 60 = decays
    return float(integral_mbq_min * 1e6 * 60.0)
