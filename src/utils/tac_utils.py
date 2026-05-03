"""
TAC / PBPK utilities shared across pipeline stages.
"""

from __future__ import annotations

import numpy as np

from src.utils.label_utils import roi_to_voi

# NumPy 2.0 renamed np.trapz → np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


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
