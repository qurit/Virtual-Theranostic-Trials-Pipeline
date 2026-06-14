"""
TAC / PBPK utilities shared across pipeline stages.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

# NumPy 2.0 renamed np.trapz → np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def get_pbpk_voi_name_for_roi(context: Any, roi_name: str) -> str | None:
    """
    Return the VOI name actually used when PBPK generated an ROI TAC.

    Prefer runtime TAC metadata, then the saved TAC JSON. This keeps downstream
    post-processing aligned with the TAC file, even if the ROI map changes later.
    """
    tac_data = getattr(context, "pbpk_tacs_by_organ", None)
    if isinstance(tac_data, dict):
        roi_data = tac_data.get(roi_name)
        if isinstance(roi_data, dict):
            voi_name = roi_data.get("voi_name")
            if isinstance(voi_name, str) and voi_name.strip():
                return voi_name.strip()

    tac_json_path = getattr(context, "pbpk_tac_json_path", None)
    if isinstance(tac_json_path, str) and tac_json_path and os.path.exists(tac_json_path):
        try:
            with open(tac_json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except (OSError, TypeError, json.JSONDecodeError):
            json_data = {}

        roi_to_voi_used = json_data.get("roi_to_voi_used", {})
        if isinstance(roi_to_voi_used, dict):
            voi_name = roi_to_voi_used.get(roi_name)
            if isinstance(voi_name, str) and voi_name.strip():
                return voi_name.strip()

        tacs = json_data.get("tacs", {})
        if isinstance(tacs, dict):
            roi_data = tacs.get(roi_name)
            if isinstance(roi_data, dict):
                voi_name = roi_data.get("voi_name")
                if isinstance(voi_name, str) and voi_name.strip():
                    return voi_name.strip()

    return None


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
