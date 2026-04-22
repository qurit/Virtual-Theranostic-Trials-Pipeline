"""
Label map and segmentation mask utilities shared across pipeline stages.

The VTT_Pipeline section of vtt_map.json maps integer label IDs to organ names.
All stages load and use this map through the canonical functions below.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from json_minify import json_minify


@lru_cache(maxsize=1)
def load_isotope_config() -> Dict[str, Any]:
    """Load src/data/isotope_config.json (cached after first call)."""
    p = Path(__file__).resolve().parent.parent / "data" / "isotope_config.json"
    return json.loads(p.read_text())


def load_vtt_label_map(path: str | Path) -> Dict[str, int]:
    """
    Load the VTT_Pipeline section of vtt_map.json as {roi_name: label_id}.

    The JSON format is:  {"VTT_Pipeline": {"1": "kidney", "2": "liver", ...}}
    This function inverts it to {"kidney": 1, "liver": 2, ...}.

    Parameters
    ----------
    path : str or Path
        Path to the label map JSON (comments allowed via json_minify).

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    KeyError           if the JSON is missing the 'VTT_Pipeline' key.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"VTT label map not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.loads(json_minify(f.read()))
    if "VTT_Pipeline" not in data:
        raise KeyError(f"Label map JSON at '{p}' is missing the 'VTT_Pipeline' key")
    return {name: int(label_id) for label_id, name in data["VTT_Pipeline"].items()}


def build_class_map(seg_arr: np.ndarray, id_to_name: Dict[int, str]) -> Dict[str, int]:
    """
    Return {roi_name: label_id} for labels actually present in `seg_arr`.

    Ignores background (label 0) and any label IDs not in `id_to_name`.

    Parameters
    ----------
    seg_arr    : integer segmentation array (any shape)
    id_to_name : {label_id: roi_name} mapping (i.e. inverse of the label map)
    """
    class_map: Dict[str, int] = {}
    for lab in np.unique(seg_arr.astype(int)):
        if lab == 0:
            continue
        name = id_to_name.get(int(lab))
        if name is not None:
            class_map[name] = int(lab)
    return class_map


def filter_roi_seg_to_subset(
    seg_arr: np.ndarray,
    roi_subset: Sequence[str],
    name2id: Dict[str, int],
) -> np.ndarray:
    """
    Zero out labels not in `roi_subset`, then recompute remaining_body.

    remaining_body always covers: body_outline − union(roi_subset_masks).
    This corrects the phase-1 remaining_body (computed with the phase-1 roi_subset)
    for stages that use a smaller subset.

    Parameters
    ----------
    seg_arr    : integer segmentation array (any shape, in-memory copy is made)
    roi_subset : ROI names for this stage ("remaining_body" is handled automatically)
    name2id    : {roi_name: label_id} from the VTT label map
    """
    remaining_body_id = name2id.get("remaining_body")

    roi_ids: set = set()
    for name in roi_subset:
        if name == "remaining_body":
            continue
        lab = name2id.get(name)
        if lab is not None:
            roi_ids.add(int(lab))
    roi_ids.discard(0)

    body_outline = seg_arr != 0
    out = seg_arr.copy()
    keep_ids = roi_ids.copy()
    if remaining_body_id is not None:
        keep_ids.add(int(remaining_body_id))
    out[~np.isin(out, list(keep_ids))] = 0

    if remaining_body_id is not None:
        stage_roi_mask = np.isin(seg_arr, list(roi_ids)) if roi_ids else np.zeros_like(body_outline)
        remaining_body_mask = body_outline & ~stage_roi_mask
        out[remaining_body_mask] = int(remaining_body_id)

    return out


def build_label_masks(arr: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Build a boolean mask for each non-zero label in a multilabel segmentation.

    Parameters
    ----------
    arr : integer segmentation array (any shape)

    Returns
    -------
    {label_id: bool_mask}  one entry per unique non-zero label

    Raises
    ------
    ValueError  if the array contains only background (all zeros).
    """
    labels = np.unique(arr)
    labels = labels[labels != 0]
    if labels.size == 0:
        raise ValueError(
            "Segmentation has no non-zero labels. "
            "Segmentation likely failed or the ROI subset is empty or mismatched."
        )
    return {int(lab): (arr == lab) for lab in labels}
