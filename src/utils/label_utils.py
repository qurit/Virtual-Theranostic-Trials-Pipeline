"""
Label map and segmentation mask utilities shared across pipeline stages.

The TDT_Pipeline label map (vtt_map.json) maps integer label IDs to organ names.
Three stages all load and use this map — this module provides one canonical
implementation used by all of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from json_minify import json_minify


def load_tdt_label_map(path: str | Path) -> Dict[str, int]:
    """
    Load the TDT_Pipeline section of a label map JSON as {roi_name: label_id}.

    The JSON format is:  {"TDT_Pipeline": {"1": "kidney", "2": "liver", ...}}
    This function inverts it to {"kidney": 1, "liver": 2, ...}.

    Parameters
    ----------
    path : str or Path
        Path to the label map JSON (comments allowed via json_minify).

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    KeyError           if the JSON is missing the 'TDT_Pipeline' key.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TDT label map not found: {p}")
    with open(p, encoding="utf-8") as f:
        import json
        data = json.loads(json_minify(f.read()))
    if "TDT_Pipeline" not in data:
        raise KeyError(f"Label map JSON at '{p}' is missing the 'TDT_Pipeline' key")
    return {name: int(label_id) for label_id, name in data["TDT_Pipeline"].items()}


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
