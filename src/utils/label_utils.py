"""
Label map and segmentation mask utilities shared across pipeline stages.

The VTT_Pipeline section of pipeline_roi_naming_map.json maps integer label IDs
to organ names. All stages load and use this map through the canonical functions below.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from json_minify import json_minify


RESERVED_ROIS = frozenset({"background", "remaining_body", "synthetic_lesion"})


@lru_cache(maxsize=1)
def load_isotope_config() -> Dict[str, Any]:
    """Load src/data/isotope_config.json (cached after first call)."""
    p = Path(__file__).resolve().parent.parent / "data" / "isotope_config.json"
    return json.loads(p.read_text())


def _normalise_roi_map_path(path: str | Path | None = None) -> str:
    """Return a stable absolute path string for ROI metadata cache keys."""
    if path is None or not str(path).strip():
        raise ValueError(
            "Pipeline ROI metadata path must be provided from src/data/pipeline_paths.json"
        )
    p = Path(path)
    return str(p.expanduser().resolve())


@lru_cache(maxsize=8)
def _load_pipeline_roi_map_cached(path_str: str) -> Dict[str, Any]:
    """Load a JSONC ROI metadata file from an already-normalised path string."""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Pipeline ROI metadata map not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.loads(json_minify(f.read()))


def load_pipeline_roi_map(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load the pipeline ROI metadata map.

    This is the single source of truth for VTT ROI labels, TotalSegmentator source
    labels, PBPK VOI mappings, and UI grouping metadata.
    """
    return _load_pipeline_roi_map_cached(_normalise_roi_map_path(path))


def load_vtt_to_totseg(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Return the VTT ROI -> TotalSegmentator/PBPK metadata table."""
    data = load_pipeline_roi_map(path)
    raw = data.get("VTT_to_totseg", {})
    return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}


def load_pbpk_observables(path: str | Path | None = None) -> list[str]:
    """
    Return the PyCNO observables the PBPK stage should request.

    Prefer the explicit developer-maintained ``PBPK_observables`` list from the
    ROI map. If absent, derive a minimal ordered list from non-null ``pbpk_voi``
    entries.
    """
    data = load_pipeline_roi_map(path)
    observables = data.get("PBPK_observables")
    if isinstance(observables, list):
        out = [str(v).strip() for v in observables if str(v).strip()]
        if out:
            return out

    seen: set[str] = set()
    out: list[str] = []
    for entry in load_vtt_to_totseg(path).values():
        voi = entry.get("pbpk_voi")
        if isinstance(voi, str) and voi.strip() and voi not in seen:
            out.append(voi)
            seen.add(voi)
    return out


def load_roi_to_pbpk_voi(
    path: str | Path | None = None,
    *,
    include_internal: bool = True,
) -> Dict[str, str]:
    """
    Return {VTT ROI name: PyCNO VOI name} for ROIs with a non-null PBPK mapping.

    ``include_internal=False`` excludes reserved/internal labels such as
    ``remaining_body`` and ``synthetic_lesion``; this is useful for user-facing
    PBPK/simulation eligibility checks.
    """
    mapping: Dict[str, str] = {}
    for roi_name, entry in load_vtt_to_totseg(path).items():
        if not include_internal and roi_name in RESERVED_ROIS:
            continue
        voi = entry.get("pbpk_voi")
        if isinstance(voi, str) and voi.strip():
            mapping[roi_name] = voi.strip()
    return mapping


def load_pbpk_compatible_rois(path: str | Path | None = None) -> set[str]:
    """
    Return user-selectable ROIs with dedicated non-Rest PBPK VOIs.

    These are the only user-selectable ROIs that can be used as independent
    SIMIND/OpenGATE sources or synthetic-lesion host organs.
    """
    return {
        roi
        for roi, voi in load_roi_to_pbpk_voi(path, include_internal=False).items()
        if voi != "Rest"
    }


def roi_to_voi(roi_name: str, path: str | Path | None = None) -> str | None:
    """Map a VTT ROI name to its PyCNO VOI name, or None if no dedicated mapping exists."""
    return load_roi_to_pbpk_voi(path, include_internal=True).get(roi_name)


def load_vtt_label_map(path: str | Path) -> Dict[str, int]:
    """
    Load the VTT_Pipeline section of pipeline_roi_naming_map.json as {roi_name: label_id}.

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
    data = load_pipeline_roi_map(path)
    if "VTT_Pipeline" not in data:
        raise KeyError(f"Label map JSON at '{path}' is missing the 'VTT_Pipeline' key")
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
