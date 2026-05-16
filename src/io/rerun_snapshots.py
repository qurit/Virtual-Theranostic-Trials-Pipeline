"""
Stage-specific config snapshot builders for rerun compatibility checks.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.utils.label_utils import SYNTHETIC_LESION_LABELS, TUMOR_CLASS_TO_LABEL



def _normalize_roi_list(value: Any) -> List[str]:
    """Normalize an ROI name or ROI sequence into a clean string list."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def synthetic_lesions_enabled(config: Dict[str, Any]) -> bool:
    """Return True when synthetic lesion specs are present and non-empty in config."""
    specs = (
        config.get("phase_1", {})
        .get("synthetic_lesions_stage", {})
        .get("specs")
    )
    return bool(specs)


def synthetic_lesion_host_rois(config: Dict[str, Any]) -> List[str]:
    """Return ROI names that have synthetic-lesion specs in config."""
    specs = (
        config.get("phase_1", {})
        .get("synthetic_lesions_stage", {})
        .get("specs")
    )
    if not isinstance(specs, dict):
        return []
    return [str(roi).strip() for roi in specs if str(roi).strip()]


def _active_lesion_labels_from_specs(config: Dict[str, Any]) -> List[str]:
    """Return the unique synthetic lesion label names referenced by the lesion specs."""
    specs = (
        config.get("phase_1", {})
        .get("synthetic_lesions_stage", {})
        .get("specs") or {}
    )
    seen: List[str] = []
    for spec in specs.values():
        if not isinstance(spec, dict):
            continue
        tc = str(spec.get("pbpk_label", "TumorRest"))
        lbl = TUMOR_CLASS_TO_LABEL.get(tc, TUMOR_CLASS_TO_LABEL["TumorRest"])
        if lbl not in seen:
            seen.append(lbl)
    return seen


def downstream_roi_subset(config: Dict[str, Any], *, synthetic_enabled: bool) -> List[str]:
    """Return the effective downstream ROI subset used by later stages."""
    rois = _normalize_roi_list(
        config.get("phase_1", {}).get("segmentation_stage", {}).get("roi_subset", [])
    )
    if "remaining_body" not in rois:
        rois.append("remaining_body")
    if synthetic_enabled:
        for lbl in _active_lesion_labels_from_specs(config):
            if lbl not in rois:
                rois.append(lbl)
    return rois


def resolved_simulation_roi_subset(
    config: Dict[str, Any],
    *,
    stage_key: str,
    synthetic_enabled: bool,
) -> List[str]:
    """
    Return the effective SIMIND/OpenGATE ROI subset, including synthetic lesion sources.

    When synthetic lesions are enabled, the active lesion labels are always appended
    to the simulation ROI subset automatically — regardless of which host ROIs the
    user listed.
    """
    stage_cfg = dict(config.get("phase_2", {}).get(stage_key, {}))
    roi_subset = stage_cfg.get("roi_subset")
    if roi_subset is None:
        roi_subset = downstream_roi_subset(config, synthetic_enabled=synthetic_enabled)
    rois = _normalize_roi_list(roi_subset)
    if synthetic_enabled:
        for lbl in _active_lesion_labels_from_specs(config):
            if lbl not in rois:
                rois.append(lbl)
    return rois


def build_segmentation_rerun_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the segmentation-stage rerun snapshot from a full config."""
    return {
        "segmentation_stage": dict(config.get("phase_1", {}).get("segmentation_stage", {})),
    }


def build_synthetic_lesions_rerun_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the synthetic-lesions rerun snapshot from a full config."""
    return {
        "synthetic_lesions_stage": dict(
            config.get("phase_1", {}).get("synthetic_lesions_stage", {})
        ),
    }


def build_pbpk_rerun_snapshot(config: Dict[str, Any], *, synthetic_enabled: bool) -> Dict[str, Any]:
    """Build the PBPK TAC rerun snapshot from a full config."""
    stage_cfg = dict(config.get("phase_1", {}).get("pbpk_tac_stage", {}))
    return {
        "pbpk_tac_stage": stage_cfg,
        "downstream_roi_subset": downstream_roi_subset(
            config, synthetic_enabled=synthetic_enabled
        ),
    }


def build_simind_rerun_snapshot(config: Dict[str, Any], *, synthetic_enabled: bool) -> Dict[str, Any]:
    """Build the SIMIND rerun snapshot from a full config."""
    stage_cfg = dict(config.get("phase_2", {}).get("simind_stage", {}))
    return {
        "simind_stage": stage_cfg,
        "resolved_roi_subset": resolved_simulation_roi_subset(
            config,
            stage_key="simind_stage",
            synthetic_enabled=synthetic_enabled,
        ),
    }


def build_opengate_rerun_snapshot(config: Dict[str, Any], *, synthetic_enabled: bool) -> Dict[str, Any]:
    """Build the OpenGATE rerun snapshot from a full config."""
    stage_cfg = dict(config.get("phase_2", {}).get("opengate_stage", {}))
    return {
        "opengate_stage": stage_cfg,
        "resolved_roi_subset": resolved_simulation_roi_subset(
            config,
            stage_key="opengate_stage",
            synthetic_enabled=synthetic_enabled,
        ),
    }


def build_spect_rerun_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the SPECT post-processing rerun snapshot from a full config."""
    stage_cfg = dict(config.get("phase_3", {}).get("spect_postprocess_stage", {}))
    stage_cfg.pop("apply_frame_duration", None)
    return {
        "frame_duration_applied": True,
        "saved_image_spacing_unit": "mm",
        "spect_postprocess_stage": stage_cfg,
    }


def build_dosemap_rerun_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the dose-map post-processing rerun snapshot from a full config."""
    return {
        "dosemap_postprocess_stage": dict(
            config.get("phase_3", {}).get("dosemap_postprocess_stage", {})
        ),
    }
