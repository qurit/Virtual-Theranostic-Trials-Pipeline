"""
Stage-specific config snapshot builders for rerun compatibility checks.
"""

from __future__ import annotations

from typing import Any, Dict, List



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


def downstream_roi_subset(config: Dict[str, Any], *, synthetic_enabled: bool) -> List[str]:
    """Return the effective downstream ROI subset used by later stages."""
    rois = _normalize_roi_list(
        config.get("phase_1", {}).get("segmentation_stage", {}).get("roi_subset", [])
    )
    if "remaining_body" not in rois:
        rois.append("remaining_body")
    if synthetic_enabled and "synthetic_lesion" not in rois:
        rois.append("synthetic_lesion")
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
    return {
        "pbpk_tac_stage": dict(config.get("phase_1", {}).get("pbpk_tac_stage", {})),
        "downstream_roi_subset": downstream_roi_subset(
            config, synthetic_enabled=synthetic_enabled
        ),
    }


def build_simind_rerun_snapshot(config: Dict[str, Any], *, synthetic_enabled: bool) -> Dict[str, Any]:
    """Build the SIMIND rerun snapshot from a full config."""
    stage_cfg = dict(config.get("phase_2", {}).get("simind_stage", {}))
    roi_subset = stage_cfg.get("roi_subset")
    if roi_subset is None:
        roi_subset = downstream_roi_subset(config, synthetic_enabled=synthetic_enabled)
    return {
        "simind_stage": stage_cfg,
        "resolved_roi_subset": _normalize_roi_list(roi_subset),
    }


def build_opengate_rerun_snapshot(config: Dict[str, Any], *, synthetic_enabled: bool) -> Dict[str, Any]:
    """Build the OpenGATE rerun snapshot from a full config."""
    stage_cfg = dict(config.get("phase_2", {}).get("opengate_stage", {}))
    roi_subset = stage_cfg.get("roi_subset")
    if roi_subset is None:
        roi_subset = downstream_roi_subset(config, synthetic_enabled=synthetic_enabled)
    return {
        "opengate_stage": stage_cfg,
        "resolved_roi_subset": _normalize_roi_list(roi_subset),
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
