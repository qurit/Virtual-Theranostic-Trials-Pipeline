"""
Shared rerun-safety checks for the VTT pipeline.

The low-level fingerprinting, metadata storage, and per-stage snapshot builders
live in smaller helper modules so this file can stay focused on the actual
rerun-guard decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.io.rerun_fingerprints import (
    any_existing_paths,
    fingerprint_optional_file,
    fingerprint_path,
    fingerprints_equal,
    flatten_paths,
    json_digest,
    normalize_jsonable,
)
from src.io.rerun_snapshots import (
    build_dosemap_rerun_snapshot,
    build_opengate_rerun_snapshot,
    build_pbpk_rerun_snapshot,
    build_segmentation_rerun_snapshot,
    build_simind_rerun_snapshot,
    build_spect_rerun_snapshot,
    build_synthetic_lesions_rerun_snapshot,
    downstream_roi_subset,
    synthetic_lesions_enabled,
)
from src.io.stage_metadata import (
    METADATA_SCHEMA_VERSION,
    STAGE_LABELS,
    build_ct_identity,
    build_stage_metadata,
    ct_input_metadata_path,
    ensure_metadata_dir,
    load_json,
    stage_metadata_path,
    utc_now_iso,
    write_json,
)


class RerunConflictError(RuntimeError):
    """Raised when cached outputs cannot be safely reused on rerun."""


def ensure_ct_matches_saved_copy(
    *,
    current_input_path: str,
    saved_copy_path: str,
    output_folder_path: str | Path,
    input_type: str,
) -> Dict[str, Any]:
    """
    Verify that the current CT matches the saved CT copy for this output folder.

    Returns the canonical CT identity block to store on the context and in stage
    metadata.
    """
    current_fp = fingerprint_path(current_input_path)
    saved_path = Path(saved_copy_path)
    saved_fp: Optional[Dict[str, Any]] = None

    if saved_path.exists():
        saved_fp = fingerprint_path(saved_path)
        if not fingerprints_equal(current_fp, saved_fp):
            raise RerunConflictError(
                "The selected CT input does not match the CT already saved in this output "
                "folder. Delete the old patient output folder (or use a new project name) "
                "before running with a different CT."
            )

    meta_path = Path(ct_input_metadata_path(output_folder_path))
    if meta_path.exists():
        meta = load_json(meta_path)
        stored_fp = meta.get("ct_identity", {}).get("fingerprint")
        if stored_fp and not fingerprints_equal(stored_fp, current_fp):
            raise RerunConflictError(
                "The selected CT input does not match the CT provenance recorded for this "
                "output folder. Delete the old patient output folder (or use a new project "
                "name) before running with a different CT."
            )

    ct_identity = build_ct_identity(
        current_input_path=current_input_path,
        saved_copy_path=saved_copy_path,
        fingerprint=current_fp,
        input_type=input_type,
    )

    write_json(
        meta_path,
        {
            "stage": "ct_input",
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "created_utc": utc_now_iso(),
            "ct_identity": ct_identity,
            "saved_copy_fingerprint": normalize_jsonable(saved_fp or current_fp),
        },
    )

    return ct_identity


def _format_stage_conflict(stage_name: str, reasons: List[str], metadata_path: str | Path) -> str:
    """Format a clear user-facing rerun conflict message."""
    label = STAGE_LABELS.get(stage_name, stage_name)
    lines = [f"{label}: existing outputs are not safe to reuse."]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.append(
        f"Delete the old stage outputs for this patient (and '{metadata_path}') and rerun."
    )
    return "\n".join(lines)


def _missing_required_outputs(required_outputs: Any) -> List[str]:
    """Return required output paths that do not exist on disk."""
    return [path for path in flatten_paths(required_outputs) if not Path(path).exists()]


def _compare_upstream_fingerprints(
    stored_upstream: Any,
    current_upstream: Optional[Dict[str, Any]],
) -> List[str]:
    """Return rerun-conflict reasons for changed upstream dependencies."""
    stored_norm = normalize_jsonable(stored_upstream or {})
    current_norm = normalize_jsonable(current_upstream or {})
    if not isinstance(stored_norm, dict) or not isinstance(current_norm, dict):
        return ["upstream dependency fingerprints changed"]

    reasons: List[str] = []
    for key in sorted(set(stored_norm) | set(current_norm)):
        stored_value = stored_norm.get(key)
        current_value = current_norm.get(key)
        if stored_value == current_value:
            continue
        if isinstance(stored_value, dict) and isinstance(current_value, dict):
            if fingerprints_equal(stored_value, current_value):
                continue
        reasons.append(f"upstream dependency changed: {key}")
    return reasons


def compare_stage_rerun_state(
    *,
    stage_name: str,
    metadata_path: str | Path,
    current_config_snapshot: Any,
    current_ct_identity: Any,
    current_upstream_fingerprints: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return a list of rerun-compatibility problems for one stage metadata file."""
    meta = load_json(metadata_path)
    rerun_guard = meta.get("rerun_guard")
    if not isinstance(rerun_guard, dict):
        return [
            "metadata is missing the rerun_guard section, so cached outputs cannot be verified"
        ]

    reasons: List[str] = []

    if meta.get("stage") != stage_name:
        reasons.append(f"metadata stage name is {meta.get('stage')!r}, expected {stage_name!r}")

    current_config_norm = normalize_jsonable(current_config_snapshot)
    current_digest = json_digest(current_config_norm)
    if rerun_guard.get("config_digest") != current_digest:
        reasons.append("configuration changed")

    current_ct_fp = normalize_jsonable(current_ct_identity).get("fingerprint")
    stored_ct_fp = normalize_jsonable(rerun_guard.get("ct_identity", {})).get("fingerprint")
    if stored_ct_fp and not fingerprints_equal(stored_ct_fp, current_ct_fp):
        reasons.append("the current CT input does not match the CT used to create these outputs")

    reasons.extend(
        _compare_upstream_fingerprints(
            rerun_guard.get("upstream_fingerprints", {}),
            current_upstream_fingerprints,
        )
    )

    return reasons


def assert_stage_rerun_safe(
    *,
    stage_name: str,
    metadata_path: str | Path,
    required_outputs: Any,
    current_config_snapshot: Any,
    current_ct_identity: Any,
    current_upstream_fingerprints: Optional[Dict[str, Any]] = None,
    context: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate that existing cached outputs for a stage are safe to reuse.

    Returns the loaded metadata dict when cached outputs exist and are safe,
    and sets ``context.stage_skipped = True`` so the profiler can skip logging.
    Returns ``None`` when no relevant cached outputs exist.
    """
    if not any_existing_paths(required_outputs):
        return None

    meta_path = Path(metadata_path)
    missing_outputs = _missing_required_outputs(required_outputs)
    if missing_outputs:
        preview = ", ".join(missing_outputs[:5])
        if len(missing_outputs) > 5:
            preview += f", ... ({len(missing_outputs)} missing total)"
        raise RerunConflictError(
            _format_stage_conflict(
                stage_name,
                [f"cached outputs are incomplete; missing required output(s): {preview}"],
                meta_path,
            )
        )

    if not meta_path.exists():
        raise RerunConflictError(
            _format_stage_conflict(
                stage_name,
                ["cached outputs exist but the persistent stage metadata file is missing"],
                meta_path,
            )
        )

    if reasons := compare_stage_rerun_state(
        stage_name=stage_name,
        metadata_path=meta_path,
        current_config_snapshot=current_config_snapshot,
        current_ct_identity=current_ct_identity,
        current_upstream_fingerprints=current_upstream_fingerprints,
    ):
        raise RerunConflictError(_format_stage_conflict(stage_name, reasons, meta_path))

    if context is not None:
        context.stage_skipped = True
    return load_json(meta_path)
