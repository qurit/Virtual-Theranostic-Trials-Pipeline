"""
Shared provenance and rerun-safety helpers for the VTT pipeline.

These helpers keep lightweight, persistent metadata outside stage ``work_dir``
folders so reruns can safely decide whether existing outputs belong to the same
CT input and the same effective configuration.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


METADATA_DIRNAME = "pipeline_metadata"
METADATA_SCHEMA_VERSION = 1

STAGE_LABELS: Dict[str, str] = {
    "segmentation_stage": "Segmentation stage",
    "synthetic_lesions_stage": "Synthetic lesions stage",
    "pbpk_tac_stage": "PBPK TAC stage",
    "simind_simulation_stage": "SIMIND stage",
    "opengate_simulation_stage": "OpenGATE stage",
    "spect_postprocess_stage": "SPECT post-processing stage",
    "dosemap_postprocess_stage": "Dose-map post-processing stage",
}

_SEGMENTATION_TDT_TO_TOTSEG = {
    "kidney": ("total", ["kidney_left", "kidney_right"]),
    "liver": ("total", ["liver"]),
    "prostate": ("total", ["prostate"]),
    "spleen": ("total", ["spleen"]),
    "heart": ("total", ["heart"]),
    "salivary_glands": (
        "head_glands_cavities",
        [
            "parotid_gland_left",
            "parotid_gland_right",
            "submandibular_gland_left",
            "submandibular_gland_right",
        ],
    ),
}


class RerunConflictError(RuntimeError):
    """Raised when cached outputs cannot be safely reused on rerun."""


def utc_now_iso() -> str:
    """Return a UTC timestamp in a stable ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_metadata_dir(output_folder_path: str | Path) -> Path:
    """Create and return the persistent metadata directory for one CT output."""
    meta_dir = Path(output_folder_path) / METADATA_DIRNAME
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir


def stage_metadata_path(output_folder_path: str | Path, stage_name: str) -> str:
    """Return the persistent JSON metadata path for one stage."""
    return str(ensure_metadata_dir(output_folder_path) / f"{stage_name}.json")


def ct_input_metadata_path(output_folder_path: str | Path) -> str:
    """Return the persistent JSON metadata path for the saved CT provenance."""
    return str(ensure_metadata_dir(output_folder_path) / "ct_input.json")


def normalize_jsonable(value: Any) -> Any:
    """Convert values into a stable JSON-serializable structure."""
    if isinstance(value, dict):
        return {
            str(k): normalize_jsonable(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_jsonable(v) for v in value]
    if isinstance(value, set):
        return [normalize_jsonable(v) for v in sorted(value, key=lambda x: repr(x))]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    return value


def json_digest(value: Any) -> str:
    """Return a SHA256 digest of a normalized JSON-compatible value."""
    payload = json.dumps(
        normalize_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_file_into(hasher: "hashlib._Hash", file_path: Path) -> int:
    """Update ``hasher`` with file bytes and return the byte count."""
    total = 0
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            total += len(chunk)
            hasher.update(chunk)
    return total


def fingerprint_file(path: str | Path) -> Dict[str, Any]:
    """Fingerprint a single file by content."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found for fingerprinting: {file_path}")

    hasher = hashlib.sha256()
    size_bytes = _hash_file_into(hasher, file_path)
    return {
        "kind": "file",
        "path": str(file_path.resolve()),
        "sha256": hasher.hexdigest(),
        "size_bytes": int(size_bytes),
    }


def fingerprint_path(path: str | Path) -> Dict[str, Any]:
    """
    Fingerprint a file or directory by content.

    Directories are hashed recursively using sorted relative file paths plus each
    file's byte content.
    """
    p = Path(path)
    if p.is_file():
        return fingerprint_file(p)
    if not p.is_dir():
        raise FileNotFoundError(f"Path not found for fingerprinting: {p}")

    files = sorted(fp for fp in p.rglob("*") if fp.is_file())
    hasher = hashlib.sha256()
    total_bytes = 0
    for fp in files:
        rel = fp.relative_to(p).as_posix().encode("utf-8")
        size = fp.stat().st_size
        hasher.update(b"FILE\0")
        hasher.update(rel)
        hasher.update(b"\0")
        hasher.update(str(size).encode("utf-8"))
        hasher.update(b"\0")
        total_bytes += _hash_file_into(hasher, fp)

    return {
        "kind": "directory",
        "path": str(p.resolve()),
        "sha256": hasher.hexdigest(),
        "file_count": len(files),
        "size_bytes": int(total_bytes),
    }


def fingerprint_optional_file(path: str | Path | None) -> Optional[Dict[str, Any]]:
    """Return a file fingerprint when the path exists, else ``None``."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return fingerprint_file(p)


def fingerprints_equal(lhs: Any, rhs: Any) -> bool:
    """Return True when two fingerprint dicts describe the same content."""
    lhs_n = normalize_jsonable(lhs)
    rhs_n = normalize_jsonable(rhs)
    if not isinstance(lhs_n, dict) or not isinstance(rhs_n, dict):
        return False
    return (
        lhs_n.get("kind") == rhs_n.get("kind")
        and lhs_n.get("sha256") == rhs_n.get("sha256")
    )


def flatten_paths(value: Any) -> List[str]:
    """Flatten nested dict/list path containers into one list of path strings."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, Mapping):
        out: List[str] = []
        for item in value.values():
            out.extend(flatten_paths(item))
        return out
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        out = []
        for item in value:
            out.extend(flatten_paths(item))
        return out
    return []


def any_existing_paths(paths: Any) -> bool:
    """Return True when any path in ``paths`` currently exists."""
    return any(Path(p).exists() for p in flatten_paths(paths))


def _diff_paths(old: Any, new: Any, prefix: str = "") -> List[str]:
    """Return a small list of changed JSON paths between ``old`` and ``new``."""
    old_n = normalize_jsonable(old)
    new_n = normalize_jsonable(new)

    if type(old_n) is not type(new_n):
        return [prefix or "<root>"]

    if isinstance(old_n, dict):
        keys = sorted(set(old_n) | set(new_n))
        out: List[str] = []
        for key in keys:
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in old_n or key not in new_n:
                out.append(child)
                continue
            out.extend(_diff_paths(old_n[key], new_n[key], child))
        return out

    if isinstance(old_n, list):
        if old_n == new_n:
            return []
        return [prefix or "<root>"]

    if old_n != new_n:
        return [prefix or "<root>"]
    return []


def _summarize_changes(old: Any, new: Any, *, max_items: int = 6) -> str:
    """Return a concise human-readable summary of changed config paths."""
    diffs = _diff_paths(old, new)
    if not diffs:
        return "configuration changed"
    shown = diffs[:max_items]
    suffix = " ..." if len(diffs) > max_items else ""
    return "configuration changed: " + ", ".join(shown) + suffix


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file and return the parsed dictionary."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write a JSON dictionary with stable indentation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(normalize_jsonable(payload), fh, indent=2)


def build_stage_metadata(
    *,
    stage_name: str,
    config_snapshot: Any,
    ct_identity: Any,
    upstream_fingerprints: Optional[Dict[str, Any]],
    outputs: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a stage metadata dictionary with a shared rerun-guard section."""
    payload: Dict[str, Any] = {
        "stage": stage_name,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "created_utc": utc_now_iso(),
        "rerun_guard": {
            "config_snapshot": normalize_jsonable(config_snapshot),
            "config_digest": json_digest(config_snapshot),
            "ct_identity": normalize_jsonable(ct_identity),
            "upstream_fingerprints": normalize_jsonable(upstream_fingerprints or {}),
            "outputs": normalize_jsonable(outputs),
        },
    }
    if extra:
        payload.update(normalize_jsonable(extra))
    return payload


def build_ct_identity(
    *,
    current_input_path: str,
    saved_copy_path: str,
    fingerprint: Dict[str, Any],
    input_type: str,
) -> Dict[str, Any]:
    """Build the canonical CT identity block stored in metadata."""
    return {
        "current_input_path": str(Path(current_input_path).resolve()),
        "saved_copy_path": str(Path(saved_copy_path).resolve()),
        "input_type": str(input_type),
        "fingerprint": normalize_jsonable(fingerprint),
    }


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
    stored_config = rerun_guard.get("config_snapshot")
    stored_digest = rerun_guard.get("config_digest")
    current_digest = json_digest(current_config_norm)
    if stored_digest != current_digest:
        reasons.append(_summarize_changes(stored_config, current_config_norm))

    current_ct_fp = normalize_jsonable(current_ct_identity).get("fingerprint")
    stored_ct_fp = normalize_jsonable(rerun_guard.get("ct_identity", {})).get("fingerprint")
    if stored_ct_fp and not fingerprints_equal(stored_ct_fp, current_ct_fp):
        reasons.append("the current CT input does not match the CT used to create these outputs")

    stored_upstream = normalize_jsonable(rerun_guard.get("upstream_fingerprints", {}))
    current_upstream = normalize_jsonable(current_upstream_fingerprints or {})
    for key in sorted(set(stored_upstream) | set(current_upstream)):
        if not fingerprints_equal(stored_upstream.get(key), current_upstream.get(key)):
            reasons.append(f"upstream dependency changed: {key}")

    return reasons


def assert_stage_rerun_safe(
    *,
    stage_name: str,
    metadata_path: str | Path,
    required_outputs: Any,
    current_config_snapshot: Any,
    current_ct_identity: Any,
    current_upstream_fingerprints: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate that existing cached outputs for a stage are safe to reuse.

    Returns the loaded metadata dict when cached outputs exist and are safe.
    Returns ``None`` when no relevant cached outputs exist.
    """
    if not any_existing_paths(required_outputs):
        return None

    meta_path = Path(metadata_path)
    if not meta_path.exists():
        raise RerunConflictError(
            _format_stage_conflict(
                stage_name,
                ["cached outputs exist but the persistent stage metadata file is missing"],
                meta_path,
            )
        )

    reasons = compare_stage_rerun_state(
        stage_name=stage_name,
        metadata_path=meta_path,
        current_config_snapshot=current_config_snapshot,
        current_ct_identity=current_ct_identity,
        current_upstream_fingerprints=current_upstream_fingerprints,
    )
    if reasons:
        raise RerunConflictError(_format_stage_conflict(stage_name, reasons, meta_path))
    return load_json(meta_path)


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
    roi_subset = config.get("phase_1", {}).get("segmentation_stage", {}).get("roi_subset", [])
    if isinstance(roi_subset, str):
        roi_subset = [roi_subset]
    rois = [str(r).strip() for r in roi_subset if str(r).strip()]
    if "remaining_body" not in rois:
        rois.append("remaining_body")
    if synthetic_enabled and "synthetic_lesion" not in rois:
        rois.append("synthetic_lesion")
    return rois


def build_segmentation_rerun_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the segmentation-stage rerun snapshot from a full config."""
    seg_cfg = dict(config.get("phase_1", {}).get("segmentation_stage", {}))
    roi_subset = seg_cfg.get("roi_subset", [])
    if isinstance(roi_subset, str):
        roi_subset = [roi_subset]
    rois = [str(r).strip() for r in roi_subset if str(r).strip()]

    total_rois: List[str] = []
    head_rois: List[str] = []
    seen_total: set[str] = set()
    seen_head: set[str] = set()
    for roi_name in rois:
        task_info = _SEGMENTATION_TDT_TO_TOTSEG.get(roi_name)
        if task_info is None:
            continue
        task_name, expanded = task_info
        if task_name == "total":
            for item in expanded:
                if item not in seen_total:
                    total_rois.append(item)
                    seen_total.add(item)
        elif task_name == "head_glands_cavities":
            for item in expanded:
                if item not in seen_head:
                    head_rois.append(item)
                    seen_head.add(item)

    return {
        "segmentation_stage": seg_cfg,
        "totalseg_plan": {
            "run_body": True,
            "run_total": bool(total_rois),
            "run_head_glands_cavities": bool(head_rois),
            "total_roi_subset": total_rois,
            "head_roi_subset": head_rois,
            "tdt_roi_subset": rois,
        },
    }


def build_synthetic_lesions_rerun_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the synthetic-lesions rerun snapshot from a full config."""
    return {
        "synthetic_lesions_stage": dict(config.get("phase_1", {}).get("synthetic_lesions_stage", {})),
    }


def build_pbpk_rerun_snapshot(config: Dict[str, Any], *, synthetic_enabled: bool) -> Dict[str, Any]:
    """Build the PBPK TAC rerun snapshot from a full config."""
    spect_cfg = dict(config.get("phase_3", {}).get("spect_postprocess_stage", {}))
    return {
        "pbpk_tac_stage": dict(config.get("phase_1", {}).get("pbpk_tac_stage", {})),
        "downstream_roi_subset": downstream_roi_subset(config, synthetic_enabled=synthetic_enabled),
        "spect_frame_timing": {
            "FrameStartTimes": list(spect_cfg.get("FrameStartTimes", []) or []),
            "FrameDurations": list(spect_cfg.get("FrameDurations", []) or []),
        },
    }


def build_simind_rerun_snapshot(config: Dict[str, Any], *, synthetic_enabled: bool) -> Dict[str, Any]:
    """Build the SIMIND rerun snapshot from a full config."""
    stage_cfg = dict(config.get("phase_2", {}).get("simind_stage", {}))
    roi_subset = stage_cfg.get("roi_subset")
    if roi_subset is None:
        roi_subset = downstream_roi_subset(config, synthetic_enabled=synthetic_enabled)
    if isinstance(roi_subset, str):
        roi_subset = [roi_subset]
    resolved = [str(r).strip() for r in roi_subset if str(r).strip()]
    return {
        "simind_stage": stage_cfg,
        "resolved_roi_subset": resolved,
    }


def build_opengate_rerun_snapshot(config: Dict[str, Any], *, synthetic_enabled: bool) -> Dict[str, Any]:
    """Build the OpenGATE rerun snapshot from a full config."""
    stage_cfg = dict(config.get("phase_2", {}).get("opengate_stage", {}))
    roi_subset = stage_cfg.get("roi_subset")
    if roi_subset is None:
        roi_subset = downstream_roi_subset(config, synthetic_enabled=synthetic_enabled)
    if isinstance(roi_subset, str):
        roi_subset = [roi_subset]
    resolved = []
    if "remaining_body" in downstream_roi_subset(config, synthetic_enabled=synthetic_enabled):
        resolved.append("remaining_body")
    resolved.extend(str(r).strip() for r in roi_subset if str(r).strip() and str(r).strip() not in resolved)
    return {
        "opengate_stage": stage_cfg,
        "resolved_roi_subset": resolved,
    }


def build_spect_rerun_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the SPECT post-processing rerun snapshot from a full config."""
    return {
        "spect_postprocess_stage": dict(config.get("phase_3", {}).get("spect_postprocess_stage", {})),
    }


def build_dosemap_rerun_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the dose-map post-processing rerun snapshot from a full config."""
    return {
        "dosemap_postprocess_stage": dict(config.get("phase_3", {}).get("dosemap_postprocess_stage", {})),
    }
