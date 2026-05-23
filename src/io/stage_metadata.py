"""
Persistent stage-metadata helpers used by rerun guards and provenance logging.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.io.rerun_fingerprints import json_digest, normalize_jsonable


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
