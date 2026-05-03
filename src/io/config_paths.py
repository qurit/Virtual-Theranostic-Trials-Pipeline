"""
Shared helpers for pipeline-path injection and config cleanup.

The pipeline keeps developer-controlled naming fields in
``src/data/pipeline_paths.json`` rather than in user-edited config files.
This module centralizes the logic that loads those fields, injects them into
live configs, and strips them back out for user-facing views.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

from json_minify import json_minify


DEVELOPER_CONFIG_FIELDS = frozenset(
    {
        "output_folder_title",
        "sub_dir_name",
        "file_prefix",
        "unification_prefix",
        "SIMINDDirectory",
    }
)

_PHASE_KEYS = ("phase_1", "phase_2", "phase_3")
_STAGE_FIELDS = ("file_prefix", "unification_prefix", "sub_dir_name")


def load_pipeline_paths(repo_root: str | Path) -> Dict[str, Any]:
    """Read ``src/data/pipeline_paths.json`` and return the parsed dict."""
    paths_file = Path(repo_root) / "src" / "data" / "pipeline_paths.json"
    try:
        return json.loads(json_minify(paths_file.read_text(encoding="utf-8")))
    except Exception:
        return {}


def get_pipeline_input_paths(repo_root: str | Path) -> Dict[str, str]:
    """
    Return resolved developer-controlled input paths.

    Paths are read from ``pipeline_paths.json`` only; keep that JSON updated
    when developer-controlled file locations change.
    """
    repo_root = Path(repo_root)
    input_paths = load_pipeline_paths(repo_root).get("input_paths", {})

    simind_directory = str(input_paths.get("SIMINDDirectory", "")).strip()

    return {
        "label_map_path": str(input_paths.get("label_map_path", "")).strip(),
        "SIMINDDirectory": simind_directory,
    }


def get_label_map_path(repo_root: str | Path | None = None) -> str:
    """Return the label map path from ``pipeline_paths.json``."""
    if repo_root is None:
        repo_root = Path(__file__).parent.parent.parent
    return get_pipeline_input_paths(repo_root)["label_map_path"]


def strip_developer_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of ``cfg`` with developer-controlled fields removed."""
    cfg_copy = copy.deepcopy(cfg)

    def _strip(obj: Any) -> None:
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if key in DEVELOPER_CONFIG_FIELDS:
                    del obj[key]
                else:
                    _strip(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                _strip(item)

    _strip(cfg_copy)
    return cfg_copy


def inject_pipeline_paths(
    cfg: Dict[str, Any],
    repo_root: str | Path,
    *,
    include_input_paths: bool,
) -> Dict[str, Any]:
    """
    Return a deep copy of ``cfg`` with developer-controlled fields injected.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline config.
    repo_root : str or Path
        Repository root used to locate ``pipeline_paths.json``.
    include_input_paths : bool
        When true, also inject ``SIMINDDirectory`` from pipeline_paths.json.
        When false, only structural naming fields are injected.
    """
    cfg_copy = copy.deepcopy(cfg)
    repo_root = Path(repo_root)
    pipeline_paths = load_pipeline_paths(repo_root)

    if include_input_paths:
        input_paths = get_pipeline_input_paths(repo_root)
        simind_directory = input_paths["SIMINDDirectory"]
        if simind_directory:
            simind_stage = cfg_copy.setdefault("phase_2", {}).setdefault("simind_stage", {})
            simind_stage["SIMINDDirectory"] = simind_directory

    for phase_key in _PHASE_KEYS:
        phase_paths = pipeline_paths.get(phase_key, {})
        if not isinstance(phase_paths, dict):
            continue

        cfg_phase = cfg_copy.setdefault(phase_key, {})
        if "sub_dir_name" in phase_paths:
            cfg_phase["sub_dir_name"] = phase_paths["sub_dir_name"]

        for stage_key, stage_data in phase_paths.items():
            if stage_key == "sub_dir_name" or not isinstance(stage_data, dict):
                continue

            cfg_stage = cfg_phase.setdefault(stage_key, {})
            for field in _STAGE_FIELDS:
                if field in stage_data:
                    cfg_stage[field] = stage_data[field]

    return cfg_copy
