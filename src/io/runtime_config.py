"""
Runtime config loading and validation helpers.

These helpers keep JSON parsing, developer-path injection, and pre-flight
validation out of ``main.py`` so the entrypoint can stay focused on orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from json_minify import json_minify

from src.io.config_paths import inject_pipeline_paths
from src.tests.validate_config import validate_config


def load_user_config(config_path: str | Path) -> Dict[str, Any]:
    """Load one user config JSON file (comments allowed via ``json_minify``)."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.loads(json_minify(fh.read()))


def load_runtime_config(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    include_input_paths: bool = True,
) -> Dict[str, Any]:
    """Load a config and inject developer-controlled pipeline paths."""
    cfg = load_user_config(config_path)
    return inject_pipeline_paths(
        cfg,
        repo_root=repo_root,
        include_input_paths=include_input_paths,
    )


def load_and_validate_runtime_config(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    run_spect: bool,
    run_dosimetry: bool,
    run_postprocess: bool,
    include_input_paths: bool = True,
) -> Dict[str, Any]:
    """Load a config, inject runtime paths, and run the shared config validator."""
    cfg = load_runtime_config(
        config_path,
        repo_root=repo_root,
        include_input_paths=include_input_paths,
    )
    validate_config(
        cfg,
        run_spect=run_spect,
        run_dosimetry=run_dosimetry,
        run_postprocess=run_postprocess,
    )
    return cfg
