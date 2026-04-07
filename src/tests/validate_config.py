"""
Pre-flight configuration validator for the VTT pipeline.

This module is called by ``main.py`` immediately after the config JSON is
parsed, before any CT is touched.  All problems are collected into a single
``ValueError`` so the user can fix everything in one pass instead of
discovering issues one run at a time.

Checks performed
----------------
- ``output_folder_title``   — non-empty string
- ``label_map_path``        — non-empty string; file must exist on disk
- ``roi_subset`` (seg.)     — non-empty list of strings
- ``pbpk_tac_stage.isotope``— string; must be in pipeline_options.json allowed list
- ``pbpk_tac_stage.VOIs``   — non-empty list
- Synthetic-lesion numerics — only when ``--synthetic_lesions`` is set
- ``SIMINDDirectory``       — read from ``pipeline_paths.json`` input_paths; path must exist (``--spect`` only)
- SIMIND ``roi_subset``     — subset of segmentation ``roi_subset``
- SIMIND ``Collimator``     — string; must be in allowed list
- SIMIND ``Isotope``        — string
- SIMIND numeric fields     — ``NumPhotons``, ``NumProjections``, etc. must be numbers
- ``xyz_dim``               — null or a list of 3 positive ints
- OpenGATE ``roi_subset``   — subset of segmentation ``roi_subset``
- OpenGATE gate numerics    — ``total_histories``, ``num_threads`` must be numbers
- ``FrameStartTimes`` /
  ``FrameDurations``        — same length; all elements must be numbers
- SPECT reconstruction
  numerics                  — ``Iterations``, ``Subsets`` must be numbers

Usage
-----
Import and call ``validate_config`` from your run script::

    from src.tests.validate_config import validate_config
    validate_config(config, run_spect=True, run_dosimetry=False, ...)

For questions or issues, contact: pyazdi@bccrc.ca
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# json_minify strips // comments from JSONC-style files before json.loads
try:
    from json_minify import json_minify
except ImportError:
    def json_minify(text: str) -> str:  # type: ignore[misc]
        """Fallback no-op when json_minify is not installed."""
        return text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_allowed_options(repo_root: str) -> Dict[str, List]:
    """
    Load src/data/pipeline_options.json and flatten it to {key: [values]}.

    The options file may be nested by phase/stage for readability, so we
    recursively walk the dict and collect every leaf list.  Keys beginning
    with ``_`` are treated as comment placeholders and ignored.

    Returns an empty dict if the file cannot be read.
    """
    opts_path = os.path.join(repo_root, "src", "data", "pipeline_options.json")
    allowed: Dict[str, List] = {}
    try:
        with open(opts_path, encoding="utf-8") as fh:
            raw = json.loads(json_minify(fh.read()))

        def _flatten(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, list):
                        allowed[k] = v
                    elif isinstance(v, dict):
                        _flatten(v)

        _flatten(raw)
    except Exception:
        pass  # Enum checks simply skip if the file is unreadable

    return allowed


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def validate_config(
    config: Dict[str, Any],
    run_spect: bool = False,
    run_dosimetry: bool = False,
    run_postprocess: bool = False,
    synthetic_lesions: bool = False,
    repo_root: str | None = None,
) -> None:
    """
    Validate the parsed config dict before any pipeline stage runs.

    Collects every problem into a single ``ValueError`` so all issues are
    surfaced at once rather than one at a time.

    Parameters
    ----------
    config : dict
        Parsed pipeline configuration (output of ``json.loads``).
    run_spect : bool
        True when ``--spect`` was passed; enables SIMIND-specific checks.
    run_dosimetry : bool
        True when ``--dosimetry`` was passed; enables OpenGATE checks.
    run_postprocess : bool
        True when ``--postprocess`` was passed; enables Phase-3 checks.
    synthetic_lesions : bool
        True when ``--synthetic_lesions`` was passed; validates lesion params.
    repo_root : str or None
        Absolute path to the repository root used to locate
        ``src/data/pipeline_options.json``.  Defaults to the grandparent
        directory of this file.

    Raises
    ------
    ValueError
        With a bullet-point summary of every problem found.
    """
    if repo_root is None:
        # src/tests/validate_config.py → src/ → repo_root/
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

    errors: List[str] = []
    _allowed = _load_allowed_options(repo_root)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _err(msg: str) -> None:
        errors.append(msg)

    def _check_number(val: Any, field: str) -> None:
        """Append an error if *val* is not a genuine number (bools excluded)."""
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            _err(
                f"'{field}' must be a number, "
                f"got {type(val).__name__}: {val!r}"
            )

    def _check_int(val: Any, field: str) -> None:
        """Append an error if *val* is not a genuine integer (bools excluded)."""
        if isinstance(val, bool) or not isinstance(val, int):
            _err(
                f"'{field}' must be an integer, "
                f"got {type(val).__name__}: {val!r}"
            )

    def _check_xyz_dim(xyz: Any, field: str) -> None:
        """Validate xyz_dim: must be null or a list of 3 positive integers."""
        if xyz is None:
            return
        if not isinstance(xyz, list) or len(xyz) != 3:
            _err(
                f"'{field}' must be null or a list of exactly 3 integers "
                f"[x, y, z], got: {xyz!r}"
            )
            return
        for i, v in enumerate(xyz):
            if v is not None and (
                isinstance(v, bool) or not isinstance(v, int) or v <= 0
            ):
                _err(
                    f"'{field}[{i}]' must be a positive integer or null, "
                    f"got {type(v).__name__}: {v!r}"
                )

    # ── Top-level ──────────────────────────────────────────────────────────────

    title = config.get("output_folder_title")
    if not isinstance(title, str) or not title.strip():
        _err("'output_folder_title' must be a non-empty string")

    # ── Phase 1 ────────────────────────────────────────────────────────────────

    p1 = config.get("phase_1", {})
    if not isinstance(p1, dict):
        _err("'phase_1' must be a dict")
        # Cannot safely inspect sub-keys without a valid dict; raise immediately
        raise ValueError(
            f"Config validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    # Segmentation stage
    seg = p1.get("segmentation_stage", {})

    # label_map_path is injected automatically at runtime from the repo root;
    # only validate it when it is explicitly present in the config.
    lmp = seg.get("label_map_path", "")
    if lmp and not os.path.exists(lmp):
        _err(
            f"'phase_1.segmentation_stage.label_map_path' does not exist: "
            f"{lmp!r}"
        )

    roi_subset = seg.get("roi_subset", [])
    if not isinstance(roi_subset, list) or len(roi_subset) == 0:
        _err(
            "'phase_1.segmentation_stage.roi_subset' must be a non-empty "
            "list of ROI name strings"
        )
    else:
        for i, r in enumerate(roi_subset):
            if not isinstance(r, str):
                _err(
                    f"'phase_1.segmentation_stage.roi_subset[{i}]' must be "
                    f"a string, got {type(r).__name__}: {r!r}"
                )
    # Guard downstream cross-checks if roi_subset is malformed
    if not isinstance(roi_subset, list):
        roi_subset = []

    # PBPK stage
    pbpk = p1.get("pbpk_tac_stage", {})

    isotope_p1 = pbpk.get("isotope")
    if not isinstance(isotope_p1, str):
        _err(
            f"'phase_1.pbpk_tac_stage.isotope' must be a string, "
            f"got {type(isotope_p1).__name__}: {isotope_p1!r}"
        )
    elif _allowed.get("isotope") and isotope_p1 not in _allowed["isotope"]:
        _err(
            f"'phase_1.pbpk_tac_stage.isotope' value {isotope_p1!r} is not "
            f"in the allowed list: {_allowed['isotope']}"
        )

    vois = pbpk.get("VOIs", [])
    if not isinstance(vois, list) or len(vois) == 0:
        _err(
            "'phase_1.pbpk_tac_stage.VOIs' must be a non-empty list of "
            "PyCNO VOI name strings"
        )

    # Synthetic lesions stage (only validated when the flag is active)
    if synthetic_lesions:
        sl = p1.get("synthetic_lesions_stage", {})
        for int_field in (
            "default_seed", "auto_max_shrink_iters", "max_lesion_placement_attempts"
        ):
            v = sl.get(int_field)
            if v is not None:
                _check_int(v, f"phase_1.synthetic_lesions_stage.{int_field}")
        for num_field in ("auto_shrink_factor", "auto_start_frac"):
            v = sl.get(num_field)
            if v is not None:
                _check_number(v, f"phase_1.synthetic_lesions_stage.{num_field}")

    # ── Phase 2 ────────────────────────────────────────────────────────────────

    p2 = config.get("phase_2", {})

    # SIMIND SPECT simulation (only when --spect is set)
    if run_spect:
        simind = p2.get("simind_stage", {})

        # SIMINDDirectory is no longer in the user config — it is set once in
        # src/data/pipeline_paths.json under "input_paths.SIMINDDirectory".
        # Check that source directly; fall back to the config field for any
        # legacy configs that still carry it.
        _pp_file = os.path.join(repo_root, "src", "data", "pipeline_paths.json")
        _pp_input_paths: Dict[str, Any] = {}
        try:
            with open(_pp_file, encoding="utf-8") as _ppf:
                _pp_input_paths = json.loads(json_minify(_ppf.read())).get("input_paths", {})
        except Exception:
            pass
        simind_dir = (
            _pp_input_paths.get("SIMINDDirectory", "")
            or simind.get("SIMINDDirectory", "")
        )
        if not isinstance(simind_dir, str) or not simind_dir.strip():
            _err(
                "SIMINDDirectory is required when --spect is enabled. "
                "Set 'input_paths.SIMINDDirectory' in src/data/pipeline_paths.json."
            )
        elif not os.path.exists(simind_dir):
            _err(
                f"SIMINDDirectory does not exist: {simind_dir!r} "
                f"(set in src/data/pipeline_paths.json → input_paths.SIMINDDirectory)"
            )

        # roi_subset must be a subset of the segmentation roi_subset
        simind_rois = simind.get("roi_subset", [])
        if not isinstance(simind_rois, list):
            _err("'phase_2.simind_stage.roi_subset' must be a list")
        else:
            bad = [r for r in simind_rois if r not in roi_subset]
            if bad:
                _err(
                    f"'phase_2.simind_stage.roi_subset' contains ROIs not in "
                    f"'phase_1.segmentation_stage.roi_subset': {bad}. "
                    "Add them to segmentation roi_subset or remove them here."
                )

        collimator = simind.get("Collimator")
        if not isinstance(collimator, str):
            _err(
                f"'phase_2.simind_stage.Collimator' must be a string, "
                f"got {type(collimator).__name__}: {collimator!r}"
            )
        elif _allowed.get("collimator") and collimator not in _allowed["collimator"]:
            _err(
                f"'phase_2.simind_stage.Collimator' value {collimator!r} is "
                f"not in the allowed list: {_allowed['collimator']}"
            )

        isotope_simind = simind.get("Isotope")
        if not isinstance(isotope_simind, str):
            _err(
                f"'phase_2.simind_stage.Isotope' must be a string, "
                f"got {type(isotope_simind).__name__}: {isotope_simind!r}"
            )

        for nf in (
            "NumProjections", "NumPhotons", "EnergyWindowWidth",
            "DetectorDistance", "DetectorWidth",
            "OutputImgSize", "OutputPixelWidth", "OutputSliceWidth",
        ):
            v = simind.get(nf)
            if v is not None:
                _check_number(v, f"phase_2.simind_stage.{nf}")

        num_photons = simind.get("NumPhotons")
        if (
            isinstance(num_photons, (int, float))
            and not isinstance(num_photons, bool)
            and num_photons <= 0
        ):
            _err(
                f"'phase_2.simind_stage.NumPhotons' must be positive, "
                f"got {num_photons}"
            )

        _check_xyz_dim(simind.get("xyz_dim"), "phase_2.simind_stage.xyz_dim")

    # OpenGATE dosimetry simulation (only when --dosimetry is set)
    if run_dosimetry:
        og = p2.get("opengate_stage", {})

        og_rois = og.get("roi_subset", [])
        if not isinstance(og_rois, list):
            _err("'phase_2.opengate_stage.roi_subset' must be a list")
        else:
            bad = [r for r in og_rois if r not in roi_subset]
            if bad:
                _err(
                    f"'phase_2.opengate_stage.roi_subset' contains ROIs not in "
                    f"'phase_1.segmentation_stage.roi_subset': {bad}. "
                    "Add them to segmentation roi_subset or remove them here."
                )

        gate = og.get("gate", {})
        for nf in ("total_histories", "num_threads"):
            v = gate.get(nf)
            if v is not None:
                _check_number(v, f"phase_2.opengate_stage.gate.{nf}")

        total_hist = gate.get("total_histories")
        if (
            isinstance(total_hist, (int, float))
            and not isinstance(total_hist, bool)
            and total_hist <= 0
        ):
            _err(
                f"'phase_2.opengate_stage.gate.total_histories' must be "
                f"positive, got {total_hist}"
            )

        _check_xyz_dim(og.get("xyz_dim"), "phase_2.opengate_stage.xyz_dim")

    # ── Phase 3 ────────────────────────────────────────────────────────────────

    # SPECT post-processing (only when --postprocess + --spect are both set)
    if run_postprocess and run_spect:
        p3 = config.get("phase_3", {})
        spect_pp = p3.get("spect_postprocess_stage", {})

        frame_starts = spect_pp.get("FrameStartTimes", [])
        frame_durs   = spect_pp.get("FrameDurations",   [])

        if not isinstance(frame_starts, list):
            _err(
                "'phase_3.spect_postprocess_stage.FrameStartTimes' must be "
                "a list of numbers"
            )
        if not isinstance(frame_durs, list):
            _err(
                "'phase_3.spect_postprocess_stage.FrameDurations' must be "
                "a list of numbers"
            )

        if isinstance(frame_starts, list) and isinstance(frame_durs, list):
            if len(frame_starts) != len(frame_durs):
                _err(
                    f"'FrameStartTimes' and 'FrameDurations' must have the "
                    f"same length (got {len(frame_starts)} vs {len(frame_durs)})"
                )
            for i, v in enumerate(frame_starts):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    _err(
                        f"'phase_3.spect_postprocess_stage.FrameStartTimes[{i}]' "
                        f"must be a number, got {type(v).__name__}: {v!r}"
                    )
            for i, v in enumerate(frame_durs):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    _err(
                        f"'phase_3.spect_postprocess_stage.FrameDurations[{i}]' "
                        f"must be a number, got {type(v).__name__}: {v!r}"
                    )

        for nf in ("Iterations", "Subsets"):
            v = spect_pp.get(nf)
            if v is not None:
                _check_number(v, f"phase_3.spect_postprocess_stage.{nf}")

    # ── Raise with full error summary ──────────────────────────────────────────

    if errors:
        raise ValueError(
            f"Config validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )
