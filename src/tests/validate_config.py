"""
Pre-flight configuration validator for the PyTheraTwin pipeline.

This module is called by ``main.py`` immediately after the config JSON is
parsed, before any CT is touched.  All problems are collected into a single
``ValueError`` so the user can fix everything in one pass instead of
discovering issues one run at a time.

Checks performed
----------------
- ``output_folder_title``      — non-empty string
- ``label_map_path``           — loaded from ``pipeline_paths.json`` and must exist
- ``roi_subset`` (seg.)        — non-empty list of recognised and enabled
                                 PyTheraTwin ROI names (defined in
                                 ``pipeline_roi_naming_map.json`` and curated by
                                 ``pipeline_options.json``)
- ``pbpk_tac_stage.isotope``   — string; must be in pipeline_options.json allowed list
- PBPK VOIs                   — loaded from the configured ROI metadata map
- Synthetic-lesion specs       — validated when ``phase_1.synthetic_lesions_stage.specs``
                                 is set; each ROI must be PBPK-compatible (non-null
                                 ``pbpk_voi`` in ``pipeline_roi_naming_map.json``)
- ``SIMINDDirectory``          — read from ``pipeline_paths.json`` input_paths; path must
                                 exist (``--spect`` only)
- SIMIND ``roi_subset``        — subset of segmentation ``roi_subset``; each ROI must be
                                 PBPK-compatible; synthetic-lesion hosts are all-or-none
                                 (``--spect`` only)
- SIMIND ``Collimator``        — string; must be in allowed list
- SIMIND ``isotope``           — string
- SIMIND ``NumProjections``    — must be ≥ 64 (fewer causes angular undersampling)
- SIMIND ``NumPhotons``        — must be positive
- SIMIND numeric fields        — ``EnergyWindowWidth``, ``DetectorDistance``, etc.
                                 must be numbers
- ``xyz_spacing_mm``           — null or a 3-element list ``[sxy, sxy, sz]`` of positive
                                 floats in mm; X and Y must be equal (square in-plane)
- OpenGATE ``roi_subset``      — subset of segmentation ``roi_subset``; each ROI must be
                                 PBPK-compatible; synthetic-lesion hosts are all-or-none
                                 (``--dosimetry`` only)
- OpenGATE adaptive gate       — ``total_histories_per_batch`` positive, ``num_threads``
                                 non-negative, ``target_uncertainty_percent`` in [0, 100]
- ``dosemap_postprocess_stage.apply_tac`` — must be true when ``--postprocess
                                 --dosimetry`` are both set (required for absolute dose)
- ``FrameStartTimes`` /
  ``FrameDurations``           — same length; all elements must be numbers
- SPECT reconstruction         — ``Iterations``, ``Subsets`` must be numbers;
                                 ``Subsets × Iterations`` must not exceed
                                 ``NumProjections``
- Cross-stage isotope          — ``pbpk_tac_stage.isotope`` must match
                                 ``simind_stage.isotope`` and
                                 ``opengate_stage.source.isotope``

Usage
-----
Import and call ``validate_config`` from your run script::

    from src.tests.validate_config import validate_config
    validate_config(config, run_spect=True, run_dosimetry=False, ...)
"""

from __future__ import annotations

import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Set

from json_minify import json_minify

from src.io.config_paths import get_pipeline_input_paths
from src.utils.label_utils import RESERVED_ROIS, SYNTHETIC_LESION_LABELS, TUMOR_CLASS_TO_LABEL, load_pbpk_compatible_rois, load_pipeline_roi_map


def _load_roi_map(repo_root: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load the pipeline ROI metadata map from pipeline_paths.json; returns None on failure."""
    path = get_pipeline_input_paths(repo_root).get("label_map_path")
    try:
        return load_pipeline_roi_map(path)
    except Exception:
        return None


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
        warnings.warn(
            f"Could not read pipeline_options.json at '{opts_path}'. "
            "Enum validation (collimator, isotope, reconstruction algorithm) will be skipped.",
            RuntimeWarning,
            stacklevel=2,
        )

    return allowed


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def validate_config(
    config: Dict[str, Any],
    run_spect: bool = False,
    run_dosimetry: bool = False,
    run_postprocess: bool = False,
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

    # Load ROI map for validation (supported ROI names + PBPK-compatible subset).
    # pipeline_options.json may further narrow the user-selectable ROI list.
    _roi_map = _load_roi_map(repo_root, config)
    _pytheratwin_allowed_rois: Set[str] = set(_roi_map.get("PyTheraTwin_allowed_rois", [])) if _roi_map else set()
    _pytheratwin_selectable_rois: Set[str] = {
        roi for roi in _allowed.get("roi_subset", []) if isinstance(roi, str)
    }
    _label_map_path = get_pipeline_input_paths(repo_root).get("label_map_path")
    try:
        _pbpk_compatible: Set[str] = load_pbpk_compatible_rois(_label_map_path)
    except Exception:
        _pbpk_compatible = set()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _err(msg: str) -> None:
        errors.append(msg)

    if _roi_map is None:
        _err(
            "Could not read pipeline ROI metadata map. Check "
            "'input_paths.label_map_path' in src/data/pipeline_paths.json."
        )
    elif _pytheratwin_selectable_rois:
        invalid_selectable_rois = sorted(_pytheratwin_selectable_rois - _pytheratwin_allowed_rois)
        if invalid_selectable_rois:
            _err(
                "src/data/pipeline_options.json roi_subset contains ROI(s) that are not "
                "defined in pipeline_roi_naming_map.json PyTheraTwin_allowed_rois: "
                f"{invalid_selectable_rois}"
            )

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

    def _check_xyz_spacing_mm(val: Any, field: str) -> None:
        """Validate xyz_spacing_mm: must be null or a [sxy, sxy, sz] list of positive floats."""
        if val is None:
            return
        if not isinstance(val, list) or len(val) != 3:
            _err(
                f"'{field}' must be null or a list of exactly 3 positive numbers "
                f"[sxy, sxy, sz] in mm, got: {val!r}"
            )
            return
        for i, v in enumerate(val):
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(float(v))
                or float(v) <= 0
            ):
                _err(
                    f"'{field}[{i}]' must be a positive number (mm), "
                    f"got {type(v).__name__}: {v!r}"
                )
        if (
            all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v)) for v in val)
            and abs(float(val[0]) - float(val[1])) > 1e-6
        ):
            _err(
                f"'{field}' must use the same value for X and Y spacing "
                f"(square in-plane voxels), got x={float(val[0])!r} mm and y={float(val[1])!r} mm"
            )

    def _is_finite_number(val: Any) -> bool:
        """Return True for finite ints/floats and False for bools or non-numbers."""
        return (
            not isinstance(val, bool)
            and isinstance(val, (int, float))
            and math.isfinite(float(val))
        )

    def _is_integer_like(val: Any) -> bool:
        """Return True for integer-valued numeric coordinates."""
        return _is_finite_number(val) and float(val).is_integer()

    def _check_synthetic_lesion_specs(specs: Any, allowed_rois: List[str]) -> None:
        """Validate per-ROI synthetic lesion specs against stage expectations."""
        if specs in (None, {}):
            return
        if not isinstance(specs, dict):
            _err(
                "'phase_1.synthetic_lesions_stage.specs' must be a dict keyed by ROI name"
            )
            return

        allowed_probabilities = {"uniform", "gaussian", "user_defined"}
        allowed_roi_names = {roi for roi in allowed_rois if isinstance(roi, str)}

        for roi_name, spec in specs.items():
            prefix = f"phase_1.synthetic_lesions_stage.specs.{roi_name}"

            if not isinstance(roi_name, str) or not roi_name.strip():
                _err(
                    f"ROI key {roi_name!r} in 'phase_1.synthetic_lesions_stage.specs' "
                    "must be a non-empty string"
                )
                continue
            if roi_name in SYNTHETIC_LESION_LABELS:
                _err(
                    f"'{prefix}' uses a reserved synthetic lesion label as a host ROI key — "
                    "use an actual organ ROI name (e.g. 'liver', 'spine')."
                )
            if roi_name not in allowed_roi_names:
                _err(
                    f"'{prefix}' is defined for ROI {roi_name!r}, but that ROI is not present in "
                    "'phase_1.segmentation_stage.roi_subset'"
                )

            if not isinstance(spec, dict):
                _err(f"'{prefix}' must be a dict")
                continue

            n_lesions = spec.get("n_lesions")
            if isinstance(n_lesions, bool) or not isinstance(n_lesions, int) or n_lesions <= 0:
                _err(f"'{prefix}.n_lesions' must be an integer > 0, got {n_lesions!r}")
                n_lesions = None

            prob = spec.get("prob", "uniform")
            if not isinstance(prob, str):
                _err(f"'{prefix}.prob' must be a string, got {type(prob).__name__}: {prob!r}")
                prob_l = ""
            else:
                prob_l = prob.lower()
                if prob_l not in allowed_probabilities:
                    _err(
                        f"'{prefix}.prob' must be one of {sorted(allowed_probabilities)}, got {prob!r}"
                    )

            sigma_mm = spec.get("sigma_mm")
            if prob_l == "gaussian":
                if sigma_mm is None:
                    _err(f"'{prefix}.sigma_mm' is required when prob='gaussian'")
                elif not _is_finite_number(sigma_mm) or float(sigma_mm) <= 0:
                    _err(f"'{prefix}.sigma_mm' must be a finite number > 0, got {sigma_mm!r}")
            elif sigma_mm is not None and (not _is_finite_number(sigma_mm) or float(sigma_mm) <= 0):
                _err(
                    f"'{prefix}.sigma_mm' must be a finite number > 0 when provided, got {sigma_mm!r}"
                )

            margin_mm = spec.get("margin_mm")
            if margin_mm is not None and (not _is_finite_number(margin_mm) or float(margin_mm) < 0):
                _err(f"'{prefix}.margin_mm' must be a finite number >= 0, got {margin_mm!r}")

            seed = spec.get("seed")
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
                _err(f"'{prefix}.seed' must be an integer when provided, got {seed!r}")

            radii_present = "radii_mm" in spec and spec.get("radii_mm") is not None
            radii_mm = spec.get("radii_mm")
            if radii_present:
                if not isinstance(radii_mm, list):
                    _err(f"'{prefix}.radii_mm' must be a list when provided")
                else:
                    if n_lesions is not None and len(radii_mm) != n_lesions:
                        _err(
                            f"'{prefix}.radii_mm' length must match n_lesions "
                            f"({n_lesions}), got {len(radii_mm)}"
                        )
                    for i, radius in enumerate(radii_mm):
                        if not _is_finite_number(radius) or float(radius) <= 0:
                            _err(
                                f"'{prefix}.radii_mm[{i}]' must be a finite number > 0, "
                                f"got {radius!r}"
                            )
            elif prob_l == "user_defined":
                _err(f"'{prefix}.radii_mm' is required when prob='user_defined'")

            centers = spec.get("user_centers_zyx")
            if prob_l == "user_defined":
                if not isinstance(centers, list):
                    _err(f"'{prefix}.user_centers_zyx' must be a list when prob='user_defined'")
                else:
                    if n_lesions is not None and len(centers) != n_lesions:
                        _err(
                            f"'{prefix}.user_centers_zyx' length must match n_lesions "
                            f"({n_lesions}), got {len(centers)}"
                        )
                    for i, center in enumerate(centers):
                        if not isinstance(center, (list, tuple)) or len(center) != 3:
                            _err(
                                f"'{prefix}.user_centers_zyx[{i}]' must be a [z, y, x] triplet, "
                                f"got {center!r}"
                            )
                            continue
                        for axis, coord in zip(("z", "y", "x"), center):
                            if not _is_integer_like(coord):
                                _err(
                                    f"'{prefix}.user_centers_zyx[{i}].{axis}' must be an integer-valued "
                                    f"coordinate, got {coord!r}"
                                )
            elif centers is not None:
                _err(
                    f"'{prefix}.user_centers_zyx' is only used when prob='user_defined'"
                )

            pbpk_label = spec.get("pbpk_label", "TumorRest")
            if not isinstance(pbpk_label, str) or pbpk_label not in TUMOR_CLASS_TO_LABEL:
                _err(
                    f"'{prefix}.pbpk_label' must be one of "
                    f"{sorted(TUMOR_CLASS_TO_LABEL)}, got {pbpk_label!r}"
                )

    def _check_no_overlapping_totseg_sources(rois: List[str], field: str) -> None:
        """Reject ROI selections that would paint the same TotalSegmentator label twice."""
        if not _roi_map:
            return
        pytheratwin_to_totseg = _roi_map.get("PyTheraTwin_to_totseg", {})
        seen: Dict[tuple[str, str], str] = {}
        overlaps: List[str] = []
        for roi_name in rois:
            entry = pytheratwin_to_totseg.get(roi_name)
            if not isinstance(entry, dict):
                continue
            task = str(entry.get("task", ""))
            if task in {"body", "synthetic"}:
                continue
            for source_name in entry.get("totseg_rois", []) or []:
                key = (task, str(source_name))
                if key in seen:
                    overlaps.append(
                        f"{roi_name!r} overlaps {seen[key]!r} on {task}:{source_name}"
                    )
                else:
                    seen[key] = roi_name
        if overlaps:
            _err(
                f"'{field}' contains overlapping ROI groups: {overlaps}. "
                "Choose either the grouped PBPK ROI (for simulation/lesions) or its "
                "individual anatomical child ROIs, not both."
            )

    def _check_simulation_roi_subset(stage_field: str, stage_rois: Any, lesion_rois: List[str]) -> None:
        """Validate SIMIND/OpenGATE user ROI subset contract."""
        if not isinstance(stage_rois, list):
            _err(f"'{stage_field}' must be a list")
            return
        if len(stage_rois) == 0:
            _err(f"'{stage_field}' must contain at least one PBPK-compatible ROI")
            return

        manual_internal = [r for r in stage_rois if r in RESERVED_ROIS]
        if manual_internal:
            _err(
                f"'{stage_field}' contains internal/reserved ROI(s) {manual_internal}. "
                "'remaining_body' and synthetic lesion labels are added internally; "
                "do not list them in config."
            )

        bad = [r for r in stage_rois if r not in roi_subset]
        if bad:
            _err(
                f"'{stage_field}' contains ROIs not in "
                f"'phase_1.segmentation_stage.roi_subset': {bad}. "
                "Add them to segmentation roi_subset or remove them here."
            )
        if _pbpk_compatible:
            not_pbpk = [r for r in stage_rois if r not in _pbpk_compatible]
            if not_pbpk:
                _err(
                    f"'{stage_field}' contains ROIs without a dedicated non-Rest "
                    f"PBPK VOI: {not_pbpk}. Simulation requires a distinct "
                    f"time-activity curve per ROI. Use a grouped PBPK ROI such as "
                    "'bone', 'gi_tract', or 'muscle' where applicable, or remove "
                    "the ROI from simulation."
                )
        # No all-or-none check for lesion host ROIs: synthetic lesion labels are always
        # added automatically to the simulation roi_subset when specs are present,
        # regardless of which host organ ROIs the user selects here.

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

    # label_map_path is developer-controlled; read from pipeline_paths.json.
    lmp = str(_label_map_path or "").strip()
    if not lmp:
        _err(
            "'input_paths.label_map_path' in src/data/pipeline_paths.json must "
            "point to the pipeline ROI metadata map"
        )
    elif not os.path.exists(lmp):
        _err(
            "'input_paths.label_map_path' in src/data/pipeline_paths.json does not exist: "
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
            elif _pytheratwin_allowed_rois and r not in _pytheratwin_allowed_rois:
                _err(
                    f"'phase_1.segmentation_stage.roi_subset[{i}]' value {r!r} is not "
                    f"a recognised PyTheraTwin ROI name. See src/data/pipeline_roi_naming_map.json "
                    f"PyTheraTwin_allowed_rois for the full list."
                )
            elif _pytheratwin_selectable_rois and r not in _pytheratwin_selectable_rois:
                parent = (
                    _roi_map.get("PyTheraTwin_to_totseg", {}).get(r, {}).get("parent_group")
                    if _roi_map else None
                )
                if parent:
                    _err(
                        f"'phase_1.segmentation_stage.roi_subset[{i}]' value {r!r} is a "
                        f"component of '{parent}', which is the user-selectable grouped ROI. "
                        f"Select '{parent}' to include all its structures as a unit, or add "
                        f"{r!r} to src/data/pipeline_options.json roi_subset to enable it individually."
                    )
                else:
                    _err(
                        f"'phase_1.segmentation_stage.roi_subset[{i}]' value {r!r} is defined "
                        "in the ROI map but is not enabled for user selection. Add it to "
                        "src/data/pipeline_options.json roi_subset, or remove it from this config."
                    )
        _check_no_overlapping_totseg_sources(roi_subset, "phase_1.segmentation_stage.roi_subset")
    # Guard downstream cross-checks if roi_subset is malformed
    if not isinstance(roi_subset, list):
        roi_subset = []

    # PBPK stage
    pbpk = p1.get("pbpk_tac_stage", {})

    isotope_p1 = pbpk.get("isotope")
    if not isinstance(isotope_p1, str):
        _err(
            "'phase_1.pbpk_tac_stage.isotope' must be a string, "
            f"got {type(isotope_p1).__name__}: {isotope_p1!r}"
        )
    elif _allowed.get("isotope") and isotope_p1 not in _allowed["isotope"]:
        _err(
            f"'phase_1.pbpk_tac_stage.isotope' value {isotope_p1!r} is not "
            f"in the allowed list: {_allowed['isotope']}"
        )

    # PBPK VOIs are developer-controlled in the configured ROI metadata map.
    # A legacy config may still contain pbpk_tac_stage.VOIs, but it is ignored.

    # Synthetic lesions stage — validated whenever specs are present in config
    sl = p1.get("synthetic_lesions_stage", {})
    lesion_rois: List[str] = []
    if sl.get("specs"):
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
        _check_synthetic_lesion_specs(sl.get("specs"), roi_subset)
        lesion_rois = [r for r in sl["specs"] if isinstance(r, str)]

    # ── Phase 2 ────────────────────────────────────────────────────────────────

    p2 = config.get("phase_2", {})

    # SIMIND SPECT simulation (only when --spect is set)
    if run_spect:
        simind = p2.get("simind_stage", {})

        # SIMINDDirectory is set once in src/data/pipeline_paths.json under
        # "input_paths.SIMINDDirectory"; fall back to the config field if absent.
        simind_dir = (
            get_pipeline_input_paths(repo_root).get("SIMINDDirectory", "")
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
                "(set in src/data/pipeline_paths.json → input_paths.SIMINDDirectory)"
            )

        # roi_subset must be a subset of the segmentation roi_subset AND PBPK-compatible.
        # Synthetic lesion source labels are resolved internally from lesion host ROIs.
        simind_rois = simind.get("roi_subset", [])
        _check_simulation_roi_subset("phase_2.simind_stage.roi_subset", simind_rois, lesion_rois)

        collimator = simind.get("Collimator")
        if not isinstance(collimator, str):
            _err(
                "'phase_2.simind_stage.Collimator' must be a string, "
                f"got {type(collimator).__name__}: {collimator!r}"
            )
        elif _allowed.get("collimator") and collimator not in _allowed["collimator"]:
            _err(
                f"'phase_2.simind_stage.Collimator' value {collimator!r} is "
                f"not in the allowed list: {_allowed['collimator']}"
            )

        isotope_simind = simind.get("isotope")
        if not isinstance(isotope_simind, str):
            _err(
                "'phase_2.simind_stage.isotope' must be a string, "
                f"got {type(isotope_simind).__name__}: {isotope_simind!r}"
            )

        for nf in (
            "NumProjections", "NumPhotons", "EnergyWindowWidth",
            "DetectorDistance", "DetectorWidth",
            "OutputImgSize", "OutputPixelWidth", "OutputSliceWidth",
            "num_cpu",
        ):
            v = simind.get(nf)
            if v is not None:
                _check_number(v, f"phase_2.simind_stage.{nf}")

        for _pw_field in ("OutputPixelWidth", "OutputSliceWidth"):
            _pw_val = simind.get(_pw_field)
            if _is_finite_number(_pw_val):
                if float(_pw_val) <= 0:
                    _err(
                        f"'phase_2.simind_stage.{_pw_field}' must be > 0 mm, "
                        f"got {_pw_val}"
                    )
                elif float(_pw_val) > 50:
                    _err(
                        f"'phase_2.simind_stage.{_pw_field}' is {_pw_val}, which exceeds 50 mm. "
                        "This field is in mm — did you accidentally enter a value in cm?"
                    )

        simind_num_cpu = simind.get("num_cpu")
        if (
            isinstance(simind_num_cpu, (int, float))
            and not isinstance(simind_num_cpu, bool)
            and simind_num_cpu < 0
        ):
            _err(
                "'phase_2.simind_stage.num_cpu' must be >= 0 (0 = use all available), "
                f"got {simind_num_cpu}"
            )

        num_photons = simind.get("NumPhotons")
        if (
            isinstance(num_photons, (int, float))
            and not isinstance(num_photons, bool)
            and num_photons <= 0
        ):
            _err(
                "'phase_2.simind_stage.NumPhotons' must be positive, "
                f"got {num_photons}"
            )

        num_proj = simind.get("NumProjections")
        if (
            isinstance(num_proj, (int, float))
            and not isinstance(num_proj, bool)
            and num_proj < 64
        ):
            _err(
                f"'phase_2.simind_stage.NumProjections' is {int(num_proj)}, "
                "but the minimum for valid OSEM reconstruction is 64. "
                "Fewer projections causes severe angular undersampling and streak "
                "artifacts. Rule of thumb: Subsets × Iterations must not exceed "
                "NumProjections, and NumProjections must be >= 64."
            )

        _check_xyz_spacing_mm(simind.get("xyz_spacing_mm"), "phase_2.simind_stage.xyz_spacing_mm")

    # OpenGATE dosimetry simulation (only when --dosimetry is set)
    if run_dosimetry:
        og = p2.get("opengate_stage", {})

        og_rois = og.get("roi_subset", [])
        _check_simulation_roi_subset("phase_2.opengate_stage.roi_subset", og_rois, lesion_rois)

        gate = og.get("gate", {})
        deprecated_gate_fields = {
            "total_histories": "total_histories_per_batch",
            "num_cpu": "num_threads",
            "random_seed": None,
        }
        for old_field, new_field in deprecated_gate_fields.items():
            if old_field in gate:
                if new_field is None:
                    _err(
                        f"'phase_2.opengate_stage.gate.{old_field}' is no longer user-configurable; "
                        "OpenGATE adaptive batches derive deterministic independent seeds internally."
                    )
                else:
                    _err(
                        f"'phase_2.opengate_stage.gate.{old_field}' has been replaced by "
                        f"'phase_2.opengate_stage.gate.{new_field}'"
                    )

        for nf in ("total_histories_per_batch", "num_threads", "target_uncertainty_percent"):
            v = gate.get(nf)
            if v is not None:
                _check_number(v, f"phase_2.opengate_stage.gate.{nf}")

        total_hist_batch = gate.get("total_histories_per_batch")
        if (
            isinstance(total_hist_batch, (int, float))
            and not isinstance(total_hist_batch, bool)
        ):
            if not float(total_hist_batch).is_integer() or total_hist_batch <= 0:
                _err(
                    "'phase_2.opengate_stage.gate.total_histories_per_batch' must be "
                    f"a positive whole number, got {total_hist_batch}"
                )

        og_num_threads = gate.get("num_threads")
        if (
            isinstance(og_num_threads, (int, float))
            and not isinstance(og_num_threads, bool)
        ):
            if not float(og_num_threads).is_integer() or og_num_threads < 0:
                _err(
                    "'phase_2.opengate_stage.gate.num_threads' must be a whole number >= 0 "
                    f"(0 = use all available), got {og_num_threads}"
                )

        target_unc = gate.get("target_uncertainty_percent")
        if (
            isinstance(target_unc, (int, float))
            and not isinstance(target_unc, bool)
            and not 0 <= float(target_unc) <= 100
        ):
            _err(
                "'phase_2.opengate_stage.gate.target_uncertainty_percent' must be between "
                f"0 and 100, got {target_unc}"
            )

        _check_xyz_spacing_mm(og.get("xyz_spacing_mm"), "phase_2.opengate_stage.xyz_spacing_mm")

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
                    "'FrameStartTimes' and 'FrameDurations' must have the "
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

        # Cross-check: Subsets × Iterations must not exceed NumProjections.
        if run_spect:
            iters   = spect_pp.get("Iterations")
            subsets = spect_pp.get("Subsets")
            num_proj_pp = p2.get("simind_stage", {}).get("NumProjections")
            if (
                isinstance(iters, (int, float)) and not isinstance(iters, bool)
                and isinstance(subsets, (int, float)) and not isinstance(subsets, bool)
                and isinstance(num_proj_pp, (int, float)) and not isinstance(num_proj_pp, bool)
                and subsets * iters > num_proj_pp
            ):
                _err(
                    f"Subsets × Iterations ({int(subsets)} × {int(iters)} = "
                    f"{int(subsets * iters)}) exceeds NumProjections "
                    f"({int(num_proj_pp)}). This over-updates the reconstruction "
                    "relative to available angular data and amplifies artifacts. "
                    "Reduce Subsets or Iterations, or increase NumProjections, so "
                    "that Subsets × Iterations ≤ NumProjections."
                )

    # Dosemap post-processing: apply_tac must be True when --postprocess + --dosimetry
    if run_postprocess and run_dosimetry:
        p3 = config.get("phase_3", {})
        dose_pp = p3.get("dosemap_postprocess_stage", {})
        if not dose_pp.get("apply_tac", True):
            _err(
                "'phase_3.dosemap_postprocess_stage.apply_tac' must be true when "
                "--postprocess and --dosimetry are both enabled. TAC weighting is "
                "required to convert OpenGATE Gy/decay outputs to absolute dose (Gy). "
                "There is no other post-processing applied to the dose map."
            )

    # ── Cross-stage isotope consistency ───────────────────────────────────────

    pbpk_iso = pbpk.get("isotope")
    if isinstance(pbpk_iso, str):
        if run_spect:
            simind_iso = p2.get("simind_stage", {}).get("isotope", "")
            if isinstance(simind_iso, str) and pbpk_iso.lower() != simind_iso.lower():
                _err(
                    f"Isotope mismatch: 'phase_1.pbpk_tac_stage.isotope' ({pbpk_iso!r}) must "
                    f"match 'phase_2.simind_stage.isotope' ({simind_iso!r})"
                )
        if run_dosimetry:
            og_iso = p2.get("opengate_stage", {}).get("source", {}).get("isotope", "lu177")
            if isinstance(og_iso, str) and pbpk_iso.lower() != og_iso.lower():
                _err(
                    f"Isotope mismatch: 'phase_1.pbpk_tac_stage.isotope' ({pbpk_iso!r}) must "
                    f"match 'phase_2.opengate_stage.source.isotope' ({og_iso!r})"
                )

    # ── Raise with full error summary ──────────────────────────────────────────

    if errors:
        raise ValueError(
            f"Config validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )
