"""
PBPK TAC generation (PyCNO PSMA model) for the PyTheraTwin pipeline.

Generates time-activity curves (TACs) for all segmented ROIs and writes
them to disk for downstream simulation and post-processing stages.

- Validate PBPK configuration inputs and CT identity.
- Optionally randomize kidney/salivary-gland parameters (lognormal sampling).
- Optionally extract patient height/weight from a DICOM directory.
- Run the PyCNO PSMA model to obtain TACs (time, tacs).
- Map each ROI in the downstream subset to its VOI and store TAC data.
- Save TACs as JSON (human-readable) + npz (full-resolution arrays).

Stop time behaviour
-------------------
The PBPK simulation stop time is derived from the isotope half-life. The TAC
is simulated for PBPK_HALF_LIFE_MULTIPLIER x the half-life (default 10x),
which captures effectively all decay. If any SPECT frame time extends beyond
this, the stop time is extended to cover it.

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.subdir_paths["phase_1"] : str
- context.ct_input_path : str  (NIfTI path or DICOM directory for height/weight extraction)
- context.config["phase_1"]["pbpk_tac_stage"] with keys:
    - "file_prefix" : str
    - "model_type" : str  (e.g. "PSMA")
    - "isotope" : str  (e.g. "lu177")
    - "Randomization_Kidney_SG_Para" : bool
- context.downstream_roi_subset : list[str]

On success, this stage sets:
- context.pbpk_tacs_by_organ : dict[str, dict]  {roi_name: {time, values, voi_name, ...}}
- context.pbpk_tac_json_path : str
- context.pbpk_tac_npz_path : str
- context.pbpk_tac_time : np.ndarray
- context.pbpk_tac_values : dict[str, np.ndarray]  {roi_name: full TAC array}
- context.pbpk_height_m : Optional[float]
- context.pbpk_weight_kg : Optional[float]
- context.pbpk_parameters : Optional[dict[str, float]]
- context.pbpk_vois : list[str]
- context.pbpk_isotope : str
- context.pbpk_stop_time_min : float
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pycno

from src.io.config_paths import get_label_map_path
from src.io.rerun_guard import (
    assert_stage_rerun_safe,
    build_stage_metadata,
    build_pbpk_rerun_snapshot,
    fingerprint_optional_file,
    stage_metadata_path,
    write_json,
)
from src.utils.dicom_utils import extract_height_weight
from src.utils.label_utils import load_isotope_config, load_pbpk_observables, roi_to_voi


class PbpkTacStage:
    """
    Run PBPK (PyCNO PSMA) and generate time-activity curves for all segmented ROIs.

    This stage only generates and saves TACs. The application of TACs to
    SIMIND projections or dose maps is done in the respective post-processing stages.
    """

    def __init__(self, context: Any) -> None:
        context.require(
            "subdir_paths",
            "config",
            "ct_input_path",
            "downstream_roi_subset",
            "output_folder_path",
            "ct_input_identity",
        )
        self.context = context
        self.debug: bool = getattr(context, "mode", "").upper() == "DEBUG"

        self.phase_output_dir: str = context.subdir_paths["phase_1"]
        self.stage_cfg: Dict[str, Any] = context.config["phase_1"]["pbpk_tac_stage"]
        self.output_dir: str = os.path.join(
            self.phase_output_dir,
            self.stage_cfg.get("sub_dir_name", "pbpk_tac_stage"),
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.work_dir: str = os.path.join(self.output_dir, "work_dir")
        os.makedirs(self.work_dir, exist_ok=True)
        self.metadata_path: str = stage_metadata_path(context.output_folder_path, "pbpk_tac_stage")

        self.ct_input_path: str = context.ct_input_path
        self.ct_input_identity: Dict[str, Any] = context.ct_input_identity

        self.prefix: str = self.stage_cfg.get("file_prefix", self.stage_cfg.get("name", "pbpk"))
        self.model_type: str = self.stage_cfg.get("model_type", "PSMA")
        self.label_map_path: str = get_label_map_path()
        self.vois_pbpk: List[str] = load_pbpk_observables(self.label_map_path)
        self.randomize_kidney_sg_para: bool = self.stage_cfg["Randomization_Kidney_SG_Para"]

        # Isotope configuration — loaded from src/data/isotope_config.json
        raw_isotope = str(self.stage_cfg.get("isotope", "lu177")).lower().strip().replace("-", "")
        _iso_cfg = load_isotope_config()
        if raw_isotope not in _iso_cfg["half_life_min"]:
            raise ValueError(
                f"Unsupported isotope '{raw_isotope}' in pbpk_tac_stage config. "
                f"Supported: {sorted(_iso_cfg['half_life_min'])}"
            )
        self.isotope: str = raw_isotope
        self.half_life_min: float = _iso_cfg["half_life_min"][raw_isotope]
        self._half_life_multiplier: int = _iso_cfg["half_life_multiplier"]

        self.downstream_roi_subset: List[str] = list(context.downstream_roi_subset)
        if "remaining_body" not in self.downstream_roi_subset:
            self.downstream_roi_subset.insert(0, "remaining_body")

        self.height: Optional[float] = None
        self.weight: Optional[float] = None
        self.parameters: Dict[str, float] = {}
        self.roi_to_voi_used: Dict[str, str] = {}

        self.tac_json_path: str = os.path.join(self.output_dir, f"{self.prefix}_tacs.json")
        self.tac_npz_path: str = os.path.join(self.output_dir, f"{self.prefix}_tacs.npz")

    def _rerun_config_snapshot(self) -> Dict[str, Any]:
        """Return the PBPK settings that must match for cached TACs to remain valid."""
        return build_pbpk_rerun_snapshot(
            self.context.config,
            synthetic_enabled=bool(getattr(self.context, "synthetic_lesions_enabled", False)),
        )

    def _current_dependency_fingerprints(self) -> Dict[str, Any]:
        """Return developer metadata fingerprints that affect generated TACs."""
        return {
            "label_map_json": fingerprint_optional_file(self.label_map_path),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sample_lognormal_from_mean_sd(self, mean: float, sd: float) -> float:
        """
        Sample from a lognormal distribution parameterized by desired mean and SD.

        If X ~ LogNormal(mu, sigma^2):
          sigma^2 = ln(1 + (sd/mean)^2)
          mu      = ln(mean) - sigma^2/2

        Parameters
        ----------
        mean, sd : float  (both must be > 0)
        """
        mean, sd = float(mean), float(sd)
        if mean <= 0 or sd <= 0:
            raise ValueError(f"mean and sd must be > 0 (got mean={mean}, sd={sd})")
        sigma2 = np.log(1.0 + (sd * sd) / (mean * mean))
        mu = np.log(mean) - 0.5 * sigma2
        return float(np.random.lognormal(mean=mu, sigma=np.sqrt(sigma2)))

    def _parameter_check(self) -> Dict[str, float]:
        """
        Validate PBPK config inputs and build the PyCNO parameter override dict.

        Returns
        -------
        dict[str, float]  parameter overrides (may be empty if no overrides apply)

        Raises
        ------
        AttributeError, ValueError
        """
        if not os.path.exists(self.ct_input_path):
            raise ValueError(f"CT input path does not exist: {self.ct_input_path}")
        if not isinstance(self.vois_pbpk, (list, tuple)) or len(self.vois_pbpk) == 0:
            raise ValueError("PBPK observables must be a non-empty list/tuple")

        if not isinstance(self.randomize_kidney_sg_para, bool):
            raise ValueError("Randomization_Kidney_SG_Para must be True or False")

        parameters: Dict[str, float] = {}

        vois_set = {str(v).strip().lower() for v in (self.vois_pbpk or [])}
        has_kidney = "kidney" in vois_set
        has_sg = "sg" in vois_set or any("salivary" in v for v in vois_set)

        if self.randomize_kidney_sg_para:
            if has_sg:
                parameters["Rden_SG"] = self._sample_lognormal_from_mean_sd(60.0, 20.0)
                parameters["lambdaRel_SG"] = self._sample_lognormal_from_mean_sd(3.9e-4, 0.63e-4)
            if has_kidney:
                parameters["Rden_Kidney"] = self._sample_lognormal_from_mean_sd(30.0, 10.0)
                parameters["lambdaRel_Kidney"] = self._sample_lognormal_from_mean_sd(2.88e-4, 0.55e-4)

        # Config-supplied values take priority; fall back to DICOM extraction.
        cfg_height = self.stage_cfg.get("height_m") or None
        cfg_weight = self.stage_cfg.get("weight_kg") or None
        try:
            cfg_height = float(cfg_height) if cfg_height is not None else None
            if cfg_height is not None and cfg_height <= 0:
                cfg_height = None
        except (TypeError, ValueError):
            cfg_height = None
        try:
            cfg_weight = float(cfg_weight) if cfg_weight is not None else None
            if cfg_weight is not None and cfg_weight <= 0:
                cfg_weight = None
        except (TypeError, ValueError):
            cfg_weight = None

        self.height = cfg_height
        self.weight = cfg_weight
        if (self.height is None or self.weight is None) and os.path.isdir(self.ct_input_path):
            dicom_h, dicom_w = extract_height_weight(self.ct_input_path)
            if self.height is None:
                self.height = dicom_h
            if self.weight is None:
                self.weight = dicom_w

        if self.height is not None:
            parameters["bodyHeight"] = float(self.height)
        if self.weight is not None:
            parameters["bodyWeight"] = float(self.weight)

        return parameters

    def _compute_stop_time(self) -> float:
        """
        Derive PBPK simulation stop time from isotope half-life.

        The base stop time is PBPK_HALF_LIFE_MULTIPLIER x the isotope half-life,
        which captures effectively all decays (>99.9% at 10x). If any SPECT frame
        time extends beyond this, the stop time is increased to cover it.

        Returns
        -------
        float  stop time in minutes
        """
        stop = self.half_life_min * self._half_life_multiplier

        # Check if SPECT frame times require a longer TAC
        cfg = self.context.config
        try:
            spect = cfg["phase_3"]["spect_postprocess_stage"]
            starts = spect.get("FrameStartTimes", [])
            durations = spect.get("FrameDurations", [])
            for s, d in zip(starts, durations):
                t_end = float(s) + float(d) / 60.0
                if t_end > stop:
                    stop = t_end
        except (KeyError, TypeError):
            pass

        if self.debug:
            print(
                f"[PbpkTacStage] Isotope: {self.isotope} | "
                f"half-life: {self.half_life_min:.1f} min ({self.half_life_min / 60.0 / 24.0:.2f} days) | "
                f"stop time: {stop:.0f} min ({stop / 60.0:.1f} hr, {stop / 60.0 / 24.0:.1f} days)"
            )

        return stop

    def _run_psma_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate parameters, run the PyCNO PSMA PBPK model, and return (time, tacs).

        Returns
        -------
        time : np.ndarray  shape (T,), time in minutes
        tacs : np.ndarray  typically shape (1, T, n_vois) from PyCNO
        """
        self.parameters = self._parameter_check()

        self.context.pbpk_height_m = self.height
        self.context.pbpk_weight_kg = self.weight
        self.context.pbpk_parameters = dict(self.parameters) if self.parameters else None

        model = pycno.Model(
            model_name=self.model_type,
            hotamount=10,
            coldamount=100,
            parameters=self.parameters,
        )

        stop = self._compute_stop_time()
        self._stop_time = stop
        steps = int(np.ceil(stop)) if stop > 0 else 1
        time, tacs = model.simulate(stop=stop, steps=steps, observables=self.vois_pbpk)
        return np.asarray(time, dtype=float), np.asarray(tacs, dtype=float)

    def _extract_tac_for_voi(self, tacs: np.ndarray, voi_index: int) -> np.ndarray:
        """Extract one VOI TAC from PyCNO output (handles 2D and 3D array shapes)."""
        if tacs.ndim == 3:
            return np.asarray(tacs[0, :, voi_index], dtype=float)
        if tacs.ndim == 2:
            return np.asarray(tacs[:, voi_index], dtype=float)
        raise ValueError(f"Unexpected TAC array shape from PyCNO: {tacs.shape}")

    def _save_tacs_json_npz(
        self,
        time: np.ndarray,
        tac_data: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, str]:
        """
        Save TAC data as JSON (human-readable) and npz (full-resolution arrays).

        JSON contains: per-ROI metadata, VOI mapping, sampled values.
        npz contains: time array and per-ROI full TAC arrays.

        Returns
        -------
        (json_path, npz_path)
        """
        json_data: Dict[str, Any] = {
            "model_type": self.model_type,
            "isotope": self.isotope,
            "half_life_min": self.half_life_min,
            "stop_time_min": float(time[-1]),
            "pbpk_height_m": None if self.height is None else float(self.height),
            "pbpk_weight_kg": None if self.weight is None else float(self.weight),
            "pbpk_parameters": {k: float(v) for k, v in self.parameters.items()},
            "vois_pbpk": list(self.vois_pbpk),
            "roi_to_voi_used": dict(self.roi_to_voi_used),
            "time_points": len(time),
            "time_min_range": [float(time[0]), float(time[-1])],
            "tacs": {},
        }
        npz_arrays: Dict[str, np.ndarray] = {"time": np.asarray(time, dtype=np.float32)}

        for roi_name, roi_data in tac_data.items():
            json_data["tacs"][roi_name] = {
                "voi_name": roi_data["voi_name"],
                "voi_index": roi_data["voi_index"],
                "n_time_points": len(roi_data["tac_full"]),
            }
            npz_arrays[f"tac_{roi_name}"] = np.asarray(roi_data["tac_full"], dtype=np.float32)

        with open(self.tac_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)

        np.savez_compressed(self.tac_npz_path, **npz_arrays)

        return self.tac_json_path, self.tac_npz_path

    def _save_stage_metadata(
        self,
        tac_data: Dict[str, Dict[str, Any]],
    ) -> None:
        """Save PBPK TAC stage metadata for debugging / provenance."""
        outputs: Dict[str, Any] = {
            "tac_json_path": self.tac_json_path,
            "tac_npz_path": self.tac_npz_path,
        }
        extra: Dict[str, Any] = {
            "stage": "pbpk_tac_stage",
            "phase_output_dir": self.phase_output_dir,
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "file_prefix": self.prefix,
            "model_type": self.model_type,
            "isotope": self.isotope,
            "half_life_min": self.half_life_min,
            "half_life_multiplier": self._half_life_multiplier,
            "ct_input_path": self.ct_input_path,
            "vois_pbpk": list(self.vois_pbpk),
            "randomize_kidney_sg_para": bool(self.randomize_kidney_sg_para),
            "pbpk_height_m": None if self.height is None else float(self.height),
            "pbpk_weight_kg": None if self.weight is None else float(self.weight),
            "pbpk_parameters": {k: float(v) for k, v in self.parameters.items()},
            "roi_to_voi_used": dict(self.roi_to_voi_used),
            "tac_json_path": self.tac_json_path,
            "tac_npz_path": self.tac_npz_path,
            "downstream_roi_subset": list(self.downstream_roi_subset),
        }
        metadata = build_stage_metadata(
            stage_name="pbpk_tac_stage",
            config_snapshot=self._rerun_config_snapshot(),
            ct_identity=self.ct_input_identity,
            upstream_fingerprints=self._current_dependency_fingerprints(),
            outputs=outputs,
            extra=extra,
        )
        write_json(self.metadata_path, metadata)

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------
    def run(self) -> Any:
        """
        Execute PBPK TAC generation and save TACs for all segmented ROIs.

        Returns
        -------
        context : Context-like
        """
        assert_stage_rerun_safe(
            stage_name="pbpk_tac_stage",
            metadata_path=self.metadata_path,
            required_outputs=[self.tac_json_path, self.tac_npz_path],
            current_config_snapshot=self._rerun_config_snapshot(),
            current_ct_identity=self.ct_input_identity,
            current_upstream_fingerprints=self._current_dependency_fingerprints(),
            context=self.context,
        )

        if os.path.exists(self.tac_json_path) and os.path.exists(self.tac_npz_path):
            if self.debug:
                print("[PbpkTacStage] TAC outputs already exist, loading from disk.")
            with open(self.tac_json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            npz_data = np.load(self.tac_npz_path)
            time_arr = npz_data["time"]
            tac_data: Dict[str, Dict[str, Any]] = {}
            tac_values: Dict[str, np.ndarray] = {}
            for roi_name, roi_info in json_data["tacs"].items():
                tac_key = f"tac_{roi_name}"
                tac_arr = npz_data[tac_key] if tac_key in npz_data else None
                tac_data[roi_name] = {
                    "voi_name": roi_info["voi_name"],
                    "voi_index": roi_info["voi_index"],
                    "tac_full": tac_arr,
                }
                if tac_arr is not None:
                    tac_values[roi_name] = tac_arr
            self.roi_to_voi_used = json_data.get("roi_to_voi_used", {})
            self.context.pbpk_tacs_by_organ = tac_data
            self.context.pbpk_tac_json_path = self.tac_json_path
            self.context.pbpk_tac_npz_path = self.tac_npz_path
            self.context.pbpk_tac_time = time_arr
            self.context.pbpk_tac_values = tac_values
            self.context.pbpk_height_m = json_data.get("pbpk_height_m")
            self.context.pbpk_weight_kg = json_data.get("pbpk_weight_kg")
            self.context.pbpk_parameters = json_data.get("pbpk_parameters")
            self.context.pbpk_vois = json_data.get("vois_pbpk", [])
            self.context.pbpk_isotope = json_data.get("isotope", self.isotope)
            self.context.pbpk_stop_time_min = json_data.get("stop_time_min", float(time_arr[-1]))
            return self.context

        time, tacs = self._run_psma_model()

        tac_data: Dict[str, Dict[str, Any]] = {}
        tac_values: Dict[str, np.ndarray] = {}

        for roi_name in self.downstream_roi_subset:
            voi_name = roi_to_voi(roi_name, self.label_map_path)

            if voi_name is None or voi_name not in self.vois_pbpk:
                if "Rest" in self.vois_pbpk:
                    voi_name = "Rest"
                else:
                    raise ValueError(
                        f"No VOI mapping for ROI '{roi_name}' and no 'Rest' VOI in PBPK model "
                        f"(available: {sorted(self.vois_pbpk)})."
                    )

            self.roi_to_voi_used[roi_name] = voi_name
            voi_index = self.vois_pbpk.index(voi_name)
            tac_voi = self._extract_tac_for_voi(tacs, voi_index)

            tac_data[roi_name] = {
                "voi_name": voi_name,
                "voi_index": voi_index,
                "tac_full": tac_voi,
            }
            tac_values[roi_name] = tac_voi

        self._save_tacs_json_npz(time, tac_data)
        self._save_stage_metadata(tac_data)

        self.context.pbpk_tacs_by_organ = tac_data
        self.context.pbpk_tac_json_path = self.tac_json_path
        self.context.pbpk_tac_npz_path = self.tac_npz_path
        self.context.pbpk_tac_time = time
        self.context.pbpk_tac_values = tac_values
        self.context.pbpk_height_m = self.height
        self.context.pbpk_weight_kg = self.weight
        self.context.pbpk_parameters = dict(self.parameters)
        self.context.pbpk_vois = list(self.vois_pbpk)
        self.context.pbpk_isotope = self.isotope
        self.context.pbpk_stop_time_min = float(time[-1])

        self.context.extras["pbpk_tac_stage"] = {
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "tac_json_path": self.tac_json_path,
            "tac_npz_path": self.tac_npz_path,
            "metadata_path": self.metadata_path,
        }

        return self.context
