"""
Command-line entry point for the Virtual Theranostic Trials pipeline.

This module defines the per-patient pipeline runner, injects developer-controlled
paths into the live config, and provides the batch CLI used by both local runs
and the web UI subprocess launcher.
"""

from __future__ import annotations

import os
import json
from json_minify import json_minify
import logging
import time
import shutil
import argparse
import copy
from typing import Any, Dict, List, Literal, Optional

from src.io.config_paths import inject_pipeline_paths
from src.io.context import Context
from src.io.profiler import StageProfiler
from src.io.rerun_guard import ensure_ct_matches_saved_copy, ensure_metadata_dir
from src.tests.validate_config import validate_config

from src.stages.segmentation_stage import SegmentationStage
from src.stages.synthetic_lesions_stage import SyntheticLesionsStage
from src.stages.pbpk_tac_stage import PbpkTacStage
from src.stages.simind_simulation_stage import SimindSimulationStage
from src.stages.spect_postprocess_stage import SpectPostprocessStage
from src.stages.opengate_simulation_stage import OpenGateSimulationStage
from src.stages.dosemap_postprocess_stage import DosemapPostprocessStage
from src.utils.dicom_utils import extract_height_weight


CTInputType = Literal["nii", "dicom"]

# ANSI colour codes (DEBUG terminal output only)
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _debug_print(msg: str, phase: str = "", logger: Optional[logging.Logger] = None) -> None:
    """Print a colored debug message to terminal and optionally log it.""" 
    colored = f"{_GREEN}[DEBUG]{_RESET} {_CYAN}[{phase}]{_RESET} {msg}" if phase else f"{_GREEN}[DEBUG]{_RESET} {msg}" 
    print(colored) 
    if logger: 
        logger.debug(f"[{phase}] {msg}" if phase else msg) 


class TdtPipeline:
    """
    Orchestrates the full VTT pipeline for a single CT input.

    Parameters
    ----------
    config_path : str
        Path to the JSON config file (comments allowed via `json_minify`).
    ct_input : str
        Path to a CT input (either a .nii/.nii.gz file OR a DICOM directory).
    ct_index : int
        Index used for naming (e.g., output folder suffix "_CT_{ct_index}").
    logging_on : bool, default=True
        If True, writes a per-CT log file into the CT output folder.
        The log records launched_via, config path, output folder, and timing.
    save_config : bool, default=False
        If True, saves a copy of the config JSON into the output folder.
    mode : {"DEBUG", "PRODUCTION"}, default="PRODUCTION"
        Controls verbosity and whether intermediate files are cleaned up.
    run_spect : bool, default=False
        If True, runs SIMIND SPECT simulation in Phase 2.
    run_dosimetry : bool, default=False
        If True, runs OpenGATE dosimetry simulation in Phase 2.
    run_postprocess : bool, default=False
        If True, runs post-processing in Phase 3 for whichever simulations ran.
    profile : bool, default=False
        If True, samples CPU % and RAM every 2 s per stage and writes
        ``profiling_CT_<ct_index>.json`` into the output folder.
    """

    def __init__(
        self,
        config_path: str,
        ct_input: str,
        ct_index: int,
        logging_on: bool = True,
        save_config: bool = False,
        mode: Literal["DEBUG", "PRODUCTION"] = "PRODUCTION",
        run_spect: bool = False,
        run_dosimetry: bool = False,
        run_postprocess: bool = False,
        launched_via: str = "cli",
        startup_banner_lines: Optional[List[str]] = None,
        profile: bool = False,
    ) -> None:
        self.config_path: str = config_path
        self.ct_input: str = ct_input
        self.ct_index: int = ct_index
        self.current_dir_path: str = os.path.abspath(os.path.dirname(__file__))

        self.logging_on: bool = logging_on
        self.save_config: bool = save_config
        self.launched_via: str = launched_via
        self.mode: Literal["DEBUG", "PRODUCTION"] = mode
        self.run_spect: bool = run_spect
        self.run_dosimetry: bool = run_dosimetry
        self.run_postprocess: bool = run_postprocess
        self.startup_banner_lines: List[str] = startup_banner_lines or []
        self.profile: bool = profile

        self.config: Dict[str, Any] = {}
        self.output_folder_path: str = ""
        self.metadata_dir_path: str = ""
        self.ct_input_type: CTInputType = "dicom"
        self.ct_saved_copy_path: str = ""
        self.ct_input_identity: Dict[str, Any] = {}
        self.run_synthetic_lesions: bool = False
        self.sub_dir_names: Dict[str, str] = {}

        self.logger: logging.Logger = logging.getLogger(f"VTT_PIPELINE_CT_{self.ct_index}")
        self.logger.setLevel(logging.DEBUG if self.mode == "DEBUG" else logging.INFO)
        self.logger.propagate = False

        self._config_setup(config_path)

        if self.logging_on:
            self._log_setup()
        else:
            self.logger.disabled = True

        self.context: Context
        self._context_setup()

        if not self.logging_on:
            self.context._log_enabled = False

    def _save_config_copy(self, config_path: str) -> None:
        """Copy the config JSON into the output folder for provenance."""
        dst = os.path.join(self.output_folder_path, "config.json")
        shutil.copy2(config_path, dst)

    def _save_ct_scan_copy(self) -> None: 
        """Copy the CT input into the output folder for provenance/debugging."""
        dst_name = os.path.basename(self.ct_input)
        dst = os.path.join(self.output_folder_path, dst_name)
        if os.path.isdir(self.ct_input):
            if not os.path.exists(dst):
                shutil.copytree(self.ct_input, dst)
        else:
            if not os.path.exists(dst):
                shutil.copy2(self.ct_input, dst)

        self.ct_saved_copy_path = dst
        self.ct_input_identity = ensure_ct_matches_saved_copy(
            current_input_path=self.ct_input,
            saved_copy_path=dst,
            output_folder_path=self.output_folder_path,
            input_type=self.ct_input_type,
        )

    def _log_setup(self) -> logging.Logger:
        """
        Configure a per-CT log file handler.

        Writes to: <output_folder_path>/logging_file_CT_<ct_index>.log
        Avoids adding duplicate handlers if the pipeline is constructed multiple times.
        """
        log_path = os.path.join(self.output_folder_path, f"logging_file_CT_{self.ct_index}.log")
        logger = self.logger

        if not any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path
            for h in logger.handlers
        ):
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(fh)

        if self.startup_banner_lines:
            for line in self.startup_banner_lines:
                logger.info(line)

        logger.info("----Log started----")
        logger.info("Launched via: %s", self.launched_via)
        logger.info("Config file:  %s", self.config_path)
        logger.info("Output folder: %s", self.output_folder_path)
        return logger

    def _config_setup(self, config_path: str) -> None:
        """
        Load config from disk and prepare output folder + phase subdirectory paths.

        Parameters
        ----------
        config_path : str
            Path to JSON config (may include // comments, stripped via json_minify).

        Raises
        ------
        FileNotFoundError
            If the config file or CT input path does not exist.
        ValueError
            If the CT input file is not a supported NIfTI extension.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            self.config = json.loads(json_minify(f.read()))

        output_folder_title = f"{self.config['output_folder_title']}_CT_{self.ct_index}"
        self.output_folder_path = os.path.join(self.current_dir_path, output_folder_title)
        os.makedirs(self.output_folder_path, exist_ok=True)
        self.metadata_dir_path = str(ensure_metadata_dir(self.output_folder_path))

        if os.path.isfile(self.ct_input):
            ct_lower = self.ct_input.lower()
            if not (ct_lower.endswith(".nii") or ct_lower.endswith(".nii.gz")):
                raise ValueError(f"CT input file must be .nii or .nii.gz, got: {self.ct_input}")
            self.ct_input_type = "nii"
        elif os.path.isdir(self.ct_input):
            self.ct_input_type = "dicom"
        else:
            raise FileNotFoundError(f"CT input not found: {self.ct_input}")

        # Inject developer-controlled paths that are no longer stored in user configs.
        self.config = inject_pipeline_paths(
            self.config,
            repo_root=self.current_dir_path,
            include_input_paths=True,
        )

        if self.save_config:
            self._save_config_copy(config_path)

        self._save_ct_scan_copy()

        # Run synthetic lesions automatically when specs are defined in config.
        lesion_specs = self.config["phase_1"]["synthetic_lesions_stage"].get("specs")
        self.run_synthetic_lesions = bool(lesion_specs)

        # Create phase subdirs
        phases = ["phase_1", "phase_2", "phase_3"]                                     
        self.sub_dir_paths: Dict[str, str] = {}
        self.sub_dir_names: Dict[str, str] = {}
        for phase in phases:
            sub_dir_path = os.path.join(self.output_folder_path, self.config[phase]["sub_dir_name"])
            os.makedirs(sub_dir_path, exist_ok=True)
            self.sub_dir_paths[phase] = sub_dir_path
            self.sub_dir_names[phase] = self.config[phase]["sub_dir_name"]

    def _context_setup(self) -> None:
        """
        Create and populate the shared Context object.

        Sets runtime metadata, config snapshot, subdir paths, and the
        downstream_roi_subset that flows through all phases.
        """
        context = Context(logger=self.logger)
        self.context = context

        context.config = copy.deepcopy(self.config)
        context.subdir_paths = copy.deepcopy(self.sub_dir_paths)
        context.subdir_names = copy.deepcopy(self.sub_dir_names)

        context.mode = self.mode
        context.ct_input_path = self.ct_input
        context.ct_input_type = self.ct_input_type
        context.ct_input_identity = copy.deepcopy(self.ct_input_identity)
        context.ct_saved_copy_path = self.ct_saved_copy_path
        context.ct_index = self.ct_index
        context.output_folder_path = self.output_folder_path
        context.metadata_dir = self.metadata_dir_path
        context.synthetic_lesions_enabled = self.run_synthetic_lesions
        context.run_spect = self.run_spect           
        context.run_dosimetry = self.run_dosimetry   
        context.run_postprocess = self.run_postprocess 

        # downstream_roi_subset flows from phase-1 config into all downstream stages.
        # 'remaining_body' (the body outline minus other ROIs) is always auto-added
        # here so downstream stages never require the user to list it explicitly.
        roi_subset = self.config["phase_1"]["segmentation_stage"]["roi_subset"]
        if isinstance(roi_subset, str):
            roi_subset = [roi_subset]
        normalized_roi_subset = [str(r).strip() for r in roi_subset if str(r).strip()]
        if "remaining_body" not in normalized_roi_subset:
            normalized_roi_subset.append("remaining_body")
        context.downstream_roi_subset = normalized_roi_subset

        self.logger.debug("Context initialized for CT_%s", self.ct_index)

    def _read_patient_hw(self) -> Dict[str, Any]:
        """
        Extract PatientSize (m) and PatientWeight (kg) from DICOM headers.

        Returns dict with keys: height_m, weight_kg, missing, source.
        """
        if self.ct_input_type != "dicom":
            return {"height_m": None, "weight_kg": None,
                    "missing": ["height", "weight"], "source": "not_dicom"}

        try:
            height, weight = extract_height_weight(self.ct_input)
        except Exception:
            return {"height_m": None, "weight_kg": None,
                    "missing": ["height", "weight"], "source": "no_files"}

        missing = (["height"] if height is None else []) + (["weight"] if weight is None else [])
        return {"height_m": height, "weight_kg": weight,
                "missing": missing, "source": "dicom"}

    def _patient_banner(self) -> None:
        """
        Print a per-patient summary banner to stdout and append it to the log.

        Shows patient index/name, height/weight, and a config summary covering
        every phase that will run.  Printed in both PRODUCTION and DEBUG modes
        (DEBUG adds ANSI colour).
        """
        cfg = self.config
        avail = os.cpu_count() or 1
        w = 68
        lines: List[str] = []

        def add(text: str = "") -> None:
            lines.append(text)

        ct_name = os.path.basename(self.ct_input.rstrip("/\\"))
        ct_type_str = "DICOM directory" if self.ct_input_type == "dicom" else "NIfTI"

        border = "═" * w
        add(f"╔{border}╗")
        add(f"║  {'Patient  CT_' + str(self.ct_index) + '  ·  ' + ct_name:<{w - 4}}  ║")
        add(f"╚{border}╝")

        add(f"  Input      : {self.ct_input}  ({ct_type_str})")
        add(f"  Output     : {self.output_folder_path}")

        # ── Height / weight ────────────────────────────────────────────
        hw = self._read_patient_hw()
        if hw["source"] == "not_dicom":
            add("  Height/Wt  : not available  (NIfTI input — no DICOM tags)")
        elif hw["source"] in ("no_files", "pydicom_missing"):
            add("  Height/Wt  : not available  (could not read DICOM files)")
        else:
            h_str = f"{hw['height_m']:.2f} m" if hw["height_m"] is not None else "not found"
            w_str = f"{hw['weight_kg']:.1f} kg" if hw["weight_kg"] is not None else "not found"
            add(f"  Height     : {h_str:<14}  Weight : {w_str}")
            if hw["missing"]:
                add(f"  Missing    : {', '.join(hw['missing'])}  (tag absent in DICOM headers)")

        add("  " + "─" * (w - 2))

        # ── Phase 1 ────────────────────────────────────────────────────
        seg_cfg   = cfg.get("phase_1", {}).get("segmentation_stage", {})
        pbpk_cfg  = cfg.get("phase_1", {}).get("pbpk_tac_stage", {})
        les_cfg   = cfg.get("phase_1", {}).get("synthetic_lesions_stage", {})

        roi_list = seg_cfg.get("roi_subset", [])
        roi_str  = ", ".join(roi_list) if roi_list else "—"
        add(f"  Phase 1  ·  Digital Twin & Ground Truth")
        add(f"    ROI subset   : {roi_str}")
        add(f"    PBPK model   : {pbpk_cfg.get('model_type', '?')}  ·  isotope : {pbpk_cfg.get('isotope', '?')}")

        if self.run_synthetic_lesions:
            specs = les_cfg.get("specs") or {}
            organs = ", ".join(specs.keys()) if specs else "none"
            add(f"    Syn. lesions : ON  ({organs})")
        else:
            add(f"    Syn. lesions : OFF")

        # ── Phase 2 ────────────────────────────────────────────────────
        add(f"  Phase 2  ·  Simulations")
        if self.run_spect:
            s = cfg.get("phase_2", {}).get("simind_stage", {})
            nc   = s.get("num_cpu", 0)
            eff  = avail if nc == 0 else min(nc, avail)
            cpu_note = f"all {avail}" if nc == 0 else str(eff)
            add(f"    SPECT        : collimator={s.get('Collimator','?')}  "
                f"photons={s.get('NumPhotons','?')}  "
                f"proj={s.get('NumProjections','?')}  "
                f"CPUs={cpu_note}")
            add(f"                   ROIs : {', '.join(s.get('roi_subset', roi_list))}")
        else:
            add(f"    SPECT        : skipped")

        if self.run_dosimetry:
            g  = cfg.get("phase_2", {}).get("opengate_stage", {})
            gc = g.get("gate", {})
            nc2  = gc.get("num_cpu", 0)
            eff2 = avail if nc2 == 0 else min(nc2, avail)
            cpu_note2 = f"all {avail}" if nc2 == 0 else str(eff2)
            add(f"    Dosimetry    : histories={gc.get('total_histories','?')}  "
                f"CPUs={cpu_note2}  "
                f"seed={gc.get('random_seed','?')}")
            add(f"                   ROIs : {', '.join(g.get('roi_subset', roi_list))}")
        else:
            add(f"    Dosimetry    : skipped")

        # ── Phase 3 ────────────────────────────────────────────────────
        add(f"  Phase 3  ·  Post-Processing")
        if self.run_postprocess and self.run_spect:
            sp = cfg.get("phase_3", {}).get("spect_postprocess_stage", {})
            add(f"    SPECT recon  : {sp.get('ReconstructionAlgorithm','OSEM')}  "
                f"iter={sp.get('Iterations','?')}  "
                f"subsets={sp.get('Subsets','?')}  "
                f"noise={'ON' if sp.get('apply_poisson_noise') else 'OFF'}")
        elif self.run_postprocess:
            add(f"    SPECT recon  : skipped  (no SPECT simulation)")
        else:
            add(f"    SPECT recon  : skipped")

        if self.run_postprocess and self.run_dosimetry:
            dp = cfg.get("phase_3", {}).get("dosemap_postprocess_stage", {})
            add(f"    Dose map     : TAC-weighted  "
                f"apply_tac={'ON' if dp.get('apply_tac', True) else 'OFF'}")
        elif self.run_postprocess:
            add(f"    Dose map     : skipped  (no dosimetry simulation)")
        else:
            add(f"    Dose map     : skipped")

        add("  " + "─" * (w - 2))

        # Print with colour in DEBUG mode, plain in PRODUCTION
        if self.mode == "DEBUG":
            print(f"{_BOLD}{_CYAN}", end="")
        for line in lines:
            print(line)
        if self.mode == "DEBUG":
            print(_RESET, end="")

        # Always write the plain text to the log
        for line in lines:
            self.logger.info(line)

    def _phase_banner(self, phase_num: int, phase_name: str) -> None:
        """Print a phase banner with optional color in DEBUG mode.""" 
        banner = f"-----------------------------Phase {phase_num}: {phase_name}-----------------------------" 
        if self.mode == "DEBUG": 
            print(f"{_BOLD}{_YELLOW}{banner}{_RESET}") 
        else: 
            print(banner) 

    def _cleanup_work_dir(self, work_dir: str) -> None:
        """Remove a work directory in PRODUCTION mode to save disk space."""
        if self.mode == "PRODUCTION" and work_dir and os.path.exists(work_dir):
            work_dir_abs = os.path.abspath(work_dir)
            shutil.rmtree(work_dir_abs, ignore_errors=True)
        elif self.mode == "DEBUG":
            _debug_print(f"Skipping cleanup of work_dir (DEBUG mode): {work_dir}", "CLEANUP", self.logger)

    def run(self) -> Context:
        """
        Execute all pipeline stages sequentially for this CT input.

        Phases
        ------
        1. Digital Twin & Ground Truth:
            1.1  TotalSegmentator segmentation + ROI unification to the shared label space
            1.2  Synthetic lesion generation (optional)
            1.3  PBPK TAC generation
        2. Simulations:
            2.1  SIMIND preprocessing + Monte Carlo projection simulation (optional, --spect)
            2.2  OpenGATE Monte Carlo dosimetry simulation (optional, --dosimetry)
        3. Post-Processing:
            3.1  SPECT post-processing (optional, --postprocess with --spect)
            3.2  Dosimetry post-processing (optional, --postprocess with --dosimetry)

        Returns
        -------
        Context
            The updated context after all stages complete.
        """
        logger = self.logger
        t_pipeline = time.perf_counter()
        profiler = StageProfiler() if self.profile else None

        self._patient_banner()

        if self.mode == "DEBUG":
            _debug_print(f"run_spect={self.run_spect} | run_dosimetry={self.run_dosimetry} | run_postprocess={self.run_postprocess}", "INIT", logger)
            _debug_print(f"synthetic_lesions={self.run_synthetic_lesions}", "INIT", logger)

        context = self.context

        logger.info("Pipeline start | mode=%s", self.mode)
        logger.info("CT input | path=%s | type=%s", self.ct_input, self.ct_input_type)

        # ── Phase 1: Digital Twin & Ground Truth ──────────────────────────────────
        self._phase_banner(1, "Digital Twin & Ground Truth")

        t_stage = time.perf_counter()
        print("Running Segmentation + ROI Unification Stage...")
        logger.info("Stage start: Segmentation + ROI Unification")
        if profiler: profiler.start_stage("segmentation")
        context = SegmentationStage(context).run()
        if profiler: profiler.end_stage()
        print("Segmentation + ROI Unification Stage completed.")
        logger.info("Stage end: Segmentation + ROI Unification | elapsed=%.2fs", time.perf_counter() - t_stage)

        if self.run_synthetic_lesions:
            t_stage = time.perf_counter()
            print("Running Synthetic Lesions Generation Stage...")
            logger.info("Stage start: Synthetic Lesions Generation")
            if profiler: profiler.start_stage("synthetic_lesions")
            context = SyntheticLesionsStage(context).run()
            if profiler: profiler.end_stage()
            print("Synthetic Lesions Generation Stage completed.")
            logger.info("Stage end: Synthetic Lesions Generation | elapsed=%.2fs", time.perf_counter() - t_stage)

        t_stage = time.perf_counter()
        print("Running PBPK TAC Generation Stage...")
        logger.info("Stage start: PBPK TAC Generation")
        if profiler: profiler.start_stage("pbpk_tac")
        context = PbpkTacStage(context).run()
        if profiler: profiler.end_stage()
        print("PBPK TAC Generation Stage completed.")
        logger.info("Stage end: PBPK TAC Generation | elapsed=%.2fs", time.perf_counter() - t_stage)

        # ── Phase 2: Simulations ───────────────────────────────────────────────
        if self.run_spect or self.run_dosimetry:
            self._phase_banner(2, "Simulations")
        else:
            print("Phase 2: Simulations (skipped — no --spect or --dosimetry)")
            logger.info("Phase 2 skipped: no --spect or --dosimetry flags")

        if self.run_spect:
            t_stage = time.perf_counter()
            print("Running SIMIND Simulation Stage...")
            logger.info("Stage start: SIMIND Simulation")
            if profiler: profiler.start_stage("simind")
            context = SimindSimulationStage(context).run()
            if profiler: profiler.end_stage()
            print("SIMIND Simulation Stage completed.")
            logger.info("Stage end: SIMIND Simulation | elapsed=%.2fs", time.perf_counter() - t_stage)

        if self.run_dosimetry:
            t_stage = time.perf_counter()
            print("Running OpenGATE Simulation Stage...")
            logger.info("Stage start: OpenGATE Simulation")
            if profiler: profiler.start_stage("opengate")
            context = OpenGateSimulationStage(context).run()
            if profiler: profiler.end_stage()
            print("OpenGATE Simulation Stage completed.")
            logger.info("Stage end: OpenGATE Simulation | elapsed=%.2fs", time.perf_counter() - t_stage)

        # ── Phase 3: Post-Processing ───────────────────────────────────────────
        if self.run_postprocess and (self.run_spect or self.run_dosimetry):
            self._phase_banner(3, "Post-Processing")
        elif self.run_postprocess:
            print("Phase 3: Post-Processing (skipped — no simulations ran)")
            logger.info("Phase 3 skipped: --postprocess set but no simulations ran")
        else:
            print("Phase 3: Post-Processing (skipped — no --postprocess)")
            logger.info("Phase 3 skipped: no --postprocess flag")

        if self.run_postprocess and self.run_spect:
            t_stage = time.perf_counter()
            print("Running SPECT Post-Processing Stage...")
            logger.info("Stage start: SPECT Post-Processing")
            if profiler: profiler.start_stage("spect_postprocess")
            context = SpectPostprocessStage(context).run()
            if profiler: profiler.end_stage()
            print("SPECT Post-Processing Stage completed.")
            logger.info("Stage end: SPECT Post-Processing | elapsed=%.2fs", time.perf_counter() - t_stage)

        if self.run_postprocess and self.run_dosimetry:
            t_stage = time.perf_counter()
            print("Running Dosimetry Post-Processing Stage...")
            logger.info("Stage start: Dosimetry Post-Processing")
            if profiler: profiler.start_stage("dosemap_postprocess")
            context = DosemapPostprocessStage(context).run()
            if profiler: profiler.end_stage()
            print("Dosimetry Post-Processing Stage completed.")
            logger.info("Stage end: Dosimetry Post-Processing | elapsed=%.2fs", time.perf_counter() - t_stage)

        # ── PRODUCTION cleanup ─────────────────────────────────────────────────
        if self.mode == "PRODUCTION":
            simind_work = getattr(context, "simind_work_dir", None)
            if simind_work and os.path.exists(simind_work):
                self._cleanup_work_dir(simind_work)

        pipeline_elapsed = time.perf_counter() - t_pipeline
        logger.info("Pipeline end | total_elapsed=%.2fs", pipeline_elapsed)

        if profiler:
            profile_path = os.path.join(
                self.output_folder_path, f"profiling_CT_{self.ct_index}.json"
            )
            profiler.save(profile_path, pipeline_elapsed, self.config)
            print(f"Profiling data saved to: {profile_path}")

        print("VTT pipeline completed successfully.")
        return context


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the VTT pipeline."""
    parser = argparse.ArgumentParser(
        description="Virtual Theranostic Trials pipeline runner"
    )

    parser.add_argument("--config_file", required=True, type=str,
                        help="Path to JSON config file.")

    ct_group = parser.add_mutually_exclusive_group(required=True)
    ct_group.add_argument("--input_ct_dir", type=str,
                          help="Directory containing CT inputs (NIfTI files or DICOM folders).")
    ct_group.add_argument("--input_ct", type=str,
                          help="Direct path to a single CT input (NIfTI file or DICOM directory).")

    parser.add_argument(
        "--mode",
        default="PRODUCTION",
        choices=["DEBUG", "PRODUCTION"],
        help="Pipeline mode. DEBUG keeps more intermediate files. Default: PRODUCTION.",
    )
    parser.add_argument(
        "--logging_on",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-CT log file writing. Default: enabled.",
    )
    parser.add_argument(
        "--ct_index_start",
        type=int,
        default=1,
        help="Starting CT index for output folder naming (e.g. 1 → _CT_1, _CT_2, …). Default: 1.",
    )
    parser.add_argument(
        "--save_config",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy the config JSON into each CT output folder. Default: disabled.",
    )
    parser.add_argument(
        "--spect",
        action=argparse.BooleanOptionalAction, 
        default=False, 
        help="Run SIMIND SPECT projection simulation in Phase 2. Default: disabled.", 
    ) 
    parser.add_argument( 
        "--dosimetry", 
        action=argparse.BooleanOptionalAction, 
        default=False, 
        help="Run OpenGATE dosimetry simulation in Phase 2. Default: disabled.", 
    ) 
    parser.add_argument( 
        "--postprocess", 
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run post-processing in Phase 3 for whichever simulations ran. Default: disabled.",
    )
    parser.add_argument(
        "--launched_via",
        default="cli",
        choices=["cli", "web_ui"],
        help="How the pipeline was invoked — recorded in the log file. Default: cli.",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Sample CPU %% and RAM every 2 s per stage and write "
            "profiling_CT_<index>.json into each CT output folder. "
            "Requires psutil. Default: disabled."
        ),
    )

    return parser


def _print_startup_banner(
    args: Any,
    items: List[str],
    cfg: Dict[str, Any],
) -> List[str]:
    """Print a formatted startup banner to stdout and return the lines for per-patient logs."""
    import datetime

    avail = os.cpu_count() or 1
    w = 68
    border = "═" * w
    lines: List[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add(f"╔{border}╗")
    add(f"║  {'Virtual Theranostic Trials  —  VTT Pipeline'.center(w - 4)}  ║")
    add(f"║  {'BC Cancer  ·  QURIT Lab'.center(w - 4)}  ║")
    add(f"╚{border}╝")
    add()
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    add(f"  Started   : {now}")
    add(f"  Mode      : {args.mode}")
    add(f"  CT inputs : {len(items)} item(s)")
    add()

    # Phases to run
    phase1_tasks = ["segmentation", "PBPK"]
    if cfg.get("phase_1", {}).get("synthetic_lesions_stage", {}).get("specs"):
        phase1_tasks.append("synthetic lesions")
    add("  Phases to run:")
    add(f"    ✓  Phase 1  —  {', '.join(phase1_tasks)}")
    if args.spect:
        nc = cfg.get("phase_2", {}).get("simind_stage", {}).get("num_cpu", 1)
        eff = avail if nc == 0 else min(nc, avail)
        using = f"using all {avail}" if nc == 0 else str(eff)
        add(f"    ✓  Phase 2  —  SIMIND SPECT  ({using} core(s))")
    if args.dosimetry:
        gate_cfg = cfg.get("phase_2", {}).get("opengate_stage", {}).get("gate", {})
        nc2 = gate_cfg.get("num_cpu", 1)
        eff2 = avail if nc2 == 0 else min(nc2, avail)
        using2 = f"using all {avail}" if nc2 == 0 else str(eff2)
        add(f"    ✓  Phase 2  —  OpenGATE dosimetry  ({using2} thread(s))")
    if args.postprocess:
        sims = []
        if args.spect:     sims.append("SPECT")
        if args.dosimetry: sims.append("dose map")
        add(f"    ✓  Phase 3  —  Post-processing  ({', '.join(sims) if sims else 'no simulations'})")
    add()

    # CPU summary
    add(f"  CPU available : {avail} core(s)")
    if args.spect:
        nc = cfg.get("phase_2", {}).get("simind_stage", {}).get("num_cpu", 1)
        eff_nc = avail if nc == 0 else min(nc, avail)
        note = f"num_cpu={nc} → using all {avail}" if nc == 0 else f"num_cpu={nc}"
        add(f"  SIMIND        : {eff_nc} core(s)  ({note})")
    if args.dosimetry:
        gate_cfg2 = cfg.get("phase_2", {}).get("opengate_stage", {}).get("gate", {})
        nc2 = gate_cfg2.get("num_cpu", 1)
        eff_nc2 = avail if nc2 == 0 else min(nc2, avail)
        note2 = f"num_cpu={nc2} → using all {avail}" if nc2 == 0 else f"num_cpu={nc2}"
        add(f"  OpenGATE      : {eff_nc2} thread(s)  ({note2})")
    add()
    add("  Existing stage outputs are reused only when CT/config metadata still match.")
    add()
    add("─" * (w + 2))
    add()

    # Print to stdout
    for line in lines:
        print(line)

    return lines


def main() -> int:
    """
    CLI entrypoint. Iterates through all CT inputs and runs the pipeline.

    Accepts either --input_ct_dir (directory of CT items) or --input_ct (single CT path).

    Returns
    -------
    int
        Process exit code (0 = all CTs processed; individual failures logged but don't stop batch).
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    # Resolve CT input items from either --input_ct or --input_ct_dir.
    if args.input_ct:
        ct_path = os.path.abspath(args.input_ct)
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"--input_ct path not found: {ct_path}")
        ct_inputs_dir = os.path.dirname(ct_path)
        items = [os.path.basename(ct_path)]
    else:
        ct_inputs_dir = os.path.abspath(args.input_ct_dir)
        if not os.path.isdir(ct_inputs_dir):
            raise NotADirectoryError(f"input_ct_dir must be a directory: {ct_inputs_dir}")
        items = [n for n in sorted(os.listdir(ct_inputs_dir)) if not n.startswith(".")]

    # Pre-flight: validate config before touching any CT.
    # Inject computed paths first so validation sees the real values.
    try:
        with open(args.config_file, encoding="utf-8") as _f:
            _cfg_raw = json.loads(json_minify(_f.read()))

        _cfg_raw = inject_pipeline_paths(
            _cfg_raw,
            repo_root=os.path.abspath(os.path.dirname(__file__)),
            include_input_paths=True,
        )

        validate_config(
            _cfg_raw,
            run_spect=args.spect,
            run_dosimetry=args.dosimetry,
            run_postprocess=args.postprocess,
        )
    except ValueError as _ve:
        print(f"\n[ERROR] {_ve}\n")
        return 1
    except FileNotFoundError as _fnf:
        print(f"\n[ERROR] Config file not found: {_fnf}\n")
        return 1

    _banner_lines = _print_startup_banner(args, items, _cfg_raw)

    any_failed = False
    for idx, name in enumerate(items, start=args.ct_index_start):
        ct_path = os.path.join(ct_inputs_dir, name)

        try:
            pipeline = TdtPipeline(
                config_path=args.config_file,
                ct_input=ct_path,
                ct_index=idx,
                logging_on=args.logging_on,
                save_config=args.save_config,
                mode=args.mode,
                run_spect=args.spect,
                run_dosimetry=args.dosimetry,
                run_postprocess=args.postprocess,
                launched_via=args.launched_via,
                startup_banner_lines=_banner_lines,
                profile=args.profile,
            )
            pipeline.run()
        except Exception:
            import traceback
            print(f"[ERROR] CT index {idx} failed for input: {ct_path}")
            traceback.print_exc()
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


# quick run: python main.py --config_file inputs/my_config.json --input_ct_dir inputs/ct_testing --mode DEBUG --logging_on --save_config --spect --dosimetry --postprocess
